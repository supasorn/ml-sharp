"""Contains `sharp bake-layers` CLI implementation.

Bakes a SHARP splat file into textured depth meshes, saved as
`<scene>_layers.glb`, for fast rendering on standalone headsets: opaque
meshes need no sorting or blending, and early-z rejects hidden pixels.

The splats are treated as the reference: everything is built on the photo's
pixel grid from renders of the splats (gsplat), so the layers show what the
splats show, over the same range of head motion.

1. Front surface: the splats' rendered depth at each photo pixel from the
   input view. SHARP's splat grid is not the photo's pixel grid (splat centres
   drift sideways), so splat centres can't be used directly: a background
   splat can project behind a foreground object and would take its colour.
2. Edges: a grid edge is a depth discontinuity when its disparity (1/z) jump
   is large AND departs from the local slope (planes are linear in disparity,
   even at grazing angles). Tiny edge fragments are dropped as noise.
3. Front mesh: the front surface textured with the original photo. Cells
   crossing an edge are flattened to the foreground depth (not stretched
   across it), so thin structures survive, and written as a separate
   `front_edges` mesh whose per-vertex alpha fades from 1 on the foreground
   to 0 on the corners pulled in from the background.
4. Back mesh: hidden content, taken from renders of the splats around the
   input view (what the splats reveal when you move). Where no view reveals
   anything, the background is extended under the foreground, with colours
   from clean background (photo pixels right at an edge blend foreground and
   background). It is pushed slightly away from the camera so the front mesh
   wins where they coincide, and kept only near edges.

For licensing see accompanying LICENSE file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import numpy as np
from PIL import Image, ImageOps
from plyfile import PlyData
from scipy import ndimage

from sharp.utils import logging as logging_utils

LOGGER = logging.getLogger(__name__)

SH_C0 = 0.28209479177387814
# 4-neighbourhood offsets.
OFFSETS = ((0, 1), (0, -1), (1, 0), (-1, 0))


def shift(arr: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    """out[i, j] = arr[i + dy, j + dx], `fill` outside the grid."""
    out = np.full_like(arr, fill)
    h, w = arr.shape[:2]
    out[max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)] = \
        arr[max(dy, 0):h - max(-dy, 0), max(dx, 0):w - max(-dx, 0)]
    return out


def find_edges(z: np.ndarray, jump_tol: float, slope_tol: float, min_size: int):
    """Depth discontinuities between grid neighbours.

    Returns (bh (H,W-1), bv (H-1,W)) masks for edges between horizontal and
    vertical neighbours. Detection runs on a 3x3 median-filtered depth so
    per-pixel noise doesn't read as edges, and connected edge fragments with
    fewer than `min_size` pixels are dropped.
    """
    d = 1 / np.maximum(ndimage.median_filter(z, 3), 1e-6)

    def along_cols(dd):
        jump = np.diff(dd, axis=1)
        before = np.concatenate([jump[:, :1], jump[:, :-1]], 1)
        after = np.concatenate([jump[:, 1:], jump[:, -1:]], 1)
        residual = np.minimum(np.abs(jump - before), np.abs(jump - after))
        scale = np.minimum(dd[:, :-1], dd[:, 1:])
        return (np.abs(jump) > jump_tol * scale) & (residual > slope_tol * scale)

    bh, bv = along_cols(d), along_cols(d.T).T
    touched = edge_pixels(bh, bv)
    labels, _ = ndimage.label(touched, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())
    keep = sizes[labels] >= min_size
    return bh & keep[:, :-1] & keep[:, 1:], bv & keep[:-1] & keep[1:]


def edge_pixels(bh: np.ndarray, bv: np.ndarray) -> np.ndarray:
    """(H,W) mask of pixels on either side of an edge."""
    m = np.zeros((bh.shape[0], bv.shape[1]), dtype=bool)
    m[:, :-1] |= bh
    m[:, 1:] |= bh
    m[:-1] |= bv
    m[1:] |= bv
    return m


def edge_sides(bh, bv, z):
    """(near, far) masks of pixels on the nearer/farther side of an edge."""
    near = np.zeros(z.shape, dtype=bool)
    far = np.zeros(z.shape, dtype=bool)
    left_near = z[:, :-1] < z[:, 1:]
    near[:, :-1] |= bh & left_near
    far[:, 1:] |= bh & left_near
    near[:, 1:] |= bh & ~left_near
    far[:, :-1] |= bh & ~left_near
    top_near = z[:-1] < z[1:]
    near[:-1] |= bv & top_near
    far[1:] |= bv & top_near
    near[1:] |= bv & ~top_near
    far[:-1] |= bv & ~top_near
    return near, far


def crossing_edge(bh, bv, dy, dx) -> np.ndarray:
    """(H,W) mask: does the step from each pixel to its (dy,dx) neighbour cross an edge."""
    m = np.zeros((bh.shape[0], bv.shape[1]), dtype=bool)
    if dx == 1:
        m[:, :-1] = bh
    elif dx == -1:
        m[:, 1:] = bh
    elif dy == 1:
        m[:-1] = bv
    else:
        m[1:] = bv
    return m


def inpaint_band(color, z, band, bh, bv, depth_tol, iters):
    """Fill `band` pixels' colours from non-band neighbours at similar depth, without crossing edges."""
    color = color.copy()
    known = ~band
    for _ in range(iters):
        todo = ~known
        if not todo.any():
            break
        acc = np.zeros_like(color)
        cnt = np.zeros(z.shape)
        for dy, dx in OFFSETS:
            nk = shift(known, dy, dx, False)
            nz = shift(z, dy, dx, np.inf)
            ok = todo & nk & ~crossing_edge(bh, bv, dy, dx) & (np.abs(nz - z) < depth_tol * z)
            acc += np.where(ok[..., None], shift(color, dy, dx, 0.0), 0)
            cnt += ok
        got = cnt > 0
        color[got] = acc[got] / cnt[got][:, None]
        known |= got
    return color


def extend_background(ray, bz, bcol, sources, fillable, f_grid, max_move, gap_tol, max_px):
    """Grow background under the foreground, as far as a head move could reveal.

    A fillable pixel takes the depth (along its own ray) and colour of the
    farthest source/filled neighbour that is clearly behind it, while the
    distance travelled stays within f * max_move * (disparity difference).
    """
    fz = bz.copy()
    known = sources.copy()
    dist = np.zeros(bz.shape)
    filled = np.zeros(bz.shape, dtype=bool)
    for _ in range(max_px):
        best_z = np.zeros(bz.shape)
        best_col = np.zeros_like(bcol)
        best_dist = np.zeros(bz.shape)
        for dy, dx in OFFSETS:
            nz = shift(bz, dy, dx, 0.0)
            take = shift(known, dy, dx, False) & (nz > best_z)
            best_z = np.where(take, nz, best_z)
            best_col = np.where(take[..., None], shift(bcol, dy, dx, 0.0), best_col)
            best_dist = np.where(take, shift(dist, dy, dx, 0.0) + 1, best_dist)
        budget = f_grid * max_move * (1 / fz - 1 / np.maximum(best_z, 1e-6)) + 2
        grow = fillable & ~known & (best_z > fz * (1 + gap_tol)) & (best_dist <= budget)
        if not grow.any():
            break
        bz = np.where(grow, best_z, bz)
        bcol = np.where(grow[..., None], best_col, bcol)
        dist = np.where(grow, best_dist, dist)
        known |= grow
        filled |= grow
    return ray * bz[..., None], bcol, filled


def grid_mesh(xyz, valid, bh, bv, stride, flatten, depth_tol=0.05):
    """Triangulate a (H,W,3) grid of points.

    Cells touching invalid points are dropped. Cells crossing an edge are not
    stretched across it: they get their own corners, moved along their rays
    to the nearest (`flatten="near"`) or farthest (`"far"`) corner depth, so
    they still cover exactly their pixels (thin structures survive) without
    spanning the discontinuity.

    Returns vertices (N,3), faces (F,3), the (row, col) grid coordinates
    (N,2) of each vertex, the index of the first edge-cell face (edge faces
    and their vertices come last), and per-vertex alpha (N,): 1 for corners
    that were already at the flattened depth, 0 for corners moved onto it
    from across the edge (so edges can fade out softly).
    """
    s = stride
    H, W = valid.shape
    rows, cols = np.arange(0, H, s), np.arange(0, W, s)
    h, w = len(rows), len(cols)
    # A coarse cell spans full-res rows r..r+s and cols c..c+s; it's cut if
    # any full-res edge lies inside that span (summed-area tables).
    def block_any(mask, rspan, cspan):
        sat = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        r0 = rows[:-1, None]; c0 = cols[None, :-1]
        r1 = np.minimum(r0 + rspan, mask.shape[0]); c1 = np.minimum(c0 + cspan, mask.shape[1])
        return (sat[r1, c1] - sat[r0, c1] - sat[r1, c0] + sat[r0, c0]) > 0
    cell_cut = block_any(bh, s + 1, s) | block_any(bv, s, s + 1)
    sub_xyz = xyz[::s, ::s]
    ok = valid[::s, ::s].reshape(-1) & (sub_xyz[..., 2].reshape(-1) > 0)
    idx = np.arange(h * w).reshape(h, w)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    corners = np.stack([a, b, c, d], -1).reshape(-1, 4)
    cell_valid = ok[corners].all(1)
    cut = cell_cut.reshape(-1)

    # Continuous cells share grid vertices.
    plain = corners[cell_valid & ~cut]
    faces = np.concatenate([plain[:, [0, 2, 1]], plain[:, [1, 2, 3]]], 0)
    used = np.unique(faces)
    remap = np.full(h * w, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    pts = sub_xyz.reshape(-1, 3)
    verts = [pts[used]]
    rc = [np.stack(np.divmod(used, w), 1)]
    faces = [remap[faces]]
    alpha = [np.ones(len(used))]
    first_edge_face = len(faces[0])

    # Edge cells: own corners, flattened along their rays.
    edge = corners[cell_valid & cut]
    if len(edge):
        cz = pts[edge][..., 2]
        target = cz.min(1) if flatten == "near" else cz.max(1)
        cell_pts = pts[edge] * (target[:, None] / cz)[..., None]
        base = len(verts[0])
        local = base + np.arange(len(edge))[:, None] * 4
        faces.append(np.concatenate([local + [0, 2, 1], local + [1, 2, 3]], 0))
        verts.append(cell_pts.reshape(-1, 3))
        rc.append(np.stack(np.divmod(edge.reshape(-1), w), 1))
        on_surface = np.abs(cz - target[:, None]) <= depth_tol * target[:, None]
        alpha.append(on_surface.reshape(-1).astype(np.float64))
    return (np.concatenate(verts), np.concatenate(faces), np.concatenate(rc) * s,
            first_edge_face, np.concatenate(alpha))


def load_gsplat():
    """Import gsplat, working around this machine's toolchain.

    The cached gsplat CUDA extension was built with the system CUDA 13 toolkit
    and links libcudart.so.13, which PyTorch (cu128) doesn't ship; load it
    globally first. JIT builds also need the env's ninja on PATH.
    """
    import ctypes
    import os
    import sys

    cudart = "/usr/local/cuda/lib64/libcudart.so.13"
    if os.path.exists(cudart):
        try:
            ctypes.CDLL(cudart, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    import gsplat

    return gsplat


def splat_tensors(v, device):
    """SHARP ply vertex data -> gsplat rasterization kwargs (colours as displayed by the viewer)."""
    import torch

    t = lambda *keys: torch.tensor(np.stack([v[k] for k in keys], -1), device=device, dtype=torch.float32)
    return dict(
        means=t("x", "y", "z"),
        quats=torch.nn.functional.normalize(t("rot_0", "rot_1", "rot_2", "rot_3"), dim=-1),
        scales=torch.exp(t("scale_0", "scale_1", "scale_2")),
        opacities=torch.sigmoid(t("opacity")[:, 0]),
        colors=(0.5 + SH_C0 * t("f_dc_0", "f_dc_1", "f_dc_2")).clamp(0, 1),
    )


def image_rays(meta: dict, width: int, margin: float):
    """Pixel grid over the photo plus `margin` (fraction) on every side.

    Returns (W, H, f, rays (H,W,3) with z = 1, (x0, y0) offset of the photo
    inside the grid, (Wc, Hc) photo size in grid pixels).
    """
    Wc = width
    Hc = int(round(Wc * meta["height"] / meta["width"]))
    x0, y0 = int(round(margin * Wc)), int(round(margin * Hc))
    W, H = Wc + 2 * x0, Hc + 2 * y0
    f = meta["focal_length_px"] * Wc / meta["width"]
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.stack([(u - W / 2) / f, (v - H / 2) / f, np.ones_like(u)], -1)
    return W, H, f, rays, (x0, y0), (Wc, Hc)


def render_splats(gsplat, splats, eye, W, H, f):
    """Render splats from `eye` (camera-space translation): rgb (H,W,3), expected depth (H,W), alpha (H,W)."""
    import torch

    dev = splats["means"].device
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=dev, dtype=torch.float32)[None]
    view = torch.eye(4, device=dev)
    view[:3, 3] = -torch.tensor(eye, device=dev, dtype=torch.float32)
    out, alpha, _ = gsplat.rendering.rasterization(
        **splats, viewmats=view[None], Ks=K, width=W, height=H, render_mode="RGB+ED")
    return out[0, ..., :3], out[0, ..., 3], alpha[0, ..., 0]


def blended_depth_mask(depth, tol):
    """Torch (H,W) mask of pixels whose rendered depth may be a blend across an edge."""
    import torch

    disp = 1 / depth.clamp(min=1e-6)
    jump = torch.zeros_like(disp, dtype=torch.bool)
    for dim in (0, 1):
        n = disp.shape[dim]
        dj = (torch.diff(disp, dim=dim).abs()
              > tol * torch.minimum(disp.narrow(dim, 0, n - 1), disp.narrow(dim, 1, n - 1)))
        jump |= torch.nn.functional.pad(dj, [0, 1] if dim == 1 else [0, 0, 0, 1])
        jump |= torch.nn.functional.pad(dj, [1, 0] if dim == 1 else [0, 0, 1, 0])
    return torch.nn.functional.max_pool2d(jump[None, None].float(), 3, 1, 1)[0, 0] > 0


def teacher_eyes(move: float) -> list[tuple[float, float, float]]:
    """Viewpoints spanning the target baseline (camera space, metres)."""
    xs = np.linspace(-move, move, 5)
    ys = np.linspace(-move / 2, move / 2, 3)
    zs = (0.0, 0.75 * move)
    return [(x, y, z) for z in zs for y in ys for x in xs if (x, y, z) != (0, 0, 0)]


def teacher_back(gsplat, splats, fz, W, H, f, meta, args):
    """Hidden surfaces as the splats reveal them from nearby viewpoints.

    Renders the splats (RGB + expected depth) from `teacher_eyes`, drops
    semi-transparent pixels and pixels on a depth edge (their depth is a
    blend), unprojects the rest and maps them onto the reference pixel grid.
    Points clearly behind the front surface are hidden content; each grid
    pixel takes the nearest such surface (averaging samples within a small
    depth tolerance of it).

    Returns (depth (H,W), colour (H,W,3), mask (H,W)).
    """
    import torch

    dev = splats["means"].device
    Wt = args.teacher_res
    Ht = int(round(Wt * meta["height"] / meta["width"]))
    ft = meta["focal_length_px"] * Wt / meta["width"]
    u, vv = torch.meshgrid(torch.arange(Wt, device=dev) + 0.5, torch.arange(Ht, device=dev) + 0.5, indexing="xy")
    front = torch.tensor(fz, device=dev, dtype=torch.float32).reshape(-1)

    idx_all, z_all, c_all = [], [], []
    for eye in teacher_eyes(args.teacher_move):
        rgb, depth, alpha = render_splats(gsplat, splats, eye, Wt, Ht, ft)
        ok = (alpha > 0.98) & ~blended_depth_mask(depth, args.edge_tol * 0.5) & (depth > 0)
        e = torch.tensor(eye, device=dev, dtype=torch.float32)
        x = (u - Wt / 2) / ft * depth + e[0]
        y = (vv - Ht / 2) / ft * depth + e[1]
        z = depth + e[2]
        ok &= z > 1e-3
        c = torch.floor(f * x / z.clamp(min=1e-3) + W / 2).long()
        r = torch.floor(f * y / z.clamp(min=1e-3) + H / 2).long()
        ok &= (c >= 0) & (c < W) & (r >= 0) & (r < H)
        flat = (r * W + c)[ok]
        zs = z[ok]
        behind = zs > front[flat] * (1 + args.gap_tol)
        idx_all.append(flat[behind])
        z_all.append(zs[behind])
        c_all.append(rgb[ok][behind])

    idx, zs, cs = torch.cat(idx_all), torch.cat(z_all), torch.cat(c_all)
    n = W * H
    zmin = torch.full((n,), float("inf"), device=dev).scatter_reduce(0, idx, zs, "amin")
    near = zs <= zmin[idx] * (1 + args.gap_tol)
    cnt = torch.zeros(n, device=dev).index_add_(0, idx[near], torch.ones_like(zs[near]))
    zsum = torch.zeros(n, device=dev).index_add_(0, idx[near], zs[near])
    csum = torch.zeros(n, 3, device=dev).index_add_(0, idx[near], cs[near])
    has = cnt >= args.teacher_min_samples
    depth = torch.where(has, zsum / cnt.clamp(min=1), torch.zeros_like(zsum))
    color = csum / cnt.clamp(min=1)[:, None]
    return (depth.reshape(H, W).cpu().numpy(), color.reshape(H, W, 3).cpu().numpy(),
            has.reshape(H, W).cpu().numpy())


def load_photo(meta: dict, images_dir: Path, max_size: int) -> Image.Image | None:
    name = meta.get("image")
    if not name or not (images_dir / name).exists():
        return None
    img = Image.open(images_dir / name)
    upright = ImageOps.exif_transpose(img)
    # Use whichever orientation matches the size the prediction was made at.
    for candidate in (upright, img):
        if candidate.size == (meta["width"], meta["height"]):
            img = candidate
            break
    else:
        LOGGER.warning("Photo %s size %s doesn't match %sx%s; using splat colours.",
                       name, img.size, meta["width"], meta["height"])
        return None
    img = img.convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    return img


def bake(path: Path, args) -> None:
    import torch

    out_path = path.with_name(f"{path.stem}_layers.glb")
    if out_path.exists() and not args.overwrite:
        LOGGER.info("%s exists, skipping.", out_path)
        return
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        LOGGER.warning("No %s, skipping.", meta_path)
        return
    meta = json.loads(meta_path.read_text())

    gsplat = load_gsplat()
    splats = splat_tensors(PlyData.read(path)["vertex"].data, "cuda")
    W, H, f, rays, (x0, y0), (Wc, Hc) = image_rays(meta, args.grid_width, args.margin)

    # Front surface: what the splats show at each photo pixel from the input
    # view. Expected depth blends across edges, so snap with a 3x3 median.
    rgb0, depth0, alpha0 = render_splats(gsplat, splats, (0.0, 0.0, 0.0), W, H, f)
    depth0, alpha0 = depth0.cpu().numpy(), alpha0.cpu().numpy()
    covered = (alpha0 > 0.5) & (depth0 > 0)
    # Pixels with no splats take the nearest covered pixel's depth.
    nearest = ndimage.distance_transform_edt(~covered, return_distances=False, return_indices=True)
    fz = ndimage.median_filter(depth0[tuple(nearest)], 3)
    f_xyz = rays * fz[..., None]

    # Front texture: the photo, inside a margin coloured by the splat render
    # (the splats extend a little past the photo's frame).
    render_img = Image.fromarray((rgb0.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    photo = load_photo(meta, args.images_dir, args.texture_size)
    if photo is not None:
        scale = photo.size[0] / Wc
        f_tex = render_img.resize((int(round(W * scale)), int(round(H * scale))), Image.BICUBIC)
        f_tex.paste(photo, (int(round(x0 * scale)), int(round(y0 * scale))))
    else:
        f_tex = render_img
    front_color = np.asarray(f_tex.resize((W, H), Image.LANCZOS)) / 255.0

    # 1. Edges and the colour-contaminated band on their background side.
    bh, bv = find_edges(fz, args.edge_tol, args.edge_tol / 2, args.min_edge)
    near_px, far_px = edge_sides(bh, bv, fz)
    # Thin structures (poles, fronds) are on the near side of one edge and the
    # far side of another; they're foreground, so only pure far-side pixels
    # seed the background band.
    far_px &= ~near_px
    nearest_fg = ndimage.minimum_filter(np.where(near_px, fz, np.inf), size=2 * args.band + 3)
    band = (ndimage.binary_dilation(far_px, iterations=args.band) & ~near_px
            & (fz > nearest_fg * (1 + args.edge_tol)))

    # 2. Front mesh: photo-textured on the pixel grid; edge cells flattened to
    # the foreground with soft alpha.
    fv, ff, frc, f_edge0, f_alpha = grid_mesh(
        f_xyz, np.ones_like(band), bh, bv, args.stride, flatten="near", depth_tol=args.edge_tol)
    f_uv = (frc[:, ::-1] + 0.5) / [W, H]

    # 3. Back surface: what the splats reveal from nearby viewpoints; elsewhere
    # the recoloured band and background extended under the foreground.
    b_color = inpaint_band(front_color, fz, band, bh, bv, args.edge_tol, args.band + 4)
    bz = fz.copy()
    t_has = np.zeros_like(band)
    if args.teacher:
        t_z, t_color, t_has = teacher_back(gsplat, splats, fz, W, H, f, meta, args)
        bz = np.where(t_has, t_z, bz)
        b_color = np.where(t_has[..., None], t_color, b_color)
    sources = band | far_px | t_has
    ext_xyz, b_color, filled = extend_background(
        rays, bz, b_color, sources, ~sources, f, args.max_move, args.edge_tol / 2, args.max_extend)
    b_xyz = ext_xyz * (1 + args.back_offset)
    needed = ndimage.binary_dilation(band | far_px | near_px | t_has | filled, iterations=2)
    b_bh, b_bv = find_edges(b_xyz[..., 2], args.edge_tol, args.edge_tol / 2, args.min_edge)
    bv_, bf, brc, _, _ = grid_mesh(b_xyz, needed, b_bh, b_bv, args.stride, flatten="far")
    b_tex = Image.fromarray((np.clip(b_color, 0, 1) * 255).astype(np.uint8))
    b_uv = (brc[:, ::-1] + 0.5) / [W, H]
    del splats
    torch.cuda.empty_cache()

    import trimesh
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    def textured(name, verts, faces, uv, tex, alpha=None):
        material = PBRMaterial(name=name, baseColorTexture=tex, metallicFactor=0.0, roughnessFactor=1.0,
                               doubleSided=True, alphaMode="BLEND" if alpha is not None else "OPAQUE")
        # trimesh's glTF exporter flips v (it expects OpenGL-style uv), so pre-flip.
        uv = np.stack([uv[:, 0], 1 - uv[:, 1]], 1)
        visual = TextureVisuals(uv=uv, material=material)
        if alpha is not None:
            # Exported as COLOR_0 (white, alpha = per-vertex opacity).
            rgba = np.full((len(verts), 4), 255, dtype=np.uint8)
            rgba[:, 3] = np.round(np.clip(alpha, 0, 1) * 255)
            visual.vertex_attributes["color"] = rgba
        return trimesh.Trimesh(vertices=verts.astype(np.float32), faces=faces, process=False, visual=visual)

    # Split the front mesh: opaque interior, and soft-alpha edge cells (their
    # vertices come after all interior vertices).
    f_first_edge_vert = ff[f_edge0:].min() if len(ff) > f_edge0 else len(fv)
    interior = slice(0, f_first_edge_vert)
    edges = slice(f_first_edge_vert, len(fv))
    scene = trimesh.Scene()
    scene.add_geometry(textured("front", fv[interior], ff[:f_edge0], f_uv[interior], f_tex),
                       node_name="front", geom_name="front")
    if len(ff) > f_edge0:
        scene.add_geometry(textured("front_edges", fv[edges], ff[f_edge0:] - f_first_edge_vert, f_uv[edges], f_tex,
                                    alpha=np.clip(f_alpha[edges], args.edge_alpha_min, 1)),
                           node_name="front_edges", geom_name="front_edges")
    scene.add_geometry(textured("back", bv_, bf, b_uv, b_tex), node_name="back", geom_name="back")
    scene.export(out_path)
    LOGGER.info("%s: grid %dx%d, %d edge px, front %d tris (%d soft-edge), back %d tris "
                "(%d px from splat views, %d px extended) -> %s",
                path.name, W, H, int(near_px.sum()), len(ff), len(ff) - f_edge0, len(bf),
                int(t_has.sum()), int(filled.sum()), out_path)


@click.command()
@click.option("-i", "--input-path", type=click.Path(path_type=Path, exists=True), required=True,
              help="A SHARP .ply file (with its .json next to it) or a folder of them.")
@click.option("--images-dir", type=click.Path(path_type=Path), default="images", show_default=True,
              help="Folder holding the source photos named in each scene's .json.")
@click.option("--grid-width", default=1024, show_default=True, help="Width (in grid pixels) of the photo part of the grid the layers are built on.")
@click.option("--margin", default=0.1, show_default=True, help="Extend the layers this fraction past each side of the photo (splats reach past its frame).")
@click.option("--stride", default=2, show_default=True, help="Mesh every Nth grid pixel (1 = full res, 4x the triangles).")
@click.option("--edge-tol", default=0.05, show_default=True,
              help="Relative disparity jump (and departure from local slope, at half this) that counts as an edge.")
@click.option("--min-edge", default=8, show_default=True, help="Drop edge fragments touching fewer pixels than this.")
@click.option("--band", default=2, show_default=True, help="Width in pixels of the background-side band recoloured in the back layer.")
@click.option("--max-move", default=0.3, show_default=True, help="Head movement (metres) the background extension should cover.")
@click.option("--max-extend", default=48, show_default=True, help="Max background extension in grid pixels.")
@click.option("--edge-alpha-min", default=0.0, show_default=True,
              help="Opacity of the background-side corners of foreground edge cells (1 = hard edges).")
@click.option("--teacher/--no-teacher", default=True, show_default=True,
              help="Build the back layer from renders of the splats around the input view.")
@click.option("--teacher-move", default=0.4, show_default=True, help="Sideways extent (metres) of the splat views used for the back layer.")
@click.option("--teacher-res", default=1024, show_default=True, help="Width of the splat renders.")
@click.option("--teacher-min-samples", default=2, show_default=True, help="Samples a grid pixel needs to take its back surface from the splat views.")
@click.option("--gap-tol", default=0.02, show_default=True, help="Relative depth gap beyond the front surface for a splat-view sample to count as hidden content.")
@click.option("--back-offset", default=0.003, show_default=True, help="Relative push of the back mesh away from the camera.")
@click.option("--texture-size", default=2048, show_default=True, help="Max photo texture dimension.")
@click.option("--overwrite", is_flag=True, help="Rebuild existing _layers.glb files.")
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def bake_layers_cli(input_path: Path, verbose: bool, **kwargs):
    """Bake splats into two textured depth meshes (<scene>_layers.glb)."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)
    args = type("Args", (), kwargs)
    if input_path.is_dir():
        paths = sorted(p for p in input_path.glob("*.ply")
                       if not p.stem.endswith(("_quest", "_pruned", "_mesh")))
    else:
        paths = [input_path]
    for path in paths:
        bake(path, args)
