"""Contains `sharp bake-hybrid` CLI implementation.

Hybrid mesh + Gaussian representation of a SHARP scene for fast rendering on
standalone headsets. Large regions that a textured surface can reproduce are
replaced by big opaque triangles; everything else stays as Gaussians. At
render time the mesh is drawn first (opaque, depth-writing) and the remaining
splats after it with depth testing, so splat fragments behind the mesh are
rejected before shading.

Outputs, next to the input:
- `<scene>_hybrid.glb`: the mesh, textured with a render of the splats.
- `<scene>_hybrid.ply`: the Gaussians the mesh doesn't replace.

1. Meshable pixels. The splats are rendered from the input view (colour,
   depth, alpha). A pixel is meshable when the splats there are opaque, its
   depth isn't blended across an edge, and it looks the same when treated as
   a point on a textured surface: it's reprojected into renders of the
   splats from viewpoints around the input view (`--move`), and where it's
   visible there, the colour must match. Foliage, hair and fuzzy edges fail
   this and stay as Gaussians.
2. Large triangles. An adaptive quadtree over the image accepts a block when
   it contains no depth edge or uncovered pixel, at most `--max-soft` of
   colour-inconsistent pixels (their Gaussians stay and draw over the mesh),
   and a plane fits its disparity (1/z): any plane is
   linear in disparity in image space, and the fit tolerance is expressed as
   on-screen parallax error at the full head move (`--parallax-px`). Blocks
   that fail are split, down to `--min-block`. Leaf blocks are fan-
   triangulated using every vertex on their border, so the mesh has no
   T-junction cracks between blocks of different sizes.
3. Gaussian removal. A Gaussian is dropped when its centre projects well
   inside the meshed area (further from its border than the Gaussian's own
   footprint) and it either lies on the meshed surface, or lies behind it
   deep enough inside that no head move within `--move` reveals it around
   the border. Gaussians in front of the mesh are kept.

The mesh sits slightly in front of the surface (`--mesh-offset`), so at
render time the depth test (per fragment; each splat is drawn at its centre
depth) hides kept splats on the meshed surface, whose colour the texture
already contains, while splats near the border still draw where they extend
past the mesh.

For licensing see accompanying LICENSE file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy import ndimage

from sharp.cli.bake_layers import (
    edge_pixels,
    find_edges,
    load_gsplat,
    load_photo,
    render_splats,
    splat_tensors,
    teacher_eyes,
)
from sharp.utils import logging as logging_utils

LOGGER = logging.getLogger(__name__)

PROPS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
         "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def grid_camera(meta: dict, width: int):
    """Pixel grid over the photo: (W, H, f, rays (H,W,3) with z = 1)."""
    W = width
    H = int(round(W * meta["height"] / meta["width"]))
    f = meta["focal_length_px"] * W / meta["width"]
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.stack([(u - W / 2) / f, (v - H / 2) / f, np.ones_like(u)], -1)
    return W, H, f, rays


def view_consistency(gsplat, splats, depth, rgb, W, H, f, args):
    """Per-pixel max colour error when the input view is treated as a textured surface.

    Each pixel's surface point is reprojected into renders of the splats from
    `teacher_eyes`; where it's visible there (depth agrees), its colour is
    compared to the render. Images are lightly blurred first so resampling of
    fine texture doesn't count as error. Returns (error (H,W), times seen (H,W)).
    """
    import torch
    import torch.nn.functional as F

    dev = depth.device
    blur = lambda img: F.avg_pool2d(img.permute(2, 0, 1)[None], 3, 1, 1)[0].permute(1, 2, 0)
    src = blur(rgb)
    u, v = torch.meshgrid(torch.arange(W, device=dev) + 0.5, torch.arange(H, device=dev) + 0.5, indexing="xy")
    X = torch.stack([(u - W / 2) / f * depth, (v - H / 2) / f * depth, depth], -1)
    err = torch.zeros(H, W, device=dev)
    seen = torch.zeros(H, W, device=dev)
    for eye in teacher_eyes(args.move):
        rgb_t, depth_t, _ = render_splats(gsplat, splats, eye, W, H, f)
        P = X - torch.tensor(eye, device=dev, dtype=torch.float32)
        z = P[..., 2].clamp(min=1e-3)
        # Normalised sample coords for grid_sample (align_corners=False).
        gx = (f * P[..., 0] / z + W / 2) / W * 2 - 1
        gy = (f * P[..., 1] / z + H / 2) / H * 2 - 1
        grid = torch.stack([gx, gy], -1)[None]
        col = F.grid_sample(blur(rgb_t).permute(2, 0, 1)[None], grid, align_corners=False)[0].permute(1, 2, 0)
        dt = F.grid_sample(depth_t[None, None], grid, mode="nearest", align_corners=False)[0, 0]
        inside = (gx.abs() < 1) & (gy.abs() < 1)
        visible = inside & ((dt - z).abs() < args.depth_tol * z)
        e = (col - src).abs().amax(-1)
        err = torch.where(visible, torch.maximum(err, e), err)
        seen += visible
    return err, seen


def plane_fit_error(disp: np.ndarray, weight: np.ndarray, x0, y0, s):
    """RMS residual of a weighted least-squares plane over disparity in each block.

    Blocks are given by arrays of origins (x0, y0) and a common size s; each
    covers the (s+1)x(s+1) grid points from its origin (inclusive corners).
    Uses summed-area tables of the plane-fit moments, so it's vectorised.
    """
    H, W = disp.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    w = weight.astype(np.float64)
    d = np.where(weight, disp, 0.0)
    terms = {"1": w, "u": w * u, "v": w * v, "d": d, "uu": w * u * u, "uv": w * u * v,
             "vv": w * v * v, "ud": u * d, "vd": v * d, "dd": d * d}
    x1, y1 = x0 + s + 1, y0 + s + 1
    sums = {}
    for k, t in terms.items():
        sat = np.pad(t.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        sums[k] = sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]
    A = np.stack([np.stack([sums["uu"], sums["uv"], sums["u"]], -1),
                  np.stack([sums["uv"], sums["vv"], sums["v"]], -1),
                  np.stack([sums["u"], sums["v"], sums["1"]], -1)], -2)
    b = np.stack([sums["ud"], sums["vd"], sums["d"]], -1)
    coef = np.linalg.solve(A + np.eye(3) * 1e-9, b[..., None])[..., 0]
    a, bb, c = coef[:, 0], coef[:, 1], coef[:, 2]
    sse = (sums["dd"] - 2 * (a * sums["ud"] + bb * sums["vd"] + c * sums["d"])
           + a * a * sums["uu"] + bb * bb * sums["vv"] + c * c * sums["1"]
           + 2 * a * bb * sums["uv"] + 2 * a * c * sums["u"] + 2 * bb * c * sums["v"])
    return np.sqrt(np.maximum(sse, 0) / np.maximum(sums["1"], 1))


def quadtree_blocks(hard_bad, soft_bad, disp, max_err, max_soft, max_block, min_block, max_depth_ratio=0.0):
    """Accepted leaf blocks (x, y, size) of an adaptive quadtree over the grid points.

    A block is accepted when it has no `hard_bad` pixels (depth edges,
    uncovered), at most `max_soft` fraction of `soft_bad` pixels (their
    Gaussians are kept and drawn over the mesh), a plane fits the
    disparity of its remaining pixels within `max_err`, and (if
    `max_depth_ratio` > 0) its depth varies by at most that fraction.
    """
    H, W = hard_bad.shape
    sat = lambda m: np.pad(m.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    hard, soft = sat(hard_bad), sat(soft_bad)
    good = ~hard_bad & ~soft_bad
    s = max_block
    ys, xs = np.mgrid[0:H - 1:s, 0:W - 1:s]
    x0, y0 = xs.ravel(), ys.ravel()
    accepted = []
    while s >= min_block and len(x0):
        on_grid = (x0 < W - 1) & (y0 < H - 1)
        x0, y0 = x0[on_grid], y0[on_grid]
        # Blocks overhanging the right/bottom edge can't be accepted, but are
        # split (not dropped) so smaller blocks still cover up to the edge.
        fits = (x0 + s < W) & (y0 + s < H)
        x1, y1 = np.minimum(x0 + s + 1, W), np.minimum(y0 + s + 1, H)
        box = lambda t: t[y1, x1] - t[y0, x1] - t[y1, x0] + t[y0, x0]
        n_hard, n_soft = box(hard), box(soft)
        area = (x1 - x0) * (y1 - y0)
        ok = fits & (n_hard == 0) & (n_soft <= max_soft * area)
        if ok.any():
            ok[ok] = plane_fit_error(disp, good, x0[ok], y0[ok], s) <= max_err
        if max_depth_ratio > 0 and ok.any():
            # Steep blocks are mostly depth edges the slope-aware edge test took for slopes (smooth
            # predicted depth across foliage edges); from the side they become slivers.
            idx = np.nonzero(ok)[0]
            ratio = np.array([disp[y0[i]:y1[i], x0[i]:x1[i]].max() / max(disp[y0[i]:y1[i], x0[i]:x1[i]].min(), 1e-9)
                              for i in idx])
            ok[idx[ratio > 1 + max_depth_ratio]] = False
        accepted += [(x, y, s) for x, y in zip(x0[ok], y0[ok])]
        # Split the rest, unless they're entirely unusable.
        split = ~ok & (n_hard + n_soft < area)
        h = s // 2
        x0 = np.concatenate([x0[split] + dx for dx in (0, h, 0, h)])
        y0 = np.concatenate([y0[split] + dy for dy in (0, 0, h, h)])
        s = h
    return accepted


def triangulate(blocks, H, W):
    """Crack-free triangulation of quadtree leaf blocks.

    Each block's border includes every vertex of neighbouring (possibly
    smaller) blocks that lies on it; blocks with only their 4 corners become 2
    triangles, others a fan around a centre vertex. Returns grid coordinates
    of vertices (N,2) as (row, col) and faces (F,3).
    """
    is_vert = np.zeros((H, W), dtype=bool)
    for x, y, s in blocks:
        is_vert[[y, y, y + s, y + s], [x, x + s, x, x + s]] = True
    vid = -np.ones((H, W), dtype=np.int64)
    corners = np.argwhere(is_vert)
    vid[corners[:, 0], corners[:, 1]] = np.arange(len(corners))
    verts = [corners]
    n = len(corners)
    faces = []
    centres = []
    for x, y, s in blocks:
        top = [(y, c) for c in range(x, x + s + 1) if is_vert[y, c]]
        right = [(r, x + s) for r in range(y + 1, y + s + 1) if is_vert[r, x + s]]
        bottom = [(y + s, c) for c in range(x + s - 1, x - 1, -1) if is_vert[y + s, c]]
        left = [(r, x) for r in range(y + s - 1, y, -1) if is_vert[r, x]]
        ring = [vid[r, c] for r, c in top + right + bottom + left]
        if len(ring) == 4:
            a, b, c, d = ring  # TL, TR, BR, BL
            faces += [(a, d, b), (b, d, c)]
        else:
            centre = n + len(centres)
            centres.append((y + s // 2, x + s // 2))
            faces += [(ring[i], centre, ring[(i + 1) % len(ring)]) for i in range(len(ring))]
    if centres:
        verts.append(np.array(centres))
    return np.concatenate(verts), np.array(faces, dtype=np.int64)


def mesh_depth(blocks, occ, H, W):
    """Per-pixel depth of the meshed occluder (inf outside it).

    Within each quadtree block, disparity is interpolated bilinearly from the
    block's corner depths -- what its triangles render, up to the split.
    """
    out = np.full((H, W), np.inf)
    disp = 1 / np.maximum(occ, 1e-6)
    by_size = {}
    for x, y, s in blocks:
        by_size.setdefault(s, []).append((x, y))
    for s, xy in by_size.items():
        xy = np.array(xy)
        x, y = xy[:, 0], xy[:, 1]
        t = np.arange(s + 1) / s
        d00, d01 = disp[y, x], disp[y, x + s]
        d10, d11 = disp[y + s, x], disp[y + s, x + s]
        top = d00[:, None] + (d01 - d00)[:, None] * t        # (n, s+1) along x
        bot = d10[:, None] + (d11 - d10)[:, None] * t
        grid = top[:, None, :] + (bot - top)[:, None, :] * t[None, :, None]  # (n, s+1 rows, s+1 cols)
        rows = (y[:, None] + np.arange(s + 1))[:, :, None]
        cols = (x[:, None] + np.arange(s + 1))[:, None, :]
        out[rows, cols] = 1 / np.maximum(grid, 1e-9)
    return out


def hidden_behind_mesh(mdepth, means, sigma, eyes, W, H, f, std, max_radius=64):
    """Per splat: is it completely hidden behind the mesh from every eye?

    Matches the GPU: Spark draws each splat as a flat quad at its centre depth
    and every fragment is depth-tested, so a splat is hidden in a view when the
    mesh is in front of its centre depth across its whole quad (radius
    `std` sigmas). Conservative: the mesh is shrunk by 1 px at its border and
    projected into each view without filling gaps, the quad is bounded by a
    square window, and splats with a quad over `max_radius` px or a centre
    off-screen (but quad partly on-screen) are kept. `mdepth` may also be a list
    of depth maps (several mesh layers, all W x H): together they occlude.
    """
    import torch
    import torch.nn.functional as F

    dev = "cuda"
    src = []
    for layer in (mdepth if isinstance(mdepth, list) else [mdepth]):
        md = torch.tensor(np.where(np.isfinite(layer), layer, np.inf), device=dev, dtype=torch.float32)
        present = torch.isfinite(md)
        interior = -F.max_pool2d(-present.float()[None, None], 3, 1, 1)[0, 0] > 0.5
        md = torch.where(interior, md, torch.full_like(md, float("inf")))
        ys, xs = torch.nonzero(torch.isfinite(md), as_tuple=True)
        d = md[ys, xs]
        for dy in (-1 / 3, 0.0, 1 / 3):
            for dx in (-1 / 3, 0.0, 1 / 3):
                u = xs.float() + 0.5 + dx
                v = ys.float() + 0.5 + dy
                src.append(torch.stack([(u - W / 2) / f * d, (v - H / 2) / f * d, d], -1))
    src = torch.cat(src)

    M = torch.tensor(means, device=dev, dtype=torch.float32)
    sig = torch.tensor(sigma, device=dev, dtype=torch.float32)
    levels = [1, 2, 4, 8, 16, 32, max_radius]
    hidden = torch.ones(len(M), device=dev, dtype=torch.bool)
    for eye in eyes:
        e = torch.tensor(eye, device=dev, dtype=torch.float32)
        # Mesh depth in this view (nearest surface).
        Q = src - e
        qz = Q[:, 2]
        ok = qz > 1e-3
        qc = torch.floor(f * Q[:, 0] / qz.clamp(min=1e-3) + W / 2).long()
        qr = torch.floor(f * Q[:, 1] / qz.clamp(min=1e-3) + H / 2).long()
        ok &= (qc >= 0) & (qc < W) & (qr >= 0) & (qr < H)
        D = torch.full((H * W,), float("inf"), device=dev).scatter_reduce(
            0, (qr * W + qc)[ok], qz[ok], "amin").view(H, W)
        # Max mesh depth over square windows (off-screen padding counts as hidden).
        maxd = [F.max_pool2d(D[None, None], 2 * k + 1, 1, k)[0, 0] for k in levels]
        # Splats in this view.
        S = M - e
        z = S[:, 2]
        front = z > 0.05
        zc = z.clamp(min=1e-3)
        cu = f * S[:, 0] / zc + W / 2
        cv = f * S[:, 1] / zc + H / 2
        rad = std * sig * f / zc
        offscreen = (cu + rad < 0) | (cu - rad > W) | (cv + rad < 0) | (cv - rad > H)
        ci, ri = torch.floor(cu).long(), torch.floor(cv).long()
        center_on = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
        h = torch.zeros_like(hidden)
        lvl_lo = torch.zeros_like(rad)
        cic, ric = ci.clamp(0, W - 1), ri.clamp(0, H - 1)
        for k, mx in zip(levels, maxd):
            fits = (rad <= k) & (rad > lvl_lo)
            h |= fits & front & center_on & (mx[ric, cic] < z)
            lvl_lo = torch.maximum(lvl_lo, torch.where(fits, torch.full_like(rad, float(k)), lvl_lo))
        h |= ~front | (front & offscreen)  # behind the camera / entirely off-screen: not visible
        hidden &= h
    return hidden.cpu().numpy()


def bake(path: Path, args) -> None:
    import torch

    mesh_path = path.with_name(f"{path.stem}{args.name_suffix}.glb")
    splat_path = path.with_name(f"{path.stem}{args.name_suffix}.ply")
    if mesh_path.exists() and splat_path.exists() and not args.overwrite:
        LOGGER.info("%s exists, skipping.", mesh_path)
        return
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        LOGGER.warning("No %s, skipping.", meta_path)
        return
    meta = json.loads(meta_path.read_text())
    v = PlyData.read(path)["vertex"].data
    gsplat = load_gsplat()
    splats = splat_tensors(v, "cuda")
    W, H, f, rays = grid_camera(meta, args.grid_width)

    # 1. Meshable pixels.
    rgb0, depth0, alpha0 = render_splats(gsplat, splats, (0.0, 0.0, 0.0), W, H, f)
    err, seen = view_consistency(gsplat, splats, depth0, rgb0, W, H, f, args)
    depth = ndimage.median_filter(depth0.cpu().numpy(), 3)
    disp = 1 / np.maximum(depth, 1e-6)
    # Hard: depth edges (slope-aware, so grazing ground and noise don't
    # count) and uncovered pixels. Soft: doesn't look like a textured surface
    # from nearby views.
    bh, bv = find_edges(depth, args.edge_tol, args.edge_tol / 2, 8)
    hard_bad = (ndimage.binary_dilation(edge_pixels(bh, bv), iterations=1)
                | (alpha0 <= args.min_alpha).cpu().numpy() | (depth <= 0))
    soft_bad = (err >= args.color_tol).cpu().numpy() & ~hard_bad
    meshable = ~hard_bad & ~soft_bad

    # 2. Quadtree of large planar blocks; tolerance = parallax error in pixels
    # at the full head move.
    max_err = args.parallax_px / (f * args.move)
    blocks = quadtree_blocks(hard_bad, soft_bad, disp, max_err, args.max_soft, args.max_block, args.min_block,
                             args.max_depth_ratio)
    rc, faces = triangulate(blocks, H, W)
    verts = rays[rc[:, 0], rc[:, 1]] * depth[rc[:, 0], rc[:, 1], None] * (1 + args.mesh_offset)
    uv = (rc[:, ::-1] + 0.5) / [W, H]
    covered = np.zeros((H, W), dtype=bool)
    for x, y, s in blocks:
        covered[y:y + s + 1, x:x + s + 1] = True

    # Texture: the splats' own render from the input view (or the photo).
    if args.texture == "photo" and (photo := load_photo(meta, args.images_dir, args.texture_width)) is not None:
        tex = photo
    else:
        Wt = args.texture_width
        ft = f * Wt / W
        Ht = int(round(Wt * H / W))
        rgb_t, _, _ = render_splats(gsplat, splats, (0.0, 0.0, 0.0), Wt, Ht, ft)
        tex = Image.fromarray((rgb_t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

    # 3. Gaussians the mesh replaces.
    means = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)
    z = means[:, 2]
    col = f * means[:, 0] / np.maximum(z, 1e-6) + W / 2 - 0.5
    row = f * means[:, 1] / np.maximum(z, 1e-6) + H / 2 - 0.5
    ci, ri = np.round(col).astype(int), np.round(row).astype(int)
    inside = (z > 0) & (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
    ci, ri = np.clip(ci, 0, W - 1), np.clip(ri, 0, H - 1)
    dist = ndimage.distance_transform_edt(covered)[ri, ci]
    z_mesh = depth[ri, ci]
    radius = args.footprint_sigma * np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], -1).max(-1)) * f / np.maximum(z, 1e-6)
    on_surface = np.abs(z - z_mesh) < args.depth_tol * z_mesh
    behind = z >= z_mesh * (1 + args.depth_tol)
    keep_move = args.keep_move or args.move
    parallax = f * keep_move * np.maximum(1 / z_mesh - 1 / np.maximum(z, 1e-6), 0)
    margin = radius + args.border_px
    remove = inside & covered[ri, ci] & (
        (on_surface & (dist > margin)) | (behind & (dist > margin + parallax)))

    keep = ~remove
    # Of the rest, also drop splats the mesh hides from every nearby viewpoint.
    n_hidden = 0
    if args.visibility:
        sc = np.sort(np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], -1).astype(np.float64)), -1)
        sigma = np.sqrt(sc[:, 2] * sc[:, 1])
        mdepth = mesh_depth(blocks, depth * (1 + args.mesh_offset), H, W)
        eyes = [(0.0, 0.0, 0.0)] + teacher_eyes(keep_move)
        idx = np.nonzero(keep)[0]
        hid = hidden_behind_mesh(mdepth, means[idx], sigma[idx], eyes, W, H, f, args.vis_std)
        keep[idx[hid]] = False
        n_hidden = int(hid.sum())
    out = np.empty(int(keep.sum()), dtype=[(k, "f4") for k in PROPS])
    for k in PROPS:
        out[k] = v[k][keep]
    PlyData([PlyElement.describe(out, "vertex")]).write(splat_path)

    import trimesh
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    material = PBRMaterial(name="hybrid", baseColorTexture=tex, metallicFactor=0.0, roughnessFactor=1.0,
                           doubleSided=True)
    # trimesh's glTF exporter flips v (it expects OpenGL-style uv), so pre-flip.
    visual = TextureVisuals(uv=np.stack([uv[:, 0], 1 - uv[:, 1]], 1), material=material)
    mesh = trimesh.Trimesh(vertices=verts.astype(np.float32), faces=faces, process=False, visual=visual)
    trimesh.Scene({"hybrid_mesh": mesh}).export(mesh_path)
    del splats
    torch.cuda.empty_cache()

    sizes = np.bincount([s for _, _, s in blocks]) if blocks else np.zeros(1)
    LOGGER.info("%s: meshable %.0f%% of pixels, %d blocks (%s), %d tris covering %.0f%%; "
                "Gaussians %d -> %d (%.0f%% removed; %d hidden behind the mesh from every viewpoint) -> %s, %s",
                path.name, 100 * meshable.mean(), len(blocks),
                ", ".join(f"{s}px:{n}" for s, n in enumerate(sizes) if n),
                len(faces), 100 * covered.mean(), len(v), int(keep.sum()), 100 * (1 - keep.mean()), n_hidden,
                mesh_path.name, splat_path.name)


@click.command()
@click.option("-i", "--input-path", type=click.Path(path_type=Path, exists=True), required=True,
              help="A SHARP .ply file (with its .json next to it) or a folder of them.")
@click.option("--move", default=0.4, show_default=True, help="Head movement (metres) the result should hold up to.")
@click.option("--grid-width", default=1024, show_default=True, help="Width of the image grid used for analysis and meshing.")
@click.option("--color-tol", default=0.08, show_default=True,
              help="Max colour difference (0-1) between a textured surface and the splats in nearby views.")
@click.option("--edge-tol", default=0.05, show_default=True, help="Relative disparity jump (departing from the local slope) that counts as a depth edge.")
@click.option("--max-soft", default=0.3, show_default=True,
              help="Fraction of colour-inconsistent pixels a block may contain (their Gaussians are kept over the mesh).")
@click.option("--depth-tol", default=0.03, show_default=True, help="Relative depth tolerance for visibility and 'on the surface'.")
@click.option("--min-alpha", default=0.99, show_default=True, help="Splat coverage needed for a pixel to be meshable.")
@click.option("--parallax-px", default=0.5, show_default=True,
              help="Max on-screen error (pixels at grid resolution) from approximating depth with a flat triangle.")
@click.option("--max-block", default=64, show_default=True, help="Largest quadtree block (grid pixels).")
@click.option("--min-block", default=2, show_default=True, help="Smallest quadtree block (grid pixels). 1 removes more Gaussians for more triangles.")
@click.option("--max-depth-ratio", default=0.0, show_default=True,
              help="If > 0, reject mesh blocks whose depth varies by more than this fraction (e.g. 0.05): they are "
                   "mostly depth edges taken for steep slopes, which show as slivers from the side.")
@click.option("--name-suffix", default="_hybrid", show_default=True, help="Output name suffix (e.g. _hybrid_s2 for a variant).")
@click.option("--keep-move", default=1.0, show_default=True, type=float,
              help="Head movement (m) within which hidden splats that could become visible are kept. Larger than "
                   "--move so moving further than the meshing range doesn't open holes at mesh edges "
                   "(1.0 vs 0.4: +1.4% splats, far-move holes 2% -> 0.1%).")
@click.option("--visibility/--no-visibility", default=True, show_default=True,
              help="Also remove Gaussians hidden behind the mesh from every viewpoint within --keep-move.")
@click.option("--vis-std", default=float(np.sqrt(8)), show_default=True,
              help="Splat quad radius (sigmas) assumed by the visibility test (sqrt(8) = largest the viewer draws).")
@click.option("--footprint-sigma", default=2.0, show_default=True,
              help="Gaussian footprint radius (in sigmas) that must lie inside the mesh for it to be removed.")
@click.option("--border-px", default=1.0, show_default=True, help="Extra margin (pixels) keeping Gaussians near the mesh border.")
@click.option("--mesh-offset", default=-0.01, show_default=True,
              help="Relative move of the mesh along view rays (negative = toward the camera). Slightly in front of "
                   "the surface, the depth test hides the splats on it (the texture already contains them), while "
                   "border splats still draw where they extend past the mesh.")
@click.option("--texture", type=click.Choice(["splats", "photo"]), default="splats", show_default=True,
              help="Texture the mesh with a render of the splats (matches them) or the original photo (sharper).")
@click.option("--texture-width", default=2048, show_default=True, help="Texture width.")
@click.option("--images-dir", type=click.Path(path_type=Path), default="images", show_default=True,
              help="Folder holding source photos (for --texture photo).")
@click.option("--overwrite", is_flag=True, help="Rebuild existing outputs.")
@click.option("--watch", is_flag=True,
              help="Keep running and bake new scenes as they appear in the input folder (e.g. from the prediction watcher).")
@click.option("--poll", default=5.0, show_default=True, help="Seconds between folder checks in --watch mode.")
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def bake_hybrid_cli(input_path: Path, verbose: bool, watch: bool, poll: float, **kwargs):
    """Bake a hybrid mesh + Gaussians scene (<scene>_hybrid.glb + <scene>_hybrid.ply)."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)
    args = type("Args", (), kwargs)
    if watch:
        if not input_path.is_dir():
            raise click.BadParameter("--watch needs a folder", param_hint="--input-path")
        if args.overwrite:
            raise click.BadParameter("--watch can't be combined with --overwrite (it would rebake every scene repeatedly)")
        watch_folder(input_path, args, poll)
        return
    for path in scene_plys(input_path):
        bake(path, args)


def scene_plys(input_path: Path) -> list[Path]:
    """Scene .ply files in a folder (or the given file), excluding derived outputs."""
    if not input_path.is_dir():
        return [input_path]
    return sorted(p for p in input_path.glob("*.ply")
                  if not p.stem.endswith(("_quest", "_pruned", "_mesh", "_hybrid")))


def watch_folder(folder: Path, args, poll: float, settle: float = 3.0) -> None:
    """Bake scenes as they appear, until interrupted.

    A scene is baked once its .ply and .json both exist and haven't changed for
    `settle` seconds (the prediction watcher writes the .ply, then the .json),
    and it has no hybrid outputs yet. A failed scene is logged and retried only
    if its files change.
    """
    import time

    LOGGER.info("Watching %s for new scenes (every %.0f s); Ctrl+C to stop.", folder, poll)
    failed: dict[Path, tuple[float, float]] = {}
    while True:
        for ply in scene_plys(folder):
            meta = ply.with_suffix(".json")
            out_glb = ply.with_name(f"{ply.stem}{args.name_suffix}.glb")
            out_ply = ply.with_name(f"{ply.stem}{args.name_suffix}.ply")
            if (out_glb.exists() and out_ply.exists() and not args.overwrite) or not meta.exists():
                continue
            try:
                stamps = (ply.stat().st_mtime, meta.stat().st_mtime)
            except FileNotFoundError:
                continue
            if time.time() - max(stamps) < settle or failed.get(ply) == stamps:
                continue
            LOGGER.info("New scene: %s", ply.name)
            try:
                bake(ply, args)
                failed.pop(ply, None)
            except Exception:  # keep watching; retry only if the files change
                LOGGER.exception("Baking %s failed", ply.name)
                failed[ply] = stamps
        time.sleep(poll)
