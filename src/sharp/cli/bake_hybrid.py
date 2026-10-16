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


def quadtree_blocks(hard_bad, soft_bad, disp, max_err, max_soft, max_block, min_block):
    """Accepted leaf blocks (x, y, size) of an adaptive quadtree over the grid points.

    A block is accepted when it has no `hard_bad` pixels (depth edges,
    uncovered), at most `max_soft` fraction of `soft_bad` pixels (their
    Gaussians are kept and drawn over the mesh), and a plane fits the
    disparity of its remaining pixels within `max_err`.
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


def bake(path: Path, args) -> None:
    import torch

    mesh_path = path.with_name(f"{path.stem}_hybrid.glb")
    splat_path = path.with_name(f"{path.stem}_hybrid.ply")
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
    blocks = quadtree_blocks(hard_bad, soft_bad, disp, max_err, args.max_soft, args.max_block, args.min_block)
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
    parallax = f * args.move * np.maximum(1 / z_mesh - 1 / np.maximum(z, 1e-6), 0)
    margin = radius + args.border_px
    remove = inside & covered[ri, ci] & (
        (on_surface & (dist > margin)) | (behind & (dist > margin + parallax)))

    keep = ~remove
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
                "Gaussians %d -> %d (%.0f%% removed) -> %s, %s",
                path.name, 100 * meshable.mean(), len(blocks),
                ", ".join(f"{s}px:{n}" for s, n in enumerate(sizes) if n),
                len(faces), 100 * covered.mean(), len(v), int(keep.sum()), 100 * remove.mean(),
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
@click.option("--min-block", default=4, show_default=True, help="Smallest quadtree block (grid pixels).")
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
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def bake_hybrid_cli(input_path: Path, verbose: bool, **kwargs):
    """Bake a hybrid mesh + Gaussians scene (<scene>_hybrid.glb + <scene>_hybrid.ply)."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)
    args = type("Args", (), kwargs)
    if input_path.is_dir():
        paths = sorted(p for p in input_path.glob("*.ply")
                       if not p.stem.endswith(("_quest", "_pruned", "_mesh", "_hybrid")))
    else:
        paths = [input_path]
    for path in paths:
        bake(path, args)
