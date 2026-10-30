"""Mesh + splat hybrid from all smooth regions (multi-region version of scripts/mesh_hybrid.py).

1. Render the scene's splats from the input camera (gsplat): color, depth, alpha. Depth (--depth median,
   default): where the ray's accumulated alpha reaches 0.5 (scripts/depth_compare.py), not gsplat's
   alpha-weighted mean (--depth expected), which semi-transparent hair pulls far behind the skin.
2. Regions: neighbouring pixels are joined when their parallax differs by less than --crack-px, where a
   pixel's parallax is P = f * move / z (how far it slides, in pixels, when the head moves --move metres;
   a disparity in pixel units). So the allowed depth step grows with z^2: lenient far away, strict nearby.
   Flooding from every seed with this neighbour-to-neighbour rule = connected components of the link
   graph, labelled in one pass. Regions under --min-region pixels and low-alpha pixels stay splats.
   Pockets of up to --fill-holes pixels enclosed by one region (depth spikes, e.g. on faces) join it,
   with their depth inpainted from around them.
3. Each region minus its border pixels, eroded (--erode), triangulated as a pixel grid: 2 triangles per
   2x2 block of kept pixels, each only where its edges pass the --crack-px test (no curtains across depth
   jumps), on parallax smoothed only across such links (--smooth).
4. Decimation (--decimate parallax, default): simplified in (u, v, P) space, where 3D planes stay planes,
   with the error capped at --max-error-px pixels (vtkDecimatePro, absolute error); vertices are mapped
   back to 3D. --decimate pro: the original decimate_pro(--reduction) in 3D.
   Textured with the render (UVs = projection into the input view) -> <scene>_mesh<suffix>.glb.
5. Splats whose centre projects into the (further eroded) pixels the mesh covers within --prune-px of the
   surface's parallax are removed, and (--visibility) those the mesh hides from every viewpoint within
   --keep-move metres (they would still cost vertex work on the headset) -> <scene>_pruned<suffix>.ply.

6. --backdrop (with --suffix back): the splats left behind the front mesh become a second, coarse opaque
   mesh behind it (rendered alone at 1 / --back-step resolution, joined with a ring of the front surface,
   pushed --back-offset-px back, decimated to --back-error-px), and the splats on it are removed; both meshes
   go into the one GLB. Viewer: ?lod=meshback.

Viewer: ?lod=meshall (suffix "all"), e.g. /?scene=<name>&lod=full,meshflood,meshall.
Debug images (regions.png: random colour per region) go to --debug-dir (default logs/mesh_regions/<scene>/).

Usage: /conda_envs/sharp_dgx/bin/python scripts/mesh_regions.py [-i data/public/output2] <scene>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def label_regions(P, valid, crack_px):
    """Connected components of the pixel graph with an edge where both pixels are valid and |dP| < crack_px."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    h, w = P.shape
    idx = np.arange(h * w).reshape(h, w)
    rows, cols = [], []
    for a, b, pa, pb in [(idx[:, :-1], idx[:, 1:], P[:, :-1], P[:, 1:]),
                         (idx[:-1, :], idx[1:, :], P[:-1, :], P[1:, :])]:
        va = valid.reshape(-1)[a] & valid.reshape(-1)[b]
        ok = va & (np.abs(pa - pb) < crack_px)
        rows.append(a[ok])
        cols.append(b[ok])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    g = coo_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(h * w, h * w))
    _, labels = connected_components(g, directed=False)
    labels = labels.reshape(h, w)
    labels[~valid] = -1
    return labels


def grid_mesh(P, mask, step, crack, smooth, max_err):
    """Decimated grid mesh of parallax map P (grid cell (r, c) centred at full-resolution pixel
    ((c + 0.5) * step, (r + 0.5) * step)) over `mask`: triangles only across neighbour steps under
    crack * step, parallax smoothed across those links (`smooth` rounds), decimated in (u, v, P) with
    an absolute error of max_err px. Returns (points (n, 3) as (u, v, P) in full-resolution pixels,
    faces (m, 3), the pixels covered by triangles (bool, grid size))."""
    import pyvista as pv
    import vtk

    h, w = P.shape
    tol = crack * step
    lh = mask[:, :-1] & mask[:, 1:] & (np.abs(P[:, 1:] - P[:, :-1]) < tol)
    lv = mask[:-1, :] & mask[1:, :] & (np.abs(P[1:, :] - P[:-1, :]) < tol)
    Ps = np.where(mask, P, 0).astype(np.float32)
    for _ in range(smooth):
        num, cnt = Ps.copy(), np.ones_like(Ps)
        num[:, :-1] += np.where(lh, Ps[:, 1:], 0); cnt[:, :-1] += lh
        num[:, 1:] += np.where(lh, Ps[:, :-1], 0); cnt[:, 1:] += lh
        num[:-1, :] += np.where(lv, Ps[1:, :], 0); cnt[:-1, :] += lv
        num[1:, :] += np.where(lv, Ps[:-1, :], 0); cnt[1:, :] += lv
        Ps = num / cnt
    rows, cols = np.nonzero(mask)
    vidx = np.full((h, w), -1, np.int64)
    vidx[rows, cols] = np.arange(len(rows))
    quad = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:] & (np.abs(P[1:, :-1] - P[:-1, 1:]) < tol)
    good = [quad & lv[:, :-1] & lh[:-1, :], quad & lh[1:, :] & lv[:, 1:]]
    corner = [vidx[:-1, :-1], vidx[1:, :-1], vidx[:-1, 1:], vidx[1:, 1:]]
    faces = np.concatenate([np.stack([corner[k][g] for k in t], 1) for t, g in zip([(0, 1, 2), (1, 3, 2)], good)])
    full = np.zeros((h + 1, w + 1), bool)
    full[1:-1, 1:-1] = good[0] | good[1]
    covered = full[:-1, :-1] & full[1:, :-1] & full[:-1, 1:] & full[1:, 1:]
    pts = np.stack([(cols + 0.5) * step, (rows + 0.5) * step, Ps[rows, cols]], 1).astype(np.float64)
    dp = vtk.vtkDecimatePro()
    dp.SetInputData(pv.PolyData(pts, np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)))
    dp.SetTargetReduction(0.9999)
    dp.PreserveTopologyOn()
    dp.BoundaryVertexDeletionOn()
    dp.SetErrorIsAbsolute(1)
    dp.SetAbsoluteError(max_err)
    dp.Update()
    dec = pv.wrap(dp.GetOutput())
    return np.asarray(dec.points), np.asarray(dec.faces).reshape(-1, 4)[:, 1:4], covered, Ps


def step_stats(P, valid):
    d = []
    for pa, pb, va, vb in [(P[:, :-1], P[:, 1:], valid[:, :-1], valid[:, 1:]),
                           (P[:-1, :], P[1:, :], valid[:-1, :], valid[1:, :])]:
        d.append(np.abs(pa - pb)[va & vb])
    d = np.concatenate(d)
    return {q: float(np.percentile(d, q)) for q in (50, 90, 95, 99, 99.5, 99.9)}


def surface_texture(gsplat, gaussians, K, width, height, eps2d, fb, Psurf, tol, bins=256):
    """Render the splats from K with, at each pixel, only those at most `tol` parallax px in front of the
    surface Psurf (H, W; inf = all): what the mesh replaces there, without the soft edges of objects in front
    of it (kept as splats, so they would show twice). Splats composited front to back in `bins` disparity
    bins (scripts/depth_compare.py), each bin in or out per pixel by its centre's parallax.
    Returns (rgb of the included splats, un-premultiplied (H, W, 3), their alpha, the excluded ones' alpha),
    in the splats' colour space."""
    means = gaussians.mean_vectors[0]
    z = means[:, 2]
    order = torch.argsort(z)
    order = order[(z > 1e-2)[order]]
    zs = z[order]
    disp = torch.linspace(1 / zs[-1].item(), 1 / zs[0].item(), bins + 1, device=z.device)
    edges = torch.sort(1 / disp).values                                 # increasing depth
    edges[-1] = zs[-1] + 1
    ends = torch.searchsorted(zs, edges[1:]).tolist()
    dev = means.device
    C_in = torch.zeros(height, width, 3, device=dev)
    T_in = torch.ones(height, width, device=dev)
    T_ex = torch.ones(height, width, device=dev)
    start = 0
    for k, end in enumerate(ends):
        if end <= start:
            continue
        sl = order[start:end]
        cols, alph, _ = gsplat.rendering.rasterization(
            means=means[sl], quats=gaussians.quaternions[0][sl], scales=gaussians.singular_values[0][sl],
            opacities=gaussians.opacities[0][sl], colors=gaussians.colors[0][sl],
            viewmats=torch.eye(4, device=dev)[None], Ks=K[None], width=width, height=height,
            render_mode="RGB", rasterize_mode="classic", packed=False, eps2d=eps2d)
        Pbin = fb * 2 / (edges[k] + edges[k + 1]).item()
        inc = Pbin <= Psurf + tol
        a = alph[0, ..., 0]
        C_in += torch.where(inc[..., None], T_in[..., None] * cols[0], 0)
        T_in = torch.where(inc, T_in * (1 - a), T_in)
        T_ex = torch.where(inc, T_ex, T_ex * (1 - a))
        start = end
    A_in = 1 - T_in
    return C_in / A_in.clamp(min=1e-4)[..., None], A_in, 1 - T_ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("-i", "--input", type=Path, default=ROOT / "data/public/output2")
    ap.add_argument("--debug-dir", type=Path)
    ap.add_argument("--suffix", default="all", help="outputs <scene>_mesh<suffix>.glb / _pruned<suffix>.ply")
    ap.add_argument("--move", type=float, default=0.1, help="head move (m) that defines parallax pixels")
    ap.add_argument("--crack-px", type=float, default=3.0, help="max neighbour parallax step (px)")
    ap.add_argument("--smooth", type=int, default=8, help="rounds of edge-stopping vertex smoothing")
    ap.add_argument("--fill-holes", type=int, default=5000, help="enclosed pockets up to this size (px) join the region")
    ap.add_argument("--depth", choices=["median", "expected"], default="median")
    ap.add_argument("--min-alpha", type=float, default=0.5, help="pixels with less rendered alpha stay splats")
    ap.add_argument("--min-region", type=int, default=2000, help="smaller regions (px, before erosion) stay splats")
    ap.add_argument("--erode", type=int, default=5, help="region erosion iterations (3x3)")
    ap.add_argument("--jump-gap", type=int, default=2, help="no mesh within this many px of a depth jump in a region")
    ap.add_argument("--decimate", choices=["parallax", "pro"], default="parallax")
    ap.add_argument("--max-error-px", type=float, default=0.5, help="parallax decimation: max error (px)")
    ap.add_argument("--reduction", type=float, default=0.99, help="pro decimation: target reduction")
    ap.add_argument("--prune-px", type=float, default=10.0, help="splat removal: max parallax difference (px)")
    ap.add_argument("--photo-detail", action=argparse.BooleanOptionalAction, default=True,
                    help="add the photo's detail (photo - render) to the texture where the mesh is the front surface")
    ap.add_argument("--detail-edge-px", type=int, default=12,
                    help="no photo detail this close to a depth edge (the photo's and the splats' outlines differ)")
    ap.add_argument("--visibility", action=argparse.BooleanOptionalAction, default=True,
                    help="also remove splats the mesh hides from every viewpoint within --keep-move")
    ap.add_argument("--keep-move", type=float, default=1.0, help="head movement (m) the hidden-splat test covers")
    ap.add_argument("--backdrop", action="store_true", help="also mesh the splats behind the front mesh (see 6.)")
    ap.add_argument("--back-step", type=int, default=4, help="backdrop grid: one cell per N x N pixels")
    ap.add_argument("--back-crack-px", type=float, default=10.0, help="backdrop: max neighbour parallax step (px)")
    ap.add_argument("--back-error-px", type=float, default=3.0, help="backdrop decimation: max error (px)")
    ap.add_argument("--back-offset-px", type=float, default=3.0, help="backdrop pushed this far behind (parallax px)")
    ap.add_argument("--back-ring", type=int, default=8, help="front surface kept around the back content (cells)")
    ap.add_argument("--back-prune-px", type=float, default=15.0, help="backdrop splat removal: max parallax difference")
    args = ap.parse_args()

    from sharp.cli.bake_layers import load_gsplat
    gsplat = load_gsplat()
    from depth_compare import median_depth
    from sharp.utils.gaussians import Gaussians3D, load_ply, save_ply
    from sharp.utils.gsplat import GSplatRenderer

    out = args.input
    stem = args.scene
    dbg = args.debug_dir or ROOT / "logs/mesh_regions" / stem
    dbg.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    t0 = time.time()

    # 1. Render.
    gaussians, metadata, _, _ = load_ply(out / f"{stem}.ply")
    width, height = metadata.resolution_px
    f_px = metadata.focal_length_px
    # gsplat's pixel j is centred at j + 0.5, so the image centre is W / 2 (as in the viewer).
    c_x, c_y = width / 2.0, height / 2.0
    intrinsics = torch.tensor([[f_px, 0, c_x, 0], [0, f_px, c_y, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                              device=device, dtype=torch.float32)
    renderer = GSplatRenderer(color_space=metadata.color_space)
    with torch.no_grad():
        r = renderer(gaussians.to(device), extrinsics=torch.eye(4)[None].to(device),
                     intrinsics=intrinsics[None], image_width=width, image_height=height)
        if args.depth == "median":
            med, _ = median_depth(gsplat, gaussians.to(device), intrinsics[:3, :3], width, height,
                                  renderer.low_pass_filter_eps)
            r = r._replace(depth=torch.where(torch.isnan(med), r.depth[0, 0], med)[None, None])
    depth = r.depth[0, 0].cpu().numpy().astype(np.float32)
    alpha = r.alpha[0, 0].cpu().numpy()
    color = (r.color[0].permute(1, 2, 0).clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    h, w = depth.shape
    fb = f_px * args.move                                   # parallax px = fb / z
    print(f"{stem}: {w}x{h}, f={f_px:.1f}, {gaussians.mean_vectors.shape[1]} splats; "
          f"crack {args.crack_px} px = disparity step {args.crack_px / fb:.5f} /m")
    Image.fromarray(color).save(dbg / "rendered.png")

    # 2. Regions.
    valid = np.isfinite(depth) & (depth > 0) & (alpha >= args.min_alpha)
    P = np.where(valid, fb / np.where(valid, depth, 1), 0).astype(np.float32)
    print("neighbour parallax steps (px) percentiles:",
          {k: round(v, 4) for k, v in step_stats(P, valid).items()})
    labels = label_regions(P, valid, args.crack_px)
    ids, sizes = np.unique(labels[labels >= 0], return_counts=True)
    big = ids[sizes >= args.min_region]
    # Enclosed pockets join their region; their depth is interpolated from the region around them.
    from scipy import ndimage
    filled = np.zeros_like(valid)
    for rid in big:
        reg = labels == rid
        pockets, n = ndimage.label(ndimage.binary_fill_holes(reg) & ~reg)
        if not n:
            continue
        small = np.flatnonzero(np.bincount(pockets.ravel())[1:] <= args.fill_holes) + 1
        hole = np.isin(pockets, small)
        labels[hole] = rid
        filled |= hole
    # Pocket parallax: normalized blur of the region's own (unfilled) pixels, at the smallest scale that
    # reaches the pixel.
    for rid in big:
        todo = filled & (labels == rid)
        known = (labels == rid) & ~filled & valid
        sigma = 2.0
        while todo.any() and sigma < 512:
            num = cv.GaussianBlur(np.where(known, P, 0).astype(np.float32), (0, 0), sigma)
            den = cv.GaussianBlur(known.astype(np.float32), (0, 0), sigma)
            reach = todo & (den > 1e-3)
            P[reach] = num[reach] / den[reach]
            todo &= ~reach
            sigma *= 2
    depth = np.where(valid | filled, fb / np.maximum(P, 1e-6), depth).astype(np.float32)
    keep = np.isin(labels, big)
    print(f"filled {filled.sum()} px of enclosed pockets")
    print(f"{len(ids)} regions, {len(big)} with >= {args.min_region} px covering {keep.mean() * 100:.1f}% "
          f"(valid {valid.mean() * 100:.1f}%)")
    rng = np.random.default_rng(0)
    lut = rng.integers(40, 256, (labels.max() + 2, 3), dtype=np.uint8)
    lut[0] = 0
    vis = lut[labels + 1]
    vis[~keep] //= 4
    Image.fromarray(vis).save(dbg / "regions.png")

    # 3. Grid mesh per region: drop pixels touching another label, then erode.
    border = np.zeros_like(keep)
    border[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    border[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    border[1:, :] |= labels[1:, :] != labels[:-1, :]
    border[:-1, :] |= labels[:-1, :] != labels[1:, :]
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.erode((keep & ~border).astype(np.uint8), kernel, iterations=args.erode).astype(bool)
    # Depth jumps inside a region (two surfaces joined by a detour elsewhere): no mesh within --jump-gap px of
    # them on either side. Their texels would otherwise touch, and a mesh edge (straightened by decimation,
    # filtered bilinearly) samples up to a texel past its own pixels: the other surface's colour as a thin
    # strip along the edge. With the gap each side's edge texels are filled with its own colour (below).
    if args.jump_gap > 0:
        jump = np.zeros_like(mask)
        for a_, b_ in [((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                       ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                       ((slice(None, -1), slice(None, -1)), (slice(1, None), slice(1, None))),
                       ((slice(None, -1), slice(1, None)), (slice(1, None), slice(None, -1)))]:
            j = mask[a_] & mask[b_] & (np.abs(P[a_] - P[b_]) >= args.crack_px)
            jump[a_] |= j
            jump[b_] |= j
        k = 2 * args.jump_gap + 1
        near_jump = cv.dilate(jump.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        print(f"depth jumps inside regions: {int(jump.sum())} px, mesh kept {args.jump_gap} px away "
              f"({int((mask & near_jump).sum())} px less)")
        mask &= ~near_jump
    cv.imwrite(str(dbg / "mask.png"), mask.astype(np.uint8) * 255)
    # Links: neighbouring mesh pixels whose (unsmoothed) parallax passes the region test. Triangles and
    # smoothing use only these, so neither crosses a depth jump (pixels of one region can still be across a
    # jump from each other, joined by a detour; a triangle there is a stretched curtain).
    lh = mask[:, :-1] & mask[:, 1:] & (np.abs(P[:, 1:] - P[:, :-1]) < args.crack_px)
    lv = mask[:-1, :] & mask[1:, :] & (np.abs(P[1:, :] - P[:-1, :]) < args.crack_px)
    # Smoothing: --smooth rounds of averaging each pixel with its linked neighbours (in parallax, where
    # planes stay planes), ~ a Gaussian of sigma 0.6 * sqrt(rounds) px that stops at depth edges.
    Ps = np.where(mask, P, 0).astype(np.float32)
    for _ in range(args.smooth):
        num, cnt = Ps.copy(), np.ones_like(Ps)
        num[:, :-1] += np.where(lh, Ps[:, 1:], 0); cnt[:, :-1] += lh
        num[:, 1:] += np.where(lh, Ps[:, :-1], 0); cnt[:, 1:] += lh
        num[:-1, :] += np.where(lv, Ps[1:, :], 0); cnt[:-1, :] += lv
        num[1:, :] += np.where(lv, Ps[:-1, :], 0); cnt[1:, :] += lv
        Ps = num / cnt
    depth_s = np.where(mask, fb / np.maximum(Ps, 1e-6), 0).astype(np.float32)
    rows, cols = np.nonzero(mask)
    vidx = np.full((h, w), -1, np.int64)
    vidx[rows, cols] = np.arange(len(rows))
    # Two triangles per 2x2 block, (a b c) and (b e c) with a = (r, c), b = (r+1, c), c = (r, c+1),
    # e = (r+1, c+1); a triangle only where its three edges are links (the diagonal b-c tested likewise).
    quad = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:] & (np.abs(P[1:, :-1] - P[:-1, 1:]) < args.crack_px)
    good = [quad & lv[:, :-1] & lh[:-1, :], quad & lh[1:, :] & lv[:, 1:]]
    corner = [vidx[:-1, :-1], vidx[1:, :-1], vidx[:-1, 1:], vidx[1:, 1:]]
    faces = np.concatenate([np.stack([corner[k][g] for k in t], 1) for t, g in zip([(0, 1, 2), (1, 3, 2)], good)])
    # Pixels the mesh covers: all four blocks around them have both triangles.
    full = np.zeros((h + 1, w + 1), bool)
    full[1:-1, 1:-1] = good[0] & good[1]
    covered = full[:-1, :-1] & full[1:, :-1] & full[:-1, 1:] & full[1:, 1:]
    print(f"full mesh: {len(rows)} vertices, {len(faces)} faces, "
          f"{2 * int((mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]).sum()) - len(faces)} "
          f"dropped across depth jumps ({time.time() - t0:.0f} s)")

    # 4. Decimate.
    import pyvista as pv
    import trimesh
    from trimesh.visual.texture import TextureVisuals
    tri = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
    d = depth_s[rows, cols]
    if args.decimate == "parallax":
        import vtk
        pts = np.stack([cols + 0.5, rows + 0.5, fb / d], 1).astype(np.float64)   # (u, v, P): all in pixels
        dp = vtk.vtkDecimatePro()
        dp.SetInputData(pv.PolyData(pts, tri))
        dp.SetTargetReduction(0.9999)
        dp.PreserveTopologyOn()
        dp.BoundaryVertexDeletionOn()
        dp.SetErrorIsAbsolute(1)
        dp.SetAbsoluteError(args.max_error_px)
        dp.Update()
        dec = pv.wrap(dp.GetOutput())
        q = np.asarray(dec.points)
        z = fb / q[:, 2]
        dv = np.stack([(q[:, 0] - c_x) * z / f_px, (q[:, 1] - c_y) * z / f_px, z], 1).astype(np.float32)
    else:
        verts = np.stack([(cols + 0.5 - c_x) * d / f_px, (rows + 0.5 - c_y) * d / f_px, d], 1).astype(np.float32)
        dec = pv.PolyData(verts, tri).decimate_pro(args.reduction, preserve_topology=True,
                                                   boundary_vertex_deletion=True)
        dv = np.asarray(dec.points, np.float32)
    df = np.asarray(dec.faces).reshape(-1, 4)[:, 1:4]
    pu = f_px * dv[:, 0] / dv[:, 2] + c_x                    # photo px
    pv = f_px * dv[:, 1] / dv[:, 2] + c_y
    # Texture: at each pixel only the splats the mesh replaces (up to --prune-px in front of its surface), not
    # the soft edges of nearer objects seen over it: those stay splats and would otherwise show twice (baked
    # onto the surface behind, where they stay as the head moves).
    Psurf = np.where(mask, Ps, np.inf).astype(np.float32)
    with torch.no_grad():
        rgb, a_in, a_ex = surface_texture(gsplat, gaussians.to(device), intrinsics[:3, :3], w, h,
                                          renderer.low_pass_filter_eps, fb, torch.from_numpy(Psurf).to(device),
                                          args.prune_px)
        if metadata.color_space == "linearRGB":
            from sharp.utils import color_space as cs_utils
            rgb = cs_utils.linearRGB2sRGB(rgb)
    rgb, a_in, a_ex = rgb.clamp(0, 1).cpu().numpy() * 255, a_in.cpu().numpy(), a_ex.cpu().numpy()
    tex = np.where((a_in > 0.02)[..., None], rgb, color.astype(np.float32))
    print(f"texture: {(mask & (a_ex > 0.02)).mean() / max(mask.mean(), 1e-9) * 100:.1f}% of the mesh's pixels had "
          f"splats in front left out ({time.time() - t0:.0f} s)")
    # Photo detail (as MPI-only's PHOTO_DETAIL): the residual photo - render, added where the mesh is the opaque
    # front surface (its own alpha >= 0.95, 2 px in from that area's edge), fading out as the splats left out in
    # front of it get denser (all of it under faint ones, alpha <= 0.1; none under alpha >= 0.5, e.g. a nearer
    # object's soft edge, where the detail belongs to that object), and not within --detail-edge-px of a depth
    # edge (a parallax step over --crack-px): there the photo's outlines and the splats' differ by a few pixels,
    # so the residual holds the nearer object's edge, which would be painted onto the surface behind it.
    if args.photo_detail:
        from PIL import ImageOps
        meta = json.loads((out / f"{stem}.json").read_text())
        im = Image.open(out.parent / "images" / meta["image"])
        im = min((ImageOps.exif_transpose(im), im), key=lambda x: abs(x.width / x.height - w / h))
        if abs(im.width / im.height - w / h) > 0.01 * w / h:
            print(f"photo {im.width}x{im.height} does not match the scene's {w}x{h}: no photo detail")
        else:
            photo = np.asarray(im.convert("RGB").resize((w, h), Image.LANCZOS), np.float32)
            surf = cv.erode((mask & (a_in >= 0.95)).astype(np.uint8), kernel, iterations=2)
            wgt = surf * np.clip((0.5 - a_ex) / 0.4, 0, 1).astype(np.float32)
            ke = np.ones((2 * args.detail_edge_px + 1,) * 2, np.uint8)
            Pv = np.where(valid | filled, P, 0).astype(np.float32)
            calm = (cv.dilate(Pv, ke) - Ps <= args.crack_px) & (Ps - cv.erode(Pv, ke) <= args.crack_px)
            wgt = cv.erode(wgt * calm, kernel, iterations=2)
            wgt = cv.GaussianBlur(wgt, (0, 0), 1.0) * surf
            tex = tex + wgt[..., None] * (photo - color)
            print(f"photo detail on {wgt.sum() / max(mask.sum(), 1) * 100:.1f}% of the mesh's pixels, "
                  f"|photo - render| there {np.abs(photo - color)[wgt > 0.5].mean():.2f} / 255")
    tex = np.clip(np.round(tex), 0, 255).astype(np.uint8)
    Image.fromarray(tex).save(dbg / "texture.png")
    cv.imwrite(str(dbg / "texture_left_out.png"), (np.clip(a_ex, 0, 1) * 255 * mask).astype(np.uint8))
    # Texels off the mesh take the colour of the nearest mesh texel, and each vertex carries its distance (texels)
    # to the nearest texel off the mesh (glTF attribute _EDGE): the viewer caps the mipmap level by it. The headset
    # samples mipmaps (2-8 texels averaged when the texture is shrunk 2-4x), and at a depth edge the surface
    # behind would otherwise blend in the nearer object next to it in the photo, a texel or two away (a fringe of
    # its colour once the head moves and the edge separates from that object).
    from scipy import ndimage
    dist, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)
    tex = tex[iy, ix]
    edge = ndimage.distance_transform_edt(covered)
    edge_v = edge[np.clip(np.floor(pv).astype(int), 0, h - 1), np.clip(np.floor(pu).astype(int), 0, w - 1)]
    glb = out / f"{stem}_mesh{args.suffix}.glb"
    uv = np.stack([np.clip(pu / w, 0, 1), np.clip(1.0 - pv / h, 0, 1)], 1)
    meshes = {"front": trimesh.Trimesh(dv, df, visual=TextureVisuals(uv=uv, image=Image.fromarray(tex)), process=False,
                                       vertex_attributes={"_edge": edge_v.astype(np.float32)})}
    print(f"decimated ({args.decimate}): {len(dv)} vertices, {len(df)} faces ({time.time() - t0:.0f} s)")

    # 5. Remove the splats on the meshed surfaces.
    means = gaussians.mean_vectors[0].cpu().numpy()
    X, Y, Z = means[:, 0], means[:, 1], means[:, 2]
    front = Z > 0
    zs = np.where(front, Z, 1)
    px, py = f_px * X / zs + c_x, f_px * Y / zs + c_y
    inb = front & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    iu = np.clip(np.floor(px).astype(np.int64), 0, w - 1)
    iv = np.clip(np.floor(py).astype(np.int64), 0, h - 1)
    prune_mask = cv.erode(covered.astype(np.uint8), kernel, iterations=args.erode).astype(bool)
    on = inb & prune_mask[iv, iu]
    dP = np.abs(fb / zs - fb / np.maximum(depth_s[iv, iu], 1e-6))
    near = on & (dP < args.prune_px)
    print("splats over meshed pixels: |dP| percentiles (px)",
          {q: round(float(np.percentile(dP[on], q)), 2) for q in (25, 50, 75, 90, 95)} if on.any() else "-")
    keep_s = front & ~near
    occluders = [np.where(covered, depth_s, np.inf)]

    # 6. Backdrop (--backdrop): the splats left behind the front mesh (SHARP's guess at what it hides: big,
    # smooth, one colour, with imperceptible depth detail) rendered alone from the input view at 1 / back-step
    # resolution, joined with a ring of the front surface around them (so it continues the visible background
    # without a seam), pushed --back-offset-px behind, meshed leniently and drawn opaque: one shaded pixel
    # instead of many blended big splats.
    n_back = 0
    if args.backdrop:
        s_ = args.back_step
        zm0 = depth_s[iv, iu]
        behind = keep_s & inb & covered[iv, iu] & (zs >= zm0)
        bidx = np.nonzero(behind)[0]
        tb = torch.from_numpy(bidx)
        gb = Gaussians3D(mean_vectors=gaussians.mean_vectors[:, tb], singular_values=gaussians.singular_values[:, tb],
                         quaternions=gaussians.quaternions[:, tb], colors=gaussians.colors[:, tb],
                         opacities=gaussians.opacities[:, tb]).to(device)
        wb, hb = w // s_, h // s_
        Kb = torch.tensor([[f_px / s_, 0, wb / 2, 0], [0, f_px / s_, hb / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                          device=device, dtype=torch.float32)
        with torch.no_grad():
            rb = renderer(gb, extrinsics=torch.eye(4)[None].to(device), intrinsics=Kb[None], image_width=wb,
                          image_height=hb)
            medb, _ = median_depth(gsplat, gb, Kb[:3, :3], wb, hb, renderer.low_pass_filter_eps)
        ab = rb.alpha[0, 0].cpu().numpy()
        db = torch.where(torch.isnan(medb), rb.depth[0, 0], medb).cpu().numpy()
        cb = (rb.color[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        back = (ab >= 0.5) & (db > 0)
        back = cv.morphologyEx(back.astype(np.uint8), cv.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool) & (db > 0)
        # The front surface at the backdrop grid (cells fully covered by the front mesh).
        crop = (slice(0, hb * s_), slice(0, wb * s_))
        cov_b = covered[crop].reshape(hb, s_, wb, s_).all((1, 3))
        Pf_b = np.where(covered, fb / np.maximum(depth_s, 1e-6), 0)[crop].reshape(hb, s_, wb, s_).mean((1, 3))
        col_b = color[crop].reshape(hb, s_, wb, s_, 3).mean((1, 3)).astype(np.uint8)
        ring = cv.dilate(back.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=args.back_ring).astype(bool)
        ring &= cov_b & ~back
        Pb = np.where(back, fb / np.maximum(db, 1e-6), np.where(ring, Pf_b, 0)) - args.back_offset_px
        mask_b = (back | ring) & (Pb > 0)
        texb = np.where(back[..., None], cb, col_b)
        q, fbk, cov_bk, Ps_b = grid_mesh(Pb, mask_b, s_, args.back_crack_px, 4, args.back_error_px)
        zq = fb / q[:, 2]
        vb = np.stack([(q[:, 0] - c_x) * zq / f_px, (q[:, 1] - c_y) * zq / f_px, zq], 1).astype(np.float32)
        ub = np.stack([np.clip(q[:, 0] / (wb * s_), 0, 1), np.clip(1 - q[:, 1] / (hb * s_), 0, 1)], 1)
        meshes["backdrop"] = trimesh.Trimesh(vb, fbk, visual=TextureVisuals(uv=ub, image=Image.fromarray(texb)),
                                             process=False)
        # Splats on the backdrop.
        ib, jb = np.clip(iv // s_, 0, hb - 1), np.clip(iu // s_, 0, wb - 1)
        on_b = keep_s & behind & cov_bk[ib, jb] & back[ib, jb]
        dPb = np.abs(fb / zs - (Ps_b[ib, jb] + args.back_offset_px))
        rm_b = on_b & (dPb < args.back_prune_px)
        keep_s &= ~rm_b
        n_back = int(rm_b.sum())
        dfull = np.full((h, w), np.inf)
        dfull[crop] = np.repeat(np.repeat(np.where(cov_bk, fb / np.maximum(Ps_b, 1e-6), np.inf), s_, 0), s_, 1)
        occluders.append(dfull)
        print(f"backdrop: {len(bidx)} splats behind the front mesh, {back.mean() * 100:.1f}% of the image opaque "
              f"behind, ring {ring.mean() * 100:.1f}%; mesh {len(vb)} vertices, {len(fbk)} faces, texture {wb}x{hb}; "
              f"{n_back} splats removed ({time.time() - t0:.0f} s)")
        Image.fromarray(texb).save(dbg / "backdrop_texture.png")
        cv.imwrite(str(dbg / "backdrop_mask.png"), (back * 255 + ring * 128).astype(np.uint8))
    # Splats the mesh hides from every viewpoint within --keep-move would only cost vertex work (the Quest's
    # bottleneck: the depth test comes after it), so drop them too: bake-hybrid's conservative test, which
    # matches the viewer's quads (sqrt(8) sigmas) and forward-projects the mesh's depth into each view.
    n_hidden = 0
    if args.visibility:
        from sharp.cli.bake_hybrid import hidden_behind_mesh
        from sharp.cli.bake_layers import teacher_eyes
        sc = np.sort(gaussians.singular_values[0].cpu().numpy().astype(np.float64), -1)
        sigma = np.sqrt(sc[:, 2] * sc[:, 1])
        mdepth = occluders
        eyes = [(0.0, 0.0, 0.0)] + teacher_eyes(args.keep_move)
        idx = np.nonzero(keep_s)[0]
        hid = hidden_behind_mesh(mdepth, means[idx].astype(np.float64), sigma[idx], eyes, w, h, f_px, np.sqrt(8))
        keep_s[idx[hid]] = False
        n_hidden = int(hid.sum())
        print(f"hidden behind the mesh from all {len(eyes)} viewpoints within {args.keep_move} m: {n_hidden} "
              f"({time.time() - t0:.0f} s)")
    over = inb & covered[iv, iu]
    zm = depth_s[iv, iu]
    print("kept splats: not over the mesh (input view) %d, over it and in front %d, over it and behind %d"
          % ((keep_s & ~over).sum(), (keep_s & over & (zs < zm)).sum(), (keep_s & over & (zs >= zm)).sum()))
    trimesh.Scene(meshes).export(glb)
    print(f"-> {glb}: {', '.join(meshes)}")
    flag = torch.from_numpy(keep_s)
    g = gaussians.to("cpu")
    pruned = Gaussians3D(
        mean_vectors=g.mean_vectors[0][flag][None],
        singular_values=g.singular_values[0][flag][None],
        quaternions=g.quaternions[0][flag][None],
        colors=g.colors[0][flag][None],
        opacities=g.opacities[0][flag][None],
    )
    ply = out / f"{stem}_pruned{args.suffix}.ply"
    save_ply(pruned, f_px, (height, width), ply)
    print(f"splats: {int(near.sum())} removed, {int(keep_s.sum())} kept -> {ply} ({time.time() - t0:.0f} s)")
    (dbg / "stats.json").write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()},
        "regions": int(len(ids)), "meshed_regions": int(len(big)), "mask_frac": float(mask.mean()),
        "full_faces": int(len(faces)), "dec_faces": int(len(df)),
        "splats": int(len(means)), "removed": int(near.sum()), "hidden": n_hidden, "backdrop_removed": n_back, "kept": int(keep_s.sum())}, indent=1))


if __name__ == "__main__":
    main()
