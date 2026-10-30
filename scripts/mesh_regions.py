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
   surface's parallax are removed -> <scene>_pruned<suffix>.ply.

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


def step_stats(P, valid):
    d = []
    for pa, pb, va, vb in [(P[:, :-1], P[:, 1:], valid[:, :-1], valid[:, 1:]),
                           (P[:-1, :], P[1:, :], valid[:-1, :], valid[1:, :])]:
        d.append(np.abs(pa - pb)[va & vb])
    d = np.concatenate(d)
    return {q: float(np.percentile(d, q)) for q in (50, 90, 95, 99, 99.5, 99.9)}


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
    ap.add_argument("--decimate", choices=["parallax", "pro"], default="parallax")
    ap.add_argument("--max-error-px", type=float, default=0.5, help="parallax decimation: max error (px)")
    ap.add_argument("--reduction", type=float, default=0.99, help="pro decimation: target reduction")
    ap.add_argument("--prune-px", type=float, default=10.0, help="splat removal: max parallax difference (px)")
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
    u = np.clip((f_px * dv[:, 0] / dv[:, 2] + c_x) / w, 0, 1)
    v = np.clip(1.0 - (f_px * dv[:, 1] / dv[:, 2] + c_y) / h, 0, 1)
    glb = out / f"{stem}_mesh{args.suffix}.glb"
    trimesh.Trimesh(dv, df, visual=TextureVisuals(uv=np.stack([u, v], 1), image=Image.fromarray(color)),
                    process=False).export(glb)
    print(f"decimated ({args.decimate}): {len(dv)} vertices, {len(df)} faces -> {glb} ({time.time() - t0:.0f} s)")

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
        "splats": int(len(means)), "removed": int(near.sum()), "kept": int(keep_s.sum())}, indent=1))


if __name__ == "__main__":
    main()
