"""Mesh + splat hybrid from one flood-filled surface (revisit of src/sharp/cli/create_mesh.py).

Same steps as create_mesh.py (which is left untouched), vectorized:
1. Render the scene's splats from the input camera (gsplat): color, depth, normals.
2. Flood-fill the depth from the nearest pixel (neighbour depth difference < --max-diff),
   erode the region -> mask of one smooth surface.
3. Triangulate the masked pixels as a grid (two triangles per 2x2 block of masked pixels),
   unprojecting the 3x-blurred depth.
4. Decimate it (pyvista decimate_pro, --reduction) and texture it with the render
   (UVs = projection into the input view) -> <scene>_mesh.glb.
5. Drop the splats whose centre projects into the (further eroded) mask within
   --depth-threshold of the surface -> <scene>_pruned.ply.

Outputs go next to the scene, where the viewer's ?hybrid=true looks for them
(`/?scene=<name>&hybrid=true`, GUI checkboxes showMesh / showSplat). Debug images and the
full-resolution mesh go to --debug-dir (default logs/mesh_hybrid/<scene>/).

Usage: /conda_envs/sharp_dgx/bin/python scripts/mesh_hybrid.py [-i data/public/output2] <scene>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def flood_fill(depth, seed_point, max_diff):
    h, w = depth.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)  # +2 for the border required by OpenCV
    cv.floodFill(depth.copy(), mask, seedPoint=seed_point, newVal=0, loDiff=max_diff, upDiff=max_diff)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("-i", "--input", type=Path, default=ROOT / "data/public/output2")
    ap.add_argument("--debug-dir", type=Path)
    ap.add_argument("--max-diff", type=float, default=0.01, help="flood fill: max neighbour depth step")
    ap.add_argument("--erode", type=int, default=5, help="mask erosion iterations (3x3)")
    ap.add_argument("--reduction", type=float, default=0.99, help="decimate_pro target reduction")
    ap.add_argument("--depth-threshold", type=float, default=0.05, help="splat removal: |z - mesh depth|")
    args = ap.parse_args()

    from sharp.cli.bake_layers import load_gsplat
    load_gsplat()
    from sharp.utils.gaussians import Gaussians3D, load_ply, save_ply
    from sharp.utils.gsplat import GSplatRenderer

    out = args.input
    stem = args.scene
    dbg = args.debug_dir or ROOT / "logs/mesh_hybrid" / stem
    dbg.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    # 1. Render.
    gaussians, metadata, _, _ = load_ply(out / f"{stem}.ply")
    width, height = metadata.resolution_px
    f_px = metadata.focal_length_px
    f_x = f_y = f_px
    c_x, c_y = (width - 1) / 2.0, (height - 1) / 2.0
    intrinsics = torch.tensor([[f_x, 0, c_x, 0], [0, f_y, c_y, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                              device=device, dtype=torch.float32)
    renderer = GSplatRenderer(color_space=metadata.color_space)
    with torch.no_grad():
        r = renderer(gaussians.to(device), extrinsics=torch.eye(4)[None].to(device),
                     intrinsics=intrinsics[None], image_width=width, image_height=height)
    depth = r.depth[0, 0].cpu().numpy().astype(np.float32)
    color = (r.color[0].permute(1, 2, 0).clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    h, w = depth.shape
    print(f"{stem}: {w}x{h}, f={f_px:.1f}, depth {np.nanmin(depth):.3f}..{np.nanmax(depth):.3f}, "
          f"{gaussians.mean_vectors.shape[1]} splats")
    Image.fromarray(color).save(dbg / "rendered.png")
    nx = cv.Sobel(depth, cv.CV_64F, 1, 0, ksize=5)
    ny = cv.Sobel(depth, cv.CV_64F, 0, 1, ksize=5)
    normal = np.stack((nx, ny, depth.astype(np.float64)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
    Image.fromarray(((normal + 1) / 2 * 255).astype(np.uint8)).save(dbg / "normal.png")

    # 2. Flood fill from the nearest pixel.
    seed = np.unravel_index(np.nanargmin(depth), depth.shape)
    mask = flood_fill(depth, (int(seed[1]), int(seed[0])), args.max_diff)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.erode(mask, kernel, iterations=args.erode)[1:-1, 1:-1]
    cv.imwrite(str(dbg / "mask.png"), mask * 255)
    print(f"seed (row, col) {seed}, mask {mask.mean() * 100:.1f}% of the image")

    # 3. Grid mesh of the masked pixels on the blurred depth.
    depth_blurred = depth
    for _ in range(3):
        depth_blurred = cv.GaussianBlur(depth_blurred, (5, 5), 0)
    valid = (mask == 1) & np.isfinite(depth_blurred) & (depth_blurred > 0)
    if not valid.any():
        sys.exit("No masked pixels; nothing to mesh.")
    rows, cols = np.nonzero(valid)
    d = depth_blurred[rows, cols]
    vertices = np.stack([(cols - c_x) * d / f_x, (rows - c_y) * d / f_y, d], axis=1).astype(np.float32)
    vidx = np.full((h, w), -1, np.int64)
    vidx[rows, cols] = np.arange(len(rows))
    quad = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    a, b, c, e = vidx[:-1, :-1][quad], vidx[1:, :-1][quad], vidx[:-1, 1:][quad], vidx[1:, 1:][quad]
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([b, e, c], 1)])
    if not len(faces):
        sys.exit("No faces; nothing to mesh.")
    import trimesh
    trimesh.Trimesh(vertices, faces, vertex_colors=color[rows, cols], process=False).export(dbg / "mesh_full.ply")
    print(f"full mesh: {len(vertices)} vertices, {len(faces)} faces")

    # 4. Decimate and texture with the render.
    import pyvista as pv
    pv_mesh = pv.PolyData(vertices, np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64))
    dec = pv_mesh.decimate_pro(args.reduction, preserve_topology=True, boundary_vertex_deletion=True)
    dv = np.asarray(dec.points, np.float32)
    df = np.asarray(dec.faces).reshape(-1, 4)[:, 1:4]
    u = np.clip(((f_x * dv[:, 0] / dv[:, 2] + c_x) + 0.5) / w, 0, 1)
    v = np.clip(1.0 - ((f_y * dv[:, 1] / dv[:, 2] + c_y) + 0.5) / h, 0, 1)
    from trimesh.visual.texture import TextureVisuals
    glb = out / f"{stem}_mesh.glb"
    trimesh.Trimesh(dv, df, visual=TextureVisuals(uv=np.stack([u, v], 1), image=Image.fromarray(color)),
                    process=False).export(glb)
    print(f"decimated mesh: {len(dv)} vertices, {len(df)} faces -> {glb}")

    # 5. Remove the splats on the meshed surface.
    means = gaussians.mean_vectors[0].cpu().numpy()
    X, Y, Z = means[:, 0], means[:, 1], means[:, 2]
    keep = Z > 0
    zs = np.where(keep, Z, 1)
    iu = np.round(f_x * X / zs + c_x).astype(np.int64)
    iv = np.round(f_y * Y / zs + c_y).astype(np.int64)
    inb = keep & (iu >= 0) & (iu < w) & (iv >= 0) & (iv < h)
    prune_mask = cv.erode(mask, kernel, iterations=args.erode)
    iuc, ivc = np.clip(iu, 0, w - 1), np.clip(iv, 0, h - 1)
    md = depth_blurred[ivc, iuc]
    near = inb & (prune_mask[ivc, iuc] == 1) & np.isfinite(md) & (md > 0) & (np.abs(Z - md) < args.depth_threshold)
    keep &= ~near
    flag = torch.from_numpy(keep)
    g = gaussians.to("cpu")
    pruned = Gaussians3D(
        mean_vectors=g.mean_vectors[0][flag][None],
        singular_values=g.singular_values[0][flag][None],
        quaternions=g.quaternions[0][flag][None],
        colors=g.colors[0][flag][None],
        opacities=g.opacities[0][flag][None],
    )
    ply = out / f"{stem}_pruned.ply"
    save_ply(pruned, f_px, (height, width), ply)
    print(f"splats: {int(near.sum())} removed, {int(keep.sum())} kept -> {ply}")
    (dbg / "stats.json").write_text(json.dumps({
        "mask_frac": float(mask.mean()), "full_faces": int(len(faces)), "dec_faces": int(len(df)),
        "splats": int(len(means)), "removed": int(near.sum()), "kept": int(keep.sum())}, indent=1))


if __name__ == "__main__":
    main()
