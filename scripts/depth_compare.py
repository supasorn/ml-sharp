"""Compare the splats' rendered depth (gsplat's alpha-normalized expected depth, "ED") with a median depth.

Expected depth = sum(w_i z_i) / sum(w_i) over every splat on the ray, so semi-transparent splats (hair,
eyelashes, soft edges) pull it towards whatever is behind them. Median depth = the depth at which the ray's
accumulated opacity reaches 0.5: the first mostly-opaque surface.

Computed exactly by a sweep: splats sorted by depth, depth thresholds t_k evenly spaced in disparity; for each
k the splats with z < t_k (a front-to-back prefix, so their alpha is the ray's accumulated alpha up to t_k) are
rendered with depth as colour. In the bin where a pixel's alpha first reaches 0.5, its median depth is the
alpha-weighted mean depth of that bin's splats: (D_k - D_k-1) / (A_k - A_k-1). Pixels that never reach 0.5
keep the expected depth.

Writes <debug-dir>/depth_compare.png (rows: whole image, a crop; columns: render, expected-depth disparity,
median-depth disparity, |difference| in parallax px) and prints dip statistics for the crop.

Usage: /conda_envs/sharp_dgx/bin/python scripts/depth_compare.py <scene> [--crop y0,y1,x0,x1]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def median_depth(gsplat, gaussians, K, width, height, eps2d, bins=256, level=0.5):
    """Median depth (H, W) and accumulated alpha (H, W) of `gaussians` (Gaussians3D, batch 1) seen through K."""
    means = gaussians.mean_vectors[0]
    z = means[:, 2]
    front = z > 1e-2
    order = torch.argsort(z)
    order = order[front[order]]
    zs = z[order]
    disp = torch.linspace(1 / zs[-1].item(), 1 / zs[0].item(), bins, device=z.device)
    thresholds = torch.cat([torch.sort(1 / disp).values[1:], zs[-1:] + 1])   # increasing depth, last = all
    ends = torch.searchsorted(zs, thresholds).tolist()
    dev = means.device
    A_prev = torch.zeros(height, width, device=dev)
    D_prev = torch.zeros(height, width, device=dev)
    med = torch.full((height, width), float("nan"), device=dev)
    start = 0
    # Each bin is rendered on its own (its splats only) and composited behind the bins before it: same
    # front-to-back compositing as one render of the whole prefix, at the cost of one bin's splats per render.
    T = torch.ones(height, width, device=dev)
    for end in ends:
        if end <= start:
            continue
        sl = order[start:end]
        cols, alph, _ = gsplat.rendering.rasterization(
            means=means[sl], quats=gaussians.quaternions[0][sl], scales=gaussians.singular_values[0][sl],
            opacities=gaussians.opacities[0][sl], colors=z[sl, None],
            viewmats=torch.eye(4, device=dev)[None], Ks=K[None], width=width, height=height,
            render_mode="RGB", rasterize_mode="classic", packed=False, eps2d=eps2d)
        a_bin, d_bin = alph[0, ..., 0], cols[0, ..., 0]      # alpha and sum(w z) of this bin alone
        A = A_prev + T * a_bin
        D = D_prev + T * d_bin
        cross = torch.isnan(med) & (A >= level)
        med[cross] = (d_bin / a_bin.clamp(min=1e-8))[cross]
        A_prev, D_prev, T = A, D, T * (1 - a_bin)
        start = end
    return med, A_prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("-i", "--input", type=Path, default=ROOT / "data/public/output2")
    ap.add_argument("--debug-dir", type=Path)
    ap.add_argument("--bins", type=int, default=256)
    ap.add_argument("--move", type=float, default=0.1, help="head move (m) for parallax px")
    ap.add_argument("--crop", default="1150,1650,2450,3150", help="y0,y1,x0,x1 of the zoomed row")
    args = ap.parse_args()

    from sharp.cli.bake_layers import load_gsplat
    gsplat = load_gsplat()
    from sharp.utils.gaussians import load_ply
    from sharp.utils.gsplat import GSplatRenderer

    dbg = args.debug_dir or ROOT / "logs/mesh_regions" / args.scene
    dbg.mkdir(parents=True, exist_ok=True)
    gaussians, metadata, _, _ = load_ply(args.input / f"{args.scene}.ply")
    width, height = metadata.resolution_px
    f = metadata.focal_length_px
    K4 = torch.tensor([[f, 0, width / 2, 0], [0, f, height / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                      device="cuda", dtype=torch.float32)
    g = gaussians.to("cuda")
    renderer = GSplatRenderer(color_space=metadata.color_space)
    with torch.no_grad():
        r = renderer(g, extrinsics=torch.eye(4, device="cuda")[None], intrinsics=K4[None],
                     image_width=width, image_height=height)
        t0 = time.time()
        med, acc = median_depth(gsplat, g, K4[:3, :3], width, height, renderer.low_pass_filter_eps, args.bins)
        torch.cuda.synchronize()
        print(f"median depth: {args.bins} bins, {time.time() - t0:.1f} s")
    ed = r.depth[0, 0]
    med = torch.where(torch.isnan(med), ed, med)
    print(f"alpha vs sweep's accumulated alpha: max diff {(r.alpha[0, 0] - acc).abs().max().item():.2e}")
    color = (r.color[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    ed, med = ed.cpu().numpy(), med.cpu().numpy()
    fb = f * args.move
    Ped, Pmed = fb / np.maximum(ed, 1e-6), fb / np.maximum(med, 1e-6)
    np.save(dbg / "depth_median.npy", med.astype(np.float32))

    y0, y1, x0, x1 = map(int, args.crop.split(","))
    from scipy import ndimage
    for name, d in [("expected", ed), ("median", med)]:
        c = d[y0:y1, x0:x1]
        dz = (c - ndimage.median_filter(c, 21)) * 100
        print(f"crop, {name:8s} depth minus local median (cm): "
              + ", ".join(f"p{q}={np.percentile(dz, q):.2f}" for q in (1, 50, 95, 99, 99.9)))
    print(f"|parallax difference| (px): " + ", ".join(
        f"p{q}={np.percentile(np.abs(Ped - Pmed), q):.2f}" for q in (50, 90, 99, 99.9)))

    def disp_img(P, lo, hi):
        return np.repeat((np.clip((P - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)[..., None], 3, 2)

    def row(sl, scale):
        lo, hi = np.percentile(Pmed[sl], [1, 99])
        diff = np.repeat((np.clip(np.abs(Ped - Pmed)[sl] / 10, 0, 1) * 255).astype(np.uint8)[..., None], 3, 2)
        tiles = [color[sl], disp_img(Ped[sl], lo, hi), disp_img(Pmed[sl], lo, hi), diff]
        tiles = [np.array(Image.fromarray(t).resize((int(t.shape[1] * scale), int(t.shape[0] * scale)))) for t in tiles]
        return np.concatenate(tiles, 1)

    full = row((slice(None), slice(None)), 400 / height)
    crop = row((slice(y0, y1), slice(x0, x1)), full.shape[1] / 4 / (x1 - x0))
    out = dbg / "depth_compare.png"
    Image.fromarray(np.concatenate([full, crop], 0)).save(out)
    print(f"-> {out}  (columns: render | expected-depth disparity | median-depth disparity | "
          f"|diff| (white = 10+ px parallax))")


if __name__ == "__main__":
    main()
