"""Split a scene's remaining hybrid splats into background splats and foreground soft-boundary splats.

For each splat of <scene>_pruned<suffix>.ply (default: the meshall leftovers): its parallax (P = f * move / z, px)
is compared with the nearest visible surface around it, the maximum of the median-depth parallax map
(scripts/depth_compare.py) over a window as wide as the splat's quad (sqrt(5) sigmas, the viewer's) plus a
margin. A splat at about that nearest depth (within --tol-px) belongs to the occluding object (its soft edge,
or its surface); one clearly behind it is background: hidden behind the object or seen past its edge.

Writes <scene>_bgsplats.ply and <scene>_fgsplats.ply (viewer: ?lod=bgsplats,fgsplats) and, in
logs/mesh_regions/<scene>/, split_preview.png (renders of all / background / foreground from the input view)
and split_stats.json (counts, sizes and screen area of each group).

Usage: /conda_envs/sharp_dgx/bin/python scripts/split_splats.py <scene> [--suffix all] [--tol-px 15]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("-i", "--input", type=Path, default=ROOT / "data/public/output2")
    ap.add_argument("--suffix", default="all", help="splits <scene>_pruned<suffix>.ply")
    ap.add_argument("--move", type=float, default=0.1, help="head move (m) for parallax px")
    ap.add_argument("--tol-px", type=float, default=15.0, help="behind the nearest surface by more = background")
    ap.add_argument("--margin-px", type=int, default=4, help="extra window around each splat's quad (photo px)")
    args = ap.parse_args()

    from sharp.cli.bake_layers import load_gsplat
    gsplat = load_gsplat()
    from depth_compare import median_depth
    from sharp.utils.gaussians import Gaussians3D, load_ply, save_ply
    from sharp.utils.gsplat import GSplatRenderer

    dbg = ROOT / "logs/mesh_regions" / args.scene
    dbg.mkdir(parents=True, exist_ok=True)
    full, md, _, _ = load_ply(args.input / f"{args.scene}.ply")
    rest, _, _, _ = load_ply(args.input / f"{args.scene}_pruned{args.suffix}.ply")
    W, H = md.resolution_px
    f = md.focal_length_px
    fb = f * args.move
    K = torch.tensor([[f, 0, W / 2, 0], [0, f, H / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]], device="cuda",
                     dtype=torch.float32)
    renderer = GSplatRenderer(color_space=md.color_space)
    eye = torch.eye(4, device="cuda")[None]
    with torch.no_grad():
        # Nearest visible surface: median depth of all the scene's splats.
        g = full.to("cuda")
        r = renderer(g, extrinsics=eye, intrinsics=K[None], image_width=W, image_height=H)
        med, _ = median_depth(gsplat, g, K[:3, :3], W, H, renderer.low_pass_filter_eps)
        depth = torch.where(torch.isnan(med), r.depth[0, 0], med)
        P = torch.where(depth > 0, fb / depth.clamp(min=1e-6), torch.zeros_like(depth))
        levels = [2, 4, 8, 16, 32, 64, 128]
        pmax = [F.max_pool2d(P[None, None], 2 * k + 1, 1, k)[0, 0] for k in levels]

        m = rest.mean_vectors[0].cuda()
        s = rest.singular_values[0].cuda()
        z = m[:, 2]
        zc = z.clamp(min=1e-3)
        u, v = f * m[:, 0] / zc + W / 2, f * m[:, 1] / zc + H / 2
        rad = np.sqrt(5) * s.max(1).values * f / zc               # quad half-size, photo px
        on = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui, vi = u.long().clamp(0, W - 1), v.long().clamp(0, H - 1)
        need = rad + args.margin_px
        near = torch.zeros_like(z)
        lo = torch.zeros_like(z)
        for k, mx in zip(levels, pmax):
            sel = (need <= k) & (need > lo)
            near = torch.where(sel, mx[vi, ui], near)
            lo = torch.maximum(lo, torch.where(sel, torch.full_like(lo, float(k)), lo))
        near = torch.where(need > levels[-1], pmax[-1][vi, ui], near)
        Ps = fb / zc
        bg = on & (Ps < near - args.tol_px)
        fg = ~bg

        def subset(mask):
            t = mask.nonzero()[:, 0].cpu()
            return Gaussians3D(mean_vectors=rest.mean_vectors[:, t], singular_values=rest.singular_values[:, t],
                               quaternions=rest.quaternions[:, t], colors=rest.colors[:, t],
                               opacities=rest.opacities[:, t])

        parts = {"bg": subset(bg), "fg": subset(fg)}
        for name, gs in parts.items():
            save_ply(gs, f, (H, W), args.input / f"{args.scene}_{name}splats.ply")

        # Previews from the input view (quarter size).
        Kq = K.clone()
        Kq[:2] /= 4
        Kq[0, 2], Kq[1, 2] = W / 8, H / 8
        tiles = []
        for gs in [rest, parts["bg"], parts["fg"]]:
            rr = renderer(gs.to("cuda"), extrinsics=eye, intrinsics=Kq[None], image_width=W // 4, image_height=H // 4)
            tiles.append((rr.color[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy())
        Image.fromarray(np.concatenate(tiles, 1)).save(dbg / "split_preview.png")

    radn, onn = rad.cpu().numpy(), on.cpu().numpy()
    stats = {}
    for name, msk in [("all", np.ones_like(onn)), ("bg", bg.cpu().numpy()), ("fg", fg.cpu().numpy())]:
        rr = radn[msk & onn]
        stats[name] = {"splats": int(msk.sum()),
                       "quad_px_p50": round(float(np.median(2 * rr)), 1) if len(rr) else None,
                       "quad_px_p90": round(float(np.percentile(2 * rr, 90)), 1) if len(rr) else None,
                       "area_x_image": round(float(((2 * rr) ** 2).sum() / (W * H)), 1)}
    (dbg / "split_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))
    print(f"-> {args.scene}_bgsplats.ply, {args.scene}_fgsplats.ply; preview {dbg / 'split_preview.png'} "
          f"(all | background | foreground)")


if __name__ == "__main__":
    main()
