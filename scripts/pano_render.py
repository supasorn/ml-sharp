"""Render a 360 splat scene (from pano_predict.py) for evaluation.

Writes, for each eye position, an equirect render and perspective views that
straddle the view seams, and reports PSNR of the centre equirect render
against the input panorama.

Usage:
    python scripts/pano_render.py pano/out/hotel_room.ply --pano pano/inputs/hotel_room.jpg
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).parent))
from pano_predict import equirect_dirs, face_rays, look_rotation, sample_view  # noqa: E402
from sharp.cli.bake_layers import load_gsplat, splat_tensors  # noqa: E402


def render(gs, splats, R, eye, size, f, W=None):
    """RGB (H,W,3) and alpha from a camera with rotation R (cam->world) at eye."""
    dev = splats["means"].device
    W = W or size
    view = torch.eye(4, device=dev)
    Rt = torch.tensor(R, device=dev, dtype=torch.float32)
    view[:3, :3] = Rt.T
    view[:3, 3] = -Rt.T @ torch.tensor(eye, device=dev, dtype=torch.float32)
    K = torch.tensor([[f, 0, W / 2], [0, f, size / 2], [0, 0, 1]], device=dev, dtype=torch.float32)
    out, alpha, _ = gs.rendering.rasterization(
        **splats, viewmats=view[None], Ks=K[None], width=W, height=size, near_plane=0.01)
    return out[0].clamp(0, 1), alpha[0, ..., 0]


def render_equirect(gs, splats, eye, width):
    """Equirect render assembled from 6 cube faces (90 deg + margin)."""
    dev = splats["means"].device
    size = width // 4 + 16
    f = (width // 4) / 2  # 90 deg over width/4 px; the extra 16 px are margin
    dirs = torch.tensor(equirect_dirs(width, width // 2), device=dev, dtype=torch.float32)
    out = torch.zeros(width // 2, width, 3, device=dev)
    best = torch.full((width // 2, width), -2.0, device=dev)
    for d in [(0, 0, 1), (1, 0, 0), (0, 0, -1), (-1, 0, 0), (0, -1, 0), (0, 1, 0)]:
        R = look_rotation(np.array(d, float))
        rgb, _ = render(gs, splats, R, eye, size, f)
        Rt = torch.tensor(R, device=dev, dtype=torch.float32)
        c = dirs @ Rt
        z = c[..., 2]
        grid = torch.stack([c[..., 0] / z.clamp(min=1e-6) * f / (size / 2),
                            c[..., 1] / z.clamp(min=1e-6) * f / (size / 2)], -1)
        smp = F.grid_sample(rgb.permute(2, 0, 1)[None], grid[None], align_corners=False)[0].permute(1, 2, 0)
        use = z > best
        out[use] = smp[use]
        best = torch.maximum(best, z)
    return out


def to_u8(t):
    return (t.cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)


def save(path, img):
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ply")
    p.add_argument("--pano", help="Input panorama for PSNR.")
    p.add_argument("-o", "--outdir", default=None)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--move", type=float, default=0.3, help="Eye offset (m) for the moved renders.")
    args = p.parse_args()

    gs = load_gsplat()
    dev = torch.device("cuda")
    v = PlyData.read(args.ply)["vertex"].data
    splats = splat_tensors(v, dev)
    outdir = Path(args.outdir or Path(args.ply).parent / (Path(args.ply).stem + "_renders"))
    outdir.mkdir(parents=True, exist_ok=True)
    res = {"splats": len(v)}

    m = args.move
    eyes = {"center": (0, 0, 0), "right": (m, 0, 0), "fwd": (0, 0, m), "up": (0, -m / 2, 0)}
    for name, eye in eyes.items():
        eq = render_equirect(gs, splats, eye, args.width)
        save(outdir / f"eq_{name}.jpg", to_u8(eq))
        if name == "center" and args.pano:
            ref = cv2.cvtColor(cv2.imread(args.pano), cv2.COLOR_BGR2RGB)
            ref = cv2.resize(ref, (args.width, args.width // 2), interpolation=cv2.INTER_AREA)
            mse = np.mean((to_u8(eq).astype(np.float32) - ref.astype(np.float32)) ** 2)
            res["psnr_center"] = round(10 * math.log10(255 ** 2 / mse), 2)
            # Same, excluding 30 deg around each pole (cube top/bottom are hard for SHARP anyway).
            h = args.width // 2
            band = slice(h // 6, h - h // 6)
            mse = np.mean((to_u8(eq)[band].astype(np.float32) - ref[band].astype(np.float32)) ** 2)
            res["psnr_center_midlat"] = round(10 * math.log10(255 ** 2 / mse), 2)
            # High-res check: 60 deg views (~17 px/deg) around the horizon and diagonals.
            full = cv2.cvtColor(cv2.imread(args.pano), cv2.COLOR_BGR2RGB)
            fz = 1024 / 2 / math.tan(math.radians(30))
            errs = []
            for yaw in range(0, 360, 30):
                for pitch in (-40, 0, 40):
                    yr, pr = math.radians(yaw), math.radians(pitch)
                    d = np.array([math.cos(pr) * math.sin(yr), -math.sin(pr), math.cos(pr) * math.cos(yr)])
                    R = look_rotation(d)
                    rgb, _ = render(gs, splats, R, eye, 1024, fz)
                    ref = sample_view(full, R, 1024, fz).astype(np.float32)
                    errs.append(np.mean((to_u8(rgb).astype(np.float32) - ref) ** 2))
                    if yaw in (45, 90) and pitch == 0 or (yaw, pitch) == (30, 40):
                        save(outdir / f"zoom_{yaw}_{pitch}.jpg", to_u8(rgb))
            res["psnr_zoom"] = round(10 * math.log10(255 ** 2 / np.mean(errs)), 2)
        # Perspective views across seams: cube edges at yaw 45, corner at pitch ~35.
        f = 640 / 2 / math.tan(math.radians(40))
        for tag, yaw, pitch in [("y45", 45, 0), ("y45p35", 45, -35), ("y0p60", 0, 60), ("y135", 135, 0)]:
            yr, pr = math.radians(yaw), math.radians(pitch)
            d = np.array([math.cos(pr) * math.sin(yr), -math.sin(pr), math.cos(pr) * math.cos(yr)])
            rgb, alpha = render(gs, splats, look_rotation(d), eye, 640, f, W=853)
            save(outdir / f"persp_{name}_{tag}.jpg", to_u8(rgb))
    print(json.dumps(res))
    (outdir / "metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
