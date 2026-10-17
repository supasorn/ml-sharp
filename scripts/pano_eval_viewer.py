"""PSNR of 360 MSI versions as the real viewer (web/pano.html) draws them.

Two steps, two Python environments:
1. capture (needs playwright + Chromium; any venv): loads each version in headless Chromium (SwiftShader
   WebGL) through the running pano server and screenshots the same views from several eye offsets.
       python scripts/pano_eval_viewer.py capture --scene yaowarat_heng_heng \
           --versions out_ico2 out_ico2_hires out_ico2_photo out_ico2_photo2 -o pano/eval/heng
2. score (sharp env, gsplat): references for the same cameras, PSNR per version and eye offset.
       python scripts/pano_eval_viewer.py score --scene yaowarat_heng_heng -o pano/eval/heng

Views: 30 deg FOV at 640 px (~21 px/deg, close to the Quest 3), yaw every 30 deg at pitch 0 plus 6 views
at pitch +-30. Eye offsets move along each view's right axis (largest parallax): 0, 3 cm (an eye), 10 cm,
30 cm. References: the panorama for the centre (the only ground truth) and gsplat renders of the splats
for every offset (the 3D model the MSI was baked from; photo-detail differs from it on purpose).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SIZE, FOV = 640, 30.0
MOVES = (0.0, 0.03, 0.1, 0.3)


def views():
    v = [(yaw, 0.0) for yaw in range(0, 360, 30)]
    v += [(yaw, p) for yaw in (0, 120, 240) for p in (30.0, -30.0)]
    return v


def three_pose(yaw, pitch, move):
    """Viewer yaw (deg, > 0 turns left), pitch, eye position (three.js axes) moved along the view's right."""
    y = math.radians(yaw)
    right = np.array([math.cos(y), 0.0, -math.sin(y)])
    return right * move


def capture(args):
    from playwright.sync_api import sync_playwright

    out = Path(args.output)
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                    "--ignore-gpu-blocklist"])
        for ver in args.versions:
            d = out / ver
            d.mkdir(parents=True, exist_ok=True)
            pg = b.new_page(viewport={"width": SIZE, "height": SIZE})
            mode = "splats" if ver == "splats" else "msi"
            scene = f"out_ico2/{args.scene}" if ver == "splats" else f"{ver}/{args.scene}"
            pg.goto(f"{args.server}/?mode={mode}&scene={scene}")
            pg.wait_for_function("(document.getElementById('status').dataset.base || '').length > 0 || "
                                 "document.getElementById('status').textContent.includes('Fail')", timeout=600000)
            pg.evaluate("document.getElementById('panel').style.display='none';"
                        "document.getElementById('xr-buttons').style.display='none'")
            t0 = time.time()
            for i, (yaw, pitch) in enumerate(views()):
                for move in MOVES:
                    x, y, z = three_pose(yaw, pitch, move)
                    pg.evaluate(f"window.panoSetView({yaw}, {pitch}, {FOV}, {x}, {y}, {z})")
                    # Splats re-sort after a move (async): give them time; MSI needs two frames.
                    pg.evaluate("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
                    if mode == "splats":
                        time.sleep(3)
                        pg.evaluate("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
                    pg.screenshot(path=str(d / f"v{i:02d}_m{move}.png"))
            print(f"{ver}: {len(views()) * len(MOVES)} views in {time.time() - t0:.0f}s", flush=True)
            pg.close()
        b.close()


def score(args):
    import cv2
    import torch
    from plyfile import PlyData

    sys.path.insert(0, str(Path(__file__).parent))
    from pano_predict import look_rotation, sample_view
    from pano_render import render
    from sharp.cli.bake_layers import load_gsplat, splat_tensors

    out = Path(args.output)
    gs = load_gsplat()
    dev = torch.device("cuda")
    sp = splat_tensors(PlyData.read(f"pano/out_ico2/{args.scene}.ply")["vertex"].data, dev)
    pano = cv2.cvtColor(cv2.imread(f"pano/inputs/{args.scene}.jpg"), cv2.COLOR_BGR2RGB)
    f = SIZE / 2 / math.tan(math.radians(FOV / 2))
    versions = sorted(p.name for p in out.iterdir() if p.is_dir())
    err = {v: {("pano", 0.0): []} | {("splats", m): [] for m in MOVES} for v in versions}
    err["splats (gsplat)"] = {("pano", 0.0): []}
    for i, (yaw, pitch) in enumerate(views()):
        # three.js view (yaw left-positive, y up, -z forward) -> OpenCV world (y down, z forward).
        yr, pr = math.radians(-yaw), math.radians(pitch)
        R = look_rotation(np.array([math.cos(pr) * math.sin(yr), -math.sin(pr), math.cos(pr) * math.cos(yr)]))
        ref_pano = sample_view(pano, R, SIZE, f).astype(np.float32)
        for move in MOVES:
            x, y, z = three_pose(yaw, pitch, move)
            for attempt in range(3):   # gsplat's intersection count overflowed once on the shared GPU
                try:
                    rgb, _ = render(gs, sp, R, (x, -y, -z), SIZE, f)
                    break
                except RuntimeError:
                    torch.cuda.empty_cache()
                    if attempt == 2:
                        raise
            ref_spl = (rgb.cpu().numpy() * 255).round().clip(0, 255).astype(np.float32)
            if move == 0.0:
                err["splats (gsplat)"][("pano", 0.0)].append(np.mean((ref_spl - ref_pano) ** 2))
            for v in versions:
                im = cv2.cvtColor(cv2.imread(str(out / v / f"v{i:02d}_m{move}.png")), cv2.COLOR_BGR2RGB)
                im = im.astype(np.float32)
                if move == 0.0:
                    err[v][("pano", 0.0)].append(np.mean((im - ref_pano) ** 2))
                err[v][("splats", move)].append(np.mean((im - ref_spl) ** 2))
            if i == 0 and move in (0.0, 0.1):
                cv2.imwrite(str(out / f"ref_splats_v00_m{move}.png"), cv2.cvtColor(ref_spl.astype(np.uint8), cv2.COLOR_RGB2BGR))
        if i == 0:
            cv2.imwrite(str(out / "ref_pano_v00.png"), cv2.cvtColor(ref_pano.astype(np.uint8), cv2.COLOR_RGB2BGR))
    psnr = lambda e: round(10 * math.log10(255 ** 2 / np.mean(e)), 2)
    res = {v: {f"{r}@{m}": psnr(e) for (r, m), e in d.items()} for v, d in err.items()}
    print(json.dumps(res, indent=1))
    (out / "psnr.json").write_text(json.dumps(res, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["capture", "score"])
    p.add_argument("--scene", required=True)
    p.add_argument("--versions", nargs="+", default=["out_ico2", "out_ico2_hires", "out_ico2_photo", "out_ico2_photo2"])
    p.add_argument("--server", default="http://localhost:3100")
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()
    capture(args) if args.step == "capture" else score(args)


if __name__ == "__main__":
    main()
