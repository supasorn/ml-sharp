"""360 panorama -> SHARP Gaussian splats.

SHARP predicts Gaussians from one perspective photo. For an equirectangular
panorama we:

1. Split it into overlapping perspective views (`--layout ico` = 20 views at
   75 deg, default; `cube` = 6 views at 100 deg, ~2x coarser).
2. Pass 1: run SHARP's monodepth on every view and stitch one consistent
   360 distance map: a log-scale per view solved on the overlaps, then a
   feathered blend in log-distance.
3. Pass 2: run full SHARP per view with that shared depth as the alignment
   target (the checkpoint's learned `depth_alignment` net turns it into a
   smooth scale map), so neighbouring views agree on geometry.
4. Rotate each view's Gaussians into the world frame and keep a Gaussian only
   in the view whose axis is closest to its direction (Voronoi ownership).

World frame: OpenCV-style like SHARP's own output (x right, y down, z
forward), z = the panorama's centre column, camera at the origin.

Usage:
    python scripts/pano_predict.py pano/inputs/hotel_room.jpg -o pano/out
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sharp.models import PredictorParams, create_predictor
from sharp.utils import color_space as cs_utils
from sharp.utils.gaussians import (Gaussians3D, apply_transform, convert_rgb_to_spherical_harmonics,
                                   get_unprojection_matrix)

LOGGER = logging.getLogger("pano")
INTERNAL = 1536  # SHARP's input resolution


# ---------------------------------------------------------------- geometry


def look_rotation(d: np.ndarray) -> np.ndarray:
    """Camera-to-world rotation (columns = cam x, y, z) looking along d, image 'down' ~ world +y."""
    d = d / np.linalg.norm(d)
    down = np.array([0.0, 1.0, 0.0])
    ref = down if abs(d @ down) < 0.99 else np.array([0.0, 0.0, -np.sign(d @ down)])
    x = np.cross(ref, d)
    x /= np.linalg.norm(x)
    y = np.cross(d, x)
    return np.stack([x, y, d], 1)


def view_layout(name: str) -> tuple[list[np.ndarray], float]:
    """(camera-to-world rotations, ownership core half-angle hint in degrees)."""
    if name == "cube":
        dirs = [(0, 0, 1), (1, 0, 0), (0, 0, -1), (-1, 0, 0), (0, -1, 0), (0, 1, 0)]
    elif name in ("ico", "icoup"):
        # Icosahedron face centres = dodecahedron vertices (20 views).
        p = (1 + 5 ** 0.5) / 2
        dirs = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        for a in (-1, 1):
            for b in (-1, 1):
                dirs += [(0, a / p, b * p), (a / p, b * p, 0), (a * p, 0, b / p)]
        if name == "icoup":
            # rotated so one view looks straight up (and one straight down): with "ico" no view does, the two
            # highest (69 deg) split the ceiling down the middle and see it obliquely
            u = np.array((1 / p, -p, 0.0)); u /= np.linalg.norm(u)
            t = np.array((0.0, -1.0, 0.0))
            ax = np.cross(u, t); ang = math.asin(np.linalg.norm(ax)); ax /= np.linalg.norm(ax)
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            Rr = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
            dirs = [tuple(Rr @ (np.array(d, float) / np.linalg.norm(d))) for d in dirs]
    else:
        raise ValueError(name)
    return [look_rotation(np.array(d, float)) for d in dirs]


def default_fov(layout: str) -> float:
    # Square view whose inscribed circle covers the Voronoi cell plus margin.
    return {"cube": 100.0, "ico": 75.0, "icoup": 75.0}[layout]


def face_rays(size: int, f: float) -> np.ndarray:
    """(size, size, 3) camera rays with z = 1 at pixel centres."""
    c = (np.arange(size) + 0.5 - size / 2) / f
    x, y = np.meshgrid(c, c)
    return np.stack([x, y, np.ones_like(x)], -1)


def dirs_to_equirect(d: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """World directions -> continuous equirect pixel coords (u, v), pixel centres at +0.5."""
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(-d[..., 1] / np.linalg.norm(d, axis=-1), -1, 1))
    u = (lon / (2 * np.pi) + 0.5) * width
    v = (0.5 - lat / np.pi) * height
    return u, v


def equirect_dirs(width: int, height: int) -> np.ndarray:
    """(H, W, 3) unit world directions at equirect pixel centres."""
    lon = ((np.arange(width) + 0.5) / width - 0.5) * 2 * np.pi
    lat = (0.5 - (np.arange(height) + 0.5) / height) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    return np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)], -1)


def sample_view(pano: np.ndarray, R: np.ndarray, size: int, f: float) -> np.ndarray:
    """Perspective view (size x size, focal f) of an equirect image."""
    H, W = pano.shape[:2]
    d = face_rays(size, f) @ R.T
    u, v = dirs_to_equirect(d, W, H)
    # Wrap horizontally by padding one column on each side.
    padded = np.concatenate([pano[:, -1:], pano, pano[:, :1]], 1)
    return cv2.remap(padded, (u + 0.5).astype(np.float32), (v - 0.5).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def project_to_view(d: torch.Tensor, R: torch.Tensor, f: float, size: int):
    """World dirs (..., 3) -> grid_sample coords in [-1, 1] and in-front mask."""
    c = d @ R  # world -> camera (R orthonormal)
    z = c[..., 2]
    zs = z.clamp(min=1e-6)
    px = c[..., 0] / zs * f + size / 2
    py = c[..., 1] / zs * f + size / 2
    grid = torch.stack([px / size * 2 - 1, py / size * 2 - 1], -1)
    return grid, z > 1e-3


# ---------------------------------------------------------------- model


_MODELS = {}                               # loaded once per process (scripts/pano_watch.py runs many panoramas)


def load_model(checkpoint: Path, device: torch.device):
    key = (str(Path(checkpoint).resolve()), str(device))
    if key not in _MODELS:
        model = create_predictor(PredictorParams())
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        # The disparity head always in float32, also under --bf16: its bfloat16 output (8-bit mantissa) put 90% of
        # neighbouring depth pixels on the same value (fp32 25%), 10% off on average; the stitched / 360 depth
        # showed contour stripes across buildings (yaowarat). Costs 0.04 s a view (0.52 -> 0.56 s).
        head = model.monodepth_model.monodepth_predictor.head
        head_fwd = head.forward
        def head_fp32(x):
            with torch.autocast("cuda", enabled=False):
                return head_fwd(x.float())
        head.forward = head_fp32
        _MODELS[key] = model.eval().to(device)
    return _MODELS[key]


@torch.no_grad()
def monodepth(model, image: torch.Tensor, disparity_factor: float):
    """Front-layer z-depth (INTERNAL x INTERNAL) as SHARP predicts it, and the monodepth network's output for
    predict_aligned: only the fields the predictor reads, features as float16 (0.7 GB a view instead of 2.3 GB;
    splats within 1e-5 of recomputing them, opacity within 0.02)."""
    out = model.monodepth_model(image)
    disp = out.disparity[:, 0:1].clamp(min=1e-4, max=1e4)
    keep = out._replace(encoder_features=[], intermediate_features=[], decoder_features=out.decoder_features.half(),
                        output_features=[t.half() for t in out.output_features])
    return (disparity_factor / disp)[0, 0], keep


@torch.no_grad()
def predict_aligned(model, image, disparity_factor: float, depth: torch.Tensor | None, mono_out=None):
    """SHARP with the stitched depth; mono_out: pass 1's monodepth output for this view, reused instead of
    running the monodepth network (most of the forward) a second time on the same image."""
    fac = torch.tensor([disparity_factor], device=image.device)
    d = None if depth is None else depth[None, None]
    if mono_out is None:
        return model(image, fac, d)
    full = mono_out._replace(decoder_features=mono_out.decoder_features.float(),
                             output_features=[t.float() for t in mono_out.output_features])
    model.monodepth_model.forward = lambda _image: full
    try:
        return model(image, fac, d)
    finally:
        del model.monodepth_model.forward              # back to the class's forward



# ---------------------------------------------------------------- depth stitching


# ---------------------------------------------------------------- 360 depth (UniK3D)

UNIK3D_HOME = Path(os.environ.get("UNIK3D_HOME", Path.home() / ".local/opt/UniK3D"))
_UNIK3D = {}


def unik3d_distance(pano: np.ndarray, eq_w: int, device) -> torch.Tensor:
    """Distance map (eq_w / 2, eq_w) of the whole panorama from UniK3D (CVPR 2025, ViT-L; CC BY-NC 4.0), one 360
    prediction: no per-view shape disagreements to stitch (the 20 views' monodepth gave flat or straight surfaces
    piecewise, e.g. a railing in three pieces at different angles). Code in UNIK3D_HOME (git clone of
    lpiccinelli-eth/UniK3D; imports wandb / pkg_resources only for training: stand-ins in <UNIK3D_HOME>_stubs)."""
    import sys
    if "model" not in _UNIK3D:
        for p_ in (UNIK3D_HOME, Path(str(UNIK3D_HOME) + "_stubs"), UNIK3D_HOME.parent / "unik3d_stubs"):
            if p_.is_dir() and str(p_) not in sys.path:
                sys.path.append(str(p_))
        from unik3d.models import UniK3D
        _UNIK3D["model"] = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl").to(device).eval()
    from unik3d.utils.camera import Spherical
    eq_h = eq_w // 2
    rgb = torch.from_numpy(cv2.resize(pano, (eq_w, eq_h), interpolation=cv2.INTER_AREA)).permute(2, 0, 1).to(device)
    cam = Spherical(params=torch.tensor([1.0, 1.0, 1.0, 1.0, eq_w, eq_h, math.pi, math.pi / 2]))
    with torch.no_grad():
        pts = _UNIK3D["model"].infer(rgb, cam)["points"][0].float()          # (3, H, W), same pixels as the input
    return pts.norm(dim=0).clamp(min=0.05)


DAP_HOME = Path(os.environ.get("DAP_HOME", Path.home() / ".local/opt/DAP"))
_DAP = {}


def dap_distance(pano: np.ndarray, eq_w: int, device, with_sky=False):
    """Distance map (eq_w / 2, eq_w) from DAP, "Depth Any Panoramas" (Insta360, arXiv 2512.16913; Depth Anything V2 on
    DINOv3; CC BY-NC 4.0): one 360 prediction in metres at 1024x512, upsampled. Its masked region (sky) is 100 m, as
    its test/infer.py (with_sky: also return that mask). Best of the 360 models tried (docs/pano-360.md): flattest ceiling, straightest railing. Code
    in DAP_HOME (git clone of Insta360-Research-Team/DAP), weights/model.pth from Insta360-Research/DAP-weights."""
    import sys
    if "model" not in _DAP:
        import yaml
        cwd = os.getcwd()
        os.chdir(DAP_HOME)                       # its DINOv3 code is found by a path relative to the working dir
        try:
            if str(DAP_HOME) not in sys.path:
                sys.path.insert(0, str(DAP_HOME))
            from networks.models import make
            cfg = yaml.safe_load(open("config/infer.yaml"))
            m = make(cfg["model"])
            sd = torch.load("weights/model.pth", map_location="cpu", weights_only=True)
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict()}, strict=True)
            _DAP["model"], _DAP["hw"] = m.to(device).eval(), (cfg["input"]["height"], cfg["input"]["width"])
        finally:
            os.chdir(cwd)
    h, w = _DAP["hw"]
    x = torch.from_numpy(cv2.resize(pano, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255)
    with torch.inference_mode():
        o = _DAP["model"](x.permute(2, 0, 1)[None].to(device))
        d = o["pred_depth"]
        sky = (1 - o["pred_mask"]) <= 0.5 if "pred_mask" in o else torch.zeros_like(d, dtype=torch.bool)
        d = torch.where(sky, torch.ones_like(d), d)
    d = F.interpolate(d.float(), size=(eq_w // 2, eq_w), mode="bilinear", align_corners=False)[0, 0] * 100
    if not with_sky:
        return d.clamp(min=0.05)
    sky = F.interpolate(sky.float(), size=(eq_w // 2, eq_w), mode="bilinear", align_corners=False)[0, 0] > 0
    return d.clamp(min=0.05), sky


def weighted_median(x: torch.Tensor, w: torch.Tensor) -> float:
    order = torch.argsort(x)
    cw = torch.cumsum(w[order], 0)
    return float(x[order][torch.searchsorted(cw, cw[-1] / 2).clamp(max=len(x) - 1)])


def view_logdist(depths, rots, f, size, eq_w, eq_h, device, border=0.03):
    """Each view's log distance on the equirect grid: L (views, eq_h, eq_w), inside mask M, feather weight Wt
    (0 at the view's border, 1 from 30% of the half-width inwards, squared)."""
    dirs = torch.tensor(equirect_dirs(eq_w, eq_h), device=device, dtype=torch.float32)
    ray_norm = torch.tensor(np.linalg.norm(face_rays(size, f), axis=-1), device=device, dtype=torch.float32)
    n = len(depths)
    L = torch.zeros(n, eq_h, eq_w, device=device)
    M = torch.zeros(n, eq_h, eq_w, device=device, dtype=torch.bool)
    Wt = torch.zeros(n, eq_h, eq_w, device=device)
    for i, (z, R) in enumerate(zip(depths, rots)):
        logr = torch.log(z * ray_norm)[None, None]
        grid, front = project_to_view(dirs, torch.tensor(R, device=device, dtype=torch.float32), f, size)
        L[i] = F.grid_sample(logr, grid[None], mode="bilinear", align_corners=False)[0, 0]
        inside = (grid.abs().amax(-1) < 1 - border) & front
        M[i] = inside
        # Feather: distance to the image border in normalized units.
        edge = (1 - grid.abs().amax(-1)).clamp(min=0)
        Wt[i] = torch.where(inside, edge.clamp(max=0.3) / 0.3, 0) ** 2
    return L, M, Wt


def poisson_depth(depths, rots, f, size, eq_w, eq_h, device, ref, ref_weight=1e-3, iters=600, free_edges=None,
                  robust=0.0, robust_deg=10.0):
    """Gradient-domain stitching: the 360 log-distance map whose differences between neighbouring equirect pixels
    best match the views' own (blended by their feather weights), plus ref_weight x (L - ref)^2 to a reference
    log-distance map (fixes the constant; a larger weight pulls the overall shape towards it, e.g. UniK3D's).
    Per-view scale offsets have no gradient, so views can't disagree by a step anywhere; their slope
    disagreements are averaged over the overlaps. Rows weighted by cos(latitude) (equal area), u wraps around.
    Conjugate gradient on the GPU, started from ref.

    free_edges (view_edge, ratio, weight[, soft[, dis]]): where the views' step to the next pixel exceeds view_edge (log) and the
    reference jumps the same way by more than ratio x as much (over 8 px: its edges are blurred), that step's
    weight is multiplied by `weight`: the reference sets how big the jump is. Monodepth can get a real depth jump
    far too small (the UNY banner hangs below a tree canopy: one view's jump 2x, DAP's 10x), and fitting both
    spread the mismatch as ramps on either side (the banner's top pulled towards the eye). Only larger jumps: DAP's
    blurred small objects (people) jump less than the views and were pushed towards the background. Steps the
    reference doesn't have (a banner in front of its wall, a thin pole, a mirror) keep the views' size.
    A 4th value `soft` (default 1) multiplies the weight of the views' other steps > view_edge: the views disagree
    on a jump's size (R0011166's selfie head: 3 views, head vs wall), and fitted rigidly the mismatch spread over
    the surface as a slope (the head 0.60 m at the top, 0.72 m at the chin; DAP and each view: about flat).
    Softer jumps take the mismatch at the outline instead; still far stiffer than the reference's pull, so the
    jumps' sizes stay the views' (people kept their depth).
    A 5th value `dis` > 0 also frees steps > view_edge where the views that see them disagree on the jump (its size
    over 6 px, weighted mean absolute deviation > dis x the mean jump): there the views' jumps can't all be kept, and
    chained through them a thin object took the error (R0011338: window bars between glass panes whose reflected
    trees each view put at another depth, 4-7 m, bowed to 0.45 m in the middle of a 1.35 m frame).
    robust > 0: the reference's pull is dropped where the result departs from it by far more than `robust` (log)
    beyond its smooth trend (see the code)."""
    L, M, Wt = view_logdist(depths, rots, f, size, eq_w, eq_h, device)
    def blend(dim):                               # weighted mean difference to the next pixel along dim, weight
        nxt = lambda t: torch.roll(t, -1, dims=dim)
        ok = M & nxt(M)
        w = torch.where(ok, Wt * nxt(Wt), 0).sqrt()
        g = (nxt(L) - L) * w
        sw = w.sum(0)
        return g.sum(0) / sw.clamp(min=1e-8), sw
    gx, wx = blend(2)                              # along u (wraps)
    gy, wy = blend(1)                              # along v (the last row has no next row)
    wy[-1] = 0
    lat = (0.5 - (torch.arange(eq_h, device=device) + 0.5) / eq_h) * math.pi
    cw = torch.cos(lat)[:, None]
    ax = (wx > 0).float() * cw * wx.clamp(max=1)
    ay = (wy > 0).float() * torch.cos(lat - math.pi / (2 * eq_h))[:, None] * wy.clamp(max=1)
    if free_edges:
        view_edge, ratio, fw, soft, dis = (*free_edges, 1.0, 0.0)[:5]
        # the reference's jump across the same step, measured over 8 px (its edges are blurred)
        sx = torch.roll(ref, -4, 1) - torch.roll(ref, 4, 1)
        rp = F.pad(ref[None, None], (0, 0, 4, 4), mode="replicate")[0, 0]
        sy = rp[8:] - rp[:-8]
        ex, ey = gx.abs() > view_edge, gy.abs() > view_edge
        fx = ex & (gx * sx > 0) & (sx.abs() > ratio * gx.abs())
        fy = ey & (gy * sy > 0) & (sy.abs() > ratio * gy.abs())
        if dis > 0:                               # and where the views disagree on the jump
            def spread(dim):                      # each view's jump over 6 px (edges land 1-2 px apart per view)
                if dim == 2:
                    J, ok = torch.roll(L, -3, 2) - torch.roll(L, 3, 2), M & torch.roll(M, -3, 2) & torch.roll(M, 3, 2)
                else:
                    Lp = F.pad(L[:, None], (0, 0, 3, 3), mode="replicate")[:, 0]
                    Mp = F.pad(M[:, None].float(), (0, 0, 3, 3), mode="replicate")[:, 0] > 0
                    J, ok = Lp[:, 6:] - Lp[:, :-6], M & Mp[:, 6:] & Mp[:, :-6]
                w = torch.where(ok, Wt, 0) * (Wt > 0.05)
                n = (w > 0).sum(0)
                mean = (w * J).sum(0) / w.sum(0).clamp(min=1e-8)
                mad = (w * (J - mean).abs()).sum(0) / w.sum(0).clamp(min=1e-8)
                return (n >= 2) & (mad > dis * mean.abs())
            fx |= ex & spread(2)
            fy |= ey & spread(1)
        ax = torch.where(fx, ax * fw, torch.where(ex, ax * soft, ax))
        ay = torch.where(fy, ay * fw, torch.where(ey, ay * soft, ay))
        LOGGER.info("free edges: %.2f%% / %.2f%% of steps", 100 * fx.float().mean(), 100 * fy.float().mean())
    rw0 = ref_weight * cw.expand(eq_h, eq_w)
    Dx = lambda t: torch.roll(t, -1, 1) - t
    DxT = lambda t: torch.roll(t, 1, 1) - t
    def Dy(t):
        o = torch.zeros_like(t); o[:-1] = t[1:] - t[:-1]; return o
    def DyT(t):
        o = torch.zeros_like(t); o[1:] += t[:-1]; o[:-1] -= t[:-1]; return o
    def solve(rw, x):
        A = lambda t: DxT(ax * Dx(t)) + DyT(ay * Dy(t)) + rw * t
        b = DxT(ax * gx) + DyT(ay * gy) + rw * ref
        r = b - A(x); p = r.clone(); rs = (r * r).sum()
        for _ in range(iters):
            Ap = A(p); a = rs / (p * Ap).sum().clamp(min=1e-30)
            x = x + a * p; r = r - a * Ap
            rs_new = (r * r).sum()
            if rs_new.sqrt() < 1e-6 * b.norm():
                break
            p = r + (rs_new / rs) * p; rs = rs_new
        return x
    x = solve(rw0, ref.clone())
    if robust:                                   # the reference is wrong where it disagrees with the result far more
        # than its smooth trend (Cauchy on the residual minus its ~robust_deg blur, a few rounds), as in
        # dap_ratio_targets: DAP sees glass / mirrors as flat walls, and pulled towards it the window of R0011338
        # (reflected trees 4-7 m deep in every view) dragged its bars, in front of the glass, from 1.35 to 0.45 m
        ds = 8
        sig = robust_deg * eq_w / 360 / ds
        k = int(3 * sig) * 2 + 1
        g1 = torch.exp(-(torch.arange(k, device=device) - k // 2) ** 2 / (2 * sig ** 2)); g1 = g1 / g1.sum()
        def blur(t):
            t = F.conv2d(F.pad(t, (k // 2, k // 2, 0, 0), mode="circular"), g1.view(1, 1, 1, k))
            return F.conv2d(F.pad(t, (0, 0, k // 2, k // 2), mode="replicate"), g1.view(1, 1, k, 1))
        w = torch.ones_like(x)
        for _ in range(3):
            res = (x - ref)[None, None]
            wd = F.avg_pool2d(w[None, None], ds)
            trend = blur(F.avg_pool2d(w[None, None] * res, ds)) / blur(wd).clamp(min=1e-6)
            trend = F.interpolate(trend, size=(eq_h, eq_w), mode="bilinear", align_corners=False)[0, 0]
            w = 1 / (1 + ((x - ref - trend) / robust) ** 2)
            x = solve(rw0 * w, x)
        LOGGER.info("robust reference: weight < 0.5 on %.1f%% of pixels", 100 * float((w < 0.5).float().mean()))
    return x


def _diffuse(num, den, kx, ky, T, dt=0.24):
    """Normalized edge-stopping diffusion of num and den for time T: a Gaussian of sigma sqrt(2T) where the link
    conductances kx, ky (between horizontal / vertical neighbours) are 1, nothing across links near 0."""
    for _ in range(int(T / dt)):
        for t in (num, den):
            fx = kx * (t[..., :, 1:] - t[..., :, :-1])
            fy = ky * (t[..., 1:, :] - t[..., :-1, :])
            t[..., :, :-1] += dt * fx
            t[..., :, 1:] -= dt * fx
            t[..., :-1, :] += dt * fy
            t[..., 1:, :] -= dt * fy
    return num, den


def dap_ratio_targets(depths, rots, f, size, dap, sky, device, sigma_deg=10.0, tau=0.1, robust=0.1, iters=3,
                      edge_stop=0.0, dap_tau=0.05, edge_ratio=0.0):
    """Each view's own monodepth times a smooth correction towards DAP's 360 distance map (run() blends them into
    pass 2's shared target). Detail stays SHARP's (DAP alone merged the hotel_room desk chair into the desk, lost yaowarat's traffic-light
    pole); DAP only reshapes what is wider than ~sigma_deg (flat ceilings, straight railings).

    The correction is a weighted Gaussian blur (normalized convolution) of log(DAP / view): weight 0 in DAP's sky, low
    where either map has a depth edge (> tau log-distance per degree; DAP's edges are blurred), and robust (Cauchy, a
    few rounds): where the two disagree far more than `robust` (log) beyond the smooth correction, DAP is taken to be
    wrong. Without that, or with a 5-degree blur, sky openings got halos pulled towards the eye (R0011338) and mirrors
    / glass doors (hotel_room) were half-flattened onto the wall.

    edge_stop > 0: the final blur stops at depth steps > edge_stop (log, between 8 px cells) that DAP also has an
    edge near (> dap_tau per degree, within ~2 deg), so a near object's correction doesn't leak onto what is behind
    it (the selfie arm's tilted R0011166's head 0.47 -> 0.67 m). The robust weights are still judged against the
    plain blur: judged against the edge-stopped one, R0011338's window panes (framed, so DAP has edges there) became
    regions of their own and were pulled onto DAP's flat wall. Where an object's samples are all rejected, the plain
    blur. edge_ratio > 0 (with edge_stop): a step is cut only where DAP's own change across it (+-2 cells) is at least
    edge_ratio x the view's, and the robust weights are judged against the edge-stopped blur, with no fallback: the
    selfie head (DAP's step ~ the view's) gets its own correction (0.68 - 0.76 m, DAP 0.70 - 0.76), the window panes
    (the view steps 1.35 -> 5 m into the reflection, DAP stays a flat wall) are blurred across and rejected as before.
    Returns z-depth maps (size x size)."""
    rays = torch.tensor(face_rays(size, f), device=device, dtype=torch.float32)
    rn = rays.norm(dim=-1)
    logd, skyf = torch.log(dap)[None, None], sky.float()[None, None]
    px_deg = f * math.pi / 180
    ds = 8                                                   # blurred at 1/8 resolution
    sig = sigma_deg * px_deg / ds
    k = int(3 * sig) * 2 + 1
    g1 = torch.exp(-(torch.arange(k, device=device) - k // 2) ** 2 / (2 * sig ** 2))
    g1 = g1 / g1.sum()
    def blur(t):
        t = F.conv2d(F.pad(t, (k // 2, k // 2, 0, 0), mode="replicate"), g1.view(1, 1, 1, k))
        return F.conv2d(F.pad(t, (0, 0, k // 2, k // 2), mode="replicate"), g1.view(1, 1, k, 1))
    def edge(t):                                             # log-distance change per degree, widened by 4 px
        gx = F.pad(t[..., :, 1:] - t[..., :, :-1], (0, 1, 0, 0))
        gy = F.pad(t[..., 1:, :] - t[..., :-1, :], (0, 0, 0, 1))
        return F.max_pool2d(torch.sqrt(gx ** 2 + gy ** 2) * px_deg, 9, 1, 4)
    targets = []
    for z, R in zip(depths, rots):
        d = rays @ torch.tensor(R, device=device, dtype=torch.float32).T
        d = d / d.norm(dim=-1, keepdim=True)
        grid = torch.stack([torch.atan2(d[..., 0], d[..., 2]) / math.pi,
                            -torch.asin((-d[..., 1]).clamp(-1, 1)) / (math.pi / 2)], -1)[None]
        lD = F.grid_sample(logd, grid, mode="bilinear", padding_mode="border", align_corners=False)
        sk = F.grid_sample(skyf, grid, mode="bilinear", padding_mode="border", align_corners=False) > 0
        lS = torch.log(z * rn)[None, None]
        c = (~sk).float() / (1 + (torch.maximum(edge(lD), edge(lS)) / tau) ** 2)
        cd = F.avg_pool2d(c, ds)
        rd = F.avg_pool2d(c * (lD - lS), ds) / cd.clamp(min=1e-6)
        if edge_stop > 0:
            sS = F.avg_pool2d(lS, ds)
            jx, jy = (sS[..., :, 1:] - sS[..., :, :-1]).abs(), (sS[..., 1:, :] - sS[..., :-1, :]).abs()
            if edge_ratio > 0:                               # DAP's change over cells p-1 .. p+2 of each link
                pD = F.pad(F.avg_pool2d(lD, ds), (2, 2, 2, 2), mode="replicate")
                ex = ((pD[..., 2:-2, 4:-1] - pD[..., 2:-2, 1:-4]).abs() / (edge_ratio * jx).clamp(min=1e-6)).clamp(max=1)
                ey = ((pD[..., 4:-1, 2:-2] - pD[..., 1:-4, 2:-2]).abs() / (edge_ratio * jy).clamp(min=1e-6)).clamp(max=1)
            else:
                dE = (F.max_pool2d(F.max_pool2d(edge(lD), ds), 5, 1, 2) / dap_tau).clamp(max=1)
                ex = torch.maximum(dE[..., :, 1:], dE[..., :, :-1])
                ey = torch.maximum(dE[..., 1:, :], dE[..., :-1, :])
            kx, ky = 1 / (1 + ex * (jx / edge_stop) ** 4), 1 / (1 + ey * (jy / edge_stop) ** 4)
            def eblur(w):
                n, m = _diffuse(w * rd, w.clone(), kx, ky, sig ** 2 / 2)
                return n / m.clamp(min=1e-6), m
        if edge_ratio > 0 and edge_stop > 0:
            rb, _ = eblur(cd)
            for _ in range(iters):
                rb, _ = eblur(cd / (1 + ((rd - rb) / robust) ** 2))
        else:
            rb = blur(cd * rd) / blur(cd).clamp(min=1e-6)
            for _ in range(iters):
                w = cd / (1 + ((rd - rb) / robust) ** 2)
                rb = blur(w * rd) / blur(w).clamp(min=1e-6)
            if edge_stop > 0:
                n, m = _diffuse(w * rd, w.clone(), kx, ky, sig ** 2 / 2)
                eps = 0.05 * float(m.mean())
                rb = (n + eps * rb) / (m + eps)
        rb = F.interpolate(rb, size=(size, size), mode="bilinear", align_corners=False)
        targets.append(z * torch.exp(rb[0, 0]))
    return targets


def blend_near_dap(L, Wt, dap, sky, soft=0.1, agree=0.15):
    """dap_ratio's blend weights where the views disagree: a view farther from DAP than the nearest one by e (log)
    gets exp(-e / soft) x its feather weight, as far as the nearest one agrees with DAP (within ~agree): the
    feather average of a view 2.5x off and one near right tilted the UNY banner (4.4 -> 16.6 m, DAP 24). Where every
    view is far from DAP (glass, mirrors: DAP's flat wall) the plain average stays."""
    e = (L - torch.log(dap)).abs()
    e = torch.where(Wt > 0, e, torch.full_like(e, 1e3))
    emin = e.amin(0)
    w = torch.exp(-(e - emin) / soft / (1 + (emin / agree) ** 2))
    return Wt * torch.where(sky, torch.ones_like(w), w)


def dap_fit_views(depths, rots, f, size, dap, sky, device, tau=0.1, robust=0.1, ds=8):
    """--depth dap_seam, step 1: each view's monodepth fitted to DAP by 1/z' = a/z + b + c x + e y (x, y normalized
    image coordinates): a projective map along the view's rays, so planes stay planes but can tilt (one scale
    can't), and monodepth's inverse-depth offset is absorbed. Robust least squares (Cauchy, relative inverse-depth
    error, scale annealed to `robust`) on DAP's non-sky pixels away from depth edges. A pixel the fit can't explain
    keeps the view's overall correction, not its neighbours' (dap_ratio alone gave the UNY banner, 2.5x off in one
    view, the 0.55x of the tree canopy around it). Returns z-depth maps and the parameters."""
    rays = torch.tensor(face_rays(size, f), device=device, dtype=torch.float32)
    rn = rays.norm(dim=-1)
    logd, skyf = torch.log(dap)[None, None], sky.float()[None, None]
    px_deg = f * math.pi / 180
    xy = (torch.arange(size, device=device) + 0.5) / size * 2 - 1
    X, Y = xy[None, :].expand(size, size), xy[:, None].expand(size, size)
    pool = lambda t: F.avg_pool2d(t[None, None], ds)[0, 0].reshape(-1)
    out, params = [], []
    for z, R in zip(depths, rots):
        d = rays @ torch.tensor(R, device=device, dtype=torch.float32).T
        d = d / d.norm(dim=-1, keepdim=True)
        grid = torch.stack([torch.atan2(d[..., 0], d[..., 2]) / math.pi,
                            -torch.asin((-d[..., 1]).clamp(-1, 1)) / (math.pi / 2)], -1)[None]
        lD = F.grid_sample(logd, grid, mode="bilinear", padding_mode="border", align_corners=False)
        sk = F.grid_sample(skyf, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, 0] > 0
        e = torch.maximum(_edge_deg(lD, px_deg), _edge_deg(torch.log(z * rn)[None, None], px_deg))[0, 0]
        c = (~sk).float() / (1 + (e / tau) ** 2)
        v, vD = 1 / z, rn / torch.exp(lD[0, 0])                       # inverse z-depth: view, DAP
        cw = pool(c)
        A = torch.stack([pool(c * v), cw, pool(c * X), pool(c * Y)], -1) / cw.clamp(min=1e-6)[:, None]
        A = A / (pool(c * vD) / cw.clamp(min=1e-6)).clamp(min=1e-6)[:, None]   # relative to DAP's inverse depth
        w = cw
        for s in (1.0, 0.5, 0.3, 0.2, robust, robust):
            Aw = A * w[:, None]
            p = torch.linalg.solve(A.T @ Aw + 1e-6 * torch.eye(4, device=device), Aw.sum(0))
            w = cw / (1 + ((A @ p - 1) / s) ** 2)
        if p[0] < 0.05:                                                # degenerate: a scale only
            p = torch.zeros(4, device=device)
            p[0] = (cw * A[:, 0]).sum() / (cw * A[:, 0] ** 2).sum()
        vf = p[0] * v + p[1] + p[2] * X + p[3] * Y
        out.append(1 / torch.maximum(vf, 0.3 * p[0] * v))                # no sign flip far away (sky)
        params.append([round(float(t), 4) for t in p])
    return out, params


def _edge_deg(t, px_deg):
    """Log-distance change per degree, widened by 4 px (t: 1x1xHxW)."""
    gx = F.pad(t[..., :, 1:] - t[..., :, :-1], (0, 1, 0, 0))
    gy = F.pad(t[..., 1:, :] - t[..., :-1, :], (0, 0, 0, 1))
    return F.max_pool2d(torch.sqrt(gx ** 2 + gy ** 2) * px_deg, 9, 1, 4)


def seam_labels(L, M, Wt, dap, sky, device, ds=2, w_dap=4.0, s_dap=0.15, w_centre=1.0, lam0=0.05, lam=4.0,
                cap=0.5, tau=0.1, iters=40):
    """--depth dap_seam, step 3: which view each equirect pixel takes its depth from. Energy: per pixel w_dap *
    Cauchy(view vs DAP, s_dap) (0 in DAP's sky, low on DAP's depth edges) + w_centre * (1 - feather weight); per
    neighbouring pair with different views lam0 + lam * how much the two views disagree there (capped), so seams run
    where the views agree and around regions where they don't, and the view nearer DAP wins those. Solved
    approximately by annealed mean field on the GPU at 1/ds resolution (no graph-cut library here).
    Returns labels (eq_h, eq_w), long."""
    n, H, W = L.shape
    h, w = H // ds, W // ds
    Mf = M[:, None].float()
    Ld = F.avg_pool2d(L[:, None] * Mf, ds)[:, 0] / F.avg_pool2d(Mf, ds)[:, 0].clamp(min=1e-6)
    Md = -F.max_pool2d(-Mf, ds)[:, 0] > 0                              # valid in the whole block
    Wd = F.avg_pool2d(Wt[:, None], ds)[:, 0]
    lD = F.avg_pool2d(torch.log(dap)[None, None], ds)[0]
    skd = F.avg_pool2d(sky.float()[None, None], ds)[0, 0] > 0.5
    eD = _edge_deg(lD[None], w / 360)[0, 0]
    trust = (~skd).float() / (1 + (eD / tau) ** 2)
    r = Ld - lD
    D = w_dap * trust * r ** 2 / (r ** 2 + s_dap ** 2) + w_centre * (1 - Wd)
    D = torch.where(Md, D, torch.full_like(D, 1e3))
    both = Md[:, None] & Md[None]
    dis = torch.where(both, (Ld[:, None] - Ld[None]).abs().clamp(max=cap), torch.full((), cap, device=device)).half()
    def nb(t, s, dim):                                                # neighbour; wraps in longitude only
        if dim == -1:
            return torch.roll(t, s, -1)
        t2 = torch.roll(t, s, -2)
        if s > 0:
            t2[..., 0, :] = t[..., 0, :]
        else:
            t2[..., -1, :] = t[..., -1, :]
        return t2
    Q = torch.softmax(-D, 0)
    for T in np.geomspace(1.0, 0.03, iters):
        msg = torch.zeros_like(D)
        for s, dim in ((1, -1), (-1, -1), (1, -2), (-1, -2)):
            Qn = nb(Q, s, dim)
            pair = 0.5 * (dis + nb(dis, s, dim)).float()
            msg += lam0 * (1 - Qn) + lam * torch.einsum("ijhw,jhw->ihw", pair, Qn)
        Q = 0.5 * Q + 0.5 * torch.softmax(-(D + msg) / float(T), 0)
    lab = F.interpolate(Q.argmax(0)[None, None].float(), size=(H, W), mode="nearest")[0, 0].long()
    ok = M.gather(0, lab[None])[0]                                     # full-res border pixels: best valid view
    return torch.where(ok, lab, torch.where(M, Wt, -1.0).argmax(0))


def _pyr_down(t):
    k = torch.tensor([1., 4., 6., 4., 1.], device=t.device) / 16
    t = F.conv2d(F.pad(t, (2, 2, 0, 0), mode="circular"), k.view(1, 1, 1, 5))
    t = F.conv2d(F.pad(t, (0, 0, 2, 2), mode="replicate"), k.view(1, 1, 5, 1))
    return t[..., ::2, ::2]


def _pyr_up(t, shape):
    return F.interpolate(t, size=shape, mode="bilinear", align_corners=False)


def seam_blend(L, M, labels, fallback, levels=5, blend="multiband", feather_px=4.0):
    """--depth dap_seam, step 4: the labelled views' log distances merged. multiband: Laplacian blending (detail
    switches at the seam, coarse levels blend over ~2^levels px and hide the views' small remaining scale
    differences); feather: the labels blurred by feather_px. Outside a view's own pixels its depth is `fallback`."""
    n = L.shape[0]
    if blend == "feather":
        k = int(3 * feather_px) * 2 + 1
        g = torch.exp(-(torch.arange(k, device=L.device) - k // 2) ** 2 / (2 * feather_px ** 2))
        g = g / g.sum()
        oh = F.one_hot(labels, n).permute(2, 0, 1).float()[:, None]
        oh = F.conv2d(F.pad(oh, (k // 2, k // 2, 0, 0), mode="circular"), g.view(1, 1, 1, k))
        oh = F.conv2d(F.pad(oh, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1))[:, 0] * M
        return (oh * L).sum(0) / oh.sum(0).clamp(min=1e-8)
    acc = None
    for i in range(n):
        m = (labels == i).float()[None, None]
        if not m.any():
            continue
        G, Gm = [torch.where(M[i], L[i], fallback)[None, None]], [m]
        for _ in range(levels):
            G.append(_pyr_down(G[-1]))
            Gm.append(_pyr_down(Gm[-1]))
        part = [Gm[k] * (G[k] - _pyr_up(G[k + 1], G[k].shape[-2:])) for k in range(levels)] + [Gm[-1] * G[-1]]
        acc = part if acc is None else [a + p for a, p in zip(acc, part)]
    out = acc[-1]
    for k in range(levels - 1, -1, -1):
        out = acc[k] + _pyr_up(out, acc[k].shape[-2:])
    return out[0, 0]


def stitch_depth(depths, rots, f, size, eq_w, eq_h, device, border=0.03, prior=0.05):
    """Consistent 360 log-distance map from per-view z-depth maps.

    Returns (log_dist (eq_h, eq_w), per-view log scales, stats).
    """
    L, M, Wt = view_logdist(depths, rots, f, size, eq_w, eq_h, device, border)
    n = len(depths)
    # Pairwise robust offsets on overlaps -> least squares for per-view log scales.
    rows, rhs, wts, pair_stats = [], [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            ov = M[i] & M[j]
            cnt = int(ov.sum())
            if cnt < 500:
                continue
            diff = (L[i] - L[j])[ov]
            # Weight by disparity so far content (sky: depth arbitrary) barely counts.
            w = torch.exp(-torch.maximum(L[i], L[j])[ov])
            med = weighted_median(diff, w)
            mad = weighted_median((diff - med).abs(), w)
            pair_stats.append((i, j, cnt, med, mad))
            r = np.zeros(n)
            r[i], r[j] = 1, -1
            rows.append(r)
            rhs.append(-med)
            wts.append(float(w.sum()))
    wts = np.sqrt(np.array(wts) / max(wts))
    A = np.array(rows) * wts[:, None]
    b = np.array(rhs) * wts
    # Weak prior towards scale 1 (SHARP depth is roughly metric); fixes the gauge
    # and views that only overlap their neighbours in the sky.
    A = np.vstack([A, np.eye(n) * prior])
    b = np.append(b, np.zeros(n))
    s = np.linalg.lstsq(A, b, rcond=None)[0]
    Ls = L + torch.tensor(s, device=device, dtype=torch.float32)[:, None, None]
    # Residual disagreement after scaling (median abs log diff over all overlaps).
    resid = []
    for i, j, cnt, _, _ in pair_stats:
        ov = M[i] & M[j]
        w = torch.exp(-torch.maximum(L[i], L[j])[ov])
        resid.append(weighted_median((Ls[i] - Ls[j])[ov].abs(), w))
    logd = (Ls * Wt).sum(0) / Wt.sum(0).clamp(min=1e-8)
    stats = dict(pairs=len(pair_stats), raw_mad=float(np.median([p[4] for p in pair_stats])),
                 raw_offsets=float(np.median([abs(p[3]) for p in pair_stats])),
                 resid_after=float(np.median(resid)), resid_max=float(np.max(resid)),
                 scales=np.exp(s).round(3).tolist())
    return logd, s, stats


# ---------------------------------------------------------------- output


def post_align(g_ndc, unproj, target, sigma_deg=1.5, px_deg=10.24, stop=0.08, robust=0.1, iters=2, layer_tol=0.15,
               sky=None):
    """--post-align: after pass 2, scale each view's splats along their rays by a smooth correction towards the shared
    map (target: its distance, size x size), so every view puts a surface where the map does. The learned alignment
    only corrects SHARP's depth by moderate factors around the view's overall scale (the median ratio): in
    R0010712_291 (v3 depth, background 16 -> 7 m) view 19's scale fell to 0.62 and the face in front, at the
    view's own scale, stayed at 0.66x the map while view 16 had it right: the face split along their boundary.

    On the front layer of the splat grid (2 x n x n): log(target / splat distance), weighted (opacity > 0.3, low at
    the map's or the splats' depth edges), smoothed over ~sigma_deg by a diffusion that stops at the splats' own
    steps > stop (log) so a face and the wall behind it get their own factors, robust (Cauchy, `robust`). Detail
    below ~sigma_deg stays SHARP's. A back-layer splat takes its pixel's factor if within layer_tol of the front,
    else none. Not in the sky (`sky`: DAP's sky mask in the view, size x size, > 0.5 = sky) and never past the
    view's farthest front splat: the shared map's sky is SHARP's times a correction DAP has no data for (79372: up
    to 27 km, SHARP's sky <= 0.8 km), and sky splats pushed out there showed as black blotches in the MSI.
    Returns the Gaussians and the fraction of front splats moved by > 5%."""
    dev = g_ndc.mean_vectors.device
    N = g_ndc.mean_vectors.shape[0]
    n = int(round(math.sqrt(N // 2)))
    if 2 * n * n != N:
        raise ValueError(f"post_align: {N} splats are not 2 x n x n")
    M, t = unproj[:3, :3], unproj[:3, 3]
    world = g_ndc.mean_vectors @ M.T + t
    logr = torch.log(world.norm(dim=-1).clamp(min=1e-6)).view(2, n, n)
    front = logr[0][None, None]
    lt = torch.log(F.interpolate(target[None, None], size=(n, n), mode="area").clamp(min=1e-6))
    def edge(x):                                             # log-distance step to a neighbour, widened by 3 px
        gx = F.pad((x[..., :, 1:] - x[..., :, :-1]).abs(), (0, 1, 0, 0))
        gy = F.pad((x[..., 1:, :] - x[..., :-1, :]).abs(), (0, 0, 0, 1))
        return F.max_pool2d(torch.maximum(gx, gy), 7, 1, 3)
    c = (g_ndc.opacities.view(2, n, n)[0] > 0.3).float()[None, None] / (1 + (torch.maximum(edge(front), edge(lt)) / 0.02) ** 2)
    sk = None if sky is None else F.interpolate(sky[None, None].float(), size=(n, n), mode="area") > 0.5
    if sk is not None:
        c = c * (~sk).float()
    rd = lt - front
    kx = 1 / (1 + ((front[..., :, 1:] - front[..., :, :-1]).abs() / stop) ** 4)
    ky = 1 / (1 + ((front[..., 1:, :] - front[..., :-1, :]).abs() / stop) ** 4)
    T = (sigma_deg * px_deg * n / 768) ** 2 / 2
    def smooth(w):
        num, den = _diffuse(w * rd, w.clone(), kx, ky, T)
        return num / den.clamp(min=1e-6), den
    rb, den = smooth(c)
    for _ in range(iters):
        rb, den = smooth(c / (1 + ((rd - rb) / robust) ** 2))
    rb = torch.where(den > 1e-4, rb, torch.zeros_like(rb))       # no support at all: unchanged
    if sk is not None:
        rb = torch.where(sk, torch.zeros_like(rb), rb)
    rb = torch.minimum(rb, front.max() - front)[0, 0]
    back = torch.where((logr[1] - logr[0]).abs() < layer_tol, rb, torch.zeros_like(rb))
    s_ = torch.exp(torch.stack([rb, back])).reshape(-1)
    mean = (world * s_[:, None] - t) @ torch.linalg.inv(M).T
    return (g_ndc._replace(mean_vectors=mean, singular_values=g_ndc.singular_values * s_[:, None]),
            float((rb.abs() > 0.05).float().mean()))


def snap_edges(g_ndc, unproj, ramp=0.06, radius=9, sigma_c=0.08, sigma_s=6.0, layer_tol=0.15, min_move=0.05):
    """--snap-edges: SHARP blurs a depth jump over ~5 px of its 768 px splat grid and can place it a few px off
    the photo's edge (UNY banner: the tips of its letters at depths between the canopy above and the banner). Per
    view, on the splat grid (2 layers x 768 x 768, one splat per pixel and layer): each pixel on or next to a jump
    of the front layer (log-distance step > `ramp` to a neighbour) and at a depth in between the two sides (not
    within 15% of the gap from either) takes one side's depth if that moves it farther (never nearer). The sides are the off-jump
    pixels within `radius`, split at the middle of their depth range (near / far); the pixel goes with the side
    holding its best colour match (gamma-encoded splat colours, sigma_c, times proximity, sigma_s) and takes the
    depth of that best match. Best match, not a weighted vote: a large side (the canopy) outvoted the few banner
    pixels next to the edge, and its light grey matched the letters' white outlines. The pixel's back-layer splat
    moves with it when within layer_tol (log) of the front one (else it is the inpainted background and stays).
    Splats move along their rays (camera at the origin). Returns g_ndc and the fraction of splats moved by more than
    min_move (log)."""
    dev = g_ndc.mean_vectors.device
    N = g_ndc.mean_vectors.shape[0]
    n = int(round(math.sqrt(N // 2)))
    if 2 * n * n != N:
        raise ValueError(f"snap_edges: {N} splats are not 2 x n x n")
    M, t = unproj[:3, :3], unproj[:3, 3]
    world = g_ndc.mean_vectors @ M.T + t
    logr = torch.log(world.norm(dim=-1).clamp(min=1e-6)).view(2, n, n)
    front = logr[0]
    # guide: the front splats' colours, gamma-encoded (in linear RGB a dark navy banner and a dark green rim are
    # nearly equal)
    col = g_ndc.colors.reshape(2, n, n, -1)[0].permute(2, 0, 1).clamp(min=0) ** (1 / 2.2)
    op = g_ndc.opacities.reshape(2, n, n)[0]
    pad = F.pad(front[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    step = torch.stack([(pad[1:-1, 2:] - front).abs(), (pad[1:-1, :-2] - front).abs(),
                        (pad[2:, 1:-1] - front).abs(), (pad[:-2, 1:-1] - front).abs()]).amax(0)
    jump = (step > ramp).float()[None, None]
    ramp_px = F.max_pool2d(jump, 3, 1, 1)[0, 0] > 0                        # on or next to a jump
    voter = ~ramp_px & (op > 0.3)
    ys, xs = torch.nonzero(ramp_px, as_tuple=True)
    snapped = front.clone()
    off = torch.arange(-radius, radius + 1, device=dev)
    oy, ox = [o.reshape(-1) for o in torch.meshgrid(off, off, indexing="ij")]
    prox = torch.exp(-(oy ** 2 + ox ** 2) / (2 * sigma_s ** 2))
    for s in range(0, len(ys), 20000):                                      # chunks: ramp pixels x window
        y, x = ys[s:s + 20000, None], xs[s:s + 20000, None]
        qy, qx = (y + oy).clamp(0, n - 1), (x + ox).clamp(0, n - 1)
        vo = voter[qy, qx]
        val = front[qy, qx]
        big = torch.tensor(1e9, device=dev)
        lo_ = torch.where(vo, val, big).amin(1, keepdim=True)
        hi_ = torch.where(vo, val, -big).amax(1, keepdim=True)
        far = val > (lo_ + hi_) / 2
        score = torch.exp(-((col[:, qy, qx] - col[:, y, x]) ** 2).sum(0) / (2 * sigma_c ** 2)) * prox * vo
        sn, sf = torch.where(far, 0, score), torch.where(far, score, 0)
        bn, bf = sn.max(1), sf.max(1)
        pick = torch.where(bf.values > bn.values, val.gather(1, bf.indices[:, None])[:, 0],
                           val.gather(1, bn.indices[:, None])[:, 0])
        # only in-between depths: a splat already on one side (a bright skylight splat behind the canopy) moved
        # across to the other side covered it with white blobs
        own = front[y[:, 0], x[:, 0]]
        gap = (hi_ - lo_)[:, 0]
        between = (own > lo_[:, 0] + 0.15 * gap) & (own < hi_[:, 0] - 0.15 * gap)
        # and only farther: a splat moved nearer covers whatever is behind it when the colour match is wrong
        # (large skylight splats in between, matched to the canopy's light grey, landed on the canopy as white
        # blobs); moved farther, a wrong match is hidden behind the near side
        ok = vo.any(1) & (gap > ramp) & (torch.maximum(bn.values, bf.values) > 1e-4) & between & (pick > own)
        snapped[y[:, 0], x[:, 0]] = torch.where(ok, pick, front[y[:, 0], x[:, 0]])
    delta = snapped - front
    move = torch.stack([ramp_px, ramp_px & ((logr[1] - front).abs() < layer_tol)]) & (delta.abs() > min_move)
    s_ = torch.where(move, torch.exp(delta).expand(2, n, n), torch.ones(2, n, n, device=dev)).reshape(-1)
    mean = (world * s_[:, None] - t) @ torch.linalg.inv(M).T
    return g_ndc._replace(mean_vectors=mean, singular_values=g_ndc.singular_values * s_[:, None]), float(move.float().mean())


def write_ply(g: Gaussians3D, path: Path, spz: Path | None = None) -> None:
    """Minimal 3DGS ply (sRGB colours like SHARP's export), written in one go (the bytes plyfile writes, ~10x
    faster); spz: also the .spz (pano_spz.py) from the same data, without reading the ply back."""
    xyz = g.mean_vectors
    colors = convert_rgb_to_spherical_harmonics(cs_utils.linearRGB2sRGB(g.colors))
    op = torch.log(g.opacities / (1 - g.opacities)).clamp(-20, 20)[:, None]
    attrs = torch.cat([xyz, colors, op, torch.log(g.singular_values), g.quaternions], 1)
    names = (["x", "y", "z"] + [f"f_dc_{i}" for i in range(3)] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    arr = np.ascontiguousarray(attrs.cpu().numpy().astype("<f4"))
    el = arr.view(np.dtype([(k, "<f4") for k in names]))[:, 0]
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(el)}\n"
              + "".join(f"property float {k}\n" for k in names) + "end_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        el.tofile(fh)
    if spz is not None:
        import pano_spz
        pano_spz.write_spz(el, spz)


def flat(g: Gaussians3D) -> Gaussians3D:
    return Gaussians3D(*[t.flatten(0, -2) if t.dim() > 2 else t.flatten() for t in g])


def cat(gs: list[Gaussians3D]) -> Gaussians3D:
    return Gaussians3D(*[torch.cat(ts) for ts in zip(*gs)])


def select(g: Gaussians3D, m: torch.Tensor) -> Gaussians3D:
    return Gaussians3D(*[t[m] for t in g])


# ---------------------------------------------------------------- main


def run(args):
    device = torch.device("cuda")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(args.input).stem
    t0 = time.time()

    pano = cv2.cvtColor(cv2.imread(str(args.input), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    rots = view_layout(args.layout)
    fov = args.fov or default_fov(args.layout)
    size = INTERNAL
    f = size / 2 / math.tan(math.radians(fov) / 2)
    # Pre-shrink the panorama to ~1.5x the views' centre resolution (antialiasing).
    target_w = int(min(pano.shape[1], 1.5 * 2 * math.pi * f))
    if target_w < pano.shape[1]:
        pano = cv2.resize(pano, (target_w, target_w // 2), interpolation=cv2.INTER_AREA)
    LOGGER.info("%s: %d views, fov %.0f, f %.1f px, pano %dx%d", name, len(rots), fov, f, *pano.shape[1::-1])

    model = load_model(Path(args.checkpoint), device)
    factor = f / size  # disparity_factor = f_px / width

    views = []
    for R in rots:
        img = sample_view(pano, R, size, f)
        views.append(torch.from_numpy(img).to(device).permute(2, 0, 1)[None].float() / 255)
    if args.save_views:
        for k, v in enumerate(views):
            cv2.imwrite(str(out / f"{name}_view{k:02d}.jpg"),
                        cv2.cvtColor((v[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Pass 1: monodepth per view, stitched (outputs kept: pass 2 reuses them).
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16 in ("mono", "all")):
        depths, mono_outs = zip(*[monodepth(model, v, factor) for v in views])
    depths = [z.float() for z in depths]
    eq_w = args.depth_width
    sky = None                                 # DAP's sky mask (equirect), where a depth mode computed it
    if args.depth == "dap_ratio":              # each view's depth smoothly corrected towards DAP, then feather-blended
        # into one shared map: views aligned to their own corrected depth disagreed on thin objects (pano_9000's
        # lamp heads, ~30%) and showed them twice in the overlaps (final splats: 4.4% of overlap pixels > 5% apart,
        # shared: 0.55%, old stitch 0.49%). The corrections already agree at large scale, so the blend has no seams.
        t_ = time.time()
        dap, sky = dap_distance(pano, eq_w, device, with_sky=True)
        targets = dap_ratio_targets(depths, rots, f, size, dap, sky, device, args.ratio_deg,
                                    edge_stop=args.ratio_edge_stop, edge_ratio=args.ratio_edge_ratio)
        L, _, Wt = view_logdist(targets, rots, f, size, eq_w, eq_w // 2, device)
        if args.ratio_near_dap > 0:
            Wt = blend_near_dap(L, Wt, dap, sky, args.ratio_near_dap)
        logd = (Wt * L).sum(0) / Wt.sum(0).clamp(min=1e-8)
        del L, Wt, targets
        stats = {"depth": args.depth, "ratio_deg": args.ratio_deg, "ratio_edge_stop": args.ratio_edge_stop,
                 "ratio_edge_ratio": args.ratio_edge_ratio,
                 "ratio_near_dap": args.ratio_near_dap}
        LOGGER.info("depth: views corrected towards DAP (blur %g deg), %.1f s", args.ratio_deg, time.time() - t_)
    elif args.depth == "dap_seam":             # experiment: a projective fit per view to DAP, dap_ratio's smooth
        # correction on what is left, then each pixel's depth from one view (seams where views agree, the view
        # nearer DAP where they don't) instead of the feather average, which tilted the UNY banner (one view 2.5x
        # off averaged with one near right). Pass 2 then keeps a splat in the view its pixel was taken from.
        t_ = time.time()
        dap, sky = dap_distance(pano, eq_w, device, with_sky=True)
        fitted, params = dap_fit_views(depths, rots, f, size, dap, sky, device)
        targets = dap_ratio_targets(fitted, rots, f, size, dap, sky, device, args.ratio_deg)
        L, M, Wt = view_logdist(targets, rots, f, size, eq_w, eq_w // 2, device)
        fb = (Wt * L).sum(0) / Wt.sum(0).clamp(min=1e-8)
        labels = seam_labels(L, M, Wt, dap, sky, device)
        logd = seam_blend(L, M, labels, fb, blend=args.seam_blend)
        del L, M, Wt, targets, fitted, fb
        cv2.imwrite(str(out / f"{name}_seams.png"),
                    np.random.RandomState(1).randint(40, 255, (len(rots), 3)).astype(np.uint8)[labels.cpu().numpy()])
        stats = {"depth": args.depth, "ratio_deg": args.ratio_deg, "seam_blend": args.seam_blend, "fit": params}
        LOGGER.info("depth: views fitted to DAP, corrected, seams (%s), %.1f s", args.seam_blend, time.time() - t_)
    elif args.depth in ("unik3d", "dap"):        # one 360 prediction; pass 1 still gives pass 2 its monodepth features
        logd = torch.log((unik3d_distance if args.depth == "unik3d" else dap_distance)(pano, eq_w, device))
        stats = {"depth": args.depth}
        LOGGER.info("depth: %s 360, distance %.2f .. %.2f m (median %.2f)", args.depth, *torch.exp(logd).quantile(
            torch.tensor([0.001, 0.999, 0.5], device=device)).tolist())
    else:
        logd, s, stats = stitch_depth(depths, rots, f, size, eq_w, eq_w // 2, device)
        LOGGER.info("depth stitch: %s", json.dumps(stats))
        if args.depth in ("poisson", "poisson_uk", "poisson_dap"):
            ref, rw = logd, 1e-3
            if args.depth == "poisson_dap":      # large-scale shape from DAP (flat ceilings, straight rails), detail
                # from the views: DAP alone is too smooth (hotel_room: the desk chair merged into the desk, chair
                # edges and the table corner pulled towards the wall, a traffic-light pole lost)
                # In DAP's sky (a flat 100 m) the reference is the views' own sky, offset to DAP's scale: the jump
                # to 100 m disagreed with the views' and dragged building edges next to the sky towards the eye.
                dap, sky = dap_distance(pano, eq_w, device, with_sky=True)
                x0 = poisson_depth(depths, rots, f, size, eq_w, eq_w // 2, device, logd, 1e-3)
                ref = torch.log(dap)
                if sky.any() and not sky.all():
                    ref = torch.where(sky, x0 + (ref - x0)[~sky].median(), ref)
                rw = args.dap_weight
            elif args.depth == "poisson_uk":     # overall shape from UniK3D, its scale matched to the stitched map
                uk = torch.log(unik3d_distance(pano, eq_w, device))
                ref, rw = uk + (logd - uk).median(), args.uk_weight
                if args.uk_fade:                 # only near the horizon: UniK3D keeps long walls / railings straight
                    # but bends ceilings and floors, which the face-on up / down views get flat (pano_9000)
                    lat = (0.5 - (torch.arange(eq_w // 2, device=device) + 0.5) / (eq_w // 2)) * 180
                    t_ = ((lat.abs() - args.uk_fade[0]) / (args.uk_fade[1] - args.uk_fade[0])).clamp(0, 1)
                    rw = (args.uk_weight * (1 - t_ * t_ * (3 - 2 * t_)) + 1e-3)[:, None]
            t_ = time.time()
            fe = (*args.free_edges[:2], 1e-3, *args.free_edges[2:]) if args.free_edges and args.depth == "poisson_dap" else None
            logd = poisson_depth(depths, rots, f, size, eq_w, eq_w // 2, device, ref, rw, free_edges=fe)
            if fe:
                stats["free_edges"] = args.free_edges
            stats["depth"] = args.depth
            LOGGER.info("depth: gradient-domain stitching (%s, reference weight up to %g), %.1f s", args.depth,
                        float(torch.as_tensor(rw).max()), time.time() - t_)
    dist = torch.exp(logd)
    np.save(out / f"{name}_dist.npy", dist.cpu().numpy())
    vis = (1 / dist).cpu().numpy()
    vis = (255 * vis / np.percentile(vis, 99)).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(out / f"{name}_disp.png"), cv2.applyColorMap(vis, cv2.COLORMAP_TURBO))

    # Pass 2: aligned SHARP per view, rotated to world, Voronoi ownership.
    axes = torch.tensor(np.stack([R[:, 2] for R in rots]), device=device, dtype=torch.float32)
    rays = torch.tensor(face_rays(size, f), device=device, dtype=torch.float32)
    K = torch.tensor([[f, 0, size / 2, 0], [0, f, size / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                     device=device, dtype=torch.float32)
    kept = []
    # Depth edges of the shared map (log-distance change per degree, widened by ~3.5 deg): no overlap there.
    gx = torch.roll(logd, -1, 1) - logd
    gy = F.pad(logd[1:] - logd[:-1], (0, 0, 0, 1))
    edges = F.max_pool2d((torch.sqrt(gx ** 2 + gy ** 2) * (logd.shape[1] / 360))[None, None], 15, 1, 7)
    seams = labels[None, None].float() if args.depth == "dap_seam" else None
    ov = int(round(args.overlap_deg * logd.shape[1] / 360))
    # The covariance transform (CPU SVD, ~0.9 s a view) runs on a thread while the GPU predicts the next view.
    from concurrent.futures import ThreadPoolExecutor
    cpu = ThreadPoolExecutor(1)
    for k, (R, img) in enumerate(zip(rots, views)):
        Rt = torch.tensor(R, device=device, dtype=torch.float32)
        target = None
        if not args.no_align:
            dworld = rays @ Rt.T
            dworld = dworld / dworld.norm(dim=-1, keepdim=True)
            # dirs_to_equirect on the GPU (the numpy version, 2.4M rays a view, cost ~0.3 s a view)
            lon = torch.atan2(dworld[..., 0], dworld[..., 2])
            lat = torch.asin((-dworld[..., 1]).clamp(-1, 1))
            grid = torch.stack([lon / math.pi, -lat / (math.pi / 2)], -1)          # = u / W * 2 - 1, v / H * 2 - 1
            # Wrap-safe enough: logd is continuous across the seam except at the exact column.
            lr = F.grid_sample(logd[None, None], grid[None], mode="bilinear", padding_mode="border",
                               align_corners=False)[0, 0]
            target = torch.exp(lr) / rays.norm(dim=-1)  # distance -> z-depth
            sky_v = None if sky is None else F.grid_sample(sky[None, None].float(), grid[None], mode="bilinear",
                                                           padding_mode="border", align_corners=False)[0, 0]
        # The alignment only corrects SHARP's own depth by moderate factors: a view far off (a street panorama's
        # downward views had SHARP's road at 0.33x / 0.42x the target) stayed at 0.59x / 0.77x and floated as a
        # big triangle. So the view is aligned to the target divided by its overall factor, and its splats are
        # scaled by that factor afterwards (about the camera: the same image from there).
        sc = float((target / depths[k]).median()) if target is not None else 1.0
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16 == "all"):
            g_ndc = flat(predict_aligned(model, img, factor, None if target is None else target / sc, mono_outs[k]))
        g_ndc = Gaussians3D(*[t.float() for t in g_ndc])
        if sc != 1.0:
            g_ndc = g_ndc._replace(mean_vectors=g_ndc.mean_vectors * sc, singular_values=g_ndc.singular_values * sc)
        E = torch.eye(4, device=device)
        E[:3, :3] = Rt.T  # world -> camera
        unproj = get_unprojection_matrix(E, K, (size, size))
        if args.post_align and target is not None:
            g_ndc, moved = post_align(g_ndc, unproj, target * rays.norm(dim=-1), px_deg=f * math.pi / 180 / (size / 768),
                                      sky=sky_v)
            LOGGER.info("view %d: post-aligned to the shared map, %.2f%% of front splats moved > 5%%", k, 100 * moved)
        if args.snap_edges:
            g_ndc, moved = snap_edges(g_ndc, unproj)
            LOGGER.info("view %d: depth edges snapped to the photo, %.2f%% of splats moved", k, 100 * moved)
        # Ownership from the world-space centres first: only the kept third gets the (CPU, per-splat SVD)
        # covariance transform. A splat is kept by the view whose axis is nearest its direction, and also by any
        # view within overlap_deg more of it: with hard boundaries, neighbouring views' small depth disagreements
        # opened cracks along them (triangles on ceilings) as soon as the eye moved.
        mean = g_ndc.mean_vectors @ unproj[:3, :3].T + unproj[:3, 3]
        ang = torch.acos((mean / mean.norm(dim=-1, keepdim=True) @ axes.T).clamp(-1, 1))   # (N, views)
        keep = ang[:, k] <= ang.min(-1).values + math.radians(args.overlap_deg)
        if args.overlap_tol > 0:
            # Past the boundary only splats on the shared depth surface and away from its depth edges: cracks open
            # on smooth surfaces the views share (pano_9000's ceiling), while near object outlines the neighbouring
            # view's copy showed as a blurred black double of a lamp head (pano_9000, every eye, even the centre).
            u = mean / mean.norm(dim=-1, keepdim=True)
            grid = torch.stack([torch.atan2(u[:, 0], u[:, 2]) / math.pi,
                                -torch.asin((-u[:, 1]).clamp(-1, 1)) / (math.pi / 2)], -1)[None, None]
            lr = F.grid_sample(logd[None, None], grid, mode="bilinear", padding_mode="border",
                               align_corners=False)[0, 0, 0]
            es = F.grid_sample(edges, grid, mode="nearest", padding_mode="border", align_corners=False)[0, 0, 0]
            on = ((torch.log(mean.norm(dim=-1)) - lr).abs() < args.overlap_tol) & (es < args.overlap_edge)
            keep = (ang[:, k] <= ang.min(-1).values) | (keep & on)
            if seams is not None:              # dap_seam: the view the shared map took this pixel from, and within
                # overlap_deg of that region the splats on the shared surface
                mine = (seams == k).float()
                near = F.max_pool2d(F.pad(mine, (ov, ov, 0, 0), mode="circular"), 2 * ov + 1, 1, (ov, 0))
                lab_s = F.grid_sample(seams, grid, mode="nearest", padding_mode="border", align_corners=False)[0, 0, 0]
                nr = F.grid_sample(near, grid, mode="nearest", padding_mode="border", align_corners=False)[0, 0, 0]
                keep = (lab_s == k) | ((nr > 0) & on)
        owner = torch.where(keep, k, -1)
        own = select(g_ndc, owner == k)
        kept.append(cpu.submit(lambda own=own, m=unproj[:3]: flat(apply_transform(Gaussians3D(*[t[None] for t in own]), m))))
        LOGGER.info("view %d: kept %d / %d", k, int((owner == k).sum()), len(owner))
    g = cat([f.result() for f in kept])
    cpu.shutdown()
    if args.min_opacity > 0:
        g = select(g, g.opacities > args.min_opacity)
    write_ply(g, out / f"{name}.ply", None if args.no_spz else out / f"{name}.spz")
    info = dict(image=Path(args.input).name, panorama=True, layout=args.layout, fov=fov,
                views=len(rots), splats=len(g.opacities), aligned=not args.no_align,
                stitch=stats, seconds=round(time.time() - t0, 1))
    (out / f"{name}.json").write_text(json.dumps(info, indent=2))
    LOGGER.info("wrote %s (%d splats) in %.1fs", out / f"{name}.ply", len(g.opacities), time.time() - t0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("-o", "--output", default="pano/out")
    p.add_argument("--layout", default="ico", choices=["cube", "ico", "icoup"])
    p.add_argument("--fov", type=float, default=None, help="View FOV in degrees (default per layout).")
    p.add_argument("--checkpoint", default="sharp_2572gikvuh.pt")
    p.add_argument("--depth-width", type=int, default=2048, help="Stitched distance map width.")
    p.add_argument("--seam-blend", choices=["feather", "multiband"], default="feather",
                   help="dap_seam: merge along the seams with a 4 px feather or Laplacian (multiband) blending")
    p.add_argument("--depth", choices=["stitch", "unik3d", "dap", "poisson", "poisson_uk", "poisson_dap", "dap_ratio",
                                       "dap_seam"], default="stitch",
                   help="the 360 distance map pass 2 aligns to: the views' monodepth stitched (one scale per view), "
                        "UniK3D or DAP on the whole panorama (unik3d_distance, dap_distance), the views' depth "
                        "slopes integrated "
                        "(poisson_depth), or that pulled towards UniK3D's / DAP's overall shape (poisson_uk, --uk-weight; "
                        "poisson_dap, --dap-weight), or each view's own depth smoothly corrected towards DAP (dap_ratio, "
                        "dap_ratio_targets)")
    p.add_argument("--uk-weight", type=float, default=0.05, help="poisson_uk: weight of the pull towards UniK3D")
    p.add_argument("--ratio-deg", type=float, default=10.0,
                   help="dap_ratio: DAP reshapes only what is wider than about this (degrees)")
    p.add_argument("--ratio-edge-stop", type=float, default=0.0,
                   help="dap_ratio: the correction stops at depth steps > this (log) that DAP also sees, e.g. 0.08 "
                        "(dap_ratio_targets' edge_stop; 0 = plain blur)")
    p.add_argument("--ratio-edge-ratio", type=float, default=0.0,
                   help="dap_ratio with --ratio-edge-stop: cut only steps where DAP changes by >= this x the view's "
                        "step, e.g. 0.3, and judge the robust weights against the edge-stopped blur (edge_ratio)")
    p.add_argument("--ratio-near-dap", type=float, default=0.0,
                   help="dap_ratio: where views disagree, favour the one nearer DAP (blend_near_dap's soft, e.g. 0.1; "
                        "0 = feather average)")
    p.add_argument("--post-align", action="store_true",
                   help="after pass 2, scale each view's splats along their rays towards the shared map, smoothly and "
                        "stopping at the view's depth steps (post_align): views at an overall scale far from a "
                        "surface's agreed depth split it at their boundary")
    p.add_argument("--snap-edges", action="store_true",
                   help="move splats at each view's depth jumps to the side whose colour they match (snap_edges)")
    p.add_argument("--free-edges", type=float, nargs="+", default=None, metavar="VIEW RATIO [SOFT [DIS]]",
                   help="poisson_dap: where the views step by > VIEW (log) and DAP jumps the same way by > RATIO x as "
                        "much, DAP sets the jump; other steps > VIEW get SOFT x the weight; DIS > 0: also free steps the "
                        "views disagree on by > DIS x the jump (poisson_depth's free_edges); e.g. 0.05 1.5 1 0.2")
    p.add_argument("--dap-weight", type=float, default=1e-3,
                   help="poisson_dap: weight of the pull towards DAP (detail scale ~ 1/sqrt(weight) equirect px)")
    p.add_argument("--uk-fade", type=float, nargs=2, default=None, metavar=("FROM", "TO"),
                   help="poisson_uk: the pull fades out between these elevations (degrees, e.g. 30 60)")
    p.add_argument("--no-align", action="store_true", help="Skip pass 2 alignment (baseline).")
    p.add_argument("--min-opacity", type=float, default=0.0)
    p.add_argument("--no-spz", action="store_true", help="don't also write <name>.spz (scripts/pano_spz.py)")
    # 5: pano_9000 ceiling from eyes 20-25 cm off-centre, in the viewer: crack pixels 15356 (hard boundaries,
    # stitched) / 943 (2 deg) / 0 (5 deg); +37% splats, MSI cells about the same (culling drops the duplicates)
    p.add_argument("--overlap-deg", type=float, default=5.0,
                   help="views keep their splats this far past their (nearest-axis) boundary: no cracks there "
                        "from a moved eye (0 = hard boundaries)")
    p.add_argument("--overlap-tol", type=float, default=0.05,
                   help="past the boundary, keep only splats within this (log distance) of the shared depth map "
                        "and away from its depth edges (--overlap-edge); 0 = all")
    p.add_argument("--overlap-edge", type=float, default=0.1,
                   help="no overlap where the shared depth map changes faster than this (log distance per degree)")
    p.add_argument("--save-views", action="store_true")
    # mono: depth pass 3x faster (0.5 vs 1.6 s a view on the GB10) and +0.6 / +1.2 dB vs the panorama
    # (R0011227 / hotel_room); all (default): ~2 s faster still, within -0.08..+0.31 dB of mono on both.
    # (Scored from the centre, which hides depth errors: bf16 depth was quantized until the head ran in fp32.)
    p.add_argument("--bf16", choices=["off", "mono", "all"], default="all",
                   help="bfloat16 autocast for the monodepth pass (mono) or the whole network (all)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
