"""360 splats (pano_predict.py) -> sparse multi-sphere image (MSI), the 360 analogue of MPI-only.

Mirrors the perspective sparse MPI (eval_alpha_patches.py MPI mode, docs/quest-xr-findings.md §4b):
- BINS concentric spheres around the capture point, uniform in disparity (0.1-99.9 percentile of the
  splats' 1/distance). A viewer inside the innermost sphere always sees them in far-to-near order, so
  the draw order is static, like the MPI's planes.
- Each bin's splats are rendered together from the centre into a cube map (faces FACE px at SS=2
  supersampling, CELL_BORDER-texel borders of real neighbour content) and cut into TILE-px cells;
  empty cells are dropped. Cell corners lie on the bin's sphere, refined to the alpha-weighted
  distance of the bin's splats in the neighbouring cells (clamped to the bin, so bins stay disjoint).
- Textures: premultiplied RGBA, half resolution (TEXDS) unless a cell loses more than TEXFULL_THR of
  detail (worst first, within MAXPAGES 4096^2 pages). All cells are 4-texel aligned squares, so the 3
  box mip levels of the KTX2 file never mix cells.
- Export: <name>_msi.glb with the viewer's layer layout (one mesh per atlas page, surface_<k>, larger k
  drawn first) and <name>_msi_ktx2.glb (scripts/glb_ktx2.py, UASTC + 3 mips).

Culling (--cull 0.05): cells that contribute less than this (alpha x transmittance in front) to every
pixel from every eye position within +-0.3 m (centre, the 6 axis ends, 8 corners of a +-0.2 m box) are
dropped, like the perspective MPI's CULL (docs/quest-xr-findings.md §4b.12). --ktx2-rdo passes a UASTC
rate-distortion lambda to glb_ktx2.py (smaller download, same GPU memory).

Eval (--eval): ray-sphere compositing of the bins (bin-centre radii, export resolution) vs gsplat
renders of the splats, from the centre and moved 0.1 / 0.3 m.

Usage:
    python scripts/pano_msi.py pano/out_ico2/hotel_room.ply [--eval]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).parent))
from pano_predict import look_rotation  # noqa: E402
from sharp.cli.bake_layers import load_gsplat, splat_tensors  # noqa: E402

FACE_DIRS = [(0, 0, 1), (1, 0, 0), (0, 0, -1), (-1, 0, 0), (0, -1, 0), (0, 1, 0)]
PAGE = 4096


def render_faces(gs, sp, rots, size, f):
    """Premultiplied RGB, alpha and expected z-depth of splats `sp` on the 6 cube faces."""
    dev = sp["means"].device
    view = torch.zeros(6, 4, 4, device=dev)
    view[:, :3, :3] = rots.transpose(1, 2)
    view[:, 3, 3] = 1
    K = torch.tensor([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]], device=dev).expand(6, 3, 3)
    out, alpha, _ = gs.rendering.rasterization(**sp, viewmats=view, Ks=K, width=size, height=size,
                                               render_mode="RGB+ED", near_plane=0.01)
    return out[..., :3], alpha[..., 0], out[..., 3]


def cube_lookup(dirs: torch.Tensor):
    """Unit world dirs (N,3) -> (face index, face x, face y in [-1,1])."""
    rots = [look_rotation(np.array(d, float)) for d in FACE_DIRS]
    axes = torch.tensor(np.stack([R[:, 2] for R in rots]), device=dirs.device, dtype=dirs.dtype)
    face = (dirs @ axes.T).argmax(-1)
    R = torch.tensor(np.stack(rots), device=dirs.device, dtype=dirs.dtype)[face]
    c = torch.einsum("nij,ni->nj", R, dirs)
    return face, c[:, 0] / c[:, 2], c[:, 1] / c[:, 2]


@torch.no_grad()
def photo_detail(pano_path, render_bin, b, B, rots_np, S, fr, dev, min_w, centres, edge_jump, edge_px,
                 edge_darken, alpha_thr=None, erode_px=2, surface_band=0.05):
    """Per render pixel: the bin that dominates it from the centre and the colour to add to that bin so the
    composite from the centre equals the panorama (splats are ~2x softer than the photo).

    Composites the bins near to far (C, transmittance T), keeps per pixel the bin with the largest weight
    T * alpha and T in front of it; residual D = pano - C goes onto that bin as D / T (clamped by the caller
    to a valid premultiplied colour), only where the bin's weight is >= min_w (no amplified noise on
    faint layers). Moving the head, the detail stays on the surface that owns the pixel.
    Not at depth edges: there the splats' soft edge differs from the photo's (e.g. a dark pole blurred
    into the bright sky), and the residual (brighten) would land on the front layer as a bright fringe
    that shows as soon as the eye is off-centre. Pixels within edge_px of a jump in the dominant bin's
    disparity by more than edge_jump (relative) get no detail; it fades in over the next edge_px.
    With edge_darken, corrections that darken are kept there too: the halos came from brightening
    (photo edges are sharper than the splats', so a dark front object needs its blurred rim brightened).
    With alpha_thr (v4), the mask is the opacity of the dominant *surface* instead of disparity jumps: the
    bins within surface_band (relative disparity, >= 1.5 bins) of the dominant bin, since a surface is sliced
    over neighbouring spheres and rarely one of them is opaque alone. Detail only where that surface's
    alpha >= alpha_thr at every pixel within erode_px (inside opaque surfaces, away from rims whose painted
    colour would slide when the eye moves), spread over the surface's bins in proportion to their alpha,
    i.e. the surface's own colour shifts by D / (its weight from the centre).
    Returns apply(k, rgb, alpha) -> bin k's premultiplied rgb with the detail added (caller clamps).
    """
    import cv2

    from pano_predict import sample_view

    pano = cv2.cvtColor(cv2.imread(str(pano_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    P = torch.stack([torch.from_numpy(sample_view(pano, R, S, fr)) for R in rots_np]).to(dev).float() / 255
    C = torch.zeros(6, S, S, 3, device=dev)
    T = torch.ones(6, S, S, device=dev)
    best_w = torch.zeros(6, S, S, device=dev)
    best_T = torch.ones(6, S, S, device=dev)
    best_a = torch.zeros(6, S, S, device=dev)
    best_k = torch.full((6, S, S), -1, device=dev, dtype=torch.int16)
    alphas = {}                            # kept for the second pass (v4), which needs only alpha (~13 GB)
    for k in range(B - 1, -1, -1):         # near first
        if not (b == k).any():
            continue
        rgb, alpha, _ = render_bin(k)
        alpha = alpha.clamp(0, 1)
        if alpha_thr is not None:
            alphas[k] = alpha
        w = T * alpha
        better = w > best_w
        best_w = torch.where(better, w, best_w)
        best_T = torch.where(better, T, best_T)
        best_a = torch.where(better, alpha, best_a)
        best_k = torch.where(better, torch.full_like(best_k, k), best_k)
        C += T[..., None] * rgb.clamp(0, 1)
        T = T * (1 - alpha)
    D = P - C
    ok = best_w >= min_w
    disp = torch.tensor(centres, device=dev, dtype=torch.float32)[best_k.long().clamp(min=0)][:, None]
    k = 2 * edge_px + 1
    jump = (F.max_pool2d(disp, k, 1, edge_px) + F.max_pool2d(-disp, k, 1, edge_px)) > edge_jump * disp
    jump = F.max_pool2d(jump.float(), k, 1, edge_px)                        # no detail within edge_px
    weight = (1 - F.avg_pool2d(jump, k, 1, edge_px, count_include_pad=False))[:, 0]   # fade back in
    inner = (slice(None), slice(S // 64, -S // 64), slice(S // 64, -S // 64))
    if alpha_thr is not None:
        cen = torch.tensor(centres, device=dev, dtype=torch.float32)
        bin_w = float(cen[1] - cen[0]) if len(centres) > 1 else 0.0
        d_best = cen[best_k.long().clamp(min=0)]
        band = torch.maximum(surface_band * d_best, torch.full_like(d_best, 1.5 * bin_w))
        # Second pass, near first: the dominant surface's opacity and its weight seen from the centre.
        T2 = torch.ones(6, S, S, device=dev)
        surf_T = torch.ones(6, S, S, device=dev)
        surf_w = torch.zeros(6, S, S, device=dev)
        for kb in range(B - 1, -1, -1):
            if not (b == kb).any():
                continue
            alpha = alphas.pop(kb)
            inb = (cen[kb] - d_best).abs() <= band
            surf_T = torch.where(inb, surf_T * (1 - alpha), surf_T)
            surf_w = surf_w + torch.where(inb, T2 * alpha, 0)
            T2 = T2 * (1 - alpha)
        opaque = (((1 - surf_T) >= alpha_thr) & (best_k >= 0)).float()[:, None]
        kw = 2 * erode_px + 1
        weight = (-F.max_pool2d(-opaque, kw, 1, erode_px))[:, 0]              # erosion: whole window opaque
        shift = weight[..., None] * D / surf_w[..., None].clamp(min=0.3)      # surface colour change (straight)
        print(f"  photo detail (surface opacity >= {alpha_thr}, band {surface_band}, erode {erode_px} px): "
              f"residual |D| mean {D[inner].abs().mean():.4f}, applied on {weight[inner].mean():.1%} of pixels, "
              f"residual left {(D * (1 - weight[..., None]))[inner].abs().mean():.4f}", flush=True)

        def apply_surface(kb, rgb, alpha):
            inb = (cen[kb] - d_best).abs() <= band
            return rgb + torch.where(inb[..., None], alpha[..., None] * shift, 0)
        return apply_surface
    w3 = weight[..., None]
    if edge_darken:
        w3 = torch.where(D < 0, 1.0, w3)
    delta = torch.where(ok[..., None], w3 * D / best_T[..., None].clamp(min=1e-3), 0)
    best_k = torch.where(ok, best_k, torch.full_like(best_k, -1))
    print(f"  photo detail: residual |D| mean {D[inner].abs().mean():.4f}, applied on {ok.float().mean():.1%} "
          f"of pixels (dominant weight >= {min_w}), {(weight == 0).float().mean():.1%} masked as depth edges, "
          f"{(weight < 1).float().mean():.1%} faded", flush=True)

    def apply_dominant(kb, rgb, alpha):
        return torch.where((best_k == kb)[..., None], rgb + delta, rgb)
    return apply_dominant


CULL_EYES = ([(0, 0, 0)] + [tuple(0.3 * s * np.eye(3)[a]) for a in range(3) for s in (-1, 1)]
             + [(x, y, z) for x in (-0.2, 0.2) for y in (-0.2, 0.2) for z in (-0.2, 0.2)])


@torch.no_grad()
def cell_contrib(alphas, centres, nc, dev, res=256, eyes=CULL_EYES):
    """Largest contribution (alpha x transmittance of nearer spheres) of each cell over `eyes`.

    alphas: {bin: (2, 6, R, R) uint8 coarse alpha of that sphere's cube map, [0] max-pooled, [1]
    min-pooled}. Each eye shoots a cube of rays (res^2 per face), intersects them with every sphere
    (bin-centre radius) near to far and keeps, per (bin, face, cell), the max of alpha x T. Both estimates
    err towards keeping cells: a cell's own alpha is the block maximum, the occlusion it causes for the
    spheres behind uses the block minimum (max-pooled occluders dropped visible slices of the ground).
    Returns (B, 6, nc, nc) float.
    """
    B = len(centres)
    best = torch.zeros(B, 6 * nc * nc, device=dev)
    c = (torch.arange(res, device=dev) + 0.5) / res * 2 - 1
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    cam = torch.stack([xx, yy, torch.ones_like(xx)], -1).reshape(-1, 3)
    rots = [look_rotation(np.array(d, float)) for d in FACE_DIRS]
    dirs = torch.cat([cam @ torch.tensor(R, device=dev, dtype=torch.float32).T for R in rots])
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    order = sorted(alphas, reverse=True)                     # near first (bin B-1 = nearest)
    for e in eyes:
        eye = torch.tensor(e, device=dev, dtype=torch.float32)
        de, ee = dirs @ eye, eye @ eye
        T = torch.ones(len(dirs), device=dev)
        for k in order:
            rk = 1 / centres[k]
            p = eye + (-de + torch.sqrt(de ** 2 - ee + rk ** 2))[:, None] * dirs
            fi, fx, fy = cube_lookup(p / p.norm(dim=-1, keepdim=True))
            a_map = alphas[k]
            R_ = a_map.shape[-1]
            ix = ((fx + 1) / 2 * R_).long().clamp(0, R_ - 1)
            iy = ((fy + 1) / 2 * R_).long().clamp(0, R_ - 1)
            a_own = a_map[0, fi, iy, ix].float() / 255
            a_occ = a_map[1, fi, iy, ix].float() / 255
            cidx = fi * nc * nc + (iy * nc // R_) * nc + (ix * nc // R_)
            best[k].scatter_reduce_(0, cidx, a_own * T, reduce="amax")
            T = T * (1 - a_occ)
    return best.reshape(B, 6, nc, nc)


def run(args, writer=None):
    """writer: optional executor; the files are then written on it (returns its future) while the caller goes on."""
    dev = torch.device("cuda")
    gs = load_gsplat()
    ply = Path(args.ply)
    t0 = time.time()
    v = PlyData.read(str(ply))["vertex"].data
    sp_all = splat_tensors(v, dev)
    r = sp_all["means"].norm(dim=-1)
    d = (1 / r).cpu().numpy()
    lo, hi = np.percentile(d, [0.1, 99.9])
    B = args.bins
    edges = lo + np.arange(1, B) / B * (hi - lo)
    centres = lo + (np.arange(B) + 0.5) / B * (hi - lo)
    bin_lo_r = 1 / (lo + (np.arange(B) + 1) / B * (hi - lo))      # near edge radius
    bin_hi_r = 1 / (lo + np.arange(B) / B * (hi - lo)).clip(1e-6)  # far edge radius
    b = torch.tensor(np.searchsorted(edges, d), device=dev)
    print(f"{ply.stem}: {len(d)} splats, {B} bins, radius {1 / hi:.2f}..{1 / max(lo, 1e-6):.1f} m", flush=True)

    SS, TEXDS, BORDER = 2, args.texds, args.border
    Fr = args.face * SS                    # render px per face (inner)
    cell = args.tile * SS                  # render px per cell
    gb = BORDER * TEXDS                    # border, render px
    S = Fr + 2 * gb
    fr = Fr / 2
    nc = Fr // cell
    rots_np = [look_rotation(np.array(dd, float)) for dd in FACE_DIRS]
    rots = torch.tensor(np.stack(rots_np), device=dev, dtype=torch.float32)
    # Camera rays of the render pixels (z = 1) -> distance factor for the expected z-depth.
    c = (torch.arange(S, device=dev) + 0.5 - S / 2) / fr
    ray_norm = torch.sqrt(c[None, :] ** 2 + c[:, None] ** 2 + 1)

    # Splats sorted by bin once: each bin is a slice (a boolean mask over all splats per bin, three passes with
    # photo detail, cost ~7 s).
    order = torch.argsort(b, stable=True)
    starts = np.r_[0, np.cumsum(torch.bincount(b, minlength=B).cpu().numpy())]
    sp_sorted = {key: val[order] for key, val in sp_all.items()}
    has_bin = starts[1:] > starts[:-1]
    b = b.cpu().numpy()                                # photo_detail's (b == k).any(): numpy, no GPU sync

    def render_bin(k):
        sp = {key: val[int(starts[k]):int(starts[k + 1])] for key, val in sp_sorted.items()}
        return render_faces(gs, sp, rots, S, fr)

    detail = None
    if args.photo_detail:
        detail = photo_detail(args.photo_detail, render_bin, b, B, rots_np, S, fr, dev, args.detail_min_w,
                              centres, args.edge_jump, args.edge_px, args.edge_darken,
                              args.detail_alpha, args.detail_erode, args.surface_band)

    cells = []        # (bin, face, cy, cx)
    half_tex, full_tex, errs, wsum, rmean = [], {}, [], [], []
    eval_faces = {} if args.eval else None
    cull_alpha = {}
    for k in range(B):                    # far first (bin 0 = lowest disparity)
        if not has_bin[k]:
            continue
        rgb, alpha, ed = render_bin(k)
        if detail is not None:
            rgb = detail(k, rgb, alpha)
            rgb = torch.minimum(rgb.clamp(min=0), alpha[..., None])               # valid premultiplied colour
        rgba = torch.cat([rgb, alpha[..., None]], -1).clamp(0, 1)                 # (6,S,S,4) premultiplied
        dist = ed * ray_norm
        inner = rgba[:, gb:gb + Fr, gb:gb + Fr]
        a_in = inner[..., 3].reshape(6, nc, cell, nc, cell)
        alive = (a_in >= 0.5 / 255).any(4).any(2)                                 # (6, nc, nc)
        if args.cull > 0:                                     # coarse alpha, max and min (4 samples per cell side)
            a_c = inner[..., 3][:, None]
            pk = Fr // (nc * 4)
            cull_alpha[k] = (torch.stack([F.max_pool2d(a_c, pk)[:, 0], -F.max_pool2d(-a_c, pk)[:, 0]]) * 255
                             ).round().to(torch.uint8)
        a_cell = a_in.sum((2, 4))
        dw = (dist[:, gb:gb + Fr, gb:gb + Fr] * inner[..., 3]).reshape(6, nc, cell, nc, cell).sum((2, 4))
        if eval_faces is not None:
            eval_faces[k] = (F.avg_pool2d(inner.permute(0, 3, 1, 2), TEXDS) * 255).round().to(torch.uint8)
        idx = alive.nonzero()
        if not len(idx):
            continue
        # Crops with borders: (n, cb, cb, 4)
        cb = cell + 2 * gb
        crops = rgba.unfold(1, cb, cell).unfold(2, cb, cell)                    # (6, nc', nc', 4, cb, cb)
        crops = crops[idx[:, 0], idx[:, 1], idx[:, 2]].permute(0, 2, 3, 1)       # (n, cb, cb, 4)
        t = crops.permute(0, 3, 1, 2)
        h = F.avg_pool2d(t, TEXDS)
        up = F.interpolate(h, scale_factor=TEXDS, mode="bilinear", align_corners=False)
        err = (up - t)[:, :3, gb:cb - gb, gb:cb - gb].abs().mean((1, 2, 3))
        n0 = sum(len(c_) for c_ in cells)
        cells.append(np.c_[np.full(len(idx), k), idx.cpu().numpy()])
        half_tex.append((h.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu())
        errs.append(err.cpu())
        cand = (err > args.texfull_thr).nonzero()[:, 0]
        if len(cand):
            ft = (crops[cand] * 255).round().to(torch.uint8).cpu()
            full_tex.update(zip((n0 + cand.cpu().numpy()).tolist(), ft))
        wsum.append(a_cell[idx[:, 0], idx[:, 1], idx[:, 2]].cpu())
        rmean.append((dw[idx[:, 0], idx[:, 1], idx[:, 2]] / a_cell[idx[:, 0], idx[:, 1], idx[:, 2]].clamp(min=1e-6)).cpu())
    cells = np.concatenate(cells)
    half_tex = torch.cat(half_tex).numpy()
    errs = torch.cat(errs).numpy()
    wsum = torch.cat(wsum).numpy()
    rmean = torch.cat(rmean).numpy()
    N = len(cells)
    print(f"  {N} cells ({N / B:.0f}/bin), {len(full_tex)} full-res candidates, {time.time() - t0:.0f}s", flush=True)

    keep = np.ones(N, bool)
    if args.cull > 0:
        contrib = cell_contrib(cull_alpha, centres, nc, dev)
        keep = (contrib[tuple(torch.tensor(cells.T, device=dev).long())] >= args.cull).cpu().numpy()
        del cull_alpha
        print(f"  cull {args.cull}: keeping {keep.sum()} of {N} cells ({1 - keep.mean():.1%} dropped)", flush=True)

    # Adaptive resolution within the page budget (kept cells only).
    hs, fs = half_tex.shape[1], (args.tile * SS + 2 * gb)
    budget = 0.95 * args.maxpages * PAGE * PAGE - keep.sum() * hs * hs
    extra = fs * fs - hs * hs
    cand = [i for i in np.argsort(-errs) if i in full_tex and keep[i]][:max(0, int(budget // extra))]
    full = np.zeros(N, bool)
    full[cand] = True
    print(f"  {full.sum()} cells at full resolution", flush=True)

    # Per-cell radius refinement: corners = alpha-weighted mean of the (up to 4) cells sharing them,
    # clamped to the bin; corners on a face border stay at the bin centre (shared with the next face).
    key = {tuple(cc): i for i, cc in enumerate(cells)}
    corner_r = np.empty((N, 4))      # (x0y0, x1y0, x1y1, x0y1)
    for i, (k, fc, cy, cx) in enumerate(cells):
        for j, (oy, ox) in enumerate([(0, 0), (0, 1), (1, 1), (1, 0)]):
            gy, gx = cy + oy, cx + ox
            if not args.refine or gy in (0, nc) or gx in (0, nc):
                corner_r[i, j] = 1 / centres[k]
                continue
            ws = rs = 0.0
            for ny, nx in [(gy - 1, gx - 1), (gy - 1, gx), (gy, gx - 1), (gy, gx)]:
                n = key.get((k, fc, ny, nx))
                if n is not None:
                    ws += wsum[n]
                    rs += wsum[n] * rmean[n]
            rr = rs / ws if ws > 0 else 1 / centres[k]
            corner_r[i, j] = np.clip(rr, bin_lo_r[k], bin_hi_r[k])

    # Corner directions (world, OpenCV frame).
    gx0 = cells[:, 3] * cell / fr - 1
    gy0 = cells[:, 2] * cell / fr - 1
    step = cell / fr
    cx_ = np.stack([gx0, gx0 + step, gx0 + step, gx0], 1)
    cy_ = np.stack([gy0, gy0, gy0 + step, gy0 + step], 1)
    cam = np.stack([cx_, cy_, np.ones_like(cx_)], -1)
    R = np.stack(rots_np)[cells[:, 1]]                                           # (N,3,3)
    dirs = np.einsum("nij,nkj->nki", R, cam)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    verts = dirs * corner_r[..., None]

    # Drop culled cells (corners above were refined with all cells, so kept ones match their neighbours).
    idx = np.nonzero(keep)[0]
    cells, verts, full, half_tex = cells[idx], verts[idx], full[idx], half_tex[idx]
    full_tex = {new: full_tex[old] for new, old in enumerate(idx) if old in full_tex}
    N = len(idx)

    # Atlas: 24-texel slot grid; full cells take 2x2 slots from the top, half cells single slots from
    # the bottom. Pages hold consecutive runs of the far-to-near order (cells are already far first).
    slot = hs
    ns = PAGE // slot
    pages, cur, top, bot = [], [], 0, 0          # top: double rows used; bot: single rows used
    pos = np.zeros((N, 3), int)                 # page, x, y (inner origin, texels)
    fill = {"full": 0, "half": 0}
    for i in range(N):
        while True:
            if full[i]:
                row, col = divmod(fill["full"], ns // 2)
                y = (top_rows := row * 2)
                if top_rows + 2 + bot <= ns:
                    pos[i] = (len(pages), col * 2 * slot + gb, y * slot + gb)
                    fill["full"] += 1
                    top = top_rows + 2
                    break
            else:
                row, col = divmod(fill["half"], ns)
                if top + row + 1 <= ns:
                    y = ns - 1 - row
                    pos[i] = (len(pages), col * slot + gb // TEXDS, y * slot + gb // TEXDS)
                    fill["half"] += 1
                    bot = row + 1
                    break
            pages.append(cur)
            cur, top, bot, fill = [], 0, 0, {"full": 0, "half": 0}
        cur.append(i)
    pages.append(cur)

    out_glb = Path(args.output or ply.parent) / f"{ply.stem}_msi.glb"
    meshes, texels = [], 0
    for pg, ids in enumerate(pages):
        img = np.zeros((PAGE, PAGE, 4), np.uint8)
        ids = np.array(ids)
        uu, vv = [], []
        for i in ids:
            p, x, y = pos[i]
            if full[i]:
                t, g, w = full_tex[i].numpy(), gb, cell
            else:
                t, g, w = half_tex[i], gb // TEXDS, cell // TEXDS
            img[y - g:y - g + t.shape[0], x - g:x - g + t.shape[1]] = t
            texels += t.shape[0] * t.shape[1]
            uu.append([x, x + w, x + w, x])
            vv.append([y, y, y + w, y + w])
        uv = np.stack([np.array(uu, np.float32).reshape(-1), np.array(vv, np.float32).reshape(-1)], 1) / PAGE
        faces = (np.arange(len(ids))[:, None, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]])[None]).reshape(-1, 3)
        meshes.append((f"surface_{len(pages) - 1 - pg}", verts[ids].reshape(-1, 3).astype(np.float32),
                       faces.astype(np.uint32), uv, img))
    # The photo bake's glTF writer: PNG pages encoded in parallel at fast compression (trimesh's export took
    # ~26 s for 10 pages, one at a time at the default level).
    info = dict(bins=B, cells=int(N), full_res_cells=int(full.sum()), pages=len(pages), face=args.face,
                texels_M=round(texels / 1e6, 1), r_min=float(1 / hi), r_max=float(1 / max(lo, 1e-6)))
    if eval_faces is not None:
        info["eval"] = evaluate(gs, sp_all, eval_faces, centres, args)
        print(f"  eval: {json.dumps(info['eval'])}", flush=True)

    def finish():                                            # CPU only: PNG pages, compressed copy, json
        import glb_ktx2
        from eval_alpha_patches import write_layer_glbs
        write_layer_glbs(out_glb, meshes)
        info.update(mb=round(out_glb.stat().st_size / 2 ** 20, 1), seconds=round(time.time() - t0))
        print(f"  wrote {out_glb}: {json.dumps(info)}", flush=True)
        if not args.no_ktx2:
            # ASTC (<name>_msi_astc.glb, the Quest's format; viewer falls back to the PNG file elsewhere) or Basis
            # UASTC (<name>_msi_ktx2.glb, any GPU; ~6x slower to encode); Zstandard 9 as the photo bake (+4% size)
            astc = args.tex == "astc"
            glb_ktx2.main([str(out_glb), str(out_glb.with_name(out_glb.stem + ("_astc.glb" if astc else "_ktx2.glb"))),
                           "--zstd", "9"] + (["--astc", "fast"] if astc else [])
                          + (["--rdo", str(args.ktx2_rdo)] if args.ktx2_rdo > 0 and not astc else []))
        out_glb.with_suffix(".json").write_text(json.dumps(info, indent=2))
    if writer is not None:
        return writer.submit(finish)
    finish()


@torch.no_grad()
def evaluate(gs, sp_all, faces, centres, args, size=512, fov=90):
    """PSNR of ray-sphere composited bins vs the splats, per eye offset (8 horizon + 2 tilted views)."""
    dev = sp_all["means"].device
    f = size / 2 / math.tan(math.radians(fov) / 2)
    c = (torch.arange(size, device=dev) + 0.5 - size / 2) / f
    ray = torch.stack(torch.meshgrid(c, c, indexing="xy") + (torch.ones(size, size, device=dev),), -1).reshape(-1, 3)
    ray = ray / ray.norm(dim=-1, keepdim=True)
    views = [(yaw, 0) for yaw in range(0, 360, 45)] + [(30, -50), (200, 50)]
    res = {}
    for move in (0.0, 0.1, 0.3):
        errs = []
        for vi, (yaw, pitch) in enumerate(views):
            yr, pr = math.radians(yaw), math.radians(pitch)
            dd = np.array([math.cos(pr) * math.sin(yr), -math.sin(pr), math.cos(pr) * math.cos(yr)])
            Rn = look_rotation(dd)
            Rt = torch.tensor(Rn, device=dev, dtype=torch.float32)
            # Eye offset sideways/up in view space (alternating), so parallax shows in every view.
            off = np.array([1.0, -0.5, 0.3]) if vi % 2 == 0 else np.array([-1.0, 0.3, -0.3])
            eye = torch.tensor(Rn @ (off / np.linalg.norm(off)) * move, device=dev, dtype=torch.float32)
            view = torch.eye(4, device=dev)
            view[:3, :3] = Rt.T
            view[:3, 3] = -Rt.T @ eye
            K = torch.tensor([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]], device=dev)
            ref, _, _ = gs.rendering.rasterization(**sp_all, viewmats=view[None], Ks=K[None], width=size,
                                                   height=size, near_plane=0.01)
            ref = ref[0].clamp(0, 1).reshape(-1, 3)
            dw = ray @ Rt.T
            ed = dw @ eye
            ee = eye @ eye
            out = torch.zeros(size * size, 3, device=dev)
            for k in sorted(faces):                        # far first
                rk = 1 / centres[k]
                t = -ed + torch.sqrt(ed ** 2 - ee + rk ** 2)
                p = eye + t[:, None] * dw
                p = p / p.norm(dim=-1, keepdim=True)
                fi, fx, fy = cube_lookup(p)
                tex = faces[k].float() / 255                 # (6,4,h,w)
                smp = torch.zeros(len(p), 4, device=dev)
                for q in range(6):
                    m = fi == q
                    if m.any():
                        g = torch.stack([fx[m], fy[m]], -1)[None, None]
                        smp[m] = F.grid_sample(tex[q:q + 1], g, align_corners=False)[0, :, 0].T
                out = out * (1 - smp[:, 3:]) + smp[:, :3]
            errs.append(float(((out.clamp(0, 1) - ref) ** 2).mean()))
            if args.eval_images and vi in (0, 2):
                im = torch.cat([out.reshape(size, size, 3), ref.reshape(size, size, 3)], 1)  # MSI | splats
                p_out = Path(args.output or Path(args.ply).parent) / f"{Path(args.ply).stem}_msi_eval_{move}_{vi}.jpg"
                Image.fromarray((im.cpu().numpy() * 255).astype(np.uint8)).save(p_out, quality=92)
        res[f"psnr_{move}"] = round(10 * math.log10(1 / np.mean(errs)), 2)
    return res


def main(argv=None, writer=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ply")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--bins", type=int, default=128)
    p.add_argument("--face", type=int, default=1024, help="Cube face resolution before 2x supersampling.")
    p.add_argument("--tile", type=int, default=16, help="Cell size (face px).")
    p.add_argument("--texds", type=int, default=2)
    p.add_argument("--border", type=int, default=4, help="Cell border texels (>= 2^L keeps mip level L clean).")
    p.add_argument("--texfull-thr", type=float, default=0.003)
    p.add_argument("--maxpages", type=int, default=8)
    p.add_argument("--photo-detail", metavar="PANO", default=None,
                   help="Add the panorama's detail the splats lost onto each pixel's dominant bin.")
    p.add_argument("--detail-min-w", type=float, default=0.3,
                   help="Only pixels whose dominant bin has at least this weight get the detail.")
    p.add_argument("--edge-jump", type=float, default=0.03,
                   help="Photo detail: relative disparity jump that counts as a depth edge.")
    p.add_argument("--edge-px", type=int, default=6,
                   help="Photo detail: render px around depth edges without detail (and to fade back in).")
    p.add_argument("--detail-alpha", type=float, default=None,
                   help="Photo detail v4: only where the dominant layer's alpha >= this (eroded); replaces the "
                        "disparity-jump edge mask.")
    p.add_argument("--detail-erode", type=int, default=2, help="Photo detail v4: erosion radius (render px).")
    p.add_argument("--surface-band", type=float, default=0.05,
                   help="Photo detail v4: bins within this relative disparity of the dominant one form its surface.")
    p.add_argument("--edge-darken", action="store_true",
                   help="Photo detail: keep darkening corrections at depth edges (mask only brightening).")
    p.add_argument("--no-refine", dest="refine", action="store_false")
    p.add_argument("--cull", type=float, default=0.0,
                   help="Drop cells contributing less than this from every eye within +-0.3 m (0 = off).")
    p.add_argument("--ktx2-rdo", type=float, default=0.0, help="UASTC RDO lambda for the KTX2 file (0 = off).")
    p.add_argument("--no-ktx2", action="store_true")
    p.add_argument("--tex", choices=["astc", "uastc"], default="uastc",
                   help="compressed copy: ASTC 4x4 (_msi_astc.glb) or Basis UASTC (_msi_ktx2.glb)")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--eval-images", action="store_true")
    return run(p.parse_args(argv), writer)


if __name__ == "__main__":
    main()
