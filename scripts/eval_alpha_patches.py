"""Offline quality of replacing ALL remaining hybrid splats with overlapping textured patches.

Every splat the hybrid mesh doesn't replace goes into a patch: per TILE-px
tile, greedy RANSAC planes in disparity (any 3D plane) within PX px parallax
at MOVE m, until no splat is left (small groups get a flat plane). A patch is
a quad on its plane covering the full footprints of its splats (so
neighbouring patches overlap the way their splats do); its texture is an RGBA
render of only those splats from the input view, so alpha falls off at the
patch border like the splats themselves.

Viewer order: opaque hybrid mesh, then patches back to front by the distance
of their centre from the eye (one sort of a few 10k patches per frame, or a
static order for small head moves), depth-tested against the mesh. Also
reported: exact per-pixel depth order.

Usage: python scripts/eval_alpha_patches.py <scene>.ply [...]
  env: SUFFIX (hybrid bake), TILE, PX, MOVE, FITFLAT (default 1), EXPORT (=_patches writes
  <scene>_patches.glb), EXPORT_DIR, TEXDS, TEXFULL_THR, MAXPAGES, BINS, MPI, BINMODE, REFINE, CULL, EXTEND, EVAL, DEBUG0, FLAT
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_alpha_layers as ev  # noqa: E402
import eval_alpha_surfaces as s1  # noqa: E402
import eval_alpha_surfaces2 as s2  # noqa: E402
from eval_alpha_layers import cast, dev, psnr, raster, splat_tensors  # noqa: E402

GW, SS = ev.GW, ev.SS
TILE = int(os.environ.get("TILE", 16))
PX = float(os.environ.get("PX", 0.5))
MOVE = float(os.environ.get("MOVE", 0.3))
SUFFIX = os.environ.get("SUFFIX", "_hybrid")
EXPORT = os.environ.get("EXPORT", "")
EXPORT_DIR = os.environ.get("EXPORT_DIR", "")
EVAL = int(os.environ.get("EVAL", 1))
FITFLAT = bool(os.environ.get("FITFLAT", "1") != "0")  # fronto-parallel patches (tilted RANSAC planes mix depths)
TEXDS = int(os.environ.get("TEXDS", 2))
TEXFULL_THR = float(os.environ.get("TEXFULL_THR", 0.003))   # MPI export: cells losing more detail than this at 1/TEXDS keep full res
MAXPAGES = int(os.environ.get("MAXPAGES", 4))              # ... within this many 4096^2 atlas pages
BINS = int(os.environ.get("BINS", 0))
MPI = bool(os.environ.get("MPI"))
BINMODE = os.environ.get("BINMODE", "uniform")   # MPI bins: uniform in disparity, or kmeans (where the content is)
REFINE = bool(os.environ.get("REFINE"))
CULL = float(os.environ.get("CULL", 0))          # drop cells contributing less than this from every viewpoint          # MPI: per-cell depth (shared corners) instead of the bin's centre
EXTEND = int(os.environ.get("EXTEND", 0))
BG = float(os.environ.get("BG", 0))           # eval: background grey level shown through gaps (viewer: page/passthrough)     # MPI: px each layer is continued underneath opaque content in front             # with BINS: sparse MPI (one joint render per bin, cut into cells)         # >0: patches on this many global depth bins instead of per-tile fits       # export texture downsampling      # fronto-parallel patches only (no tilted planes)
MARGIN = s1.MARGIN
OUT = Path("renderings") / "alpha_patches"
rng = np.random.default_rng(0)


def fit_patches(u, w, d, tol):
    """Per-splat patch id, plane (a, b, c) with d = a u + b w + c (grid px), all splats assigned."""
    A = np.stack([u, w, np.ones_like(u)], 1)
    key = np.floor(u / TILE).astype(np.int64) * 100000 + np.floor(w / TILE).astype(np.int64)
    o = np.argsort(key)
    groups = np.split(o, np.flatnonzero(np.diff(key[o])) + 1)
    pid = -np.ones(len(u), int)
    planes = []
    for g in groups:
        free = g
        while len(free):
            ds_o = free[np.argsort(d[free])]
            ds = d[ds_o]
            j = np.searchsorted(ds, ds + 2 * tol, "right")
            i = int(np.argmax(j - np.arange(len(ds))))
            best, bp = ds_o[i:j[i]], np.array([0.0, 0.0, ds[i] + tol])
            if len(free) >= 3 and not FITFLAT:
                for _ in range(60):
                    s = rng.choice(free, 3, replace=False)
                    try:
                        p = np.linalg.solve(A[s], d[s])
                    except np.linalg.LinAlgError:
                        continue
                    inl = free[np.abs(A[free] @ p - d[free]) < tol]
                    if len(inl) > len(best):
                        best, bp = inl, p
            pid[best] = len(planes)
            planes.append(bp)
            free = np.setdiff1d(free, best, assume_unique=True)
    return pid, np.array(planes)


def bin_patches(u, w, d):
    """Patches on BINS global depth bins (uniform in disparity over the 0.1-99.9% range): one patch per
    (tile, bin) with splats, flat at the bin's centre disparity. Bins are globally ordered, so patches of
    different bins never interleave in depth."""
    lo, hi = np.percentile(d, [0.1, 99.9])
    b = np.clip(((d - lo) / (hi - lo) * BINS).astype(int), 0, BINS - 1)
    key = (np.floor(u / TILE).astype(np.int64) * 100000 + np.floor(w / TILE).astype(np.int64)) * 4096 + b
    uk, pid = np.unique(key, return_inverse=True)
    centre = lo + (uk % 4096 + 0.5) / BINS * (hi - lo)
    err = (hi - lo) / BINS / 2
    print(f"  {BINS} bins over disparity {lo:.3f}-{hi:.3f}: max snap {err:.4f} 1/m", flush=True)
    return pid, np.stack([np.zeros_like(centre), np.zeros_like(centre), centre], 1)


def patch_rects(pid, P, us, ws, rad, Wc, Hc):
    """Integer canvas rects (x0, y0, x1, y1) covering each patch's splat footprints (render px, canvas coords)."""
    cx, cy = us + MARGIN, ws + MARGIN
    x0 = np.full(P, np.inf); y0 = np.full(P, np.inf); x1 = np.full(P, -np.inf); y1 = np.full(P, -np.inf)
    np.minimum.at(x0, pid, cx - rad); np.minimum.at(y0, pid, cy - rad)
    np.maximum.at(x1, pid, cx + rad); np.maximum.at(y1, pid, cy + rad)
    r = np.stack([np.floor(x0), np.floor(y0), np.ceil(x1), np.ceil(y1)], 1)
    r[:, [0, 2]] = r[:, [0, 2]].clip(0, Wc); r[:, [1, 3]] = r[:, [1, 3]].clip(0, Hc)
    return r.astype(np.int64)


def pack_passes(rects, Wc, Hc, cell=8):
    """Assign patches to render passes so rects in one pass never overlap (crops stay clean)."""
    occ = []
    pas = np.empty(len(rects), int)
    order = np.argsort(-(rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1]))
    gx0, gy0 = rects[:, 0] // cell, rects[:, 1] // cell
    gx1, gy1 = -(-rects[:, 2] // cell), -(-rects[:, 3] // cell)
    for i in order:
        for k, m in enumerate(occ):
            if not m[gy0[i]:gy1[i], gx0[i]:gx1[i]].any():
                break
        else:
            occ.append(np.zeros((-(-Hc // cell), -(-Wc // cell)), bool)); k = len(occ) - 1
        occ[k][gy0[i]:gy1[i], gx0[i]:gx1[i]] = True
        pas[i] = k
    return pas, len(occ)


def composite_vals(vals, pix, key, mrgb, npix):
    """Front-to-back per pixel by key (smaller = nearer), over the mesh colour."""
    o = torch.argsort(pix.double() * 1e7 + key.double())
    vals, pix = vals[o], pix[o]
    first = torch.ones(len(o), dtype=torch.bool, device=dev)
    first[1:] = pix[1:] != pix[:-1]
    seg = torch.cummax(torch.where(first, torch.arange(len(o), device=dev), 0), 0).values
    a = vals[:, 3].clamp(0, 0.999)
    la = torch.log1p(-a).double()
    cs = torch.cumsum(la, 0)
    excl = (cs - la - (cs[seg] - la[seg])).float()
    C = torch.zeros(npix, 3, device=dev).index_add_(0, pix, torch.exp(excl)[:, None] * vals[:, :3])
    L = torch.zeros(npix, device=dev).index_add_(0, pix, la.float())
    return C + torch.exp(L)[:, None] * mrgb


def run(scene: Path):
    t0 = time.time()
    meta = json.loads(scene.with_suffix(".json").read_text())
    Hg = int(round(GW * meta["height"] / meta["width"]))
    fg = meta["focal_length_px"] * GW / meta["width"]
    W, H, f = GW * SS, Hg * SS, fg * SS
    Wc, Hc = W + 2 * MARGIN, H + 2 * MARGIN
    full = splat_tensors(PlyData.read(scene)["vertex"].data, dev)
    kv = PlyData.read(scene.with_name(scene.stem + SUFFIX + ".ply"))["vertex"].data
    z = np.asarray(kv["z"], np.float64)
    ok = z > 0.05
    kv = kv[ok]; z = z[ok]
    kept = splat_tensors(kv, dev)
    u, w = fg * kv["x"] / z + GW / 2, fg * kv["y"] / z + Hg / 2
    mesh = trimesh.load(scene.with_name(scene.stem + SUFFIX + ".glb")).to_geometry()
    Tm = torch.tensor(mesh.vertices, device=dev, dtype=torch.float32)[torch.tensor(mesh.faces, device=dev)]
    rgbM, aM = raster([full], (0.0, 0.0, 0.0), W, H, f)
    M = s1.TriMesh(Tm, torch.cat([rgbM, aM[..., None]], -1), W, H, 0, f, W, H)
    Kc = ev.K_of(W, H, f); Kc[0, 0, 2] += MARGIN; Kc[0, 1, 2] += MARGIN
    if MPI:
        # Sparse MPI: BINS global depth bins (uniform in disparity); each bin's splats rendered together and
        # cut into TILE-px cells, keeping cells with any alpha. Cells of a bin abut on one plane (no margins,
        # no gaps when moving); bins are disjoint in depth, drawn back to front.
        d = 1 / z
        lo, hi = np.percentile(d, [0.1, 99.9])
        if BINMODE == "kmeans":
            # Bins where the content is: 1-D k-means (Lloyd) of the splats' disparities, i.e. least squared
            # snapping parallax over all splats, instead of equal widths over the whole range.
            dc = np.clip(d, lo, hi)
            centers = np.quantile(dc, (np.arange(BINS) + 0.5) / BINS)
            for _ in range(40):
                edges = (centers[1:] + centers[:-1]) / 2
                lab = np.searchsorted(edges, dc)
                sums, cnt = np.bincount(lab, dc, BINS), np.bincount(lab, minlength=BINS)
                centers = np.where(cnt > 0, sums / np.maximum(cnt, 1), centers)
                centers = np.maximum.accumulate(centers)
            edges = (centers[1:] + centers[:-1]) / 2
        else:
            edges = lo + np.arange(1, BINS) / BINS * (hi - lo)
            centers = lo + (np.arange(BINS) + 0.5) / BINS * (hi - lo)
        binof = lambda x: np.searchsorted(edges, x)
        b = binof(d)
        # Snapping a splat to its bin's centre depth must not move it across the mesh (at its centre pixel):
        # splats just behind the mesh (kept for disocclusions; the depth test hides them in the hybrid)
        # would otherwise be drawn over it, e.g. SHARP's blurry fill behind grass smudging the grass.
        _, mdep0 = s2.mesh_layer(M, (0.0, 0.0, 0.0), W, H, f)
        mdisp0 = (1 / mdep0).reshape(H, W).cpu().numpy()                  # 0 where no mesh
        ci = np.clip(np.round(u * SS - 0.5).astype(int), 0, W - 1)
        ri = np.clip(np.round(w * SS - 0.5).astype(int), 0, H - 1)
        inside_f = (u * SS >= 0) & (u * SS < W) & (w * SS >= 0) & (w * SS < H)
        md = np.where(inside_f, mdisp0[ri, ci], 0)
        mb = binof(md)
        same = (md > 0) & (b == mb)
        b = np.where(same & (d < md), b - 1, np.where(same & (d >= md), b + 1, b)).clip(0, BINS - 1)
        print(f"  {same.mean():.1%} of splats moved one bin to stay on their side of the mesh", flush=True)
        cell = TILE * SS
        ny, nx = -(-Hc // cell), -(-Wc // cell)
        b_t = torch.tensor(b, device=dev)
        crops, crops_b, rect_l, cen_l, cd_l = [], [], [], [], []
        gb = 2 * TEXDS                                     # border (render px) of real neighbour texels, export only
        cb = cell + 2 * gb
        # Layer extension (EXTEND render px): each bin's content is continued, opaque, underneath what is
        # already opaque in front of it (nearer bins, or the mesh), so moving the head reveals that colour
        # instead of a gap at bin boundaries (thin bright/dark contours on smooth surfaces) and mesh borders.
        # Hidden at the input view by construction.
        if EXTEND:
            _, mdep = s2.mesh_layer(M, (0.0, 0.0, 0.0), W, H, f)
            mdisp = torch.zeros(Hc, Wc, device=dev)
            mdisp[MARGIN:MARGIN + H, MARGIN:MARGIN + W] = (1 / mdep).reshape(H, W)      # 0 where no mesh
            Tfront = torch.ones(Hc, Wc, device=dev)          # transmittance of nearer bins (original alphas)
            # Pass 1: transmittance of everything farther than each bin, so an extension never covers
            # farther content (e.g. the splats the hybrid keeps for disocclusions), which is what a
            # head move should reveal there.
            Tback = {}
            run_T = torch.ones(Hc, Wc, device=dev)
            for k in range(BINS):                            # farthest first
                Tback[k] = (run_T * 255).round().to(torch.uint8)
                m = b_t == k
                if m.any():
                    sub = {kk: v[m] for kk, v in kept.items()}
                    _, al, _ = ev.gs.rendering.rasterization(**sub, viewmats=torch.eye(4, device=dev)[None], Ks=Kc,
                                                             width=Wc, height=Hc, eps2d=ev.EPS)
                    run_T = run_T * (1 - al[0, ..., 0])
        for k in reversed(range(BINS)):                      # nearest bin first
            m = b_t == k
            if not m.any():
                continue
            sub = {kk: v[m] for kk, v in kept.items()}
            out, alpha, _ = ev.gs.rendering.rasterization(**sub, viewmats=torch.eye(4, device=dev)[None], Ks=Kc,
                                                          width=Wc, height=Hc, eps2d=ev.EPS, render_mode="RGB+D")
            rgb, a, dacc = out[0, ..., :3], alpha[0, ..., 0], out[0, ..., 3]
            if EXTEND:
                dk = centers[k]
                hidden = ((Tfront < 0.02) | (mdisp > dk * 1.01)) & (Tback.pop(k).float() > 0.5 * 255)
                filled = a > 0.5
                col = torch.where(filled[..., None], rgb / a.clamp(min=1e-3)[..., None], 0)
                fl = filled.float()
                for _ in range(EXTEND):                      # push edge colours outward one px per step
                    sc = torch.nn.functional.avg_pool2d(col.permute(2, 0, 1)[None] * fl, 3, 1, 1)[0].permute(1, 2, 0)
                    sn = torch.nn.functional.avg_pool2d(fl[None, None], 3, 1, 1)[0, 0]
                    new = (sn > 0) & (fl == 0)
                    col = torch.where(new[..., None], sc / sn.clamp(min=1e-6)[..., None], col)
                    fl = torch.where(new, torch.ones_like(fl), fl)
                ext = (fl > 0) & ~filled & hidden
                Tfront = Tfront * (1 - a)
                rgb = torch.where(ext[..., None], col, rgb)
                a = torch.where(ext, torch.ones_like(a), a)
            img = torch.nn.functional.pad(torch.cat([rgb, a[..., None]], -1),
                                          (0, 0, 0, nx * cell - Wc, 0, ny * cell - Hc)).half()
            cells = img.reshape(ny, cell, nx, cell, 4).permute(0, 2, 1, 3, 4)
            occ = cells[..., 3].amax((2, 3)) >= 1 / 255
            cy, cx = torch.nonzero(occ, as_tuple=True)
            crops.append(cells[cy, cx].reshape(-1, 4))
            if EXPORT:
                pimg = torch.nn.functional.pad(img.permute(2, 0, 1)[None].float(), (gb, gb, gb, gb))
                cols = torch.nn.functional.unfold(pimg, cb, stride=cell)[0]          # (4*cb*cb, ny*nx)
                sel = cols[:, cy * nx + cx].reshape(4, cb, cb, -1).permute(3, 1, 2, 0)
                crops_b.append(sel.half().reshape(-1, 4))
            cyn, cxn = cy.cpu().numpy(), cx.cpu().numpy()
            rect_l.append(np.stack([cxn * cell, cyn * cell, (cxn + 1) * cell, (cyn + 1) * cell], 1))
            cen_l.append(np.full(len(cyn), centers[k]))
            if REFINE:
                # Per-cell depth: alpha-weighted mean disparity of this bin's splats in the cell; corners shared
                # by neighbouring cells take the mean of those cells, so the bin stays one crack-free surface.
                pd = torch.where(a > 1e-3, a / dacc.clamp(min=1e-6), 0)
                pad_ = lambda t: torch.nn.functional.pad(t, (0, nx * cell - Wc, 0, ny * cell - Hc))
                aw = pad_(a).reshape(ny, cell, nx, cell).sum((1, 3))
                dw = pad_(a * pd).reshape(ny, cell, nx, cell).sum((1, 3))
                dcell = torch.where(occ, dw / aw.clamp(min=1e-9), 0).double()
                o = occ.double()
                G = torch.zeros(ny + 1, nx + 1, device=dev, dtype=torch.float64); Gn = torch.zeros_like(G)
                for oy in (0, 1):
                    for ox in (0, 1):
                        G[oy:oy + ny, ox:ox + nx] += dcell * o; Gn[oy:oy + ny, ox:ox + nx] += o
                G = (G / Gn.clamp(min=1)).cpu().numpy()
                cd_l.append(np.stack([G[cyn, cxn], G[cyn, cxn + 1], G[cyn + 1, cxn], G[cyn + 1, cxn + 1]], 1))
            else:
                cd_l.append(np.full((len(cyn), 4), centers[k]))
        rects = np.concatenate(rect_l).astype(np.int64)
        cen = np.concatenate(cen_l)
        planes = np.stack([np.zeros_like(cen), np.zeros_like(cen), cen], 1)
        cdisp = np.concatenate(cd_l)                      # corner disparities (x0y0, x1y0, x0y1, x1y1) per cell
        P = len(rects)
        area = np.full(P, cell * cell, np.int64)
        off = np.r_[0, np.cumsum(area)[:-1]]
        atlas = torch.cat(crops)
        border = (torch.cat(crops_b), gb) if EXPORT else None
        del crops, crops_b
        print(f"{scene.stem}{SUFFIX}: {len(kv)} splats -> sparse MPI, {BINS} {BINMODE} bins (max snap "
              f"{np.diff(np.r_[lo, edges, hi]).max() / 2:.4f} 1/m{', refined cells' if REFINE else ''}), {P} cells of {TILE} px ({2 * P} tris), "
              f"overdraw {area.sum() / (W * H):.1f}x, {time.time() - t0:.0f}s", flush=True)
    else:
        if BINS:
            pid, planes = bin_patches(u, w, 1 / z)
        else:
            pid, planes = fit_patches(u, w, 1 / z, PX / (fg * MOVE))
        if os.environ.get("FLAT"):                  # debug: replace every plane by a flat one at its splats' mean disparity
            md = np.bincount(pid, 1 / z) / np.bincount(pid)
            planes = np.stack([np.zeros_like(md), np.zeros_like(md), md], 1)
        P = len(planes)
        sc = np.exp(np.stack([kv["scale_0"], kv["scale_1"], kv["scale_2"]], -1).astype(np.float64)).max(1)
        rad = np.clip(3 * sc * f / z, 1, 256) + 1                         # render px
        rects = patch_rects(pid, P, u * SS - 0.5, w * SS - 0.5, rad, Wc, Hc)
        area = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
        pas, npass = pack_passes(rects, Wc, Hc)
        print(f"{scene.stem}{SUFFIX}: {len(kv)} splats -> {P} patches ({len(kv) / P:.1f} splats each, "
              f"{2 * P} tris), overdraw {area.sum() / (W * H):.1f}x, {npass} texture passes, fit {time.time() - t0:.0f}s",
              flush=True)

        # Textures: per pass, render the union of its patches' splats, crop each rect into a flat atlas.
        off = np.r_[0, np.cumsum(area)[:-1]]
        atlas = torch.zeros(int(area.sum()), 4, device=dev, dtype=torch.float16)
        pid_t = torch.tensor(pid, device=dev)
        Kc = ev.K_of(W, H, f); Kc[0, 0, 2] += MARGIN; Kc[0, 1, 2] += MARGIN
        for k in range(npass):
            ps = np.nonzero(pas == k)[0]
            m = torch.isin(pid_t, torch.tensor(ps, device=dev))
            sub = {kk: v[m] for kk, v in kept.items()}
            out, alpha, _ = ev.gs.rendering.rasterization(**sub, viewmats=torch.eye(4, device=dev)[None], Ks=Kc,
                                                          width=Wc, height=Hc, eps2d=ev.EPS)
            img = torch.cat([out[0], alpha[0]], -1).half()
            for i in ps:
                x0, y0, x1, y1 = rects[i]
                atlas[off[i]:off[i] + area[i]] = img[y0:y1, x0:x1].reshape(-1, 4)
    print(f"  textures {atlas.shape[0] / 1e6:.0f}M texels, {time.time() - t0:.0f}s", flush=True)

    # Patch geometry: plane n·p = 1 (see eval_alpha_layers) and the rect corners on it.
    a_, b_, c_ = (torch.tensor(planes[:, i], device=dev, dtype=torch.float32) for i in range(3))
    nP = torch.stack([a_ * fg, b_ * fg, a_ * GW / 2 + b_ * Hg / 2 + c_], 1)
    R_ = torch.tensor(rects, device=dev, dtype=torch.float32)
    cu = (torch.stack([R_[:, 0], R_[:, 2], R_[:, 0], R_[:, 2]], 1) - MARGIN) / SS     # grid px of rect corners
    cv = (torch.stack([R_[:, 1], R_[:, 1], R_[:, 3], R_[:, 3]], 1) - MARGIN) / SS
    dmin = torch.tensor(1 / z.max() / 2, device=dev, dtype=torch.float32)
    dc = (a_[:, None] * cu + b_[:, None] * cv + c_[:, None]).clamp(min=dmin)
    corners = torch.stack([(cu - GW / 2) / fg, (cv - Hg / 2) / fg, torch.ones_like(cu)], -1) / dc[..., None]
    if MPI:   # per-cell corner disparities: each cell is two triangles (diagonal x0y0-x1y1, as exported)
        cdt = torch.tensor(cdisp, device=dev, dtype=torch.float32).clamp(min=dmin)
        corners = torch.stack([(cu - GW / 2) / fg, (cv - Hg / 2) / fg, torch.ones_like(cu)], -1) / cdt[..., None]
    centre = corners.mean(1)
    rx0, ry0, rw, rh = R_[:, 0], R_[:, 1], R_[:, 2] - R_[:, 0], R_[:, 3] - R_[:, 1]
    off_t, rw_i = torch.tensor(off, device=dev), (R_[:, 2] - R_[:, 0]).long()

    def inside(prim, ui, vi):
        lx, ly = ui + MARGIN - 0.5 - rx0[prim], vi + MARGIN - 0.5 - ry0[prim]
        return (lx >= -0.5) & (lx < rw[prim] - 0.5) & (ly >= -0.5) & (ly < rh[prim] - 0.5)

    if MPI:
        tri = torch.stack([corners[:, [0, 1, 3]], corners[:, [0, 3, 2]]], 1).reshape(-1, 3, 3)   # prim = 2*cell + half
        geomN, geomC = torch.linalg.solve(tri, torch.ones(len(tri), 3, device=dev)), tri
        to_cell = lambda prim: prim // 2

        def geom_inside(prim, ui, vi):
            c = prim // 2
            lx, ly = ui + MARGIN - rx0[c], vi + MARGIN - ry0[c]                # from the rect's corner (px)
            sx, ty = lx / rw[c], ly / rh[c]
            upper = ty <= sx                                                   # triangle (x0y0, x1y0, x1y1)
            return inside(c, ui, vi) & torch.where(prim % 2 == 0, upper, ~upper)
    else:
        geomN, geomC, geom_inside, to_cell = nP, corners, inside, (lambda prim: prim)

    def patch_samples(eye):
        V, PIX, T, PR = [], [], [], []
        for prim, pix, p, ui, vi, t in cast(geomN, geomC, geom_inside, eye, W, H, f):
            prim = to_cell(prim)
            lx = (ui + MARGIN - 0.5 - rx0[prim]).clamp(0, None)
            ly = (vi + MARGIN - 0.5 - ry0[prim]).clamp(0, None)
            wi, hi = rw_i[prim], rh[prim].long()
            x0 = lx.floor().long().clamp(max=wi - 1); y0 = ly.floor().long().clamp(max=hi - 1)
            x1 = (x0 + 1).clamp(max=wi - 1); y1 = (y0 + 1).clamp(max=hi - 1)
            fx, fy = (lx - x0).clamp(0, 1)[:, None], (ly - y0).clamp(0, 1)[:, None]
            base = off_t[prim]
            g = lambda yy, xx: atlas[base + yy * wi + xx].float()
            c = (g(y0, x0) * (1 - fx) + g(y0, x1) * fx) * (1 - fy) + (g(y1, x0) * (1 - fx) + g(y1, x1) * fx) * fy
            k = (c[:, 3] > 1 / 255) & alive[prim]
            V.append(c[k]); PIX.append(pix[k]); T.append(t[k]); PR.append(prim[k])
        return torch.cat(V), torch.cat(PIX), torch.cat(T), torch.cat(PR)


    alive = torch.ones(P, dtype=torch.bool, device=dev)
    if CULL > 0:
        # Drop cells that are invisible from every viewpoint in the head range: per viewpoint, composite in
        # depth order (after the mesh depth test) and take each cell's largest visible contribution
        # alpha * transmittance of everything in front; cells below CULL everywhere are removed.
        best = torch.zeros(P, device=dev)
        eyes_c = [(x, y, zz) for x in (-MOVE, 0.0, MOVE) for y in (-MOVE / 2, 0.0, MOVE / 2) for zz in (0.0, 0.4 * MOVE)]
        for eye in eyes_c:
            vals, pix, t, prim = patch_samples(eye)
            _, mdep_c = s2.mesh_layer(M, eye, W, H, f)
            k = t < mdep_c[pix]
            vals, pix, t, prim = vals[k], pix[k], t[k], prim[k]
            o = torch.argsort(pix.double() * 1e7 + t.double())
            a, pix, prim = vals[o, 3].clamp(0, 0.999), pix[o], prim[o]
            first = torch.ones(len(o), dtype=torch.bool, device=dev); first[1:] = pix[1:] != pix[:-1]
            seg = torch.cummax(torch.where(first, torch.arange(len(o), device=dev), 0), 0).values
            la = torch.log1p(-a).double(); cs = torch.cumsum(la, 0)
            contrib = (torch.exp(cs - la - (cs[seg] - la[seg])) * a).float()
            best.scatter_reduce_(0, prim, contrib, "amax")
            if eye == (0.0, 0.0, 0.0):
                prim0 = prim.clone()
            del vals, pix, t, prim, o, a, la, cs, contrib
            torch.cuda.empty_cache()
        alive = best >= CULL
        print(f"  input view: {len(prim0) / (W * H):.2f} -> {alive[prim0].sum().item() / (W * H):.2f} layer fragments "
              f"per pixel after the mesh depth test", flush=True)
        fill = area.astype(np.float64)
        print(f"  culling: {int((~alive).sum())} of {P} cells ({(~alive).float().mean().item():.0%}, "
              f"{fill[~alive.cpu().numpy()].sum() / fill.sum():.0%} of cell area) never contribute >= {CULL} "
              f"from {len(eyes_c)} viewpoints within +-{MOVE} m -> dropped", flush=True)

    if os.environ.get("DEBUG0"):
        Rk, _ = raster([kept], (0.0, 0.0, 0.0), W, H, f)
        vals, pix, t, prim = patch_samples((0.0, 0.0, 0.0))
        ex = composite_vals(vals, pix, t, torch.zeros(W * H, 3, device=dev), W * H).reshape(H, W, 3)
        cnt = torch.zeros(W * H, device=dev).index_add_(0, pix, torch.ones(len(pix), device=dev))
        cov = torch.zeros(W * H, device=dev).index_add_(0, pix, vals[:, 3])
        print(f"  DEBUG input view, patches vs kept splats (no mesh): {psnr(Rk, ex):.2f} dB; samples/px mean "
              f"{cnt.mean().item():.1f}; mean summed alpha {cov.mean().item():.2f}; "
              f"tilted planes {(np.abs(planes[:, :2]).sum(1) > 0).mean():.2f}", flush=True)
        # Direct: rect texel (x, y) is frame pixel (x - MARGIN, y - MARGIN) at the input view; key = patch depth.
        pz = torch.tensor(1 / np.maximum(planes[:, 2], 1e-6), device=dev, dtype=torch.float32)
        pidx = torch.repeat_interleave(torch.arange(P, device=dev), torch.tensor(area, device=dev))
        loc = torch.arange(len(pidx), device=dev) - torch.tensor(off, device=dev)[pidx]
        xx = R_[pidx, 0].long() + loc % rw_i[pidx] - MARGIN
        yy = R_[pidx, 1].long() + loc // rw_i[pidx] - MARGIN
        okk = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H) & (atlas[:, 3] > 1 / 255)
        D = composite_vals(atlas[okk].float(), (yy * W + xx)[okk], pz[pidx[okk]], torch.zeros(W * H, 3, device=dev), W * H)
        _, aRk = raster([kept], (0.0, 0.0, 0.0), W, H, f)
        la_ = torch.log1p(-atlas[okk, 3].float().clamp(0, 0.999))
        Lsum = torch.zeros(W * H, device=dev).index_add_(0, (yy * W + xx)[okk], la_)
        aD = 1 - torch.exp(Lsum).reshape(H, W)
        print(f"  DEBUG coverage: ref mean {aRk.mean().item():.4f} patches {aD.mean().item():.4f}, "
              f"mean |diff| {(aRk - aD).abs().mean().item():.4f}, max {(aRk - aD).abs().max().item():.3f}", flush=True)
        rngD = np.bincount(pid, minlength=P)
        dmx = np.full(P, -np.inf); dmn = np.full(P, np.inf)
        np.maximum.at(dmx, pid, 1 / z); np.minimum.at(dmn, pid, 1 / z)
        span = (dmx - dmn) / (PX / (fg * MOVE))
        print(f"  DEBUG member disparity span in tol units: p50 {np.percentile(span, 50):.1f} p90 {np.percentile(span, 90):.1f} "
              f"p99 {np.percentile(span, 99):.1f}; patches {P}", flush=True)
        print(f"  DEBUG direct atlas composite: {psnr(Rk, D.reshape(H, W, 3)):.2f} dB; vs cast {psnr(D.reshape(H, W, 3), ex):.2f}", flush=True)
        px_, py_ = (pix % W).float() + 0.5, (pix // W).float() + 0.5
        pts = torch.stack([(px_ - W / 2) / f * t, (py_ - H / 2) / f * t, t], 1)
        OUT.mkdir(parents=True, exist_ok=True)
        Image.fromarray((torch.cat([Rk, ex], 1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).resize(
            (2 * GW, Hg)).save(OUT / "debug0.jpg")
        return
    if EXPORT:
        export_glb(rects, planes, atlas, off, area, fg, Hg, scene, centre, border=border if MPI else None,
                   cdisp=cdisp if MPI else None, alive=alive.cpu().numpy())
        if not EVAL:
            return

    eyes = {"0": (0.0, 0.0, 0.0)}
    for mv in (0.1, 0.2, 0.3):
        eyes |= {f"x+{mv}": (mv, 0, 0), f"x-{mv}": (-mv, 0, 0), f"y+{mv/2}": (0, mv / 2, 0),
                 f"z+{0.4*mv:.2f}": (0, 0, 0.4 * mv)}
    res = {}
    for name, eye in eyes.items():
        R, _ = raster([full], eye, W, H, f)
        ms = M.samples(eye, W, H, f, opaque=True)
        Hy, _ = raster([kept, ms], eye, W, H, f)
        mrgb, mdep = s2.mesh_layer(M, eye, W, H, f)
        if BG:                                  # background behind everything (the viewer's canvas is transparent)
            mrgb = torch.where(torch.isinf(mdep)[:, None], torch.full_like(mrgb, BG), mrgb)
        vals, pix, t, prim = patch_samples(eye)
        k = t < mdep[pix]
        vals, pix, t, prim = vals[k], pix[k], t[k], prim[k]
        ex = composite_vals(vals, pix, t, mrgb, W * H).reshape(H, W, 3)
        dist = (centre - torch.tensor(eye, device=dev)).norm(dim=1)
        rank = torch.empty(P, device=dev); rank[torch.argsort(dist)] = torch.arange(P, device=dev, dtype=torch.float32)
        so = composite_vals(vals, pix, rank[prim], mrgb, W * H).reshape(H, W, 3)
        res[name] = {"H": psnr(R, Hy), "P_exact": psnr(R, ex), "P_sorted": psnr(R, so)}
        print(f"  {name:8s} " + "  ".join(f"{kk} {v:5.2f}" for kk, v in res[name].items()), flush=True)
        if name in ("0", "x+0.2", "x+0.3"):
            OUT.mkdir(parents=True, exist_ok=True)
            row = torch.cat([R, Hy, so], 1).clamp(0, 1)
            Image.fromarray((row.cpu().numpy() * 255).astype(np.uint8)).resize((3 * GW, Hg)).save(
                OUT / f"{scene.stem}{SUFFIX}_t{TILE}_b{BINS}{BINMODE[0]}{"r" if REFINE else ""}_c{CULL}_e{EXTEND}_bg{BG}_{name}.jpg", quality=92)
        del vals, pix, t, prim, ms
        torch.cuda.empty_cache()
    (OUT / f"{scene.stem}{SUFFIX}_t{TILE}_b{BINS}{BINMODE[0]}{"r" if REFINE else ""}_c{CULL}.json").write_text(json.dumps(
        {"patches": P, "overdraw": float(area.sum() / (W * H)), "psnr": res}, indent=1))


def export_glb(rects, planes, atlas, off, area, fg, Hg, scene, centre, page=4096, border=None, cdisp=None,
               alive=None):
    """Patches as quads in one mesh per atlas page (premultiplied RGBA PNGs), back to front as seen from the
    input view (static order; fine for small head moves): pages hold consecutive runs of the far-to-near
    order and each page's triangles follow it, while texels are placed by height (shelf packing). Textures
    are box-downsampled by TEXDS and trimmed to the texels with alpha. With `border` (atlas of the same cells
    with a margin of gb render px of real neighbour texels, MPI), cells keep that margin instead of a
    duplicated-edge gutter and aren't trimmed, so filtering across cell seams sees the right colours."""
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals
    ds = TEXDS
    at = atlas.float().clamp(0, 1).cpu().numpy()
    def down(t):                                                     # box filter (premultiplied, so exact)
        if ds == 1:
            return t
        h, w = -(-t.shape[0] // ds), -(-t.shape[1] // ds)
        t = np.pad(t, ((0, h * ds - t.shape[0]), (0, w * ds - t.shape[1]), (0, 0)))
        return t.reshape(h, ds, w, ds, 4).mean((1, 3))

    texs, geo, pad = {}, {}, {}                    # texs: padded texture block; pad: its margin (texels)
    if border is not None:
        atb, gb = border
        cb = (rects[0, 2] - rects[0, 0]) + 2 * gb
        assert gb % ds == 0 and cb % ds == 0
        atb = atb.reshape(-1, cb, cb, 4)
        # Adaptive resolution: cells that lose the most detail at 1/ds resolution (mean |premultiplied
        # error| after down- and up-sampling, inner part) keep full resolution, worst first, while
        # their error is above TEXFULL_THR and the atlas stays within MAXPAGES pages.
        full = np.zeros(len(rects), bool)
        if ds > 1 and TEXFULL_THR > 0:
            err = torch.empty(len(rects), device=atb.device)
            for c0 in range(0, len(rects), 8192):
                t1 = atb[c0:c0 + 8192].float()
                t2 = torch.nn.functional.avg_pool2d(t1.permute(0, 3, 1, 2), ds)
                up = torch.nn.functional.interpolate(t2, scale_factor=ds, mode="bilinear", align_corners=False)
                d = (up - t1.permute(0, 3, 1, 2))[:, :3, gb:cb - gb, gb:cb - gb].abs().mean((1, 2, 3))
                err[c0:c0 + 8192] = d
            err = err.cpu().numpy()
            budget = 0.75 * MAXPAGES * page * page - len(rects) * (cb // ds) ** 2
            extra = cb * cb - (cb // ds) ** 2
            cand = np.argsort(-err)
            cand = cand[err[cand] > TEXFULL_THR][:max(0, int(budget // extra))]
            full[cand] = True
            print(f"  adaptive textures: {full.sum()} of {len(rects)} cells at full resolution "
                  f"(error > {TEXFULL_THR}, budget {MAXPAGES} pages)", flush=True)
        atb = atb.cpu().float().clamp(0, 1).numpy()                  # on the CPU: a float copy of the whole border atlas can exceed GPU memory
    for i in range(len(rects)):
        if alive is not None and not alive[i]:
            continue
        x0, y0, x1, y1 = rects[i]
        t = down(at[off[i]:off[i] + area[i]].reshape(y1 - y0, x1 - x0, 4))
        m = t[..., 3] >= 0.5 / 255
        if not m.any():
            continue
        if border is not None:
            # Shrink the quad to the texels with alpha (the rest is fill the GPU would blend for nothing),
            # keeping `g` texels of real data around it for filtering, so the result is unchanged.
            tb = atb[i] if full[i] else down(atb[i])
            g, sp = (gb, 1) if full[i] else (gb // ds, ds)               # border texels, render px per texel
            inner = tb[g:tb.shape[0] - g, g:tb.shape[1] - g, 3] >= 0.5 / 255
            if not inner.any():
                continue
            ys, xs = np.nonzero(inner.any(1))[0], np.nonzero(inner.any(0))[0]
            texs[i] = tb[ys[0]:ys[-1] + 1 + 2 * g, xs[0]:xs[-1] + 1 + 2 * g]
            pad[i] = g
            geo[i] = (x0 + xs[0] * sp, y0 + ys[0] * sp, x0 + (xs[-1] + 1) * sp, y0 + (ys[-1] + 1) * sp)
            continue
        ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
        texs[i] = np.pad(t[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1], ((1, 1), (1, 1), (0, 0)), mode="edge")
        pad[i] = 1
        geo[i] = (x0 + xs[0] * ds, y0 + ys[0] * ds, x0 + (xs[-1] + 1) * ds, y0 + (ys[-1] + 1) * ds)
    order = [i for i in np.argsort(-centre[:, 2].cpu().numpy()) if i in texs]   # far first

    def shelf(ids):
        place, x, y, sh = {}, 0, 0, 0
        for i in sorted(ids, key=lambda i: -texs[i].shape[0]):
            h, w = texs[i].shape[:2]                                 # includes the margin
            if x + w > page:
                x, y, sh = 0, y + sh, 0
            if y + h > page:
                return None
            place[i] = (x + pad[i], y + pad[i]); x += w; sh = max(sh, h)   # inner (unpadded) origin
        return place

    pages, start = [], 0
    while start < len(order):
        end, acc = start, 0
        while end < len(order):
            h, w = texs[order[end]].shape[:2]
            if acc + h * w > 0.95 * page * page:
                break
            acc += h * w; end += 1
        end = max(end, start + 1)
        while (pl := shelf(order[start:end])) is None:
            end = start + max(1, int((end - start) * 0.97))
        pages.append((order[start:end], pl)); start = end
    meshes, texels = {}, 0
    for pg, (ids, pl) in enumerate(pages):
        img = np.zeros((page, page, 4), np.uint8)
        for i in ids:
            px, py = pl[i]
            t, g = texs[i], pad[i]
            img[py - g:py - g + t.shape[0], px - g:px - g + t.shape[1]] = (t * 255 + 0.5).astype(np.uint8)
            texels += t.shape[0] * t.shape[1]
        r = np.array([geo[i] for i in ids], np.float64)
        cu = (np.stack([r[:, 0], r[:, 2], r[:, 2], r[:, 0]], 1) - MARGIN) / SS
        cv = (np.stack([r[:, 1], r[:, 1], r[:, 3], r[:, 3]], 1) - MARGIN) / SS
        if cdisp is None:
            plq = planes[ids]
            d = np.maximum(plq[:, :1] * cu + plq[:, 1:2] * cv + plq[:, 2:3], 1e-3)
        else:   # corner disparities of the full cell, on its two triangles (diagonal x0y0-x1y1)
            fr = rects[ids].astype(np.float64)
            sx = (np.stack([r[:, 0], r[:, 2], r[:, 2], r[:, 0]], 1) - fr[:, :1]) / (fr[:, 2:3] - fr[:, :1])
            ty = (np.stack([r[:, 1], r[:, 1], r[:, 3], r[:, 3]], 1) - fr[:, 1:2]) / (fr[:, 3:4] - fr[:, 1:2])
            c00, c10, c01, c11 = (cdisp[ids, j][:, None] for j in range(4))
            d = np.where(ty <= sx, c00 + sx * (c10 - c00) + ty * (c11 - c10), c00 + ty * (c01 - c00) + sx * (c11 - c01))
            d = np.maximum(d, 1e-3)
        V = np.stack([(cu - GW / 2) / fg, (cv - Hg / 2) / fg, np.ones_like(cu)], -1) / d[..., None]
        px = np.array([pl[i][0] for i in ids]); py = np.array([pl[i][1] for i in ids])
        tw = np.array([texs[i].shape[1] - 2 * pad[i] for i in ids]); th = np.array([texs[i].shape[0] - 2 * pad[i] for i in ids])
        uu = np.stack([px, px + tw, px + tw, px], 1) / page
        vv = np.stack([py, py, py + th, py + th], 1) / page
        faces = (np.arange(len(ids))[:, None, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]])[None]).reshape(-1, 3)
        mat = PBRMaterial(name=f"patches_{pg}", baseColorTexture=Image.fromarray(img, "RGBA"), metallicFactor=0.0,
                          roughnessFactor=1.0, doubleSided=True, alphaMode="BLEND")
        vis = TextureVisuals(uv=np.stack([uu.reshape(-1), 1 - vv.reshape(-1)], 1), material=mat)
        # The viewer draws surface_<k> with larger k first (k = 0 is the front).
        meshes[f"surface_{len(pages) - 1 - pg}"] = trimesh.Trimesh(
            vertices=V.reshape(-1, 3).astype(np.float32), faces=faces, process=False, visual=vis)
    path = (Path(EXPORT_DIR) if EXPORT_DIR else scene.parent) / f"{scene.stem}{EXPORT}.glb"
    trimesh.Scene(meshes).export(path)
    quad_px = sum((geo[i][2] - geo[i][0]) * (geo[i][3] - geo[i][1]) for i in texs)
    print(f"  quads cover {quad_px / (GW * SS * Hg * SS):.1f}x the frame (fill before blending/depth test)", flush=True)
    print(f"  wrote {path} ({path.stat().st_size / 2**20:.0f} MB, {len(pages)} atlas pages of {page}^2, "
          f"{len(texs)} patches, {texels / 1e6:.0f}M texels = {texels / (len(pages) * page * page):.0%} of the pages)",
          flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        run(Path(s))
