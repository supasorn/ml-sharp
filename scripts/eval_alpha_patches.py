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
  <scene>_patches.glb), EXPORT_DIR, TEXDS, TEXFULL_THR, MAXPAGES, BINS, MPI, BINMODE, REFINE, CULL, OCC, OCC_BLK, OCC_EPS, NOMESH, SPLIT, EXTEND, EVAL, DEBUG0, FLAT
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
CELL_BORDER = int(os.environ.get("CELL_BORDER", 2))
EXPORT_KTX2 = bool(os.environ.get("EXPORT_KTX2"))        # also write <name>_ktx2.glb directly (compressed pages)
KTX2_LEVELS = int(os.environ.get("KTX2_LEVELS", 3))      # mipmap levels in the KTX2 pages
CULL_SCALE = float(os.environ.get("CULL_SCALE", 1.0))    # resolution of the culling renders (0.5 = half)   # MPI export: border texels per cell (>= 2^L keeps mip level L clean)
CELL_BORDER = int(os.environ.get("CELL_BORDER", 2))
EXPORT_KTX2 = bool(os.environ.get("EXPORT_KTX2"))        # also write <name>_ktx2.glb directly (compressed pages)
KTX2_LEVELS = int(os.environ.get("KTX2_LEVELS", 3))      # mipmap levels in the KTX2 pages
CULL_SCALE = float(os.environ.get("CULL_SCALE", 1.0))    # resolution of the culling renders (0.5 = half)   # MPI export: border texels per cell (>= 2^L keeps mip level L clean)
TEXFULL_THR = float(os.environ.get("TEXFULL_THR", 0.003))   # MPI export: cells losing more detail than this at 1/TEXDS keep full res
MAXPAGES = int(os.environ.get("MAXPAGES", 4))              # ... within this many 4096^2 atlas pages
BINS = int(os.environ.get("BINS", 0))
MPI = bool(os.environ.get("MPI"))
BINMODE = os.environ.get("BINMODE", "uniform")   # MPI bins: uniform in disparity, or kmeans (where the content is)
REFINE = bool(os.environ.get("REFINE"))
# MPI + REFINE, SNAP=1: cell corners near the front surface (seen from the input view, all splats rendered
# together) take that surface's depth instead of their bin's own, so strips of neighbouring bins on one surface
# (a face, a receding wall) meet at the same depth instead of separating into a visible stack when the viewer
# moves closer. A corner may snap only where the surface is covered and smooth around it (its 4 cells agree
# within SNAP_REL: no depth edge) and the bin's own corner is within min(SNAP_TOL bin widths, SNAP_REL of the
# disparity) of it. SNAP_WHERE=all snaps every such corner; =seams only corners where >= 2 bins qualify (where
# strips meet; elsewhere a bin keeps its own relief). SNAP_VALUE=cell: the surface averaged over the 4 cells
# around the corner; =point: averaged over SNAP_BOX px at the corner (keeps more relief).
# REFINE_MODE=mean (default): a cell's depth is the alpha-weighted mean disparity of its bin's splats and a
# corner the mean of the cells around it. =plane: a weighted least-squares plane of disparity per cell (ridge
# PLANE_REG on the slopes, so tiny or thin patches fall back to the mean), evaluated at the corners (clamped to
# the bin's range, see PLANE_MARGIN) and averaged over the cells around each corner by alpha. A strip ending
# inside a cell then gets the depth the surface has at its end, so strips of neighbouring bins meet (no crack
# when moving) while each cell's content stays at its own depth.
REFINE_MODE = os.environ.get("REFINE_MODE", "mean")
# MPI: cell size per bin by distance (parallax): bins nearer than NEAR_M metres use TILE_NEAR px cells (finer
# geometry where head moves shift things most), bins farther than FAR_M use TILE_FAR (fewer triangles where
# nothing moves); others TILE. Each bin stays uniform inside (crack-free); 0 = off.
TILE_NEAR = int(os.environ.get("TILE_NEAR", 0))
NEAR_M = float(os.environ.get("NEAR_M", 2.0))
TILE_FAR = int(os.environ.get("TILE_FAR", 0))
FAR_M = float(os.environ.get("FAR_M", 20.0))
PLANE_REG = float(os.environ.get("PLANE_REG", 0.01))
PLANE_MARGIN = float(os.environ.get("PLANE_MARGIN", 0.0))   # clamp corners to the bin's range +- this many bin
# widths; 0 keeps bins disjoint in depth, so the fixed far-to-near draw order stays exact (layers never cross)
# MPI + REFINE, SURF=1: surface pieces. Per cell, runs of occupied bins (gaps of at most SURF_GAP empty bins,
# never across a multiple of SURF_SPAN bins, so neighbouring cells cut a surface at the same depths) become one
# piece: their bins' textures composited front to back (exact at the input view), one least-squares plane of
# disparity weighted by visible alpha. Pieces of neighbouring cells share a corner where their planes meet there
# within SURF_JOIN bin widths (pieces with the same bins first, so a thick shell's sheets aren't sheared; one
# piece per cell), so a face is one connected sheet and tears only at depth jumps; content within a run (hair
# over skin) is pressed onto its plane. Found on two scenes (hand, faces): cutting runs from each cell's own
# nearest bin (SURF_ALIGN=0) chops steep surfaces differently per cell and tears them; one bin per piece
# (SURF_SPAN=1, SURF_JOIN=0.3) scores best but strips the forehead like plain MPI; cutting where a bin stops
# adding visible surface (SURF_VIS) tears faces; joins of 1 bin leave tears, of 3 smooth.
SURF = bool(os.environ.get("SURF"))
SURF_GAP = int(os.environ.get("SURF_GAP", 1))
SURF_SPAN = int(os.environ.get("SURF_SPAN", 8))
SURF_JOIN = float(os.environ.get("SURF_JOIN", 3.0))
SURF_VIS = float(os.environ.get("SURF_VIS", 0))
SURF_ALIGN = os.environ.get("SURF_ALIGN", "1") != "0"
SURF_LINK = float(os.environ.get("SURF_LINK", 0.5))   # > 1: never link
OVERLAP = float(os.environ.get("OVERLAP", 2 if os.environ.get("SURF") else 0))   # see export_glb
SURF_GROUP = os.environ.get("SURF_GROUP", "match")   # chain: the first version's corner grouping
SURF_FILL = int(os.environ.get("SURF_FILL", 0))      # render px of the seam band filled (0 = off)
SURF_LINK_PLANE = bool(os.environ.get("SURF_LINK_PLANE"))   # linked pieces share one plane (tested: worse)
SNAP = bool(os.environ.get("SNAP"))
SNAP_TOL = float(os.environ.get("SNAP_TOL", 2.0))
SNAP_REL = float(os.environ.get("SNAP_REL", 0.03))
SNAP_WHERE = os.environ.get("SNAP_WHERE", "all")
SNAP_VALUE = os.environ.get("SNAP_VALUE", "cell")
SNAP_BOX = int(os.environ.get("SNAP_BOX", 5))
SNAP_TAG = ((f"n{TILE_NEAR}-{NEAR_M:g}" if TILE_NEAR else "") + (f"f{TILE_FAR}-{FAR_M:g}" if TILE_FAR else "")
            + ("p" + (f"m{PLANE_MARGIN:g}" if PLANE_MARGIN else "") if REFINE_MODE == "plane" else "")
            + (f"s{SNAP_WHERE[0]}{SNAP_VALUE[0]}{SNAP_BOX if SNAP_VALUE == 'point' else ''}" if SNAP else "")
            + (f"S{SURF_SPAN}g{SURF_GAP}j{SURF_JOIN:g}" + (f"v{SURF_VIS:g}" if SURF_VIS else "") + ("a" if SURF_ALIGN else "") + f"f{SURF_FILL}" + (f"L{SURF_LINK:g}" if SURF_LINK_PLANE else "") + (f"o{OVERLAP:g}" if OVERLAP else "") + ("c" if SURF_GROUP == "chain" else "") if SURF else ""))
EVAL_EYES = [e for e in os.environ.get("EVAL_EYES", "").split(",") if e]   # eval: only these eyes (default all)
ZEYE = float(os.environ.get("ZEYE", 0))           # eval: extra eyes this far forward (and forward + sideways)
CULL = float(os.environ.get("CULL", 0))
OCC = bool(os.environ.get("OCC"))
NOMESH = bool(os.environ.get("NOMESH"))          # MPI-only: all splats of the original .ply, no hybrid mesh
SPLIT = [int(x) for x in os.environ.get("SPLIT", "").split(",") if x]   # opaque-block split stats, block sizes (px)                # MPI: depth-only occluder quads for fully opaque blocks
OCC_BLK = int(os.environ.get("OCC_BLK", 8))      # occluder block size (render px; multiple of TEXDS)
OCC_EPS = float(os.environ.get("OCC_EPS", 0.002))  # occluders sit this fraction farther than their cell          # drop cells contributing less than this from every viewpoint          # MPI: per-cell depth (shared corners) instead of the bin's centre
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


T0 = time.time()                                   # per-run clock for the timing in log lines


def snap_corners(bins, cd_l, surf, binw):
    """SNAP: replace bins' corner disparities by the front surface's where allowed (see SNAP above).

    bins: per bin with cells, (bin, G (ny+1, nx+1) corner disparities, used (ny+1, nx+1), cyn, cxn);
    cd_l: per-bin corner lists to fill (None placeholders, same order); surf: (Gs value, Gs_ok).
    """
    Gs, ok = surf
    D = np.stack([g for _, g, _, _, _ in bins])
    used = np.stack([u for _, _, u, _, _ in bins])
    tol = np.minimum(SNAP_TOL * binw, SNAP_REL * Gs)
    near = used & ok[None] & (np.abs(D - Gs[None]) <= tol[None])
    if SNAP_WHERE == "seams":
        near &= (near.sum(0) >= 2)[None]
    D = np.where(near, Gs[None], D)
    slots = [i for i, c in enumerate(cd_l) if c is None]
    for slot, (n, (_, _, _, cyn, cxn)) in zip(slots, enumerate(bins)):
        G = D[n]
        cd_l[slot] = np.stack([G[cyn, cxn], G[cyn, cxn + 1], G[cyn + 1, cxn], G[cyn + 1, cxn + 1]], 1)
    print(f"  surface snap ({SNAP_WHERE}, {SNAP_VALUE}): {int(near.sum())} of {int(used.sum())} bin corners "
          f"({near.sum() / max(1, used.sum()):.1%}) took the front surface's depth", flush=True)


def surface_pieces(surf_l, crops, crops_b, rect_l, cen_l, cd_l, cell, nx, lo, hi, edges):
    """SURF: merge the bins' cells into surface pieces (see SURF); appends to the cell lists like one bin."""
    ks = np.concatenate([np.full(len(e[1]), e[0]) for e in surf_l])
    cid = np.concatenate([e[1] for e in surf_l])
    o = np.lexsort((-ks, cid))                            # by cell, nearest bin first
    ks, cid = ks[o], cid[o]
    tex = torch.cat([e[2] for e in surf_l])[torch.tensor(o, device=dev)]
    texb = torch.cat([e[3] for e in surf_l])[torch.tensor(o)] if surf_l[0][3] is not None else None
    pdisp = torch.cat([e[4] for e in surf_l])[torch.tensor(o, device=dev)]
    surf_l.clear()
    N = len(ks)
    # Pieces, all cells at once, walking each cell's bins near to far: a bin joins the cell's current piece
    # while it adds visible surface there (at least SURF_VIS of its alpha lands where the piece is still
    # transparent: a steep surface reaching other pixels at other depths, skin between hair strands), within
    # SURF_GAP empty bins and SURF_SPAN bins; else (mostly behind what the piece already covers: the fill SHARP
    # puts behind surfaces) the piece is closed and the bin starts the next one. Textures: front-to-back "over"
    # of premultiplied RGBA; plane statistics: each texel's disparity weighted by its visible alpha.
    ucid, cstart, ccount = np.unique(cid, return_index=True, return_counts=True)
    nC = len(ucid)
    tc = (torch.arange(cell, device=dev, dtype=torch.float64) + 0.5) / cell - 0.5
    Xc, Yc = tc[None, None, :], tc[None, :, None]
    A = torch.zeros(nC, cell, cell, 4, device=dev); T = torch.ones(nC, cell, cell, 1, device=dev)
    if texb is not None:
        cbs = texb.shape[1:3]
        Ab = torch.zeros((nC,) + tuple(texb.shape[1:]), device=dev); Tb = torch.ones((nC,) + tuple(cbs) + (1,), device=dev)
    ST = torch.zeros(nC, 9, device=dev, dtype=torch.float64)
    has = np.zeros(nC, bool); kn = np.zeros(nC, np.int64); kf = np.zeros(nC, np.int64)
    lk = np.zeros(nC, bool)                               # current piece continues the one before (SURF_LINK)
    P_tex, P_texb, P_st, P_kn, P_kf, P_c, P_lk = [], [], [], [], [], [], []

    def emit(ci):                                         # close the current pieces of cells ci
        if len(ci) == 0:
            return
        ct = torch.tensor(ci, device=dev)
        P_tex.append(A[ct].half()); P_st.append(ST[ct].cpu().numpy())
        if texb is not None:
            P_texb.append(Ab[ct].half().cpu())
        P_kn.append(kn[ci].copy()); P_kf.append(kf[ci].copy()); P_c.append(ucid[ci]); P_lk.append(lk[ci].copy())
        A[ct] = 0; T[ct] = 1; ST[ct] = 0
        if texb is not None:
            Ab[ct] = 0; Tb[ct] = 1
        has[ci] = False

    for r in range(ccount.max()):
        ci = np.nonzero(ccount > r)[0]
        e = cstart[ci] + r
        k = ks[e]
        et = torch.tensor(e, device=dev); ct = torch.tensor(ci, device=dev)
        c = tex[et].float()
        a = c[..., 3]
        vis = ((T[ct, ..., 0] * a).sum((1, 2)) / a.sum((1, 2)).clamp(min=1e-9)).cpu().numpy()
        deep = (kn[ci] // SURF_SPAN != k // SURF_SPAN) if SURF_ALIGN else (kn[ci] - k >= SURF_SPAN)
        gap_ = (vis < SURF_VIS) | (kf[ci] - k > SURF_GAP + 1)
        split = has[ci] & (gap_ | deep)
        fresh = ~has[ci]
        emit(ci[split])
        lk[ci[fresh]] = False
        lk[ci[split]] = (~gap_ & (vis >= SURF_LINK))[split]
        new_ = ~has[ci]
        kn[ci[new_]] = k[new_]
        wv = (T[ct, ..., 0] * a).double()
        pv = pdisp[et].double()
        Sc = lambda f_: (wv * f_).sum((1, 2))
        ST[ct] += torch.stack([Sc(1.0), Sc(Xc), Sc(Yc), Sc(Xc * Xc), Sc(Yc * Yc), Sc(Xc * Yc),
                               Sc(pv), Sc(pv * Xc), Sc(pv * Yc)], 1)
        A[ct] += T[ct] * c
        T[ct] *= 1 - c[..., 3:]
        if texb is not None:
            cb_ = texb[torch.tensor(e)].to(dev).float()
            Ab[ct] += Tb[ct] * cb_
            Tb[ct] *= 1 - cb_[..., 3:]
        kf[ci] = k; has[ci] = True
    emit(np.nonzero(has)[0])
    del A, T, tex
    pc = np.concatenate(P_c); k_near = np.concatenate(P_kn); k_far = np.concatenate(P_kf)
    o3 = np.lexsort((-k_near, pc))                        # by cell, nearest first
    pc, k_near, k_far = pc[o3], k_near[o3], k_far[o3]
    st = np.concatenate(P_st)[o3]
    lnk = np.concatenate(P_lk)[o3]
    out = torch.cat(P_tex)[torch.tensor(o3, device=dev)]
    outb = torch.cat(P_texb)[torch.tensor(o3)] if texb is not None else None
    # Seam fill: a piece continuing the visible surface of the one before it in its cell (cut only by an
    # aligned depth boundary, see SURF_LINK) also carries that piece's colour where it is opaque, underneath
    # it: hidden at the input view, it fills the seam that opens between the two when the head moves (the
    # cheek's own colour instead of what is behind it).
    if SURF_FILL:
        for i in np.nonzero(lnk)[0]:
            for t_ in (out, outb):
                if t_ is None:
                    continue
                f_, b_ = t_[i - 1].float(), t_[i].float()
                near = torch.nn.functional.max_pool2d((b_[None, None, ..., 3] > 0.1).float(), 2 * SURF_FILL + 1, 1,
                                                      SURF_FILL)[0, 0, ..., None] > 0      # band along the seam
                t_[i] = torch.where((f_[..., 3:] > 0.9) & near, f_ + (1 - f_[..., 3:]) * b_, b_).half()
    nS = len(pc)
    # A piece cut off only by an aligned depth boundary (SURF_ALIGN) while its first bin still adds visible
    # surface (at least SURF_LINK of its alpha where the piece before is transparent: a cheek curving across
    # the boundary) continues that surface: linked pieces keep their textures but share one plane and their
    # corners (else the boundary is a seam inside the cell, as between MPI bins). Geometry and corner sharing
    # below are per surface (linked pieces together).
    ent = np.cumsum(~(lnk & SURF_LINK_PLANE)) - 1
    fE = np.r_[0, np.nonzero(np.diff(ent))[0] + 1]
    lE = np.r_[fE[1:] - 1, nS - 1]
    chain = (np.cumsum(~lnk) - 1)[fE]                    # pieces continuing one visible surface in a cell
    pieces = (pc, k_near, k_far, st, nS)
    pc, k_near, k_far, st, nS = pc[fE], k_near[fE], k_far[lE], np.add.reduceat(st, fE, axis=0), len(fE)
    # Geometry: one plane per piece from the summed statistics, corners clamped to its bins' range +- 1 bin.
    sw, sx, sy, sxx, syy, sxy, sp, spx, spy = st.T
    lam = PLANE_REG * sw
    Mp = np.stack([np.stack([sw, sx, sy], -1), np.stack([sx, sxx + lam, sxy], -1),
                   np.stack([sy, sxy, syy + lam], -1)], -2) + 1e-9 * np.eye(3)
    c0, cxs, cys = np.linalg.solve(Mp, np.stack([sp, spx, spy], -1)[..., None])[..., 0].T
    bw = (hi - lo) / BINS
    e_all = np.r_[lo, edges, hi]
    d_lo, d_hi = e_all[k_far] - bw, e_all[k_near + 1] + bw
    pcy, pcx = pc // nx, pc % nx
    pred = np.stack([np.clip(c0 + cxs * (ox - 0.5) + cys * (oy - 0.5), d_lo, d_hi)
                     for oy in (0, 1) for ox in (0, 1)], 1)           # x0y0, x1y0, x0y1, x1y1
    # Shared corners: at each grid corner, groups of the pieces around it whose planes meet within SURF_JOIN
    # bins there. The most visible pieces anchor groups (faint fuzz in front of a surface must not split it).
    # Each piece joins, among the groups within SURF_JOIN without a piece of its cell yet (pieces of one cell
    # are separate surfaces), the one sharing most of its bins (a thick shell is several sheets; pairing its
    # bins arbitrarily would shear them), else the nearest (a strip ending where its neighbour continues in the
    # next bin), else starts its own.
    key = np.stack([(pcy + oy) * (nx + 1) + pcx + ox for oy in (0, 1) for ox in (0, 1)], 1).ravel()
    pv, wv, cv, chv = pred.ravel(), np.repeat(sw, 4), np.repeat(pc, 4), np.repeat(chain, 4)
    klo, khi = np.repeat(k_far, 4), np.repeat(k_near, 4)
    o2 = np.lexsort((-wv, key))                           # per corner, most visible first
    ko, po, co, lo_, hi_, cho = key[o2], pv[o2], cv[o2], klo[o2], khi[o2], chv[o2]
    gid = np.empty(len(o2), np.int64)
    starts = np.r_[0, np.nonzero(np.diff(ko))[0] + 1, len(o2)]
    ng = 0
    if SURF_GROUP == "chain":                             # first version: near to far, chained within SURF_JOIN
        o2 = np.lexsort((pv, key))
        brk = np.r_[True, (np.diff(key[o2]) != 0) | (np.diff(pv[o2]) > SURF_JOIN * bw)]
        gid = np.cumsum(brk) - 1
        starts = starts[:0]
    for a_, b_ in zip(starts[:-1], starts[1:]):
        groups = []                                       # [anchor disparity, klo, khi, cells, id, chains]
        for i in range(a_, b_):
            best, bs = None, None
            for g in groups:
                dd = abs(po[i] - g[0])
                if dd <= SURF_JOIN * bw and (co[i] not in g[3] or cho[i] in g[5]):
                    sc_ = (min(hi_[i], g[2]) - max(lo_[i], g[1]), -dd)
                    if bs is None or sc_ > bs:
                        best, bs = g, sc_
            if best is None:
                best = [po[i], lo_[i], hi_[i], set(), ng, set()]; ng += 1; groups.append(best)
            best[3].add(co[i]); best[5].add(cho[i]); gid[i] = best[4]
    cl = gid
    m = np.bincount(cl, pv[o2] * wv[o2]) / np.maximum(np.bincount(cl, wv[o2]), 1e-12)
    shared = np.empty_like(pv); shared[o2] = m[cl]
    cd = shared.reshape(nS, 4)[ent]
    nE = nS
    pred = pred[ent]
    pc, k_near, k_far, st, nS = pieces
    pcy, pcx = pc // nx, pc % nx
    sw, sp = st[:, 0], st[:, 6]
    # Keep each cell's pieces in their depth order at every corner (a back piece pulled in front of the one
    # before it by its corners would show through it, e.g. a dark strap behind a shirt).
    if os.environ.get("SURF_ORDER", "1") != "0":
        for i in np.nonzero(pc[1:] == pc[:-1])[0] + 1:   # pieces are by cell, nearest first
            cd[i] = np.minimum(cd[i], cd[i - 1] - 1e-3 * bw)
    nb = np.bincount(cl)
    if os.environ.get("SURF_DEBUG"):                     # x0,y0,x1,y1 in GW-wide photo px: dump the pieces there
        x0_, y0_, x1_, y1_ = (float(v) for v in os.environ["SURF_DEBUG"].split(","))
        gsz = np.empty_like(pv); gsz[o2] = nb[cl]
        gsz = gsz.reshape(nE, 4)[ent]
        for i in range(nS):
            px0, py0 = (pcx[i] * cell - MARGIN) / SS, (pcy[i] * cell - MARGIN) / SS
            if x0_ <= px0 <= x1_ and y0_ <= py0 <= y1_:
                print(f"  piece {i} cell ({pcx[i]},{pcy[i]}) px ({px0:.0f},{py0:.0f}) bins {k_far[i]}-{k_near[i]} "
                      f"w {sw[i]:.0f} pred {np.round(pred[i] / bw, 1)} cd {np.round(cd[i] / bw, 1)} group {gsz[i]}")
    print(f"  surface pieces: {N} bin cells -> {nS} pieces (mean {N / nS:.1f} bins) on {nE} surfaces, "
          f"{np.mean(nb[cl] > 1):.0%} of piece corners shared", flush=True)
    crops.append(out.reshape(-1, 4))
    if outb is not None:
        crops_b.append((0, outb))
    rect_l.append(np.stack([pcx * cell, pcy * cell, (pcx + 1) * cell, (pcy + 1) * cell], 1))
    cen_l.append(sp / np.maximum(sw, 1e-12))
    cd_l.append(cd)


def run(scene: Path):
    global T0
    t0 = T0 = time.time()
    meta = json.loads(scene.with_suffix(".json").read_text())
    Hg = int(round(GW * meta["height"] / meta["width"]))
    fg = meta["focal_length_px"] * GW / meta["width"]
    W, H, f = GW * SS, Hg * SS, fg * SS
    Wc, Hc = W + 2 * MARGIN, H + 2 * MARGIN
    full = splat_tensors(PlyData.read(scene)["vertex"].data, dev)
    kv = PlyData.read(scene if NOMESH else scene.with_name(scene.stem + SUFFIX + ".ply"))["vertex"].data
    z = np.asarray(kv["z"], np.float64)
    ok = z > 0.05
    kv = kv[ok]; z = z[ok]
    kept = splat_tensors(kv, dev)
    u, w = fg * kv["x"] / z + GW / 2, fg * kv["y"] / z + Hg / 2
    if NOMESH:                                  # MPI-only: no hybrid mesh (one triangle behind the camera, never hit)
        Tm = torch.tensor([[[0.0, 0.0, -10.0], [1.0, 0.0, -10.0], [0.0, 1.0, -10.0]]], device=dev)
    else:
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
        gb = CELL_BORDER * TEXDS                          # border (render px) of real neighbour texels, export only
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
        snap_bins = []                                    # SNAP: (bin, corner disparities, used corners, cells)
        surf_l = []                                       # SURF: (bin, cell ids, textures, bordered textures, stats)
        assert not SURF or (REFINE and not (SNAP or TILE_NEAR or TILE_FAR or EXTEND)), "SURF: REFINE, one cell size"
        if REFINE and SNAP:
            # Front surface: all splats rendered together (expected depth where alpha > 0.5).
            out_s, al_s, _ = ev.gs.rendering.rasterization(**kept, viewmats=torch.eye(4, device=dev)[None], Ks=Kc,
                                                           width=Wc, height=Hc, eps2d=ev.EPS, render_mode="RGB+ED")
            w_s = (al_s[0, ..., 0] > 0.5).float()
            disp_s = torch.where(w_s > 0, 1 / out_s[0, ..., 3].clamp(min=1e-6), 0)
            del out_s, al_s
            padc = lambda t: torch.nn.functional.pad(t, (0, nx * cell - Wc, 0, ny * cell - Hc))
            sw = padc(w_s).reshape(ny, cell, nx, cell).sum((1, 3))
            sd = padc(w_s * disp_s).reshape(ny, cell, nx, cell).sum((1, 3))
            covered = (sw > 0.5 * cell * cell).double()
            dsurf = torch.where(sw > 0, sd / sw.clamp(min=1e-9), 0).double()
            Gc = torch.zeros(ny + 1, nx + 1, device=dev, dtype=torch.float64); Gn_ = torch.zeros_like(Gc)
            Gmax = torch.zeros_like(Gc); Gmin = torch.full_like(Gc, float("inf"))
            for oy in (0, 1):
                for ox in (0, 1):
                    sl = (slice(oy, oy + ny), slice(ox, ox + nx))
                    Gc[sl] += dsurf * covered; Gn_[sl] += covered
                    Gmax[sl] = torch.maximum(Gmax[sl], torch.where(covered > 0, dsurf, 0))
                    Gmin[sl] = torch.minimum(Gmin[sl], torch.where(covered > 0, dsurf, float("inf")))
            snap_ok = ((Gn_ == 4) & (Gmax - Gmin <= SNAP_REL * Gmax)).cpu().numpy()
            if SNAP_VALUE == "point":
                box = lambda t: torch.nn.functional.avg_pool2d(t[None, None], SNAP_BOX, 1, SNAP_BOX // 2,
                                                               count_include_pad=False)[0, 0]
                dsm = box(w_s * disp_s) / box(w_s).clamp(min=1e-6)
                gy = (torch.arange(ny + 1, device=dev) * cell).clamp(max=Hc - 1)
                gx = (torch.arange(nx + 1, device=dev) * cell).clamp(max=Wc - 1)
                snap_val = dsm[gy][:, gx].double().cpu().numpy()
            else:
                snap_val = (Gc / Gn_.clamp(min=1)).cpu().numpy()
        for k in reversed(range(BINS)):                      # nearest bin first
            m = b_t == k
            if not m.any():
                continue
            sub = {kk: v[m] for kk, v in kept.items()}
            out, alpha, _ = ev.gs.rendering.rasterization(**sub, viewmats=torch.eye(4, device=dev)[None], Ks=Kc,
                                                          width=Wc, height=Hc, eps2d=ev.EPS, render_mode="RGB+D")
            rgb, a, dacc = out[0, ..., :3], alpha[0, ..., 0], out[0, ..., 3]
            if TILE_NEAR or TILE_FAR:                       # this bin's cell size (see TILE_NEAR)
                zk = 1 / max(centers[k], 1e-6)
                tile_k = TILE_NEAR if TILE_NEAR and zk < NEAR_M else TILE_FAR if TILE_FAR and zk > FAR_M else TILE
                cell = tile_k * SS
                ny, nx = -(-Hc // cell), -(-Wc // cell)
                cb = cell + 2 * gb
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
            if SURF:                                         # keep this bin's cells for the pieces (see SURF)
                padS = lambda t: torch.nn.functional.pad(t, (0, nx * cell - Wc, 0, ny * cell - Hc))
                pd = torch.where(a > 1e-3, a / dacc.clamp(min=1e-6), 0)
                Pc = padS(pd).reshape(ny, cell, nx, cell).permute(0, 2, 1, 3)[cy, cx]      # disparity per px
                selb = None
                if EXPORT:
                    pimg = torch.nn.functional.pad(img.permute(2, 0, 1)[None].float(), (gb, gb, gb, gb))
                    cols = torch.nn.functional.unfold(pimg, cb, stride=cell)[0]
                    selb = cols[:, cy * nx + cx].reshape(4, cb, cb, -1).permute(3, 1, 2, 0).half().cpu()
                surf_l.append((k, (cy * nx + cx).cpu().numpy(), cells[cy, cx], selb, Pc))
                continue
            crops.append(cells[cy, cx].reshape(-1, 4))
            if EXPORT:
                pimg = torch.nn.functional.pad(img.permute(2, 0, 1)[None].float(), (gb, gb, gb, gb))
                cols = torch.nn.functional.unfold(pimg, cb, stride=cell)[0]          # (4*cb*cb, ny*nx)
                sel = cols[:, cy * nx + cx].reshape(4, cb, cb, -1).permute(3, 1, 2, 0)
                crops_b.append((sum(len(r) for r in rect_l), sel.half().cpu()))   # (first cell index, (n, cb, cb, 4)), on the CPU
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
                if REFINE_MODE == "plane":
                    tc = (torch.arange(cell, device=dev, dtype=torch.float64) + 0.5) / cell - 0.5   # cell coords
                    A_ = pad_(a).double().reshape(ny, cell, nx, cell)
                    P_ = pad_(pd).double().reshape(ny, cell, nx, cell)
                    Xc, Yc = tc[None, None, None, :], tc[None, :, None, None]
                    Sc = lambda f_: (A_ * f_).sum((1, 3))
                    sw, sx, sy = Sc(1.0), Sc(Xc), Sc(Yc)
                    sxx, syy, sxy = Sc(Xc * Xc), Sc(Yc * Yc), Sc(Xc * Yc)
                    lam = PLANE_REG * sw
                    Mp = torch.stack([torch.stack([sw, sx, sy], -1), torch.stack([sx, sxx + lam, sxy], -1),
                                     torch.stack([sy, sxy, syy + lam], -1)], -2)
                    Mp = Mp + 1e-9 * torch.eye(3, device=dev, dtype=torch.float64)
                    rhs = torch.stack([Sc(P_), Sc(P_ * Xc), Sc(P_ * Yc)], -1)
                    c0, cxs, cys = torch.linalg.solve(Mp, rhs[..., None])[..., 0].unbind(-1)
                    d_lo = (edges[k - 1] if k > 0 else lo) - PLANE_MARGIN * (hi - lo) / BINS
                    d_hi = (edges[k] if k < BINS - 1 else hi) + PLANE_MARGIN * (hi - lo) / BINS
                    wcell = torch.where(occ, sw, 0)
                    for oy in (0, 1):
                        for ox in (0, 1):
                            pred = (c0 + cxs * (ox - 0.5) + cys * (oy - 0.5)).clamp(d_lo, d_hi)
                            G[oy:oy + ny, ox:ox + nx] += pred * wcell; Gn[oy:oy + ny, ox:ox + nx] += wcell
                    G = torch.where(Gn > 0, G / Gn.clamp(min=1e-12), 0).cpu().numpy()
                else:
                    for oy in (0, 1):
                        for ox in (0, 1):
                            G[oy:oy + ny, ox:ox + nx] += dcell * o; Gn[oy:oy + ny, ox:ox + nx] += o
                    G = (G / Gn.clamp(min=1)).cpu().numpy()
                if SNAP:                                     # decided after all bins (seams need the others)
                    snap_bins.append((k, G, (Gn > 0).cpu().numpy(), cyn, cxn))
                    cd_l.append(None)
                else:
                    cd_l.append(np.stack([G[cyn, cxn], G[cyn, cxn + 1], G[cyn + 1, cxn], G[cyn + 1, cxn + 1]], 1))
            else:
                cd_l.append(np.full((len(cyn), 4), centers[k]))
        if SURF:
            surface_pieces(surf_l, crops, crops_b, rect_l, cen_l, cd_l, cell, nx, lo, hi, edges)
        if REFINE and SNAP:
            assert not (TILE_NEAR or TILE_FAR), "SNAP needs one cell size"
            snap_corners(snap_bins, cd_l, (snap_val, snap_ok), (hi - lo) / BINS)
        rects = np.concatenate(rect_l).astype(np.int64)
        cen = np.concatenate(cen_l)
        planes = np.stack([np.zeros_like(cen), np.zeros_like(cen), cen], 1)
        cdisp = np.concatenate(cd_l)                      # corner disparities (x0y0, x1y0, x0y1, x1y1) per cell
        P = len(rects)
        area = ((rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])).astype(np.int64)
        off = np.r_[0, np.cumsum(area)[:-1]]
        atlas = torch.cat(crops)
        border = (crops_b, gb) if EXPORT else None
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

    ov = OVERLAP                                          # quads reach this many render px past their cell
    def inside(prim, ui, vi):
        lx, ly = ui + MARGIN - 0.5 - rx0[prim], vi + MARGIN - 0.5 - ry0[prim]
        return (lx >= -0.5 - ov) & (lx < rw[prim] - 0.5 + ov) & (ly >= -0.5 - ov) & (ly < rh[prim] - 0.5 + ov)

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

    def patch_samples(eye, with_local=False, scale=1.0):
        V, PIX, T, PR, LX, LY = [], [], [], [], [], []
        Ws, Hs, fs = (W, H, f) if scale == 1.0 else (round(W * scale), round(H * scale), f * scale)
        ins = geom_inside if scale == 1.0 else (lambda prim, ui, vi: geom_inside(prim, ui / scale, vi / scale))
        for prim, pix, p, ui, vi, t in cast(geomN, geomC, ins, eye, Ws, Hs, fs):
            if scale != 1.0:
                ui, vi = ui / scale, vi / scale                         # input-view coords at render resolution
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
            if with_local:
                LX.append(lx[k]); LY.append(ly[k])
        if with_local:
            return torch.cat(V), torch.cat(PIX), torch.cat(T), torch.cat(PR), torch.cat(LX), torch.cat(LY)
        return torch.cat(V), torch.cat(PIX), torch.cat(T), torch.cat(PR)


    alive = torch.ones(P, dtype=torch.bool, device=dev)
    if CULL > 0:
        # Drop cells that are invisible from every viewpoint in the head range: per viewpoint, composite in
        # depth order (after the mesh depth test) and take each cell's largest visible contribution
        # alpha * transmittance of everything in front; cells below CULL everywhere are removed.
        best = torch.zeros(P, device=dev)
        eyes_c = [(x, y, zz) for x in (-MOVE, 0.0, MOVE) for y in (-MOVE / 2, 0.0, MOVE / 2) for zz in (0.0, 0.4 * MOVE)]
        for eye in eyes_c:
            vals, pix, t, prim = patch_samples(eye, scale=CULL_SCALE)
            _, mdep_c = s2.mesh_layer(M, eye, round(W * CULL_SCALE), round(H * CULL_SCALE), f * CULL_SCALE)
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
        alive = best >= CULL
        torch.cuda.empty_cache()
        npx = round(W * CULL_SCALE) * round(H * CULL_SCALE)
        print(f"  input view: {len(prim0) / npx:.2f} -> {alive[prim0].sum().item() / npx:.2f} layer fragments "
              f"per pixel after the mesh depth test", flush=True)
        fill = area.astype(np.float64)
        print(f"  culling: {int((~alive).sum())} of {P} cells ({(~alive).float().mean().item():.0%}, "
              f"{fill[~alive.cpu().numpy()].sum() / fill.sum():.0%} of cell area) never contribute >= {CULL} "
              f"from {len(eyes_c)} viewpoints within +-{MOVE} m -> dropped [{time.time() - t0:.0f}s]", flush=True)

    # Occluders (MPI, OCC=1): fully opaque OCC_BLK-px blocks of the kept cells (every texel alpha > 0.99 at render
    # resolution, so also after box downsampling), as depth-only quads drawn with the opaque mesh. They sit
    # flat at the cell's farthest corner pushed OCC_EPS farther, so they never hide their own layer, only layers
    # behind; fragments there then fail the depth test before shading. A cell opaque in all blocks gets one quad.
    occ_rects = np.zeros((0, 4), np.int64); occ_disp = np.zeros(0)
    if MPI and OCC:
        cellpx = rects[0, 2] - rects[0, 0]
        nb = cellpx // OCC_BLK
        al = atlas[:, 3].reshape(P, nb, OCC_BLK, nb, OCC_BLK).float().amin((2, 4)) > 0.99      # (P, nb, nb)
        al &= alive[:, None, None]
        full_c = al.all(2).all(1)
        far = (cdisp.min(1) * (1 - OCC_EPS))
        ci_, by_, bx_ = (x.cpu().numpy() for x in torch.nonzero(al & ~full_c[:, None, None], as_tuple=True))
        fc = torch.nonzero(full_c)[:, 0].cpu().numpy()
        r0 = rects[ci_]
        occ_rects = np.concatenate([rects[fc], np.stack([r0[:, 0] + bx_ * OCC_BLK, r0[:, 1] + by_ * OCC_BLK,
                                                         r0[:, 0] + (bx_ + 1) * OCC_BLK, r0[:, 1] + (by_ + 1) * OCC_BLK], 1)])
        occ_disp = np.concatenate([far[fc], far[ci_]])
        print(f"  occluders: {len(occ_rects)} quads ({len(fc)} whole cells + {len(ci_)} blocks), covering "
              f"{((occ_rects[:, 2] - occ_rects[:, 0]) * (occ_rects[:, 3] - occ_rects[:, 1])).sum() / (W * H):.2f}x "
              f"the frame", flush=True)

    def occluder_depth(eye):
        """Per-pixel depth (along the eye ray, inf = none) of the nearest occluder quad."""
        dep = torch.full((W * H,), float("inf"), device=dev)
        if not len(occ_rects):
            return dep
        Ro = torch.tensor(occ_rects, device=dev, dtype=torch.float32)
        do = torch.tensor(occ_disp, device=dev, dtype=torch.float32).clamp(min=float(dmin))
        n_o = torch.stack([torch.zeros_like(do), torch.zeros_like(do), do], 1)   # flat: n.p = 1 <=> z = 1/d
        ou = (torch.stack([Ro[:, 0], Ro[:, 2], Ro[:, 0], Ro[:, 2]], 1) - MARGIN) / SS
        ov = (torch.stack([Ro[:, 1], Ro[:, 1], Ro[:, 3], Ro[:, 3]], 1) - MARGIN) / SS
        oc = torch.stack([(ou - GW / 2) / fg, (ov - Hg / 2) / fg, torch.ones_like(ou)], -1) / do[:, None, None]

        def o_inside(prim, ui, vi):
            lx, ly = ui + MARGIN - Ro[prim, 0], vi + MARGIN - Ro[prim, 1]
            return (lx >= 0) & (lx < Ro[prim, 2] - Ro[prim, 0]) & (ly >= 0) & (ly < Ro[prim, 3] - Ro[prim, 1])
        for prim, pix, p, ui, vi, t in cast(n_o, oc, o_inside, eye, W, H, f):
            dep.scatter_reduce_(0, pix, t, "amin")
        return dep

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
    if ZEYE:                                     # moving closer (forward), and closer + sideways
        eyes |= {f"z+{ZEYE}": (0, 0, ZEYE), f"x+0.2z+{ZEYE}": (0.2, 0, ZEYE)}
    for e_ in os.environ.get("EXTRA_EYES", "").split(";"):   # name:x,y,z;...
        if e_:
            n_, xyz = e_.split(":")
            eyes[n_] = tuple(float(v) for v in xyz.split(","))
    if EVAL_EYES:
        eyes = {k: v for k, v in eyes.items() if k in EVAL_EYES}
    res = {}
    print(f"  size: {int(alive.sum())} cells kept -> {2 * int(alive.sum())} layer triangles"
          f"{'' if NOMESH else f' + {len(M.T)} mesh triangles'}", flush=True)
    # Opaque-block split (stats only): blocks of SPLIT render px whose texels all have alpha > 0.99 (render res),
    # drawn as opaque quads with depth write, front to back; their texels leave the transparent layers.
    split_masks = {}
    if MPI and SPLIT:
        cellpx = rects[0, 2] - rects[0, 0]
        for blk in SPLIT:
            nbk = cellpx // blk
            m_ = atlas[:, 3].reshape(P, nbk, blk, nbk, blk).float().amin((2, 4)) > 0.99
            split_masks[blk] = m_ & alive[:, None, None]
    for name, eye in eyes.items():
        R, _ = raster([full], eye, W, H, f)
        if NOMESH:
            Hy, ms = R, None
        else:
            ms = M.samples(eye, W, H, f, opaque=True)
            Hy, _ = raster([kept, ms], eye, W, H, f)
        mrgb, mdep = s2.mesh_layer(M, eye, W, H, f)
        if BG:                                  # background behind everything (the viewer's canvas is transparent)
            mrgb = torch.where(torch.isinf(mdep)[:, None], torch.full_like(mrgb, BG), mrgb)
        vals, pix, t, prim, lxs, lys = patch_samples(eye, with_local=True)
        k = t < mdep[pix]
        vals, pix, t, prim, lxs, lys = vals[k], pix[k], t[k], prim[k], lxs[k], lys[k]
        n_mesh = len(t)
        res_split = {}
        for blk, msk in split_masks.items():
            bx = (lxs.round().long() // blk).clamp(0, msk.shape[2] - 1)
            by = (lys.round().long() // blk).clamp(0, msk.shape[1] - 1)
            is_o = msk[prim, by, bx]
            dep_o = torch.full((W * H,), float("inf"), device=dev).scatter_reduce_(0, pix[is_o], t[is_o], "amin")
            front = dep_o[pix]
            keep_t = ~is_o & (t < front)                      # transparent fragments still drawn
            keep_o = is_o & (t <= front)                      # the nearest opaque block per pixel (front to back)
            n_t, n_o = int(keep_t.sum()), int(torch.isfinite(dep_o).sum())
            vv = vals.clone(); vv[is_o, :3] = vv[is_o, :3] / vv[is_o, 3:].clamp(min=1e-3); vv[is_o, 3] = 1.0
            ks = keep_t | keep_o
            img_s = composite_vals(vv[ks], pix[ks], t[ks], mrgb, W * H).reshape(H, W, 3)
            res_split[blk] = psnr(R, img_s)
            print(f"  {name:8s} split {blk}px: layer fragments/px {n_mesh / (W * H):.2f} -> transparent "
                  f"{n_t / (W * H):.2f} + opaque {n_o / (W * H):.2f} = {(n_t + n_o) / (W * H):.2f} "
                  f"({1 - (n_t + n_o) / max(n_mesh, 1):.0%} fewer); PSNR {res_split[blk]:.2f}", flush=True)
        if not SPLIT:
            print(f"  {name:8s} layer fragments/px: {n_mesh / (W * H):.2f}", flush=True)
        if OCC and len(occ_rects):                # depth pre-pass: drop fragments behind an occluder
            k = t <= occluder_depth(eye)[pix]
            vals, pix, t, prim = vals[k], pix[k], t[k], prim[k]
            print(f"  {name:8s} layer fragments/px: {n_mesh / (W * H):.2f} after mesh test -> "
                  f"{len(t) / (W * H):.2f} after occluders ({1 - len(t) / max(n_mesh, 1):.0%} fewer)", flush=True)
        ex = composite_vals(vals, pix, t, mrgb, W * H).reshape(H, W, 3)
        dist = (centre - torch.tensor(eye, device=dev)).norm(dim=1)
        rank = torch.empty(P, device=dev); rank[torch.argsort(dist)] = torch.arange(P, device=dev, dtype=torch.float32)
        so = composite_vals(vals, pix, rank[prim], mrgb, W * H).reshape(H, W, 3)
        res[name] = {"H": psnr(R, Hy), "P_exact": psnr(R, ex), "P_sorted": psnr(R, so)}
        print(f"  {name:8s} " + "  ".join(f"{kk} {v:5.2f}" for kk, v in res[name].items()), flush=True)
        if name in ("0", "x+0.2", "x+0.3") or (ZEYE and name.endswith(f"z+{ZEYE}")) or EVAL_EYES:
            OUT.mkdir(parents=True, exist_ok=True)
            row = torch.cat([R, Hy, so], 1).clamp(0, 1)
            Image.fromarray((row.cpu().numpy() * 255).astype(np.uint8)).resize((3 * GW, Hg)).save(
                OUT / f"{scene.stem}{SUFFIX}_t{TILE}_b{BINS}{BINMODE[0]}{"r" if REFINE else ""}{SNAP_TAG}_c{CULL}{"_occ" if OCC else ""}{"_nomesh" if NOMESH else ""}_e{EXTEND}_bg{BG}_{name}.jpg", quality=92)
        del vals, pix, t, prim, ms
        torch.cuda.empty_cache()
    (OUT / f"{scene.stem}{SUFFIX}_t{TILE}_b{BINS}{BINMODE[0]}{"r" if REFINE else ""}{SNAP_TAG}_c{CULL}{"_occ" if OCC else ""}{"_nomesh" if NOMESH else ""}.json").write_text(json.dumps(
        {"patches": P, "overdraw": float(area.sum() / (W * H)), "psnr": res}, indent=1))


def export_glb(rects, planes, atlas, off, area, fg, Hg, scene, centre, page=4096, border=None, cdisp=None,
               alive=None):
    """Patches as quads in one mesh per atlas page (premultiplied RGBA PNGs), back to front as seen from the
    input view (static order; fine for small head moves): pages hold consecutive runs of the far-to-near
    order and each page's triangles follow it, while texels are placed by height (shelf packing). Textures
    are box-downsampled by TEXDS and trimmed to the texels with alpha. With `border` (atlas of the same cells
    with a margin of gb render px of real neighbour texels, MPI), cells keep that margin instead of a
    duplicated-edge gutter and aren't trimmed, so filtering across cell seams sees the right colours."""
    ds = TEXDS
    def down(t):                                                     # box filter (premultiplied, so exact)
        if ds == 1:
            return t
        h, w = -(-t.shape[0] // ds), -(-t.shape[1] // ds)
        t = np.pad(t, ((0, h * ds - t.shape[0]), (0, w * ds - t.shape[1]), (0, 0)))
        return t.reshape(h, ds, w, ds, 4).mean((1, 3))

    texs, geo, pad, eo = {}, {}, {}, {}            # texs: padded texture block; pad: its margin (texels)
    if border is not None:
        groups, gb = border                                           # [(first cell index, (n, cb, cb, 4))]: cells
        assert gb % ds == 0 and all(t.shape[1] % ds == 0 for _, t in groups)   # of one size per group (bin)
        N = len(rects)
        live = np.ones(N, bool) if alive is None else alive.astype(bool)
        # Adaptive resolution: cells that lose the most detail at 1/ds resolution (mean |premultiplied
        # error| after down- and up-sampling, inner part) keep full resolution, worst first, while
        # their error is above TEXFULL_THR and the atlas stays within MAXPAGES pages.
        full = np.zeros(N, bool)
        if ds > 1 and TEXFULL_THR > 0:
            err = np.full(N, -1.0)
            half_px = np.zeros(N); extra = np.zeros(N)
            for st, atb in groups:
                cb = atb.shape[1]
                for c0 in range(0, len(atb), 8192):
                    t1 = atb[c0:c0 + 8192].to(dev).float()
                    t2 = torch.nn.functional.avg_pool2d(t1.permute(0, 3, 1, 2), ds)
                    up = torch.nn.functional.interpolate(t2, scale_factor=ds, mode="bilinear", align_corners=False)
                    d = (up - t1.permute(0, 3, 1, 2))[:, :3, gb:cb - gb, gb:cb - gb].abs().mean((1, 2, 3))
                    err[st + c0:st + c0 + len(t1)] = d.cpu().numpy()
                half_px[st:st + len(atb)] = (cb // ds) ** 2
                extra[st:st + len(atb)] = cb * cb - (cb // ds) ** 2
            err[~live] = -1                                          # culled cells: never upgraded, not budgeted
            budget = 0.75 * MAXPAGES * page * page - half_px[live].sum()
            cand = np.argsort(-err)
            cand = cand[err[cand] > TEXFULL_THR]
            cand = cand[np.cumsum(extra[cand]) <= budget]
            full[cand] = True
            print(f"  adaptive textures: {full.sum()} of {N} cells at full resolution "
                  f"(error > {TEXFULL_THR}, budget {MAXPAGES} pages)", flush=True)
        # Batched (GPU, chunked) per cell size: every cell's texture at its chosen resolution (full, or
        # box-downsampled by ds; premultiplied, so averaging is exact), and the bounding box of its texels with
        # alpha inside the cell. Textures stay 8-bit from here on.
        g2 = gb // ds

        def first_last(m):                                            # m: (n, L) bool -> first, last index (or -1)
            anyv = m.any(1)
            L = m.shape[1]
            idx = torch.arange(L, device=m.device)
            first = torch.where(m, idx, L).amin(1)
            last = torch.where(m, idx, -1).amax(1)
            return torch.where(anyv, first, -1), torch.where(anyv, last, -1)
        for st, atb in groups:
            cb = atb.shape[1]
            n_g = len(atb)
            half_u8 = np.zeros((n_g, cb // ds, cb // ds, 4), np.uint8)
            bbox = np.full((n_g, 4), -1, np.int64)                    # first row, last row, first col, last col
            full_t = torch.tensor(full[st:st + n_g], device=dev)
            for c0 in range(0, n_g, 16384):
                t1 = atb[c0:c0 + 16384].to(dev).float().clamp(0, 1)              # (n, cb, cb, 4)
                th = torch.nn.functional.avg_pool2d(t1.permute(0, 3, 1, 2), ds).permute(0, 2, 3, 1) if ds > 1 else t1
                half_u8[c0:c0 + len(t1)] = (th * 255 + 0.5).byte().cpu().numpy()
                inner_h = th[:, g2:th.shape[1] - g2, g2:th.shape[2] - g2, 3] >= 0.5 / 255
                inner_f = t1[:, gb:cb - gb, gb:cb - gb, 3] >= 0.5 / 255
                fsel = full_t[c0:c0 + len(t1)]
                for inner, sel in ((inner_h, ~fsel), (inner_f, fsel)):
                    if not sel.any():
                        continue
                    r0, r1 = first_last(inner[sel].any(2))
                    k0, k1 = first_last(inner[sel].any(1))
                    ids = torch.nonzero(sel)[:, 0].cpu().numpy() + c0
                    bbox[ids] = torch.stack([r0, r1, k0, k1], 1).cpu().numpy()
            full_j = np.nonzero(full[st:st + n_g] & live[st:st + n_g])[0]
            full_u8 = {}
            for c0 in range(0, len(full_j), 16384):                   # full-resolution cells only, batched
                ids = full_j[c0:c0 + 16384]
                full_u8.update(zip(ids, (atb[torch.tensor(ids, device=atb.device)].float().clamp(0, 1) * 255 + 0.5)
                                   .byte().cpu().numpy()))
            for j in np.nonzero(live[st:st + n_g] & (bbox[:, 0] >= 0))[0]:
                i = st + j
                x0, y0 = rects[i, 0], rects[i, 1]
                r0, r1, k0, k1 = bbox[j]
                tb, g, sp = (full_u8[j], gb, 1) if full[i] else (half_u8[j], g2, ds)
                texs[i] = tb[r0:r1 + 1 + 2 * g, k0:k1 + 1 + 2 * g]
                pad[i] = g
                geo[i] = (x0 + k0 * sp, y0 + r0 * sp, x0 + (k1 + 1) * sp, y0 + (r1 + 1) * sp)
                # OVERLAP: the quad reaches this many render px into its border texels (its neighbours' own
                # content), covering hairline cracks between pieces whose corners differ slightly.
                eo[i] = min(g, int(OVERLAP // sp))
    else:
        at = atlas.float().clamp(0, 1).cpu().numpy()
        for i in range(len(rects)):
            if alive is not None and not alive[i]:
                continue
            x0, y0, x1, y1 = rects[i]
            t = down(at[off[i]:off[i] + area[i]].reshape(y1 - y0, x1 - x0, 4))
            m = t[..., 3] >= 0.5 / 255
            if not m.any():
                continue
            ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
            texs[i] = (np.pad(t[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1], ((1, 1), (1, 1), (0, 0)), mode="edge")
                       * 255 + 0.5).astype(np.uint8)
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
    meshes, texels = [], 0
    for pg, (ids, pl) in enumerate(pages):
        img = np.zeros((page, page, 4), np.uint8)
        for i in ids:
            px, py = pl[i]
            t, g = texs[i], pad[i]
            img[py - g:py - g + t.shape[0], px - g:px - g + t.shape[1]] = t
            texels += t.shape[0] * t.shape[1]
        e_ = np.array([eo.get(i, 0) for i in ids], np.float64)
        sp_ = np.array([pad[i] and (geo[i][2] - geo[i][0]) / max(1, texs[i].shape[1] - 2 * pad[i]) for i in ids])
        r = np.array([geo[i] for i in ids], np.float64) + (e_ * sp_)[:, None] * [-1, -1, 1, 1]
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
        px = np.array([pl[i][0] for i in ids]) - e_; py = np.array([pl[i][1] for i in ids]) - e_
        tw = np.array([texs[i].shape[1] - 2 * pad[i] for i in ids]) + 2 * e_; th = np.array([texs[i].shape[0] - 2 * pad[i] for i in ids]) + 2 * e_
        uu = np.stack([px, px + tw, px + tw, px], 1) / page
        vv = np.stack([py, py, py + th, py + th], 1) / page
        faces = (np.arange(len(ids))[:, None, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]])[None]).reshape(-1, 3)
        # glTF uv: origin at the top-left of the image (no flip)
        uv = np.stack([uu.reshape(-1), vv.reshape(-1)], 1).astype(np.float32)
        # The viewer draws surface_<k> with larger k first (k = 0 is the front).
        meshes.append((f"surface_{len(pages) - 1 - pg}", V.reshape(-1, 3).astype(np.float32),
                       faces.astype(np.uint32), uv, img))
    path = (Path(EXPORT_DIR) if EXPORT_DIR else scene.parent) / f"{scene.stem}{EXPORT}.glb"
    write_layer_glbs(path, meshes, ktx2=EXPORT_KTX2)
    quad_px = sum((geo[i][2] - geo[i][0]) * (geo[i][3] - geo[i][1]) for i in texs)
    print(f"  quads cover {quad_px / (GW * SS * Hg * SS):.1f}x the frame (fill before blending/depth test)", flush=True)
    print(f"  [{time.time() - T0:.0f}s] wrote {path} ({path.stat().st_size / 2**20:.0f} MB, {len(pages)} atlas pages of {page}^2, "
          f"{len(texs)} patches, {texels / 1e6:.0f}M texels = {texels / (len(pages) * page * page):.0%} of the pages)",
          flush=True)


def write_layer_glbs(path, meshes, ktx2=False):
    """Write the layer meshes [(name, V, faces, uv, page RGBA uint8)] as <path> (PNG pages, fast compression) and,
    with ktx2, also <stem>_ktx2.glb (Basis UASTC KTX2 pages with mipmaps, KHR_texture_basisu; see glb_ktx2.py),
    encoding the pages in parallel."""
    import io, tempfile
    from concurrent.futures import ThreadPoolExecutor
    from glb_ktx2 import write_glb, encode_pages

    def png(img, level):
        b = io.BytesIO(); Image.fromarray(img, "RGBA").save(b, format="PNG", compress_level=level); return b.getvalue()
    with ThreadPoolExecutor(len(meshes) or 1) as ex:
        pngs = list(ex.map(lambda m: png(m[4], 1), meshes))

    def build(images, mime, basisu):
        g = {"asset": {"version": "2.0", "generator": "ml-sharp eval_alpha_patches"}, "scene": 0,
             "scenes": [{"nodes": list(range(len(meshes)))}], "nodes": [], "meshes": [], "materials": [],
             "textures": [], "images": [], "accessors": [], "bufferViews": []}
        views = []

        def view(data, target=None):
            g["bufferViews"].append({"buffer": 0, **({"target": target} if target else {})})
            views.append(data)
            return len(views) - 1

        def accessor(arr, ctype, typ, target, minmax=False):
            a = {"bufferView": view(arr.tobytes(), target), "componentType": ctype, "count": len(arr), "type": typ}
            if minmax:
                a["min"] = arr.min(0).tolist(); a["max"] = arr.max(0).tolist()
            g["accessors"].append(a)
            return len(g["accessors"]) - 1
        for k, ((name, V, F, uv, _), im) in enumerate(zip(meshes, images)):
            pos = accessor(V, 5126, "VEC3", 34962, minmax=True)
            tex = accessor(uv, 5126, "VEC2", 34962)
            idx = accessor(F.reshape(-1), 5125, "SCALAR", 34963)
            g["images"].append({"bufferView": view(im), "mimeType": mime})
            g["textures"].append({"extensions": {"KHR_texture_basisu": {"source": k}}} if basisu else {"source": k})
            g["materials"].append({"name": f"patches_{k}", "doubleSided": True, "alphaMode": "BLEND",
                                   "pbrMetallicRoughness": {"baseColorTexture": {"index": k},
                                                            "metallicFactor": 0.0, "roughnessFactor": 1.0}})
            g["meshes"].append({"name": name, "primitives": [{"attributes": {"POSITION": pos, "TEXCOORD_0": tex},
                                                               "indices": idx, "material": k}]})
            g["nodes"].append({"name": name, "mesh": k})
        if basisu:
            g["extensionsUsed"] = g["extensionsRequired"] = ["KHR_texture_basisu"]
        return g, views
    g, views = build(pngs, "image/png", False)
    write_glb(path, g, views)
    if ktx2:
        with tempfile.TemporaryDirectory() as td:
            raw = [png(m[4], 0) for m in meshes]                      # uncompressed PNG: just a container for ktx
            kt = encode_pages(raw, td, levels=KTX2_LEVELS)
        g, views = build(kt, "image/ktx2", True)
        write_glb(Path(path).with_name(Path(path).stem + "_ktx2.glb"), g, views)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        run(Path(s))
