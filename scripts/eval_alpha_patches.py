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
from PIL import Image
from plyfile import PlyData


def _lazy_module(name):
    """Import `name` on first use (trimesh: ~2 s, only needed with a hybrid mesh); later `import name` elsewhere
    gets the same lazy module."""
    import importlib.util
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.find_spec(name)
    spec.loader = importlib.util.LazyLoader(spec.loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


trimesh = _lazy_module("trimesh")
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
KTX2_ZSTD = int(os.environ.get("KTX2_ZSTD", 18))         # Zstandard level of the KTX2 pages (9: ~7% larger, faster)
# EXPORT_KTX2 writes <stem>_ktx2.glb (Basis UASTC: any GPU, the browser transcodes) if KTX2_UASTC and <stem>_astc.glb
# (plain ASTC 4x4: Quest, Apple, phones; ~6x faster to encode, more accurate) if KTX2_ASTC (its encoder preset).
KTX2_UASTC = bool(int(os.environ.get("KTX2_UASTC", 1)))
KTX2_ASTC = os.environ.get("KTX2_ASTC", "fast")
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
ADAPT = int(os.environ.get("ADAPT", 0))                       # MPI: finer bins (x ADAPT) where stacking shows
ADAPT_T = float(os.environ.get("ADAPT_T", 25))
ADAPT_N = int(os.environ.get("ADAPT_N", 8))
ADAPT_D = int(os.environ.get("ADAPT_D", 0))                   # strip-edge px per cell flagging it alone (0 = off)
ADAPT_W = int(os.environ.get("ADAPT_W", 2))                   # contrast window radius (render px)
# LDI=1 (MPI, not SURF): layered depth image. A front layer (per pixel, the slices composited near to far until
# LDI_OPAQUE opaque) on one continuous grid mesh (a vertex every LDI_STEP render px on the median-depth map),
# cut only where its depth jumps by more than LDI_CUT parallax px at MOVE; MPI-only slices behind it for
# everything else (pixels never that opaque: wisps, soft edges; shells thicker than LDI_THICK px: fur; cut
# squares at occlusion edges; everything behind the front surface). One layer per visible surface: no stacking.
LDI = bool(os.environ.get("LDI"))
LDI_STEP = int(os.environ.get("LDI_STEP", 8))
LDI_OPAQUE = float(os.environ.get("LDI_OPAQUE", 0.95))
LDI_CUT = float(os.environ.get("LDI_CUT", 3))
LDI_THICK = float(os.environ.get("LDI_THICK", 4))
LDI_BAND = int(os.environ.get("LDI_BAND", 16))
LDI_GROUPS = int(os.environ.get("LDI_GROUPS", 24))
LDI_SPLIT = bool(int(os.environ.get("LDI_SPLIT", 0)))   # split jump squares into near/far parts (mosaic edges)
LDI_FRONT = []                                    # set by run: [(V, faces, uv, RGBA uint8 texture)] for export_glb
# ADAPT_SHARE=1 (with ADAPT): NeX-style shared textures. A refined cell's ADAPT sub-bins (between two normal bins)
# share one colour texture (their composite) and keep only their own alpha, packed as the channels of a second
# tile next to it: one double-width tile instead of ADAPT. Each sub-bin is still its own quad at its own depth;
# the viewer reads colour (un-premultiplied) from the first half and its alpha from its channel of the second
# (glTF TEXCOORD_1: the alpha tile's uv, TEXCOORD_2.x: the channel, -1 for ordinary quads).
ADAPT_SHARE = bool(os.environ.get("ADAPT_SHARE"))
# SHARE_COLOR=C (with ADAPT_SHARE): every cell shares colour across C colour planes (each spanning BINS/C normal or
# BINS*ADAPT/C refined bins, which keep only their alpha, 4 per tile); 0 = only refined cells, per normal bin.
SHARE_COLOR = int(os.environ.get("SHARE_COLOR", 0))
# LCOL=1 (MPI): colour by visibility layers. The normal bins are peeled near to far per pixel into up to 3 colour
# layers (a layer closes once LCOL_OPQ opaque; the bin where it closed is the split), stored once on a colour page;
# every plane (normal or ADAPT-refined) keeps only its alpha (4 planes per tile, per cell). A plane takes the colour
# of the layer its bin falls in at that pixel: one colour texture per visible surface instead of per plane.
LCOL = bool(os.environ.get("LCOL"))
LCOL_OPQ = float(os.environ.get("LCOL_OPQ", 0.95))
LCOL_GAP = int(os.environ.get("LCOL_GAP", 2))
LCOL_EPS = float(os.environ.get("LCOL_EPS", 0.02))
LCOL_DC = float(os.environ.get("LCOL_DC", 0.1))
LCOL_SPAN = int(os.environ.get("LCOL_SPAN", 3))
LCOL_GAPA = float(os.environ.get("LCOL_GAPA", 0.2))
LCOL_DATA = {}                                    # set by run: page (RGBA uint8), w, h, nbin per cell
# LCT=1 (MPI): colour by visibility layers in per-cell tiles, any number of layers. The bins are peeled into layers as
# for LCOL (same split rules; up to LCT_MAX layers per pixel), but nothing is stored for the whole frame: each cell's
# bin plane is split into one piece per layer present in the cell (its alpha where the bin is in that layer), and the
# pieces of one (cell, layer) share one colour tile (the layer's composite) with their alphas packed 4 per tile (the
# ADAPT_SHARE format: no new viewer shader). All quads are drawn in one far-to-near order (LDI_GROUPS).
LCT = bool(os.environ.get("LCT"))
LCT_MAX = int(os.environ.get("LCT_MAX", 8))
# LCW=1 (MPI): colour by visibility layers without cutting planes (LCT's cut pieces leave seams: two pieces' filtered
# alpha edges cover less than the plane). Bins are peeled into layers as for LCT. Every plane keeps its whole alpha
# plus a per-texel weight between its cell's two main layers for it (alpha, weight: 2 planes per RGBA tile); each
# cell has one colour tile per layer it uses: the composite of its exported planes' texels in that layer
# (un-premultiplied, spread). The viewer mixes the two layer colours by the weight. Same quads (fill) as plain MPI.
LCW = bool(os.environ.get("LCW"))
LCW_DATA = {}                                     # set by run: lab (P, 2) layers per plane, wt [(first, (n, cb, cb))]
# GAPCOPY=1 (MPI, not ADAPT): fill the gaps that open between the strips of a surface crossing planes (stacking).
# Where a cell's visible plane-k+1 pixels touch visible plane-k pixels (>= GAPCOPY_MIN px; visible: alpha x
# transmittance in front > GAPCOPY_VIS, so hidden fill doesn't count), the plane-k+1 tile is drawn a second time on
# plane k's sheet, right after plane k: from a moved eye it stays flush with the plane-k strip and shows exactly in
# the gap that opens beside it (the k+1 strip, drawn later and nearer, covers it elsewhere). A depth smear of the
# surface across the gap with one texture lookup; silhouettes (no adjacent plane) get no copy.
# PARALLAX=1 (with SEPA, REFINE): parallax-corrected planes. Each texel's disparity offset from its cell's sheet (the
# splats' own disparity there), in plane thicknesses [-1, 1], goes in the colour tile's alpha channel (unused by
# SEPA), the plane thickness (disparity) in TEXCOORD_2.y; the viewer shifts the texture lookup by
# offset x (eye_xy - x eye_z) / (1 - d eye_z) (photo-camera coordinates), so each texel appears at about its own
# depth instead of its plane's (no shear of features cut by plane boundaries). Same quads, one more lookup.
PARALLAX = bool(os.environ.get("PARALLAX"))
PAR_DATA = {}                                     # set by run: off [(first, (n, cb, cb))], dk (P,)
GAPCOPY = bool(os.environ.get("GAPCOPY"))
GAPCOPY_VIS = float(os.environ.get("GAPCOPY_VIS", 0.3))
GAPCOPY_MIN = int(os.environ.get("GAPCOPY_MIN", 4))
GAPCOPY_DATA = {}                                 # set by run: conn (P,) the plane on whose sheet a copy goes, or -1
LCOL = LCOL or LCT or LCW
LCT_LAYER = {}                                    # set by run: normal bin -> per-pixel layer index (uint8)
# UNPREM=1 (MPI export): plain cell textures hold un-premultiplied colour, spread into transparent texels, instead of
# premultiplied colour (meshes named surface_<k>_u; the viewer multiplies by alpha). Block compression (KTX2 UASTC)
# then sees smooth colour instead of colour that falls with each plane's alpha, and its errors no longer scale with
# the alpha of the ~10 planes stacked at a pixel.
UNPREM = bool(os.environ.get("UNPREM"))
# SEPA=1 (MPI export): colour and alpha in separate tiles. Each cell's planes get one colour tile each (un-premultiplied,
# spread, alpha 255: UASTC's colour-only modes, ~3x less error than for the semi-transparent RGBA texels of stacked
# planes) and their alphas packed 4 per RGBA tile (the ADAPT_SHARE format: colour tile x alpha channel; no new viewer
# shader). ~1.25x the texels. Tiles are grouped by runs of SEPA_N depth-adjacent planes of a cell (on one atlas page);
# every quad is drawn at its own plane's place in its page's far-to-near order.
SEPA = bool(os.environ.get("SEPA"))
SEPA_N = int(os.environ.get("SEPA_N", 4))
SEPA_BAND = float(os.environ.get("SEPA_BAND", 0.92))  # depth band size, in atlas pages (see export_glb)
CELL_META = []                                    # set by run: [(n, 3) level, parent bin, sub-bin] per cell
# SURF_REGION=<file.npz> (with SURF): surface pieces only inside a region of cells (npz "region", (ny, nx) bool, the
# bake's cell grid incl. its margin), MPI-only cells outside it and along its border (npz "interior": cells without
# MPI); chosen by scripts' visual test (where surface pieces don't break from moved views).
SURF_REGION = None
if os.environ.get("SURF_REGION"):
    _r = np.load(os.environ["SURF_REGION"])
    SURF_REGION = (_r["region"], _r["interior"])
BEND = float(os.environ.get("BEND", 0))                       # corner window (render px); 0 = off
BEND_TOL = float(os.environ.get("BEND_TOL", 3))
BEND_GAP = float(os.environ.get("BEND_GAP", 2))
BEND_VIS = float(os.environ.get("BEND_VIS", 0.2))   # min visible alpha around a corner for a bin to bend there
SURF_GAP = int(os.environ.get("SURF_GAP", 1))
SURF_SPAN = int(os.environ.get("SURF_SPAN", 8))
SURF_JOIN = float(os.environ.get("SURF_JOIN", 3.0))
SURF_VIS = float(os.environ.get("SURF_VIS", 0))
SURF_ALIGN = os.environ.get("SURF_ALIGN", "1") != "0"
SURF_LINK = float(os.environ.get("SURF_LINK", 0.5))   # > 1: never link
OVERLAP = float(os.environ.get("OVERLAP", 2 if os.environ.get("SURF") else 0))   # see export_glb
# SURF_VISGLUE=1: each grid corner chooses how its pieces meet (torn, or chained within each of SURF_VG_JOINS bin
# widths) by looking: the pieces are rendered from check views around the input view and compared with the splats
# there, counting only the pixels the corner's cells draw; the option closest to the splats wins. Depth thresholds
# can't tell a steep flat surface (should connect) from a depth jump (should tear); other views can.
SURF_VISGLUE = bool(os.environ.get("SURF_VISGLUE"))
SURF_SPLIT = float(os.environ.get("SURF_SPLIT", 0))           # parallax px; 0 = off
SURF_MAXMOVE = float(os.environ.get("SURF_MAXMOVE", 0))       # bin widths a shared corner may move; 0 = any
SURF_SKIRT = float(os.environ.get("SURF_SKIRT", 0))           # parallax px; 0 = no skirts (export only)
SURF_HYBRID = float(os.environ.get("SURF_HYBRID", 0))         # parallax px; 0 = off (see SURF_HYBRID)
SURF_DMAP = float(os.environ.get("SURF_DMAP", 0))             # corner window (render px); 0 = off
SURF_DMAP_GAP = float(os.environ.get("SURF_DMAP_GAP", 2))      # parallax px between surfaces at a corner
SURF_SPLIT_GAIN = float(os.environ.get("SURF_SPLIT_GAIN", 0.5))
SURF_JOIN_VIS = float(os.environ.get("SURF_JOIN_VIS", 2))   # >= 1: no joins (see SURF_SPLIT)
SURF_VG_JOINS = [float(v) for v in os.environ.get("SURF_VG_JOINS", "0,3,8").split(",")]
SURF_VG_INIT = int(os.environ.get("SURF_VG_INIT", 1))       # starting option (index into SURF_VG_JOINS)
SURF_VG_PASSES = int(os.environ.get("SURF_VG_PASSES", 2))
SURF_VG_MARGIN = float(os.environ.get("SURF_VG_MARGIN", 0))   # an option must beat the starting one by this fraction
SURF_VG_MOVE = float(os.environ.get("SURF_VG_MOVE", 0.2))   # check views: this far left/right/up/down/closer (m)
SURF_OPTS = []                                              # set by surface_pieces: [corner keys, [cd per option]]
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
            + (f"B{BEND:g}-{BEND_TOL:g}" if BEND and not SURF else "")
            + ("G" if GAPCOPY and not SURF else "") + ("X" if PARALLAX else "") + (f"LW{LCT_MAX}" if LCW and not SURF else f"LT{LCT_MAX}" if LCT and not SURF else "LC" if LCOL and not SURF else "")
            + (f"A{ADAPT}t{ADAPT_T:g}n{ADAPT_N}w{ADAPT_W}d{ADAPT_D}" + ("s" if ADAPT_SHARE else "") + (f"c{SHARE_COLOR}" if SHARE_COLOR else "") if ADAPT and not SURF else "")
            + (f"L{LDI_STEP}c{LDI_CUT:g}t{LDI_THICK:g}" if LDI and not SURF else "")
            + ("R" + os.path.basename(os.environ["SURF_REGION"]).split(".")[0] if SURF_REGION is not None else "")
            + (f"S{SURF_SPAN}g{SURF_GAP}j{SURF_JOIN:g}" + (f"v{SURF_VIS:g}" if SURF_VIS else "") + ("a" if SURF_ALIGN else "") + f"f{SURF_FILL}" + (f"L{SURF_LINK:g}" if SURF_LINK_PLANE else "") + (f"o{OVERLAP:g}" if OVERLAP else "") + ("c" if SURF_GROUP == "chain" else "") + (f"x{SURF_SPLIT:g}-{SURF_SPLIT_GAIN:g}" if SURF_SPLIT else "") + (f"M{SURF_MAXMOVE:g}" if SURF_MAXMOVE else "") + (f"J{SURF_JOIN_VIS:g}" if SURF_JOIN_VIS < 1 else "") + (f"D{SURF_DMAP:g}-{SURF_DMAP_GAP:g}" if SURF_DMAP else "") + (f"H{SURF_HYBRID:g}" if SURF_HYBRID else "") + ("V" + "-".join(f"{j:g}" for j in SURF_VG_JOINS) + (f"m{SURF_VG_MARGIN:g}" if SURF_VG_MARGIN else "") if SURF_VISGLUE else "") if SURF else ""))
EVAL_EYES = [e for e in os.environ.get("EVAL_EYES", "").split(",") if e]   # eval: only these eyes (default all)
ZEYE = float(os.environ.get("ZEYE", 0))           # eval: extra eyes this far forward (and forward + sideways)
CULL = float(os.environ.get("CULL", 0))
CULL_EYES = int(os.environ.get("CULL_EYES", 18))   # culling viewpoints: 18 (two depths) or 9 (one)
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
BAKE_TIMING = int(os.environ.get("BAKE_TIMING", 0))  # print each stage's wall time (tick); 2: also sections (sec)
_tick = [time.time()]


def cell_tiles(t, cy, cx, cell, cb):
    """Bordered tiles of cells (cy, cx) from t (C, H + 2 border, W + 2 border) (padded by the border): (n, cb, cb, C),
    as unfold(t, cb, stride=cell)[:, cy * nx + cx] but gathering only these cells (not the whole frame)."""
    ar = torch.arange(cb, device=t.device)
    r = (cy * cell)[:, None] + ar; c = (cx * cell)[:, None] + ar
    return t[:, r[:, :, None], c[:, None, :]].permute(1, 2, 3, 0)


def to_np(t):
    """GPU tensor -> numpy through pinned host memory: copies into fresh pageable memory crawl here (~190 MB/s on the
    GB10, ~15x slower), and the export copies hundreds of MB of tiles."""
    if t.device.type == "cpu":
        return t.numpy()
    b = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    b.copy_(t)
    return b.numpy()


_sec = {}                                          # BAKE_TIMING: seconds per section of the bin loop


def sec(name):
    """BAKE_TIMING=2: add the time since the last sec() mark to section `name` (syncs the GPU: slows the loop)."""
    if BAKE_TIMING >= 2:
        torch.cuda.synchronize()
        now = time.time()
        _sec[name] = _sec.get(name, 0) + now - _sec.get("_t", now)
        _sec["_t"] = now


def tick(label):
    """BAKE_TIMING: wall time since the last tick (GPU work finished first)."""
    if BAKE_TIMING:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.time()
        print(f"  [time] {label}: {now - _tick[0]:.1f}s", flush=True)
        _tick[0] = now


_PLY_T = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
          "ushort": "u2", "short": "i2", "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4"}


def read_ply_vertex(path):
    """The vertex element of a binary little-endian PLY as a numpy structured array (as plyfile's ["vertex"].data),
    read in one go (plyfile reads element by element: ~10 s for a 66 MB file on the NAS)."""
    b = Path(path).read_bytes()
    end = b.index(b"end_header") + len(b"end_header")
    end += 2 if b[end:end + 2] == b"\r\n" else 1
    lines = b[:end].decode("ascii").splitlines()
    assert "binary_little_endian" in lines[1], f"{path}: only binary little-endian PLY"
    els = []                                          # (name, count, [(prop, dtype)])
    for l in lines:
        t = l.split()
        if t and t[0] == "element":
            els.append((t[1], int(t[2]), []))
        elif t and t[0] == "property":
            assert t[1] != "list", f"{path}: list properties not supported"
            els[-1][2].append((t[2], "<" + _PLY_T[t[1]]))
    off = end
    for name, n, props in els:
        dt = np.dtype(props)
        if name == "vertex":
            return np.frombuffer(b, dt, count=n, offset=off)
        off += n * dt.itemsize
    raise ValueError(f"{path}: no vertex element")


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


def surface_pieces(surf_l, crops, crops_b, rect_l, cen_l, cd_l, cell, nx, lo, hi, edges, fpx):
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
    def run_pass(force, nocut=None, fbe=None):
        """One walk over every cell's bins (see above); force: entries that must start a new piece."""
        nonlocal_state = None
        A = torch.zeros(nC, cell, cell, 4, device=dev); T = torch.ones(nC, cell, cell, 1, device=dev)
        if texb is not None:
            cbs = texb.shape[1:3]
            Ab = torch.zeros((nC,) + tuple(texb.shape[1:]), device=dev); Tb = torch.ones((nC,) + tuple(cbs) + (1,), device=dev)
        ST = torch.zeros(nC, 10, device=dev, dtype=torch.float64)
        E_st = np.zeros((N, 10))                         # each entry's plane statistics (visible weights)
        if force is None and (SURF_SPLIT or SURF_HYBRID):   # each entry's visible weight and weight x disparity per px
            E_w = torch.zeros(N, cell * cell, device=dev, dtype=torch.float16)
            E_wp = torch.zeros(N, cell * cell, device=dev, dtype=torch.float16)
        has = np.zeros(nC, bool); kn = np.zeros(nC, np.int64); kf = np.zeros(nC, np.int64)
        lk = np.zeros(nC, bool)                               # current piece continues the one before (SURF_LINK)
        fsp = np.zeros(nC, bool)                              # current piece starts at a SURF_SPLIT cut
        fdp = np.zeros(nC, bool)                              # current piece starts at a span (SURF_SPAN) cut only
        fbp = np.zeros(nC, bool)                              # current piece is an MPI fallback slice (SURF_HYBRID)
        E_a = np.zeros(N)                                     # each entry's summed alpha
        if force is None:                                     # median-depth map (SURF_DMAP): per pixel, the disparity
            Tall = torch.ones(nC, cell, cell, device=dev)     # of the bin where the accumulated alpha crosses 0.5
            Dmed = torch.full((nC, cell, cell), float('nan'), device=dev)
        P_tex, P_texb, P_st, P_kn, P_kf, P_c, P_lk, P_fs, P_dp, P_fb = [], [], [], [], [], [], [], [], [], []

        def emit(ci):                                         # close the current pieces of cells ci
            if len(ci) == 0:
                return
            ct = torch.tensor(ci, device=dev)
            P_tex.append(A[ct].half()); P_st.append(ST[ct].cpu().numpy())
            if texb is not None:
                P_texb.append(Ab[ct].half().cpu())
            P_kn.append(kn[ci].copy()); P_kf.append(kf[ci].copy()); P_c.append(ucid[ci]); P_lk.append(lk[ci].copy()); P_fs.append(fsp[ci].copy()); P_dp.append(fdp[ci].copy()); P_fb.append(fbp[ci].copy())
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
            if nocut is not None:
                deep = deep & ~nocut[e]
            gap_ = (vis < SURF_VIS) | (kf[ci] - k > SURF_GAP + 1)
            split = has[ci] & (gap_ | deep | (force[e] if force is not None else False))
            fresh = ~has[ci]
            emit(ci[split])
            lk[ci[fresh]] = False
            lk[ci[split]] = (~gap_ & (vis >= SURF_LINK))[split]
            fsp[ci[fresh]] = False
            fsp[ci[split]] = (~gap_ & ~deep)[split]
            fdp[ci[fresh]] = False
            fdp[ci[split]] = (~gap_ & deep & ~(force[e] if force is not None else False))[split]
            if fbe is not None:
                fbp[ci[fresh | split]] = fbe[e[fresh | split]]
            new_ = ~has[ci]
            kn[ci[new_]] = k[new_]
            wv = (T[ct, ..., 0] * a).double()
            pv = pdisp[et].double()
            Sc = lambda f_: (wv * f_).sum((1, 2))
            row = torch.stack([Sc(1.0), Sc(Xc), Sc(Yc), Sc(Xc * Xc), Sc(Yc * Yc), Sc(Xc * Yc),
                               Sc(pv), Sc(pv * Xc), Sc(pv * Yc), Sc(pv * pv)], 1)
            ST[ct] += row
            E_st[e] = row.cpu().numpy()
            if force is None and (SURF_SPLIT or SURF_HYBRID):
                E_w[et] = wv.reshape(len(e), -1).half()
                E_wp[et] = (wv * pv).reshape(len(e), -1).float().half()
                E_a[e] = a.sum((1, 2)).cpu().numpy()
            A[ct] += T[ct] * c
            T[ct] *= 1 - c[..., 3:]
            if force is None:
                ta = Tall[ct]
                cross = (ta > 0.5) & (ta * (1 - a) <= 0.5)
                Dmed[ct] = torch.where(cross, pdisp[et].float(), Dmed[ct])
                Tall[ct] = ta * (1 - a)
            if texb is not None:
                cb_ = texb[torch.tensor(e)].to(dev).float()
                Ab[ct] += Tb[ct] * cb_
                Tb[ct] *= 1 - cb_[..., 3:]
            kf[ci] = k; has[ci] = True
        emit(np.nonzero(has)[0])
        return P_tex, P_texb, P_st, P_kn, P_kf, P_c, P_lk, (E_w, E_wp, E_a) if force is None and (SURF_SPLIT or SURF_HYBRID) else None, P_fs, P_dp, (Dmed if force is None else None), P_fb

    def plane_sse(S):
        """Weighted squared residual of the least-squares plane of disparity (stats rows as in run_pass)."""
        w, sx, sy, sxx, syy, sxy, sp, spx, spy, spp = np.moveaxis(S, -1, 0)
        lam = PLANE_REG * w
        M0 = np.stack([np.stack([w, sx, sy], -1), np.stack([sx, sxx, sxy], -1), np.stack([sy, sxy, syy], -1)], -2)
        Mr = M0 + np.stack([np.zeros_like(w), lam, lam], -1)[..., None] * np.eye(3) + 1e-9 * np.eye(3)
        b = np.stack([sp, spx, spy], -1)
        th = np.linalg.solve(Mr, b[..., None])[..., 0]
        return np.maximum(spp - 2 * (th * b).sum(-1) + np.einsum("...i,...ij,...j->...", th, M0, th), 0)

    passes = passes1 = run_pass(None)
    Dmed = passes[10]
    if SURF_SPLIT or SURF_HYBRID:
        px = fpx * MOVE                                   # parallax px per unit disparity
        xs = ((torch.arange(cell, device=dev, dtype=torch.float64) + 0.5) / cell - 0.5)
        X_, Y_ = xs[None, :].expand(cell, cell).reshape(-1), xs[:, None].expand(cell, cell).reshape(-1)
        basis = torch.stack([torch.ones_like(X_), X_, Y_, X_ * X_, Y_ * Y_, X_ * Y_], 1)   # (px, 6)

        def stats(Wp, WPp):                               # per-pixel weights and weight x disparity -> rows
            Pm = WPp / Wp.clamp(min=1e-9)
            return torch.cat([Wp @ basis, (WPp[..., None] * basis[:, :3]).sum(-2), (WPp * Pm).sum(-1, keepdim=True)], -1)
    if SURF_SPLIT:
        # SURF_SPLIT: a run whose pixels one plane fits badly (RMS above SURF_SPLIT parallax px for a MOVE head
        # move) is cut where two planes, one per depth range, fit them best, if that removes at least
        # SURF_SPLIT_GAIN of the error. Per pixel: each pixel's visible-weighted mean disparity within each part,
        # so a thick fuzzy shell (samples spread in depth within every pixel, one surface across pixels) stays
        # whole, while an arm and the shirt beside it (two surfaces on different pixels) split, each flat on its
        # own surface. Decided on the first walk's visible weights, then the cells are walked again with the cuts.
        E_w, E_wp, E_a = passes[7]
        P_c0, P_kn0, P_kf0 = np.concatenate(passes[5]), np.concatenate(passes[3]), np.concatenate(passes[4])
        force = np.zeros(N, bool)
        nsplit = 0
        for c_, k0, k1 in zip(P_c0, P_kn0, P_kf0):        # each run: its cell's entries in bins k0..k1
            ci_ = np.searchsorted(ucid, c_)
            ent_ = np.arange(cstart[ci_], cstart[ci_] + ccount[ci_])
            ent_ = ent_[(ks[ent_] <= k0) & (ks[ent_] >= k1)]
            if len(ent_) < 2:
                continue
            et_ = torch.tensor(ent_, device=dev)
            cw, cwp = E_w[et_].double().cumsum(0), E_wp[et_].double().cumsum(0)
            tot = stats(cw[-1], cwp[-1]).cpu().numpy()
            if tot[0] <= 0:
                continue
            one = plane_sse(tot) / tot[0]
            if np.sqrt(one) * px < SURF_SPLIT:
                continue
            A_ = stats(cw[:-1], cwp[:-1]).cpu().numpy()
            B_ = stats(cw[-1] - cw[:-1], cwp[-1] - cwp[:-1]).cpu().numpy()
            two = (plane_sse(A_) + plane_sse(B_)) / tot[0]
            ok = (A_[:, 0] > 0.1 * tot[0]) & (B_[:, 0] > 0.1 * tot[0])
            if not ok.any():
                continue
            two = np.where(ok, two, np.inf)
            j = int(np.argmin(two))
            if two[j] < (1 - SURF_SPLIT_GAIN) * one:
                force[ent_[j + 1]] = True; nsplit += 1
        # Joins (the reverse): a run cut from the one before it only by the span rule (SURF_SPAN), while one
        # plane fits both runs' pixels within SURF_SPLIT px and the later run is mostly visible (at least
        # SURF_JOIN_VIS of its alpha; not the fill hidden behind a surface), continues that surface (a leg turning
        # away covers more than SURF_SPAN bins in one cell): the cut is dropped, else its seam opens in the cell.
        nocut = np.zeros(N, bool)
        njoin = 0
        if SURF_JOIN_VIS < 1:
            P_dp0 = np.concatenate(passes[9])
            runs = sorted(zip(P_c0, -P_kn0, P_kn0, P_kf0, P_dp0))     # by cell, nearest run first
            prev = None
            for c_, _, k0, k1, dp in runs:
                ci_ = np.searchsorted(ucid, c_)
                ent_ = np.arange(cstart[ci_], cstart[ci_] + ccount[ci_])
                ent_ = ent_[(ks[ent_] <= k0) & (ks[ent_] >= k1)]
                if prev is not None and prev[0] == c_ and dp and not force[ent_[0]]:
                    both = np.r_[prev[1], ent_]
                    et_ = torch.tensor(both, device=dev)
                    tot = stats(E_w[et_].double().sum(0), E_wp[et_].double().sum(0)).cpu().numpy()
                    vis_b = E_w[torch.tensor(ent_, device=dev)].double().sum().item() / max(E_a[ent_].sum(), 1e-9)
                    if tot[0] > 0 and np.sqrt(plane_sse(tot) / tot[0]) * px < SURF_SPLIT and vis_b >= SURF_JOIN_VIS:
                        nocut[ent_] = True; njoin += 1      # (all of it: the span rule would cut again)
                        ent_ = both                           # the joined run continues
                prev = (c_, ent_)
        print(f"  surface split: {nsplit} runs cut where two planes fit their pixels better, {njoin} span cuts "
              f"dropped where one plane fits a visible continuation", flush=True)
        passes = run_pass(force, nocut)
    if SURF_HYBRID:
        # SURF_HYBRID: MPI-only everywhere a run isn't clearly one smooth surface. A run of several bins whose
        # visible pixels (each pixel's visible-weighted mean disparity) one plane doesn't fit within SURF_HYBRID
        # parallax px at MOVE (fingers over a shirt, hair edges, a frame over glass) is cut into one piece per
        # bin, drawn as MPI-only draws its cells: flat at the cell's mean disparity within the bin, corners shared
        # only with pieces of the same bin. Runs that fit (skin, the hair shell over a forehead) stay surface pieces.
        E_w, E_wp, E_a = passes1[7]
        f_ = force.copy() if SURF_SPLIT else np.zeros(N, bool)
        nc_ = nocut if SURF_SPLIT else None
        fbe = np.zeros(N, bool)
        nfb = 0
        for c_, k0, k1 in zip(np.concatenate(passes[5]), np.concatenate(passes[3]), np.concatenate(passes[4])):
            ci_ = np.searchsorted(ucid, c_)
            ent_ = np.arange(cstart[ci_], cstart[ci_] + ccount[ci_])
            ent_ = ent_[(ks[ent_] <= k0) & (ks[ent_] >= k1)]
            if len(ent_) < 2:
                continue
            et_ = torch.tensor(ent_, device=dev)
            tot = stats(E_w[et_].double().sum(0), E_wp[et_].double().sum(0)).cpu().numpy()
            if tot[0] > 0 and np.sqrt(plane_sse(tot) / tot[0]) * px <= SURF_HYBRID:
                continue
            f_[ent_[1:]] = True; fbe[ent_] = True; nfb += 1
            if nc_ is not None:
                nc_[ent_] = False
        print(f"  hybrid: {nfb} runs drawn as MPI slices (one plane doesn't fit them within {SURF_HYBRID:g} px)",
              flush=True)
        passes = run_pass(f_, nc_, fbe)
    P_tex, P_texb, P_st, P_kn, P_kf, P_c, P_lk, _, P_fs, _, _, P_fb = passes
    del tex
    pc = np.concatenate(P_c); k_near = np.concatenate(P_kn); k_far = np.concatenate(P_kf)
    o3 = np.lexsort((-k_near, pc))                        # by cell, nearest first
    pc, k_near, k_far = pc[o3], k_near[o3], k_far[o3]
    st = np.concatenate(P_st)[o3]
    lnk = np.concatenate(P_lk)[o3]
    fsplit = np.concatenate(P_fs)[o3]
    fbk = np.concatenate(P_fb)[o3]
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
    sw, sx, sy, sxx, syy, sxy, sp, spx, spy = st[:, :9].T
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
    key_pos = key.copy()                                  # the grid corner (key may get a bin added below)
    if SURF_HYBRID:                                       # MPI fallback slices: flat, grouped only within their bin
        fbE = fbk[fE]
        flat = np.clip(sp / np.maximum(sw, 1e-12), e_all[k_far], e_all[k_near + 1])
        pred[fbE] = flat[fbE, None]
        key = key + np.repeat(np.where(fbE, (k_near + 1) * 10 ** 8, 0), 4)
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
    if SURF_MAXMOVE:                                      # a corner pulled far from its own plane by a long chain
        shared = np.where(np.abs(shared - pv) > SURF_MAXMOVE * bw, pv, shared)   # stretches into a spike: keep own
    cd = shared.reshape(nS, 4)[ent]
    if SURF_VISGLUE:                                      # corner options for the visual test (see SURF_VISGLUE)
        opts = []
        for J in SURF_VG_JOINS:
            o_ = np.lexsort((pv, key))
            br = np.r_[True, (np.diff(key[o_]) != 0) | (np.diff(pv[o_]) > J * bw)]
            g_ = np.cumsum(br) - 1
            m_ = np.bincount(g_, pv[o_] * wv[o_]) / np.maximum(np.bincount(g_, wv[o_]), 1e-12)
            sh = np.empty_like(pv); sh[o_] = m_[g_]
            opts.append(sh.reshape(nS, 4)[ent])
        SURF_OPTS[:] = [key.reshape(nS, 4)[ent], opts]
    nE = nS
    pred = pred[ent]
    pc, k_near, k_far, st, nS = pieces
    pcy, pcx = pc // nx, pc % nx
    sw, sp = st[:, 0], st[:, 6]
    # Keep each cell's pieces in their depth order at every corner (a back piece pulled in front of the one
    # before it by its corners would show through it, e.g. a dark strap behind a shirt).
    if SURF_DMAP:
        # SURF_DMAP: corners from the median-depth map instead of the planes' gluing. At each grid corner, the map's
        # pixels within SURF_DMAP render px are grouped into surfaces (sorted, split where the disparity jumps by
        # more than SURF_DMAP_GAP parallax px at MOVE); each piece takes the median of the surface nearest its own
        # corner, if that lies within its bins' range +- 1 bin. Neighbouring pieces on one surface read the same
        # pixels, so they meet exactly; at a depth jump they tear exactly there. Hidden pieces (no surface in the
        # map near them) keep their glued value.
        ny_ = int(pcy.max()) + 2
        Mg = torch.full(((ny_ + 1) * cell, (nx + 1) * cell), float("nan"), device=dev)
        ucy, ucx = ucid // nx, ucid % nx
        for i_ in range(0, nC, 4096):                     # place each cell's median map
            sl = slice(i_, min(nC, i_ + 4096))
            for k_, (yy, xx) in enumerate(zip(ucy[sl], ucx[sl])):
                Mg[yy * cell:(yy + 1) * cell, xx * cell:(xx + 1) * cell] = Dmed[i_ + k_]
        Mg = Mg.cpu().numpy()
        R_ = int(SURF_DMAP); gapd = SURF_DMAP_GAP / (fpx * MOVE)
        ukeys = np.unique(key_pos)
        clusters = {}
        for kk in ukeys:
            gy_, gx_ = divmod(int(kk), nx + 1)
            win = Mg[max(0, gy_ * cell - R_):gy_ * cell + R_, max(0, gx_ * cell - R_):gx_ * cell + R_].ravel()
            win = np.sort(win[np.isfinite(win)])
            if len(win) == 0:
                continue
            br_ = np.r_[0, np.nonzero(np.diff(win) > gapd)[0] + 1, len(win)]
            clusters[kk] = np.array([np.median(win[a_:b_]) for a_, b_ in zip(br_[:-1], br_[1:]) if b_ - a_ >= 3])
        keyp = key_pos.reshape(nE, 4)[ent]
        d_lo_p, d_hi_p = e_all[k_far] - bw, e_all[k_near + 1] + bw
        nset = 0
        for i_ in range(nS):
            for j_ in range(4):
                cs_ = clusters.get(keyp[i_, j_])
                if cs_ is None or len(cs_) == 0:
                    continue
                c_ = cs_[np.argmin(np.abs(cs_ - cd[i_, j_]))]
                if d_lo_p[i_] <= c_ <= d_hi_p[i_]:
                    cd[i_, j_] = c_; nset += 1
        print(f"  depth-map corners: {nset} of {4 * nS} piece corners set from the median-depth map", flush=True)
    # A SURF_SPLIT cut's back piece stays behind its front piece (both nearly at one depth, their planes can
    # cross; drawn by centre depth, the back piece's hidden content would cover the front one).
    for i in np.nonzero(fsplit)[0]:
        cd[i] = np.minimum(cd[i], cd[i - 1] - 1e-3 * bw)
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
        crops_b.append((sum(len(r) for r in rect_l), outb))
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
    tick("start")
    full_v = read_ply_vertex(scene)
    full = splat_tensors(full_v, dev)
    kv = full_v if NOMESH else read_ply_vertex(scene.with_name(scene.stem + SUFFIX + ".ply"))
    tick("load splats")
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
        meta_l = []                                   # ADAPT: (level, parent bin, sub-bin) per cell
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
        ldi_m = None
        if LDI and not SURF:
            # LDI pass 1: front colour (premultiplied, the slices while less than LDI_OPAQUE is accumulated), the bin
            # each pixel's front ends at, and disparities where the accumulated alpha crosses 0.1 / 0.5 / 0.9.
            Tm = torch.ones(Hc, Wc, device=dev)
            Fc = torch.zeros(Hc, Wc, 4, device=dev)
            kc = torch.full((Hc, Wc), -1, dtype=torch.long, device=dev)
            d10, d50, d90 = (torch.full((Hc, Wc), float("nan"), device=dev) for _ in range(3))
            for k in reversed(range(BINS)):
                m = b_t == k
                if not m.any():
                    continue
                sub = {kk: v[m] for kk, v in kept.items()}
                out_, al_, _ = ev.gs.rendering.rasterization(**sub, viewmats=torch.eye(4, device=dev)[None], Ks=Kc,
                                                             width=Wc, height=Hc, eps2d=ev.EPS, render_mode="RGB+D")
                a_ = al_[0, ..., 0]
                pd_ = torch.where(a_ > 1e-3, a_ / out_[0, ..., 3].clamp(min=1e-6), float("nan"))
                inc = (Tm > 1 - LDI_OPAQUE) & (a_ > 1e-4)
                Fc = torch.where(inc[..., None], Fc + Tm[..., None] * torch.cat([out_[0, ..., :3], a_[..., None]], -1), Fc)
                kc = torch.where(inc, torch.full_like(kc, k), kc)
                Tn = Tm * (1 - a_)
                for dd, th in ((d10, 0.9), (d50, 0.5), (d90, 0.1)):
                    dd.copy_(torch.where((Tm > th) & (Tn <= th), pd_, dd))
                Tm = Tn
            par = fg * MOVE                                 # parallax px per unit disparity (photo px)
            ok = (Tm <= 1 - LDI_OPAQUE) & torch.isfinite(d50) & torch.isfinite(d10) & torch.isfinite(d90)
            ok &= (d10 - d90).abs() * par <= LDI_THICK
            # Mesh: vertices every LDI_STEP render px, at the median of the front pixels' median depths around them.
            st_ = LDI_STEP
            gy_n, gx_n = Hc // st_ + 1, Wc // st_ + 1
            Dv = torch.where(ok, d50, float("nan"))
            pad = torch.nn.functional.pad(Dv[None, None], (st_ // 2, st_ // 2 + st_, st_ // 2, st_ // 2 + st_),
                                          value=float("nan"))[0, 0]
            win = pad.unfold(0, st_, st_).unfold(1, st_, st_)[:gy_n, :gx_n].reshape(gy_n, gx_n, -1)
            vd = torch.nanmedian(win, -1).values.cpu().numpy()   # (gy_n, gx_n), nan where no front pixel near
            for _ in range(2):                                # vertices without front pixels: neighbours' mean
                pv_ = np.pad(vd, 1, constant_values=np.nan)
                nb_ = np.stack([pv_[dy:dy + gy_n, dx:dx + gx_n] for dy in range(3) for dx in range(3)])
                with np.errstate(all="ignore"):
                    vd = np.where(np.isfinite(vd), vd, np.nanmean(nb_, 0))
            vok = np.isfinite(vd)
            q = np.stack([vd[:-1, :-1], vd[:-1, 1:], vd[1:, :-1], vd[1:, 1:]])          # 00 10 01 11
            qok = vok[:-1, :-1] & vok[:-1, 1:] & vok[1:, :-1] & vok[1:, 1:]
            spread = (np.nan_to_num(q).max(0) - np.nan_to_num(q).min(0)) * par
            split = qok & (spread > LDI_CUT)
            if not LDI_SPLIT:                                 # squares at depth jumps go to the MPI entirely
                qok = qok & ~split
                split = np.zeros_like(split)
            # A square spanning a depth jump is split at the largest gap of its corner depths: a near part and a far
            # part, each flat-extended to its own side's corners and textured with its own pixels (by their median
            # depth), so the front layer tears exactly along the jump (the MPI behind fills it).
            qs = np.sort(np.nan_to_num(q), 0)
            gi = np.argmax(np.diff(qs, axis=0), 0)
            thr = np.take_along_axis(qs, gi[None], 0)[0] / 2 + np.take_along_axis(qs, gi[None] + 1, 0)[0] / 2
            thr = np.where(split, thr, -np.inf)
            qpx = np.repeat(np.repeat(qok, st_, 0), st_, 1)
            tpx = np.repeat(np.repeat(thr, st_, 0), st_, 1)
            ph, pw = min(Hc, qpx.shape[0]), min(Wc, qpx.shape[1])
            qpx_t = torch.zeros(Hc, Wc, dtype=torch.bool, device=dev); qpx_t[:ph, :pw] = torch.tensor(qpx[:ph, :pw], device=dev)
            tpx_t = torch.full((Hc, Wc), -float("inf"), device=dev); tpx_t[:ph, :pw] = torch.tensor(tpx[:ph, :pw], device=dev, dtype=torch.float32)
            ldi_m = ok & qpx_t
            far_px = ldi_m & (d50 < tpx_t)
            # A pixel joins the front only where its own depth matches the sheet there (bilinear between the
            # corners; at split squares its side's mean): else it would ride on another surface (a handle's dark
            # pixels on the face beside it).
            vt = torch.tensor(np.nan_to_num(vd, nan=0.0), device=dev, dtype=torch.float32)[None, None]
            sh = torch.nn.functional.interpolate(vt, size=((gy_n - 1) * st_ + 1, (gx_n - 1) * st_ + 1),
                                                 mode="bilinear", align_corners=True)[0, 0]
            shp = torch.zeros(Hc, Wc, device=dev); shp[:min(Hc, sh.shape[0]), :min(Wc, sh.shape[1])] = sh[:Hc, :Wc]
            spl_px = torch.isfinite(tpx_t)
            qn = np.nan_to_num(q)
            sidem = lambda sel: np.nanmean(np.where(sel, qn, np.nan), 0)
            with np.errstate(all="ignore"):
                nm_, fm_ = sidem(qn >= thr[None]), sidem(qn < thr[None])
            up = lambda arr: (lambda t: torch.nn.functional.pad(t, (0, max(0, Wc - t.shape[1]), 0, max(0, Hc - t.shape[0])))[:Hc, :Wc])(
                torch.tensor(np.repeat(np.repeat(np.nan_to_num(arr, nan=-1.0), st_, 0), st_, 1), device=dev, dtype=torch.float32))
            side_d = torch.where(d50 >= tpx_t, up(nm_), up(fm_))
            good = torch.where(spl_px, (d50 - side_d).abs() * par <= LDI_CUT, (d50 - shp).abs() * par <= LDI_CUT)
            ldi_m = ldi_m & good
            far_px = far_px & good
            # Along the front's cuts (split or dropped squares) the MPI keeps its own copy too, within LDI_BAND render
            # px: covered by the front at the input view, it fills the front's tears like MPI-only does when moving.
            cutq = (split | (vok[:-1, :-1] & ~qok)).astype(np.float32)
            cutq = torch.tensor(cutq, device=dev)[None, None]
            nb = max(1, LDI_BAND // st_)
            cutq = torch.nn.functional.max_pool2d(cutq, 2 * nb + 1, 1, nb)[0, 0] > 0
            cpx = torch.zeros(Hc, Wc, dtype=torch.bool, device=dev)
            cq = cutq.repeat_interleave(st_, 0).repeat_interleave(st_, 1)
            cpx[:min(Hc, cq.shape[0]), :min(Wc, cq.shape[1])] = cq[:Hc, :Wc]
            ldi_rm = ldi_m & ~cpx                             # front content removed from the MPI here
            kc_t = kc
            ds_ = TEXDS
            def tex_of(msk):
                ft = torch.nn.functional.avg_pool2d(torch.where(msk[..., None], Fc, 0).permute(2, 0, 1)[None], ds_)[0]
                return (ft.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
            tex_near, tex_far = tex_of(ldi_m & ~far_px), tex_of(far_px)
            th_, tw_ = tex_near.shape[:2]
            vy, vx = np.mgrid[0:gy_n, 0:gx_n]
            pxv, pyv = np.minimum(vx * st_, Wc).ravel(), np.minimum(vy * st_, Hc).ravel()
            dv = np.nan_to_num(vd, nan=1.0).clip(1e-4).ravel()
            Vx, Vy, Vd = [pxv], [pyv], [dv]
            nV = len(dv)

            def newv(x_, y_, d_):
                n0 = sum(len(v) for v in Vd)
                Vx.append(x_); Vy.append(y_); Vd.append(d_)
                return n0 + np.arange(len(x_))
            qy, qx = np.nonzero(qok & ~split)
            i00 = qy * gx_n + qx
            idx4 = lambda i: (i, i + 1, i + gx_n, i + gx_n + 1)
            tri = lambda a0, a1, a2, a3: np.stack([np.stack([a0, a1, a3], 1), np.stack([a0, a3, a2], 1)], 1).reshape(-1, 3)
            F_near = [tri(*idx4(i00))]
            F_far = []
            sy, sx = np.nonzero(split)
            if len(sy):
                c4 = np.stack(idx4(sy * gx_n + sx), 1)                       # corner vertex ids (00 10 01 11)
                cd4 = np.stack([q[j][sy, sx] for j in range(4)], 1)
                th4 = thr[sy, sx][:, None]
                for side, Fl in ((cd4 >= th4, F_near), (cd4 < th4, F_far)):
                    mean_side = np.nanmean(np.where(side, cd4, np.nan), 1)
                    ids = c4.copy()
                    for j in range(4):
                        o_ = ~side[:, j]
                        if o_.any():
                            ids[o_, j] = newv(pxv[c4[o_, j]], pyv[c4[o_, j]], mean_side[o_])
                    Fl.append(tri(*[ids[:, j] for j in range(4)]))
            Vx, Vy, Vd = (np.concatenate(v) for v in (Vx, Vy, Vd))
            u_, v_ = (Vx - MARGIN) / SS, (Vy - MARGIN) / SS
            Vg = (np.stack([(u_ - GW / 2) / fg, (v_ - Hg / 2) / fg, np.ones_like(u_)], -1) / Vd[:, None]).astype(np.float32)
            UVg = np.stack([Vx / (tw_ * ds_), Vy / (th_ * ds_)], -1).astype(np.float32)
            LDI_FRONT[:] = [(Vg, np.concatenate(F_near).astype(np.uint32), UVg, tex_near)]
            if F_far:
                LDI_FRONT.append((Vg, np.concatenate(F_far).astype(np.uint32), UVg, tex_far))
            print(f"  LDI front: {ldi_m.float().mean().item():.0%} of the canvas, {len(qy)} squares + {len(sy)} split "
                  f"at depth jumps ({len(np.concatenate(F_near)) + sum(len(f) for f in F_far)} tris), "
                  f"{(vok[:-1, :-1] & ~qok).sum()} dropped", flush=True)
            del Fc, d10, d90, Dv, pad, win
        bend_cl = None
        if BEND and not SURF:
            # BEND: each bin stays one continuous sheet (MPI-only's shared corners, never torn) but bends toward the
            # visible surface: a median-depth map (per pixel, the disparity of the bin where the accumulated alpha,
            # near to far, crosses 0.5) is grouped at every grid corner into surfaces (its pixels within BEND render
            # px, split where the disparity jumps by more than BEND_GAP parallax px at MOVE); a bin's corner takes
            # the nearest surface's median if that lies within its bin's range +- BEND_TOL bins. Bins carrying one
            # surface (strips of a forehead) then meet there instead of ending at their own depths.
            Tm = torch.ones(Hc, Wc, device=dev)
            Dm = torch.full((Hc, Wc), float("nan"), device=dev)
            for k in reversed(range(BINS)):
                m = b_t == k
                if not m.any():
                    continue
                sub = {kk: v[m] for kk, v in kept.items()}
                out_, al_, _ = ev.gs.rendering.rasterization(**sub, viewmats=torch.eye(4, device=dev)[None], Ks=Kc,
                                                             width=Wc, height=Hc, eps2d=ev.EPS, render_mode="RGB+D")
                a_ = al_[0, ..., 0]
                pd_ = torch.where(a_ > 1e-3, a_ / out_[0, ..., 3].clamp(min=1e-6), float("nan"))
                cross = (Tm > 0.5) & (Tm * (1 - a_) <= 0.5)
                Dm = torch.where(cross, pd_, Dm)
                Tm = Tm * (1 - a_)
            Dm = Dm.cpu().numpy()
            R_, gapd = int(BEND), BEND_GAP / (fg * MOVE)
            bend_cl = {}
            for gy_ in range(ny + 1):
                for gx_ in range(nx + 1):
                    win = Dm[max(0, gy_ * cell - R_):gy_ * cell + R_, max(0, gx_ * cell - R_):gx_ * cell + R_].ravel()
                    win = np.sort(win[np.isfinite(win)])
                    if len(win) < 3:
                        continue
                    br_ = np.r_[0, np.nonzero(np.diff(win) > gapd)[0] + 1, len(win)]
                    cs_ = [np.median(win[a0:b0]) for a0, b0 in zip(br_[:-1], br_[1:]) if b0 - a0 >= 3]
                    if cs_:
                        bend_cl[(gy_, gx_)] = np.array(cs_)
            nbent = 0
            Tf = torch.ones(Hc, Wc, device=dev)               # transmittance of the nearer bins (BEND visibility)
        passes_ = [(b_t, centers, edges, BINS, None, 0)]
        coarse_G = {}                                     # ADAPT: each coarse bin's corners (G, used)
        if ADAPT and not SURF:
            # ADAPT: more slices only where stacking shows. A pass over the bins gives each pixel's visible bin
            # (where the accumulated alpha, near to far, crosses 0.5) and the composited image; a strip edge is
            # where neighbouring pixels' visible bins differ by exactly one (one surface continuing in the next
            # bin; larger steps are occlusion edges), and it shows where the image has detail across it (hair
            # over skin, not skin over skin). Cells with at least ADAPT_N strip-edge pixels whose local contrast
            # (largest colour step within 2 px, 0-255) exceeds ADAPT_T are flagged (grown by one cell) and built
            # from ADAPT x finer bins; the rest from the normal bins. Each level shares its own cell corners.
            Tm = torch.ones(Hc, Wc, device=dev)
            Cc = torch.zeros(Hc, Wc, 3, device=dev)
            vb = torch.full((Hc, Wc), -10, dtype=torch.long, device=dev)
            for k in reversed(range(BINS)):
                m = b_t == k
                if not m.any():
                    continue
                sub = {kk: v[m] for kk, v in kept.items()}
                out_, al_, _ = ev.gs.rendering.rasterization(**sub, viewmats=torch.eye(4, device=dev)[None], Ks=Kc,
                                                             width=Wc, height=Hc, eps2d=ev.EPS)
                a_ = al_[0, ..., 0]
                Cc = Cc + Tm[..., None] * out_[0]
                Tn = Tm * (1 - a_)
                vb = torch.where((Tm > 0.5) & (Tn <= 0.5), torch.full_like(vb, k), vb)
                Tm = Tn
            dx = (vb[:, 1:] - vb[:, :-1]).abs() == 1
            dy = (vb[1:, :] - vb[:-1, :]).abs() == 1
            E = torch.zeros(Hc, Wc, dtype=torch.bool, device=dev)
            E[:, 1:] |= dx; E[:, :-1] |= dx; E[1:, :] |= dy; E[:-1, :] |= dy
            Gx = torch.zeros(Hc, Wc, device=dev); Gy = torch.zeros(Hc, Wc, device=dev)
            Gx[:, 1:] = (Cc[:, 1:] - Cc[:, :-1]).abs().mean(-1) * 255
            Gy[1:, :] = (Cc[1:, :] - Cc[:-1, :]).abs().mean(-1) * 255
            con = torch.nn.functional.max_pool2d(torch.maximum(Gx, Gy)[None, None], 2 * ADAPT_W + 1, 1, ADAPT_W)[0, 0]
            hit = (E & (con > ADAPT_T)).float()
            padA = torch.nn.functional.pad(hit, (0, nx * cell - Wc, 0, ny * cell - Hc))
            cnt = padA.reshape(ny, cell, nx, cell).sum((1, 3))
            flag = cnt >= ADAPT_N
            if ADAPT_D:                                       # or dense strip edges (a steep surface), any contrast
                dens = torch.nn.functional.pad(E.float(), (0, nx * cell - Wc, 0, ny * cell - Hc)).reshape(ny, cell, nx, cell).sum((1, 3))
                flag = flag | (dens >= ADAPT_D)
            flag = torch.nn.functional.max_pool2d(flag.float()[None, None], 3, 1, 1)[0, 0] > 0
            BF = BINS * ADAPT
            edges_f = lo + np.arange(1, BF) / BF * (hi - lo)
            centers_f = lo + (np.arange(BF) + 0.5) / BF * (hi - lo)
            b_tf = torch.tensor(np.clip(np.searchsorted(edges_f, d), 0, BF - 1), device=dev)
            passes_ = [(b_t, centers, edges, BINS, ~flag, 0), (b_tf, centers_f, edges_f, BF, flag, 1)]
            print(f"  adaptive: {flag.float().mean().item():.1%} of cells flagged for {ADAPT}x finer bins "
                  f"({int((E.sum()).item())} strip-edge px, {int(hit.sum().item())} with contrast > {ADAPT_T:g})",
                  flush=True)
            if os.environ.get("ADAPT_DEBUG"):             # flag grid, visible bin and strip-edge contrast maps
                np.savez(os.environ["ADAPT_DEBUG"], flag=flag.cpu().numpy(), vb=vb.cpu().numpy(),
                         E=E.cpu().numpy(), con=con.cpu().numpy(), C=Cc.cpu().numpy(), cell=cell, margin=MARGIN)
            del Cc, Tm, vb, E, Gx, Gy, con, hit
        LCOL_DATA.clear(); LCT_LAYER.clear(); LCW_DATA.clear()
        if LCOL and not SURF:
            nL = LCT_MAX if (LCT or LCW) else 3
            Tl = torch.ones(Hc, Wc, device=dev); Ll = torch.zeros(Hc, Wc, dtype=torch.long, device=dev)
            accC = torch.zeros(nL, Hc, Wc, 3, device=dev); accA = torch.zeros(nL, Hc, Wc, device=dev)
            spl = torch.zeros(2, Hc, Wc, device=dev)
            lastk = torch.full((Hc, Wc), 10 ** 6, dtype=torch.long, device=dev)
            kopq = torch.full((Hc, Wc), 10 ** 6, dtype=torch.long, device=dev)
            for k in reversed(range(BINS)):
                m = b_t == k
                if not m.any():
                    continue
                sub = {kk: v[m] for kk, v in kept.items()}
                out_, al_, _ = ev.gs.rendering.rasterization(**sub, viewmats=torch.eye(4, device=dev)[None], Ks=Kc,
                                                             width=Wc, height=Hc, eps2d=ev.EPS)
                a_ = al_[0, ..., 0]
                occ_ = a_ > LCOL_EPS
                # a new layer starts at content after a depth gap (> LCOL_GAP empty bins) behind an opaque layer: a
                # surface's own thick shell (several consecutive nearly opaque bins) stays one layer
                # ... or at content whose colour differs from the opaque layer's (LCOL_DC, mean abs, 0-1): the fill behind
                # a face, a strap behind hair, even when adjacent in depth
                Lc = Ll.clamp(max=nL - 1)
                lay_c = torch.gather(accC, 0, Lc[None, ..., None].expand(1, -1, -1, 3))[0]
                lay_a = torch.gather(accA, 0, Lc[None])[0]
                diff_ = ((out_[0] / a_.clamp(min=1e-3)[..., None]) - lay_c / lay_a.clamp(min=1e-3)[..., None]).abs().mean(-1)
                cdiff = (a_ > 0.1) & (lay_a > 0.5) & (diff_ > LCOL_DC)
                # ... or deeper than LCOL_SPAN bins past where the layer became opaque (the hidden fill behind a surface)
                deep_ = kopq - k > LCOL_SPAN
                # a depth gap separates surfaces whatever the layer's opacity (a face only 90% opaque at its eyes or
                # edges must not take the background behind it into its layer); the other rules need it opaque
                gap_ = (lastk - k > LCOL_GAP + 1) & (lay_a > LCOL_GAPA)
                new_ = occ_ & (Ll < nL - 1) & (gap_ | ((Tl < 1 - LCOL_OPQ) & (cdiff | deep_)))
                for l in (0, 1):
                    spl[l] = torch.where(new_ & (Ll == l), float(k + 1), spl[l])
                Ll = Ll + new_.long(); Tl = torch.where(new_, torch.ones_like(Tl), Tl)
                if LCT or LCW:
                    LCT_LAYER[k] = Ll.to(torch.uint8)
                for l in range(nL):
                    w_ = ((Ll == l).float() * Tl)
                    accC[l] += w_[..., None] * out_[0]; accA[l] += w_ * a_
                Tn_ = Tl * (1 - a_)
                if os.environ.get("LCOL_PROBE"):
                    for pp in os.environ["LCOL_PROBE"].split(";"):
                        px_, py_ = (int(v) for v in pp.split(","))
                        X_, Y_ = px_ * SS + MARGIN, py_ * SS + MARGIN
                        if a_[Y_, X_] > 1e-3:
                            print(f"  probe {pp} bin {k} a {a_[Y_, X_].item():.3f} T {Tn_[Y_, X_].item():.3f} L {Ll[Y_, X_].item()} "
                                  f"col {(out_[0, Y_, X_] / a_[Y_, X_]).cpu().numpy().round(2)} diff {diff_[Y_, X_].item():.3f}", flush=True)
                kopq = torch.where(new_, torch.full_like(kopq, 10 ** 6), kopq)
                kopq = torch.where((Tl >= 1 - LCOL_OPQ) & (Tn_ < 1 - LCOL_OPQ), torch.full_like(kopq, k), kopq)
                Tl = Tn_
                lastk = torch.where(occ_, torch.full_like(lastk, k), lastk)
            if LCT or LCW:
                print("  colour layers: px reaching layer " + ", ".join(
                    f"{l + 1}: {(Ll >= l).float().mean().item():.0%}" for l in range(1, nL)), flush=True)
            if LCT or LCW:
                del accC, accA, Tl, Ll, spl
        if LCOL and not SURF and not (LCT or LCW):
            dsl = TEXDS
            pool = lambda t: torch.nn.functional.avg_pool2d(t[None], dsl)[0]
            imgs_l = []
            for l in range(3):
                Pc = pool(accC[l].permute(2, 0, 1)).permute(1, 2, 0); Pa = pool(accA[l][None])[0][..., None]
                col = torch.where(Pa > 1 / 255, Pc / Pa.clamp(min=1e-6), 0); wgt = (Pa > 1 / 255).float()
                for _ in range(64):                       # spread colour into empty texels (filtering, sharing)
                    if bool(wgt.all()):
                        break
                    sm = torch.nn.functional.avg_pool2d((col * wgt).permute(2, 0, 1)[None], 3, 1, 1, count_include_pad=False)[0].permute(1, 2, 0)
                    sw = torch.nn.functional.avg_pool2d(wgt.permute(2, 0, 1)[None], 3, 1, 1, count_include_pad=False)[0].permute(1, 2, 0)
                    col = torch.where(wgt > 0, col, sm / sw.clamp(min=1e-6)); wgt = torch.where(wgt > 0, wgt, (sw > 0).float())
                imgs_l.append(col.clamp(0, 1))
            hl, wl = imgs_l[0].shape[:2]
            sp_ = spl[:, ::dsl, ::dsl][:, :hl, :wl] / 255
            ph_, pw_ = 2 * hl + (-2 * hl) % 4, 2 * wl + (-2 * wl) % 4
            pageC = torch.zeros(ph_, pw_, 4, device=dev)
            for l, (oy, ox) in enumerate([(0, 0), (0, wl), (hl, 0)]):
                pageC[oy:oy + hl, ox:ox + wl, :3] = imgs_l[l]
                if l < 2:
                    pageC[oy:oy + hl, ox:ox + wl, 3] = sp_[l]
            LCOL_DATA.update(page=(pageC.cpu().numpy() * 255 + 0.5).astype(np.uint8), w=wl, h=hl, ds=dsl)
            print(f"  colour layers: {(Ll >= 1).float().mean().item():.0%} / {(Ll >= 2).float().mean().item():.0%} of px "
                  f"reach layer 2 / 3; page {pw_}x{ph_}", flush=True)
            del accC, accA, Tl, Ll, spl
        nbin_l = []                                   # LCOL: each cell's normal bin
        # the export's plain-MPI and SEPA paths take the cells' bordered tiles to the GPU; the others read them on the CPU
        TILES_GPU = SEPA or not (SURF or LCOL or LDI or ADAPT_SHARE or LCW or LCT)
        GAPCOPY_DATA.clear(); PAR_DATA.clear()
        par_o, par_dk = [], []                        # PARALLAX: offset tiles like crops_b; plane thickness per cell
        gc_prev, gc_conn = None, {}                   # GAPCOPY: (visible px, cell -> row) of the last bin; copies
        gc_T = torch.ones(Hc, Wc, device=dev) if GAPCOPY else None   # transmittance of the nearer bins
        lct_n = [0, 0]                                # LCT: cell planes, layer pieces (LCW: planes with 2 layers)
        lcw_lab, lcw_wt = [], []                      # LCW: (lA, lB) per plane; weight tiles like crops_b
        _orig_bins = (b_t, centers, edges)
        # Plain MPI (and ADAPT): each bin's work on the window of cells its content covers (outside it everything is
        # exactly 0), not the whole frame; the other modes keep the full-frame path below.
        FASTWIN = (not SURF and ldi_m is None and not (TILE_NEAR or TILE_FAR) and not EXTEND and not GAPCOPY
                   and not PARALLAX and not LCOL and not SNAP and bend_cl is None and REFINE and REFINE_MODE == "mean"
                   and not os.environ.get("NOFASTWIN"))
        tick("adaptive detection (and earlier)")
        lvl_prev = 0
        bin_sort = {}                                 # FASTWIN: id(bin index tensor) -> (bin-sorted splats, starts)
        for b_t, centers, edges, cmask_, lvl_, k in ((p_[0], p_[1], p_[2], p_[4], p_[5], k_) for p_ in passes_
                                                     for k_ in reversed(range(p_[3]))):   # nearest bin first
            if lvl_ != lvl_prev:
                tick(f"planes, level {lvl_prev}"); lvl_prev = lvl_
            if FASTWIN:                                      # the bin's splats as a slice of the bin-sorted splats
                srt = bin_sort.get(id(b_t))                  # (stable: same order as the mask), no GPU sync
                if srt is None:
                    o_ = torch.argsort(b_t, stable=True)
                    st_ = np.r_[0, np.cumsum(torch.bincount(b_t, minlength=len(centers)).cpu().numpy())]
                    srt = bin_sort[id(b_t)] = ({kk: v[o_] for kk, v in kept.items()}, st_)
                s0_, s1_ = int(srt[1][k]), int(srt[1][k + 1])
                if s0_ == s1_:
                    continue
                sec(f"other L{lvl_}")
                sub = {kk: v[s0_:s1_] for kk, v in srt[0].items()}
            else:
                m = b_t == k
                if not m.any():
                    continue
                sec(f"other L{lvl_}")
                sub = {kk: v[m] for kk, v in kept.items()}
            out, alpha, _ = ev.gs.rendering.rasterization(**sub, viewmats=torch.eye(4, device=dev)[None], Ks=Kc,
                                                          width=Wc, height=Hc, eps2d=ev.EPS, render_mode="RGB+D")
            sec(f"raster L{lvl_}")
            rgb, a, dacc = out[0, ..., :3], alpha[0, ..., 0], out[0, ..., 3]
            if FASTWIN:
                # occupied cells from the alpha alone (max-pool per cell; the half-precision test as the full path),
                # then only their bordered tiles gathered from the rasterizer's output (0 outside the frame)
                amx = torch.nn.functional.max_pool2d(a[None, None], cell, cell, ceil_mode=True)[0, 0]
                occ_c = amx.half() >= 1 / 255
                if cmask_ is not None:                       # ADAPT: this level's cells only
                    occ_c = occ_c & cmask_[:occ_c.shape[0], :occ_c.shape[1]]
                cy, cx = torch.nonzero(occ_c, as_tuple=True)
                if len(cy) == 0:
                    continue
                ar_ = torch.arange(cb, device=dev)
                rr = cy[:, None] * cell - gb + ar_; cc = cx[:, None] * cell - gb + ar_            # (n, cb) frame px
                okr, okc = (rr >= 0) & (rr < Hc), (cc >= 0) & (cc < Wc)
                ri, ci = rr.clamp(0, Hc - 1)[:, :, None], cc.clamp(0, Wc - 1)[:, None, :]
                ok2 = (okr[:, :, None] & okc[:, None, :])[..., None]
                o4 = torch.where(ok2, out[0][ri, ci], 0)                                     # (n, cb, cb, RGB+D)
                a4 = torch.where(ok2, alpha[0][ri, ci], 0)                                   # (n, cb, cb, 1)
                tiles = torch.cat([o4[..., :3], a4], -1).half()                             # as the padded image
                crops.append(tiles[:, gb:gb + cell, gb:gb + cell].reshape(-1, 4))
                if EXPORT:
                    sel = tiles.float()
                    crops_b.append((sum(len(r) for r in rect_l), sel.half() if TILES_GPU else sel.half().cpu()))
                cyn, cxn = cy.cpu().numpy(), cx.cpu().numpy()
                rect_l.append(np.stack([cxn * cell, cyn * cell, (cxn + 1) * cell, (cyn + 1) * cell], 1))
                if ADAPT:
                    per_ = ((BINS * ADAPT if lvl_ else BINS) // SHARE_COLOR) if SHARE_COLOR else (ADAPT if lvl_ else 1)
                    meta_l.append(np.tile([lvl_, k // per_, k % per_, per_], (len(cyn), 1)))
                cen_l.append(np.full(len(cyn), centers[k]))
                # per-cell depth (as below): alpha-weighted mean disparity, corners shared by neighbouring cells
                a_i = a4[:, gb:gb + cell, gb:gb + cell, 0]
                d_i = o4[:, gb:gb + cell, gb:gb + cell, 3]
                pd_i = torch.where(a_i > 1e-3, a_i / d_i.clamp(min=1e-6), 0)
                aw = a_i.sum((1, 2)); dw = (a_i * pd_i).sum((1, 2))
                dcell = (dw / aw.clamp(min=1e-9)).double()
                G = torch.zeros(ny + 1, nx + 1, device=dev, dtype=torch.float64); Gn = torch.zeros_like(G)
                one = torch.ones_like(dcell)
                for oy in (0, 1):
                    for ox in (0, 1):
                        G.index_put_((cy + oy, cx + ox), dcell, accumulate=True)
                        Gn.index_put_((cy + oy, cx + ox), one, accumulate=True)
                G = (G / Gn.clamp(min=1)).cpu().numpy()
                if ADAPT:
                    used_ = (Gn > 0).cpu().numpy()
                    if lvl_ == 0:
                        coarse_G[k] = (G.copy(), used_)
                    elif k // ADAPT in coarse_G:
                        Gc_, uc_ = coarse_G[k // ADAPT]
                        both_ = used_ & uc_
                        G[both_] = Gc_[both_]
                cd_l.append(np.stack([G[cyn, cxn], G[cyn, cxn + 1], G[cyn + 1, cxn], G[cyn + 1, cxn + 1]], 1))
                sec(f"windowed bin L{lvl_}")
                continue
            if ldi_m is not None:                            # LDI: the front layer has this content (not near cuts)
                inf_ = ldi_rm & (kc_t <= k)
                a = torch.where(inf_, 0, a); rgb = torch.where(inf_[..., None], 0, rgb)
                dacc = torch.where(inf_, 0, dacc)
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
            img = torch.nn.functional.pad(torch.cat([rgb, a[..., None]], -1).half(),
                                          (0, 0, 0, nx * cell - Wc, 0, ny * cell - Hc))
            cells = img.reshape(ny, cell, nx, cell, 4).permute(0, 2, 1, 3, 4)
            occ = cells[..., 3].amax((2, 3)) >= 1 / 255
            if cmask_ is not None:                           # ADAPT: this level's cells only
                occ = occ & cmask_[:ny, :nx]
            cy, cx = torch.nonzero(occ, as_tuple=True)
            sec(f"c1 occupancy L{lvl_}")
            if SURF:                                         # keep this bin's cells for the pieces (see SURF)
                padS = lambda t: torch.nn.functional.pad(t, (0, nx * cell - Wc, 0, ny * cell - Hc))
                pd = torch.where(a > 1e-3, a / dacc.clamp(min=1e-6), 0)
                Pc = padS(pd).reshape(ny, cell, nx, cell).permute(0, 2, 1, 3)[cy, cx]      # disparity per px
                selb = None
                if EXPORT:
                    pimg = torch.nn.functional.pad(img.permute(2, 0, 1).float(), (gb, gb, gb, gb))
                    selb = cell_tiles(pimg, cy, cx, cell, cb).half().cpu()
                if SURF_REGION is None:
                    surf_l.append((k, (cy * nx + cx).cpu().numpy(), cells[cy, cx], selb, Pc))
                    continue
                # SURF_REGION: surface pieces only in the region's cells; MPI-only cells everywhere except the
                # region's interior, so along its border both are drawn (MPI continues under the surface's edge).
                rin = torch.tensor(SURF_REGION[0][:ny, :nx], device=dev)[cy, cx]
                if rin.any():
                    surf_l.append((k, (cy * nx + cx)[rin].cpu().numpy(), cells[cy, cx][rin],
                                   selb[rin.cpu()] if selb is not None else None, Pc[rin]))
                mo = ~torch.tensor(SURF_REGION[1][:ny, :nx], device=dev)[cy, cx]
                cy, cx = cy[mo], cx[mo]
                if len(cy) == 0:
                    continue
            lay_ = lab_ = None
            if LCW and not SURF:
                # LCW: the plane's two main layers in the cell (by its alpha there); the weight picks the second
                Lk = LCT_LAYER[(k // ADAPT) if (ADAPT and lvl_) else k].float()
                Lp = torch.nn.functional.pad(Lk, (0, nx * cell - Wc, 0, ny * cell - Hc), value=255)
                Lcell = Lp.reshape(ny, cell, nx, cell).permute(0, 2, 1, 3)[cy, cx]
                acell = cells[cy, cx][..., 3].float()
                mass = torch.stack([(acell * (Lcell == l)).sum((1, 2)) for l in range(LCT_MAX)], 1)
                t2 = mass.topk(2, 1)
                lab_ = torch.stack([t2.indices[:, 0], torch.where(t2.values[:, 1] > 0, t2.indices[:, 1], t2.indices[:, 0])], 1)
                lcw_lab.append(lab_.cpu().numpy())
                lct_n[0] += len(cy); lct_n[1] += int((t2.values[:, 1] > 0).sum())
                if len(lct_n) < 3:
                    lct_n += [0, 0.0, 0.0]
                n3_ = (mass > 0).sum(1) > 2                 # planes with a third layer: its texels take A or B
                lct_n[2] += int(n3_.sum()); lct_n[3] += float((mass.sum(1) - t2.values.sum(1)).sum()); lct_n[4] += float(mass.sum())
            if LCT and not SURF:
                # LCT: one piece per visibility layer present in the cell: the bin's content where it is in that layer
                Lk = LCT_LAYER[(k // ADAPT) if (ADAPT and lvl_) else k].float()
                Lp = torch.nn.functional.pad(Lk, (0, nx * cell - Wc, 0, ny * cell - Hc), value=255)
                Lcell = Lp.reshape(ny, cell, nx, cell).permute(0, 2, 1, 3)[cy, cx]
                on_ = cells[cy, cx][..., 3] >= 1 / 255
                ii, lay_ = torch.nonzero(torch.stack([((Lcell == l) & on_).flatten(1).any(1) for l in range(LCT_MAX)], 1),
                                         as_tuple=True)
                crops.append((cells[cy, cx][ii] * (Lcell[ii] == lay_[:, None, None])[..., None]).reshape(-1, 4))
                lct_n[0] += len(cy); lct_n[1] += len(ii)
                if os.environ.get("LCT_STATS"):            # each piece's share of its plane's alpha; layer counts
                    am_ = cells[cy, cx][..., 3].float()
                    pm_ = (am_[ii] * (Lcell[ii] == lay_[:, None, None])).sum((1, 2)) / am_[ii].sum((1, 2)).clamp(min=1e-9)
                    npc = torch.bincount(ii, minlength=len(cy))
                    lct_n.append((pm_.cpu().numpy(), npc[ii].cpu().numpy()))
                cy, cx = cy[ii], cx[ii]
            else:
                crops.append(cells[cy, cx].reshape(-1, 4))
            if EXPORT:
                sec(f"c2 interiors L{lvl_}")
                # half precision (as img), only the cells' tiles converted: same values, no full-frame float copy
                pimg = torch.nn.functional.pad(img.permute(2, 0, 1), (gb, gb, gb, gb))
                sel = cell_tiles(pimg, cy, cx, cell, cb).float()                            # (n, cb, cb, 4)
                sec(f"c3 bordered tiles L{lvl_}")
                if lay_ is not None or lab_ is not None:     # the border too: neighbours' texels of the same layer
                    Lb = cell_tiles(torch.nn.functional.pad(Lp[None], (gb, gb, gb, gb), value=255), cy, cx, cell, cb)[..., 0]
                if lay_ is not None:
                    sel = sel * (Lb == lay_[:, None, None])[..., None]
                if lab_ is not None:                         # weight 1: the second layer (nearer in layer index)
                    lA_, lB_ = lab_[:, 0, None, None].float(), lab_[:, 1, None, None].float()
                    lcw_wt.append((sum(len(r) for r in rect_l), ((Lb - lB_).abs() < (Lb - lA_).abs()).half().cpu()))
                # (first cell index, (n, cb, cb, 4)): kept on the GPU where the export takes them back there anyway
                # (plain MPI and SEPA; one copy per bin to fresh host memory costs ~10-20 s), else on the CPU
                crops_b.append((sum(len(r) for r in rect_l), sel.half() if TILES_GPU else sel.half().cpu()))
                sec(f"c4 copy to CPU L{lvl_}")
            cyn, cxn = cy.cpu().numpy(), cx.cpu().numpy()
            if GAPCOPY and not SURF and not ADAPT:   # connected to the last (nearer) bin in a cell: copy its tile here
                base_ = sum(len(r_) for r_ in rect_l)
                vis_ = gc_T * a > GAPCOPY_VIS
                if gc_prev is not None:
                    pv, rows_prev = gc_prev
                    dil = lambda m_: (torch.nn.functional.max_pool2d(m_.float()[None, None], 3, 1, 1)[0, 0] > 0)
                    tch = ((vis_ & dil(pv) & ~pv) | (pv & dil(vis_) & ~vis_)).float()   # the boundary, both sides
                    cnt = torch.nn.functional.pad(tch, (0, nx * cell - Wc, 0, ny * cell - Hc)).reshape(
                        ny, cell, nx, cell).sum((1, 3))[cy, cx].cpu().numpy()
                    for n_, (yy, xx) in enumerate(zip(cyn, cxn)):
                        pr_ = rows_prev.get((yy, xx))
                        if pr_ is not None and cnt[n_] >= GAPCOPY_MIN:
                            gc_conn[pr_] = base_ + n_
                gc_prev = (vis_, {(yy, xx): base_ + n_ for n_, (yy, xx) in enumerate(zip(cyn, cxn))})
                gc_T = gc_T * (1 - a)
            rect_l.append(np.stack([cxn * cell, cyn * cell, (cxn + 1) * cell, (cyn + 1) * cell], 1))
            if LCOL and not SURF:
                nbin_l.append(np.full(len(cyn), (k // ADAPT) if (ADAPT and lvl_) else k))
            if LCT and not SURF:                            # share key: (cell, level, layer); order: bin
                meta_l.append(np.stack([np.full(len(cyn), lvl_ or 0), lay_.cpu().numpy(), np.full(len(cyn), k),
                                        np.zeros(len(cyn), np.int64)], 1))
            elif ADAPT and not SURF:
                per_ = ((BINS * ADAPT if lvl_ else BINS) // SHARE_COLOR) if SHARE_COLOR else (ADAPT if lvl_ else 1)
                meta_l.append(np.tile([lvl_, k // per_, k % per_, per_], (len(cyn), 1)))
            cen_l.append(np.full(len(cyn), centers[k]))
            sec(f"cells, crops, copies L{lvl_}")
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
                if ADAPT and not SURF:
                    # the two levels meet exactly: a fine bin's corners shared with cells of its coarse parent bin
                    # take the coarse sheet's depth there (else a crack opens along the level boundary)
                    used_ = (Gn > 0).cpu().numpy()
                    if lvl_ == 0:
                        coarse_G[k] = (G.copy(), used_)
                    elif k // ADAPT in coarse_G:
                        Gc_, uc_ = coarse_G[k // ADAPT]
                        both_ = used_ & uc_
                        G[both_] = Gc_[both_]
                if bend_cl is not None:                      # BEND: bend this bin's sheet toward the surface
                    # only where the bin is visible (its alpha not hidden by nearer bins) around the corner: the
                    # fill hidden behind a surface must not bend forward onto it
                    vis_ = torch.nn.functional.pad(Tf * a, (0, (nx + 1) * cell - Wc, 0, (ny + 1) * cell - Hc))
                    R2 = int(BEND)
                    cv_ = torch.nn.functional.avg_pool2d(vis_[None, None], 2 * R2 + 1, 1, R2)[0, 0]
                    cvis = cv_[::cell, ::cell][:ny + 1, :nx + 1].cpu().numpy()
                    lo_k = (edges[k - 1] if k > 0 else lo) - BEND_TOL * (hi - lo) / BINS
                    hi_k = (edges[k] if k < BINS - 1 else hi) + BEND_TOL * (hi - lo) / BINS
                    for gy_, gx_ in zip(*np.nonzero((Gn > 0).cpu().numpy())):
                        cs_ = bend_cl.get((gy_, gx_))
                        if cs_ is None:
                            continue
                        c_ = cs_[np.argmin(np.abs(cs_ - G[gy_, gx_]))]
                        if lo_k <= c_ <= hi_k and cvis[gy_, gx_] >= BEND_VIS:
                            G[gy_, gx_] = c_; nbent += 1
                if SNAP:                                     # decided after all bins (seams need the others)
                    snap_bins.append((k, G, (Gn > 0).cpu().numpy(), cyn, cxn))
                    cd_l.append(None)
                else:
                    cd_l.append(np.stack([G[cyn, cxn], G[cyn, cxn + 1], G[cyn + 1, cxn], G[cyn + 1, cxn + 1]], 1))
                    if PARALLAX and EXPORT:              # each texel's disparity offset from the sheet (two triangles)
                        Gt = torch.tensor(G, device=dev, dtype=torch.float32)
                        up_ = lambda t: t.repeat_interleave(cell, 0).repeat_interleave(cell, 1)
                        g00, g10, g01, g11 = up_(Gt[:-1, :-1]), up_(Gt[:-1, 1:]), up_(Gt[1:, :-1]), up_(Gt[1:, 1:])
                        TY = ((torch.arange(ny * cell, device=dev) % cell + 0.5) / cell)[:, None]
                        SX = ((torch.arange(nx * cell, device=dev) % cell + 0.5) / cell)[None, :]
                        sheet = torch.where(TY <= SX, g00 + SX * (g10 - g00) + TY * (g11 - g10),
                                            g00 + TY * (g01 - g00) + SX * (g11 - g01))
                        dk_ = (hi - lo) / len(centers)
                        off = torch.where(pad_(a) > 1e-3, ((pad_(pd) - sheet) / dk_).clamp(-1, 1), 0)
                        po = cell_tiles(torch.nn.functional.pad(off[None], (gb, gb, gb, gb)), cy, cx, cell, cb)[..., 0]
                        par_o.append((crops_b[-1][0], po.half() if TILES_GPU else po.half().cpu()))
                        par_dk.append(np.full(len(cyn), dk_))
            else:
                cd_l.append(np.full((len(cyn), 4), centers[k]))
            sec(f"refine (depth fit) L{lvl_}")
            if bend_cl is not None:
                Tf = Tf * (1 - a)
        if SURF:
            surface_pieces(surf_l, crops, crops_b, rect_l, cen_l, cd_l, cell, nx, lo, hi, edges, fg)
        if REFINE and SNAP:
            assert not (TILE_NEAR or TILE_FAR), "SNAP needs one cell size"
            snap_corners(snap_bins, cd_l, (snap_val, snap_ok), (hi - lo) / BINS)
        b_t, centers, edges = _orig_bins
        if bend_cl is not None:
            print(f"  bend: {nbent} bin corners moved to the median-depth surface", flush=True)
        if BAKE_TIMING:
            print("  [time]   " + ", ".join(f"{k_}: {v_:.1f}s" for k_, v_ in _sec.items() if k_ != "_t"), flush=True)
        tick("bin planes into cells")
        rects = np.concatenate(rect_l).astype(np.int64)
        if PARALLAX and EXPORT and par_dk:
            assert sum(len(t) for _, t in par_o) == len(rects), "PARALLAX needs REFINE (no SNAP)"
            PAR_DATA.update(off=par_o, dk=np.concatenate(par_dk))
        if GAPCOPY and not SURF and not ADAPT:
            conn_ = np.full(len(rects), -1, np.int64)
            conn_[list(gc_conn)] = list(gc_conn.values())
            GAPCOPY_DATA["conn"] = conn_
            print(f"  GAPCOPY: {len(gc_conn)} of {len(rects)} cell planes connected to the next farther plane", flush=True)
        if LCW and not SURF:
            print(f"  LCW: {lct_n[1]} of {lct_n[0]} cell planes span two colour layers, {lct_n[2] if len(lct_n) > 2 else 0} more "
                  f"({lct_n[3] / max(lct_n[4], 1e-9) if len(lct_n) > 2 else 0:.2%} of the alpha in a third+ layer)", flush=True)
            LCW_DATA.update(lab=np.concatenate(lcw_lab), wt=lcw_wt)
        if LCT and not SURF:
            print(f"  LCT: {lct_n[0]} cell planes -> {lct_n[1]} layer pieces", flush=True)
            if len(lct_n) > 2:
                fr = np.concatenate([t[0] for t in lct_n[2:]]); sp = np.concatenate([t[1] for t in lct_n[2:]]) > 1
                print("  LCT pieces of split planes: " + ", ".join(f"<{q:g}% of the plane's alpha: {(fr[sp] < q / 100).mean():.0%}"
                      for q in (1, 5, 20, 50)) + f"; split planes: {sp.sum()} pieces", flush=True)
        if LCOL_DATA:
            LCOL_DATA['nbin'] = np.concatenate(nbin_l)
        CELL_META[:] = [np.concatenate(meta_l)] if ((ADAPT and ADAPT_SHARE or LCT) and not SURF and meta_l) else []
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

    atlas_a = None                                    # culling: the atlas's alpha only, contiguous

    def patch_samples(eye, with_local=False, scale=1.0, alpha_only=False):
        """Fragments of every cell seen from eye: values (RGBA, or alpha only: (n,)), pixel, depth, cell."""
        nonlocal atlas_a
        if alpha_only and atlas_a is None:
            atlas_a = atlas[:, 3].float().contiguous()
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
            if alpha_only:
                g = lambda yy, xx: atlas_a[base + yy * wi + xx]
                fx, fy = fx[:, 0], fy[:, 0]
                c = (g(y0, x0) * (1 - fx) + g(y0, x1) * fx) * (1 - fy) + (g(y1, x0) * (1 - fx) + g(y1, x1) * fx) * fy
                k = (c > 1 / 255) & alive[prim]
            else:
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
    if MPI and SURF and SURF_VISGLUE and SURF_OPTS:
        # SURF_VISGLUE: each corner's option chosen by looking (see SURF_VISGLUE). Corners two cells apart share
        # no piece, so the four corner parity classes are decided in turn, each by rendering every option for
        # the whole class. A corner's score: the error against the splats, summed over the pixels any option
        # draws with its pieces (so a tear is charged for the hole it leaves, a rubber sheet for what it covers).
        keys, opts = SURF_OPTS
        ukey, kinv = np.unique(keys, return_inverse=True)
        kinv = kinv.reshape(keys.shape)
        cls_k = (ukey // (nx + 1)) % 2 * 2 + ukey % (nx + 1) % 2
        O = np.stack(opts)                                    # (options, pieces, 4)
        nO, ar = len(O), np.arange(P)
        choice = np.full(len(ukey), SURF_VG_INIT)
        cd_of = lambda ch: O[ch[kinv], ar[:, None], np.arange(4)[None]]
        corner_in = np.stack([kinv[ar, j] for j in range(4)], 1)
        cls_piece = cls_k[corner_in]                          # (pieces, 4): each piece has one corner per class
        sc = 0.5
        Ws, Hs, fs = round(W * sc), round(H * sc), f * sc
        mv = SURF_VG_MOVE
        eyes_v = [(mv, 0, 0), (-mv, 0, 0), (0, mv / 2, 0), (0, -mv / 2, 0), (0, 0, mv), (0.7 * mv, 0, 0.7 * mv)]
        refs = [raster([full], e, Ws, Hs, fs)[0].reshape(-1, 3) for e in eyes_v]

        def set_cd(cd):
            nonlocal_ = torch.tensor(cd, device=dev, dtype=torch.float32).clamp(min=dmin)
            cn = torch.stack([(cu - GW / 2) / fg, (cv - Hg / 2) / fg, torch.ones_like(cu)], -1) / nonlocal_[..., None]
            tr = torch.stack([cn[:, [0, 1, 3]], cn[:, [0, 3, 2]]], 1).reshape(-1, 3, 3)
            return cn, torch.linalg.solve(tr, torch.ones(len(tr), 3, device=dev)), tr
        t_vg = time.time()
        for ps_ in range(SURF_VG_PASSES):
            for c_ in range(4):
                sel = cls_k == c_
                score = np.zeros((nO, len(ukey)))
                jsel = np.argmax(cls_piece == c_, 1)          # each piece's corner of this class
                kp = torch.tensor(corner_in[ar, jsel], device=dev)
                for e_, ref in zip(eyes_v, refs):
                    frags, errs = [], []
                    for o_ in range(nO):
                        ch = choice.copy(); ch[sel] = o_
                        corners, geomN, geomC = set_cd(cd_of(ch))
                        vals, pix, t, prim = patch_samples(e_, scale=sc)
                        img = composite_vals(vals, pix, t, torch.zeros(Ws * Hs, 3, device=dev), Ws * Hs)
                        errs.append((img - ref).abs().mean(-1))
                        frags.append(pix.long() * P + prim.long())
                    pairs = torch.unique(torch.cat(frags))
                    pp, pr = pairs // P, pairs % P
                    for o_ in range(nO):
                        score[o_] += torch.zeros(len(ukey), device=dev, dtype=torch.float64).index_add_(
                            0, kp[pr], errs[o_][pp].double()).cpu().numpy()
                best = np.argmin(score, 0)
                keep = score[best, np.arange(len(ukey))] > (1 - SURF_VG_MARGIN) * score[SURF_VG_INIT]
                best[keep] = SURF_VG_INIT                     # hysteresis: change only for a clear win
                changed = int((best[sel] != choice[sel]).sum())
                choice[sel] = best[sel]
                print(f"  visual glue pass {ps_ + 1} class {c_}: {changed} corners changed, options "
                      f"{np.bincount(choice, minlength=nO).tolist()} [{time.time() - t_vg:.0f}s]", flush=True)
        cdisp = cd_of(choice)
        corners, geomN, geomC = set_cd(cdisp)
        centre = corners.mean(1)
        print(f"  visual glue: options {dict(zip(SURF_VG_JOINS, np.bincount(choice, minlength=nO).tolist()))} "
              f"(torn = 0) over {len(ukey)} corners", flush=True)
    if CULL > 0:
        # Drop cells that are invisible from every viewpoint in the head range: per viewpoint, composite in
        # depth order (after the mesh depth test) and take each cell's largest visible contribution
        # alpha * transmittance of everything in front; cells below CULL everywhere are removed.
        best = torch.zeros(P, device=dev)
        eyes_c = [(x, y, zz) for x in (-MOVE, 0.0, MOVE) for y in (-MOVE / 2, 0.0, MOVE / 2)
                  for zz in ((0.0, 0.4 * MOVE) if CULL_EYES == 18 else (0.0,))]     # CULL_EYES=9: no forward row
        _sec.clear(); sec("cull start")
        for eye in eyes_c:
            vals, pix, t, prim = patch_samples(eye, scale=CULL_SCALE, alpha_only=True)
            sec("cull: cast + sample")
            _, mdep_c = s2.mesh_layer(M, eye, round(W * CULL_SCALE), round(H * CULL_SCALE), f * CULL_SCALE)
            sec("cull: mesh layer")
            k = t < mdep_c[pix]
            vals, pix, t, prim = vals[k], pix[k], t[k], prim[k]
            o = torch.argsort(pix.double() * 1e7 + t.double())
            a, pix, prim = vals[o].clamp(0, 0.999), pix[o], prim[o]
            first = torch.ones(len(o), dtype=torch.bool, device=dev); first[1:] = pix[1:] != pix[:-1]
            seg = torch.cummax(torch.where(first, torch.arange(len(o), device=dev), 0), 0).values
            la = torch.log1p(-a).double(); cs = torch.cumsum(la, 0)
            contrib = (torch.exp(cs - la - (cs[seg] - la[seg])) * a).float()
            best.scatter_reduce_(0, prim, contrib, "amax")
            sec("cull: sort + composite")
            if eye == (0.0, 0.0, 0.0):
                prim0 = prim.clone()
            del vals, pix, t, prim, o, a, la, cs, contrib
        alive = best >= CULL
        if BAKE_TIMING:
            print("  [time]   " + ", ".join(f"{k_}: {v_:.1f}s" for k_, v_ in _sec.items() if k_ not in ("_t", "cull start")), flush=True)
        torch.cuda.empty_cache()
        npx = round(W * CULL_SCALE) * round(H * CULL_SCALE)
        print(f"  input view: {len(prim0) / npx:.2f} -> {alive[prim0].sum().item() / npx:.2f} layer fragments "
              f"per pixel after the mesh depth test", flush=True)
        fill = area.astype(np.float64)
        print(f"  culling: {int((~alive).sum())} of {P} cells ({(~alive).float().mean().item():.0%}, "
              f"{fill[~alive.cpu().numpy()].sum() / fill.sum():.0%} of cell area) never contribute >= {CULL} "
              f"from {len(eyes_c)} viewpoints within +-{MOVE} m -> dropped [{time.time() - t0:.0f}s]", flush=True)
        tick("culling")

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
                    if SURF:                                   # per covered texel: a piece covering part of its cell
                        cov = (t1[:, gb:cb - gb, gb:cb - gb, 3] > 1 / 255).float().mean((1, 2))   # (split runs)
                        d = d / cov.clamp(min=0.05)            # keeps its detail like a whole one
                    err[st + c0:st + c0 + len(t1)] = d.cpu().numpy()
                half_px[st:st + len(atb)] = (cb // ds) ** 2
                extra[st:st + len(atb)] = cb * cb - (cb // ds) ** 2
            err[~live] = -1                                          # culled cells: never upgraded, not budgeted
            if SEPA:                                  # a group: its planes' colour tiles + 1 alpha tile per 4 planes
                lv = np.nonzero(live)[0]
                _, ci = np.unique(rects[lv, :2], axis=0, return_inverse=True); ci = ci.reshape(-1)
                o_ = np.lexsort((cdisp[lv].mean(1), ci))                      # by cell, far first
                rk = np.zeros(len(lv), np.int64)
                first = np.r_[True, ci[o_][1:] != ci[o_][:-1]]
                st_ = np.maximum.accumulate(np.where(first, np.arange(len(lv)), 0))
                rk[o_] = np.arange(len(lv)) - st_                             # rank in its cell
                run = rk // SEPA_N
                _, gi = np.unique(np.stack([ci, run], 1), axis=0, return_inverse=True)
                gi = gi.reshape(-1); ng = gi.max() + 1
                gn = np.bincount(gi, minlength=ng)
                ntile = gn + -(-gn // 4)
                gerr = np.full(ng, -1.0); np.maximum.at(gerr, gi, err[lv])
                ghalf = np.zeros(ng); ghalf[gi] = half_px[lv]; gext = np.zeros(ng); gext[gi] = extra[lv]
                budget = 0.75 * MAXPAGES * page * page - (ghalf * ntile).sum()
                cand = np.argsort(-gerr)
                cand = cand[gerr[cand] > TEXFULL_THR]
                cand = cand[np.cumsum((gext * ntile)[cand]) <= budget]
                full[lv[np.isin(gi, cand)]] = True
            elif LCW and LCW_DATA:
                lv = np.nonzero(live)[0]
                _, gi = np.unique(rects[lv, :2], axis=0, return_inverse=True)
                gi = gi.reshape(-1); ng = gi.max() + 1
                gn = np.bincount(gi, minlength=ng)
                lab = LCW_DATA["lab"][lv]
                gl_ = np.unique(np.concatenate([np.stack([gi, lab[:, 0]], 1), np.stack([gi, lab[:, 1]], 1)]), axis=0)
                ntile = np.bincount(gl_[:, 0], minlength=ng) + -(-gn // 2)
                gerr = np.full(ng, -1.0); np.maximum.at(gerr, gi, err[lv])
                ghalf = np.zeros(ng); ghalf[gi] = half_px[lv]; gext = np.zeros(ng); gext[gi] = extra[lv]
                budget = 0.75 * MAXPAGES * page * page - (ghalf * ntile).sum()
                cand = np.argsort(-gerr)
                cand = cand[gerr[cand] > TEXFULL_THR]
                cand = cand[np.cumsum((gext * ntile)[cand]) <= budget]
                full[lv[np.isin(gi, cand)]] = True
            elif LCT and CELL_META:
                # LCT: a (cell, layer) group is one tile (a single piece) or a colour tile plus its pieces' alphas, 4 per
                # tile, and goes to full resolution as a whole: choose groups (worst member's error) at that cost
                meta = CELL_META[0]
                lv = np.nonzero(live)[0]
                _, gi = np.unique(np.stack([rects[lv, 0], rects[lv, 1], meta[lv, 0], meta[lv, 1]], 1), axis=0,
                                  return_inverse=True)
                gi = gi.reshape(-1); ng = gi.max() + 1
                gn = np.bincount(gi, minlength=ng)
                ntile = np.where(gn > 1, 1 + -(-gn // 4), 1)
                gerr = np.full(ng, -1.0); np.maximum.at(gerr, gi, err[lv])
                ghalf = np.zeros(ng); ghalf[gi] = half_px[lv]; gext = np.zeros(ng); gext[gi] = extra[lv]
                budget = 0.75 * MAXPAGES * page * page - (ghalf * ntile).sum()
                cand = np.argsort(-gerr)
                cand = cand[gerr[cand] > TEXFULL_THR]
                cand = cand[np.cumsum((gext * ntile)[cand]) <= budget]
                full[lv[np.isin(gi, cand)]] = True
            else:
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

        def unprem(t):                                                # (n, h, w, 4) premultiplied -> colour / alpha,
            if not UNPREM:                                            # spread into transparent texels
                return t
            a_ = t[..., 3:]
            c_ = torch.where(a_ > 1 / 255, t[..., :3] / a_.clamp(min=1e-6), 0); w_ = (a_ > 1 / 255).float()
            for _ in range(8):
                if bool(w_.all()):
                    break
                sm = torch.nn.functional.avg_pool2d((c_ * w_).permute(0, 3, 1, 2), 3, 1, 1, count_include_pad=False)
                sw = torch.nn.functional.avg_pool2d(w_.permute(0, 3, 1, 2), 3, 1, 1, count_include_pad=False)
                sm, sw = sm.permute(0, 2, 3, 1), sw.permute(0, 2, 3, 1)
                c_ = torch.where(w_ > 0, c_, sm / sw.clamp(min=1e-6)); w_ = torch.where(w_ > 0, w_, (sw > 0).float())
            return torch.cat([c_.clamp(0, 1), a_], -1)

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
            # live cells only (culled ones are never used; converting every tile to float was ~90% of this step)
            lj_all = np.nonzero(live[st:st + n_g])[0]
            for c0 in range(0, len(lj_all), 16384):
                lj = lj_all[c0:c0 + 16384]
                ljt = torch.tensor(lj, device=atb.device)
                t1 = atb[ljt].to(dev).float().clamp(0, 1)                         # (n, cb, cb, 4)
                th = torch.nn.functional.avg_pool2d(t1.permute(0, 3, 1, 2), ds).permute(0, 2, 3, 1) if ds > 1 else t1
                half_u8[lj] = to_np((unprem(th) * 255 + 0.5).byte())
                inner_h = th[:, g2:th.shape[1] - g2, g2:th.shape[2] - g2, 3] >= 0.5 / 255
                inner_f = t1[:, gb:cb - gb, gb:cb - gb, 3] >= 0.5 / 255
                fsel = torch.tensor(full[st + lj], device=dev)
                for inner, sel in ((inner_h, ~fsel), (inner_f, fsel)):
                    if not sel.any():
                        continue
                    r0, r1 = first_last(inner[sel].any(2))
                    k0, k1 = first_last(inner[sel].any(1))
                    ids = lj[torch.nonzero(sel)[:, 0].cpu().numpy()]
                    bbox[ids] = torch.stack([r0, r1, k0, k1], 1).cpu().numpy()
            full_j = np.nonzero(full[st:st + n_g] & live[st:st + n_g])[0]
            full_u8 = {}
            for c0 in range(0, len(full_j), 16384):                   # full-resolution cells only, batched
                ids = full_j[c0:c0 + 16384]
                full_u8.update(zip(ids, to_np((unprem(atb[torch.tensor(ids, device=atb.device)].to(dev).float().clamp(0, 1))
                                               * 255 + 0.5).byte())))
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
    tick("export: cell textures")
    shared = {}                                       # ADAPT_SHARE: representative cell -> [(member cell, sub-plane)]
    shared_na = {}                                    # its number of alpha tiles
    shared_nc = {}                                    # its number of colour tiles (SEPA: one per plane; else 1)
    band_ = {}                                        # SEPA: plane -> depth band (pages break between bands)
    if CELL_META and border is not None:
        meta = CELL_META[0]
        gidx = np.zeros(len(rects), np.int64); goff = np.zeros(len(rects), np.int64)
        for gi_, (st, atb) in enumerate(groups):
            gidx[st:st + len(atb)] = gi_; goff[st:st + len(atb)] = np.arange(len(atb))
        from collections import defaultdict
        fam = defaultdict(list)
        for i in list(texs):
            if meta[i, 0] == 1 or SHARE_COLOR or LCT:
                fam[(rects[i, 0], rects[i, 1], meta[i, 0], meta[i, 1])].append(i)
        nsh, saved = 0, 0
        for key_, mem in fam.items():
            if len(mem) < 2:
                continue
            mem = sorted(mem, key=lambda i: -meta[i, 2])           # nearest sub-bin first
            tb = torch.stack([groups[gidx[i]][1][goff[i]].float() for i in mem]).clamp(0, 1)   # (m, cb, cb, 4)
            col = torch.zeros_like(tb[0]); T_ = torch.ones_like(tb[0][..., :1])
            for t_ in tb:
                col = col + T_ * t_; T_ = T_ * (1 - t_[..., 3:])
            na_ = -(-len(mem) // 4)                         # alpha tiles: the occupied sub-planes, 4 per tile
            apk = torch.zeros(tb.shape[1], tb.shape[2] * na_, 4)
            w_ = tb.shape[2]
            slot_ = {i: n_ for n_, i in enumerate(sorted(mem, key=lambda i: meta[i, 2]))}   # far first
            for i, t_ in zip(mem, tb):
                s_ = slot_[i]
                apk[:, (s_ // 4) * w_:(s_ // 4 + 1) * w_, s_ % 4] = t_[..., 3]
            isfull = any(full[i] for i in mem)
            if not isfull and ds > 1:
                dn = lambda t: torch.nn.functional.avg_pool2d(t.permute(2, 0, 1)[None], ds)[0].permute(1, 2, 0)
                col, apk = dn(col), dn(apk)
            wt_ = col.shape[1]
            # colour stored un-premultiplied (8-bit premultiplied colour at small alpha is too coarse to divide back),
            # spread into transparent texels (no dark fringes from filtering); alpha comes from the alpha tiles
            ca_ = col[..., 3:]
            st_c = torch.where(ca_ > 1 / 255, col[..., :3] / ca_.clamp(min=1e-6), 0)
            wgt = (ca_ > 1 / 255).float()
            for _ in range(8):
                if bool(wgt.all()):
                    break
                sm = torch.nn.functional.avg_pool2d((st_c * wgt).permute(2, 0, 1)[None], 3, 1, 1, count_include_pad=False)[0].permute(1, 2, 0)
                sw = torch.nn.functional.avg_pool2d(wgt.permute(2, 0, 1)[None], 3, 1, 1, count_include_pad=False)[0].permute(1, 2, 0)
                st_c = torch.where(wgt > 0, st_c, sm / sw.clamp(min=1e-6))
                wgt = torch.where(wgt > 0, wgt, (sw > 0).float())
            col = torch.cat([st_c, ca_], -1)
            g_ = gb if isfull else g2
            sp_ = 1 if isfull else ds                     # render px per texel
            u8 = lambda t: (t.clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)
            inner = col[g_:col.shape[0] - g_, g_:col.shape[1] - g_, 3] >= 0.5 / 255
            if not inner.any():
                continue
            rr = torch.nonzero(inner.any(1))[:, 0]; cc_ = torch.nonzero(inner.any(0))[:, 0]
            r0, r1, k0, k1 = int(rr[0]), int(rr[-1]), int(cc_[0]), int(cc_[-1])
            crop = lambda t: t[r0:r1 + 1 + 2 * g_, k0:k1 + 1 + 2 * g_]      # trimmed to the content, with border
            rep_ = mem[0]
            saved += sum(texs[i].shape[0] * texs[i].shape[1] for i in mem)
            for i in mem:
                del texs[i]
            texs[rep_] = np.concatenate([u8(crop(col))] + [u8(crop(apk[:, t * wt_:(t + 1) * wt_])) for t in range(na_)], 1)
            pad[rep_] = g_
            x0_, y0_ = int(rects[rep_, 0]), int(rects[rep_, 1])
            geo[rep_] = (x0_ + k0 * sp_, y0_ + r0 * sp_, x0_ + (k1 + 1) * sp_, y0_ + (r1 + 1) * sp_)
            # each member's quad covers only its own content (in the tile's inner texels; None: border only, not drawn)
            mbox = {}
            for i in mem:
                s_ = slot_[i]
                am_ = apk[g_:apk.shape[0] - g_, (s_ // 4) * wt_ + g_:(s_ // 4 + 1) * wt_ - g_, s_ % 4][r0:r1 + 1, k0:k1 + 1]
                ri_, ci_ = torch.nonzero((am_ >= 0.5 / 255).any(1))[:, 0], torch.nonzero((am_ >= 0.5 / 255).any(0))[:, 0]
                mbox[i] = (int(ri_[0]), int(ri_[-1]), int(ci_[0]), int(ci_[-1])) if len(ri_) else None
            shared[rep_] = [(i, slot_[i], mbox[i], 0) for i in mem]
            shared_na[rep_] = na_
            saved -= texs[rep_].shape[0] * texs[rep_].shape[1]
            nsh += 1
        print(f"  shared textures: {nsh} refined cells, {sum(len(v) for v in shared.values())} sub-bin quads on "
              f"{nsh} double tiles ({saved / 1e6:.1f}M texels saved)", flush=True)
    if SEPA and border is not None:
        # batched: runs of SEPA_N depth-adjacent planes of a cell, gathered as (R, SEPA_N, cb, cb, 4) on the GPU
        assert len({atb.shape[1] for _, atb in groups}) == 1 and all(
            st == sum(len(t) for _, t in groups[:j]) for j, (st, _) in enumerate(groups)), "SEPA: one cell size"
        ALL = torch.cat([atb for _, atb in groups])                               # every plane's bordered tile
        ALLO = torch.cat([t for _, t in PAR_DATA["off"]]) if PAR_DATA else None   # PARALLAX: offsets, same order
        cb = ALL.shape[1]
        from collections import defaultdict
        fam = defaultdict(list)
        for i in list(texs):
            fam[(rects[i, 0], rects[i, 1])].append(i)
        cmean = cdisp.mean(1)
        # depth bands of SEPA_BAND atlas pages each (far first): a run never crosses a band and pages hold whole
        # bands, so every plane is drawn on the page of its place in the far-to-near order (a run spanning a page
        # break had its nearer planes drawn a page too early: holes). A first pass without bands measures each
        # plane's share of its run's tiles; the bands are cut by those (shrunk if one still overflows a page).
        far0 = [i for i in np.argsort(-centre[:, 2].cpu().numpy()) if i in texs]
        texs0, pad0, geo0 = dict(texs), dict(pad), dict(geo)
        bsize, cost = None, None
        for attempt in range(8):
            texs.clear(); texs.update(texs0); pad.clear(); pad.update(pad0); geo.clear(); geo.update(geo0)
            shared.clear(); shared_na.clear(); shared_nc.clear(); band_.clear()
            if cost is None:                      # measuring pass: one band
                band_.update((i, 0) for i in far0)
                cost_m = {i: t_.shape[0] * t_.shape[1] for i, t_ in texs.items()}
            else:
                cum = np.cumsum([cost[i] for i in far0])
                band_.update(zip(far0, (cum // (bsize * page * page)).astype(int)))
            runs = []
            for key_, mem in fam.items():
                mem = sorted(mem, key=lambda i: cmean[i])                 # far first
                j = 0
                while j < len(mem):
                    r_ = [mem[j]]
                    while len(r_) < SEPA_N and j + len(r_) < len(mem) and band_[mem[j + len(r_)]] == band_[mem[j]]:
                        r_.append(mem[j + len(r_)])
                    runs.append(r_); j += len(r_)
            N_ = SEPA_N
            for isfull in (False, True):
                rs = [r for r in runs if any(full[i] for i in r) == isfull]
                g_ = gb if isfull else g2
                sp_ = 1 if isfull else ds
                for c0 in range(0, len(rs), 4096):
                    ch_r = rs[c0:c0 + 4096]
                    idx = torch.tensor([r + [-1] * (N_ - len(r)) for r in ch_r])
                    ok_ = (idx >= 0).to(dev)
                    Tm = ALL[idx.clamp(min=0).reshape(-1)].to(dev).float().clamp(0, 1)            # (R*N, cb, cb, 4)
                    if not isfull and ds > 1:
                        Tm = torch.nn.functional.avg_pool2d(Tm.permute(0, 3, 1, 2), ds).permute(0, 2, 3, 1)
                    Tm = Tm * ok_.reshape(-1)[:, None, None, None]
                    if ALLO is not None:              # offsets (alpha-weighted when downsampled) -> 0..1
                        Om = ALLO[idx.clamp(min=0).reshape(-1)].to(dev).float()
                        if not isfull and ds > 1:
                            a0 = ALL[idx.clamp(min=0).reshape(-1)].to(dev).float()[..., 3].clamp(0, 1)
                            pa_ = torch.nn.functional.avg_pool2d(a0[:, None], ds)[:, 0]
                            Om = torch.where(pa_ > 1e-4, torch.nn.functional.avg_pool2d((Om * a0)[:, None], ds)[:, 0]
                                             / pa_.clamp(min=1e-4), 0)
                        Oe = (Om.clamp(-1, 1) * 0.5 + 0.5)[..., None]
                    H_, W_ = Tm.shape[1:3]
                    am = Tm[:, g_:H_ - g_, g_:W_ - g_, 3] >= 0.5 / 255                          # (R*N, h, w)
                    mr0, mr1 = first_last(am.any(2)); mc0, mc1 = first_last(am.any(1))           # each plane's box
                    un = am.reshape(len(ch_r), N_, *am.shape[1:]).any(1)                         # the run's union
                    r0, r1 = first_last(un.any(2)); k0, k1 = first_last(un.any(1))
                    if cost is None:                  # measuring pass: only each run's tile area, shared by its planes
                        bx = torch.stack([r0, r1, k0, k1], 1).cpu().numpy()
                        for j, mem in enumerate(ch_r):
                            if bx[j, 0] >= 0:
                                ar_ = ((bx[j, 1] - bx[j, 0] + 1 + 2 * g_) * (bx[j, 3] - bx[j, 2] + 1 + 2 * g_)
                                       * (len(mem) + -(-len(mem) // 4)))
                                cost_m.update((i, ar_ / len(mem)) for i in mem)
                        continue
                    a_ = Tm[..., 3:]                      # colour un-premultiplied, spread into transparent texels
                    Cu = torch.where(a_ > 1 / 255, Tm[..., :3] / a_.clamp(min=1e-6), 0); wg_ = (a_ > 1 / 255).float()
                    for _ in range(8):
                        sm = torch.nn.functional.avg_pool2d((Cu * wg_).permute(0, 3, 1, 2), 3, 1, 1, count_include_pad=False).permute(0, 2, 3, 1)
                        sw = torch.nn.functional.avg_pool2d(wg_.permute(0, 3, 1, 2), 3, 1, 1, count_include_pad=False).permute(0, 2, 3, 1)
                        Cu = torch.where(wg_ > 0, Cu, sm / sw.clamp(min=1e-6)); wg_ = torch.where(wg_ > 0, wg_, (sw > 0).float())
                    Cu8 = (torch.cat([Cu.clamp(0, 1), Oe if ALLO is not None else torch.ones_like(a_)], -1) * 255 + 0.5).byte()
                    Cu8 = to_np(Cu8.reshape(len(ch_r), N_, H_, W_, 4))
                    A8 = to_np((a_[..., 0] * 255 + 0.5).byte().reshape(len(ch_r), N_, H_, W_))
                    mb_ = torch.stack([mr0, mr1, mc0, mc1], 1).reshape(len(ch_r), N_, 4).cpu().numpy()
                    bx = torch.stack([r0, r1, k0, k1], 1).cpu().numpy()
                    for j, mem in enumerate(ch_r):
                        R0, R1, K0, K1 = (int(v) for v in bx[j])
                        if R0 < 0:
                            continue
                        sl = (slice(R0, R1 + 1 + 2 * g_), slice(K0, K1 + 1 + 2 * g_))
                        m = len(mem); na_ = -(-m // 4)
                        apk = np.zeros((na_, R1 - R0 + 1 + 2 * g_, K1 - K0 + 1 + 2 * g_, 4), np.uint8)
                        for s_ in range(m):
                            apk[s_ // 4, ..., s_ % 4] = A8[j, s_][sl]
                        rep_ = mem[0]
                        for i in mem:
                            del texs[i]
                        texs[rep_] = np.concatenate([Cu8[j, s_][sl] for s_ in range(m)] + list(apk), 1)
                        pad[rep_] = g_
                        geo[rep_] = (int(rects[rep_, 0]) + K0 * sp_, int(rects[rep_, 1]) + R0 * sp_,
                                     int(rects[rep_, 0]) + (K1 + 1) * sp_, int(rects[rep_, 1]) + (R1 + 1) * sp_)
                        shared[rep_] = [(i, s_, (int(mb_[j, s_, 0]) - R0, int(mb_[j, s_, 1]) - R0, int(mb_[j, s_, 2]) - K0,
                                                 int(mb_[j, s_, 3]) - K0) if mb_[j, s_, 0] >= 0 else None, s_)
                                        for s_, i in enumerate(mem)]
                        shared_na[rep_] = na_; shared_nc[rep_] = m
                    del Tm, Cu, wg_, a_
            if cost is None:                      # each plane's share of its run's tiles (plain tiles: their own)
                cost = cost_m
                bsize = SEPA_BAND
                continue
            barea = defaultdict(int)
            for i, t_ in texs.items():
                barea[band_[i]] += t_.shape[0] * t_.shape[1]
            worst = max(barea.values()) / (page * page)
            if worst <= 0.95:
                break
            print(f"  SEPA: a band of {bsize:.2f} pages needs {worst:.2f} pages of tiles; smaller bands", flush=True)
            bsize *= 0.92 / worst
        print(f"  SEPA: {len(shared)} groups, {sum(len(v) for v in shared.values())} planes", flush=True)
    lw = {}                                           # LCW: representative cell -> (n layers, [(member, slot, box, iA, iB)])
    if LCW_DATA and border is not None:
        lab = LCW_DATA["lab"]
        wmap = dict(LCW_DATA["wt"])
        gidx = np.zeros(len(rects), np.int64); goff = np.zeros(len(rects), np.int64)
        for gi_, (st, atb) in enumerate(groups):
            gidx[st:st + len(atb)] = gi_; goff[st:st + len(atb)] = np.arange(len(atb))
        from collections import defaultdict
        fam = defaultdict(list)
        for i in list(texs):
            fam[(rects[i, 0], rects[i, 1])].append(i)
        cmean = cdisp.mean(1)
        u8 = lambda t: (t.clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)
        p2 = lambda t: torch.nn.functional.avg_pool2d(t[:, None], ds)[:, 0]
        for key_, mem in fam.items():
            mem = sorted(mem, key=lambda i: cmean[i])                 # far first
            isfull = any(full[i] for i in mem)
            Tm = torch.stack([groups[gidx[i]][1][goff[i]].float() for i in mem]).clamp(0, 1)      # (m, cb, cb, 4)
            A = Tm[..., 3]
            Wt = torch.stack([wmap[groups[gidx[i]][0]][goff[i]].float() for i in mem])
            lays = sorted({int(l) for i in mem for l in lab[i]})
            x0_, y0_ = int(rects[mem[0], 0]), int(rects[mem[0], 1])
            # each layer's colour: the composite (near to far) of the cell's planes' texels in that layer
            lt = torch.stack([torch.where(Wt[s_] > 0.5, float(lays.index(int(lab[i, 1]))), float(lays.index(int(lab[i, 0]))))
                              for s_, i in enumerate(mem)])
            C = torch.zeros(len(lays), *A.shape[1:], 4)
            for j_ in range(len(lays)):
                T_ = torch.ones_like(A[0])[..., None]
                for s_ in reversed(range(len(mem))):
                    t_ = Tm[s_] * (lt[s_] == j_)[..., None]
                    C[j_] += T_ * t_; T_ = T_ * (1 - t_[..., 3:])
            if not isfull and ds > 1:
                Ad = p2(A)
                Wt = torch.where(Ad > 1e-4, p2(A * Wt) / Ad.clamp(min=1e-4), p2(Wt))   # alpha-weighted
                A = Ad
                C = torch.nn.functional.avg_pool2d(C.permute(0, 3, 1, 2), ds).permute(0, 2, 3, 1)
            ca_ = C[..., 3:]                          # un-premultiplied, spread into transparent texels
            Cu = torch.where(ca_ > 1 / 255, C[..., :3] / ca_.clamp(min=1e-6), 0); wg_ = (ca_ > 1 / 255).float()
            for _ in range(8):
                if bool(wg_.all()):
                    break
                sm = torch.nn.functional.avg_pool2d((Cu * wg_).permute(0, 3, 1, 2), 3, 1, 1, count_include_pad=False).permute(0, 2, 3, 1)
                sw = torch.nn.functional.avg_pool2d(wg_.permute(0, 3, 1, 2), 3, 1, 1, count_include_pad=False).permute(0, 2, 3, 1)
                Cu = torch.where(wg_ > 0, Cu, sm / sw.clamp(min=1e-6)); wg_ = torch.where(wg_ > 0, wg_, (sw > 0).float())
            C = Cu
            g_ = gb if isfull else g2
            sp_ = 1 if isfull else ds
            inner = (A[:, g_:A.shape[1] - g_, g_:A.shape[2] - g_] >= 0.5 / 255).any(0)
            if not inner.any():
                continue
            rr = torch.nonzero(inner.any(1))[:, 0]; cc_ = torch.nonzero(inner.any(0))[:, 0]
            r0, r1, k0, k1 = int(rr[0]), int(rr[-1]), int(cc_[0]), int(cc_[-1])
            A, Wt = A[:, r0:r1 + 1 + 2 * g_, k0:k1 + 1 + 2 * g_], Wt[:, r0:r1 + 1 + 2 * g_, k0:k1 + 1 + 2 * g_]
            C = C[:, r0:r1 + 1 + 2 * g_, k0:k1 + 1 + 2 * g_]
            aw = torch.zeros(-(-len(mem) // 2), A.shape[1], A.shape[2], 4)
            mbox = []
            for s_, i in enumerate(mem):
                aw[s_ // 2, ..., 2 * (s_ % 2)] = A[s_]; aw[s_ // 2, ..., 2 * (s_ % 2) + 1] = Wt[s_]
                am_ = A[s_, g_:A.shape[1] - g_, g_:A.shape[2] - g_] >= 0.5 / 255
                ri_, ci_ = torch.nonzero(am_.any(1))[:, 0], torch.nonzero(am_.any(0))[:, 0]
                mbox.append((int(ri_[0]), int(ri_[-1]), int(ci_[0]), int(ci_[-1])) if len(ri_) else None)
            rep_ = mem[0]
            for i in mem:
                del texs[i]
            texs[rep_] = np.concatenate([u8(torch.cat([c_, torch.ones_like(c_[..., :1])], -1)) for c_ in C]
                                        + [u8(t_) for t_ in aw], 1)
            pad[rep_] = g_
            geo[rep_] = (x0_ + k0 * sp_, y0_ + r0 * sp_, x0_ + (k1 + 1) * sp_, y0_ + (r1 + 1) * sp_)
            lw[rep_] = (len(lays), [(i, s_, mbox[s_], lays.index(int(lab[i, 0])), lays.index(int(lab[i, 1])))
                                    for s_, i in enumerate(mem)])
        print(f"  LCW: {len(lw)} cells, {sum(len(v[1]) for v in lw.values())} planes, "
              f"{sum(v[0] for v in lw.values())} colour tiles", flush=True)
    lq = {}                                           # LCOL: representative cell -> [(member, tile, channel)]
    if LCOL_DATA and border is not None:
        nbin = LCOL_DATA["nbin"]
        gidx = np.zeros(len(rects), np.int64); goff = np.zeros(len(rects), np.int64)
        for gi_, (st, atb) in enumerate(groups):
            gidx[st:st + len(atb)] = gi_; goff[st:st + len(atb)] = np.arange(len(atb))
        from collections import defaultdict
        fam = defaultdict(list)
        for i in list(texs):
            fam[(rects[i, 0], rects[i, 1])].append(i)
        cmean = cdisp.mean(1)
        for key_, mem in fam.items():
            mem = sorted(mem, key=lambda i: cmean[i])              # far first
            isfull = any(full[i] for i in mem)
            tb = torch.stack([groups[gidx[i]][1][goff[i]][..., 3].float() for i in mem]).clamp(0, 1)
            if not isfull and ds > 1:
                tb = torch.nn.functional.avg_pool2d(tb[:, None], ds)[:, 0]
            nt = -(-len(mem) // 4)
            tiles = torch.zeros(nt, tb.shape[1], tb.shape[2], 4)
            for n_, t_ in enumerate(tb):
                tiles[n_ // 4, ..., n_ % 4] = t_
            rep_ = mem[0]
            for i in mem:
                del texs[i]
            texs[rep_] = (torch.cat(list(tiles), 1).numpy() * 255 + 0.5).astype(np.uint8)
            pad[rep_] = gb if isfull else g2
            geo[rep_] = tuple(int(v) for v in rects[rep_])
            lq[rep_] = [(i, n_ // 4, n_ % 4) for n_, i in enumerate(mem)]
        print(f"  colour layers: {len(lq)} cell columns, {sum(len(v) for v in lq.values())} planes as alpha", flush=True)
    tick("export: SEPA tiles")
    far_ = np.argsort(-centre[:, 2].cpu().numpy())
    order = [i for i in far_ if i in texs]   # far first
    rank = np.empty(len(far_), np.int64); rank[far_] = np.arange(len(far_))      # every plane's place, far first

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
    bstart = [j for j in range(len(order)) if j == 0 or band_.get(order[j]) != band_.get(order[j - 1])] if band_ else []
    while start < len(order):
        end, acc = start, 0
        while end < len(order):
            h, w = texs[order[end]].shape[:2]
            if acc + h * w > 0.95 * page * page:
                break
            acc += h * w; end += 1
        end = max(end, start + 1)
        if band_ and end < len(order):               # SEPA: break at a band boundary (whole bands per page)
            end = max([b for b in bstart if start < b <= end], default=end)
        while (pl := shelf(order[start:end])) is None:
            end_ = start + max(1, int((end - start) * 0.97))
            end = max([b for b in bstart if start < b <= end_], default=end_) if band_ else end_
        pages.append((order[start:end], pl)); start = end
    if shared:                                        # planes drawn on a page that comes before their place in the order
        nxt = [rank[ids_[0]] for ids_, _ in pages[1:]] + [len(rank)]
        bad = sum(1 for pg, (ids_, _) in enumerate(pages) for i in ids_ if i in shared
                  for m_, *_ in shared[i] if rank[m_] > nxt[pg])
        print(f"  shared tiles: {bad} of {sum(len(v) for v in shared.values())} planes past their page's end "
              f"in the far-to-near order", flush=True)
    # SURF_SKIRT: skirts between neighbouring pieces that meet at their shared cell edge (both have texels
    # there) at nearly one depth but aren't glued: a quad from one piece's edge to the other's, zero-width at the
    # input view, stretched over the crack that opens between them when the head moves (a mesh's behaviour)
    # instead of showing what is behind. Coloured with the farther piece's edge texels and drawn right after it.
    # Not across real depth edges: only where the pieces' depths differ by at most SURF_SKIRT parallax px at MOVE.
    skirts = {}                                       # farther piece -> [(vertical?, X, lo, hi, dF(lo,hi), dO(lo,hi), side)]
    if SURF and SURF_SKIRT and cdisp is not None and border is not None:
        from collections import defaultdict
        by_cell = defaultdict(list)
        for i in texs:
            by_cell[(rects[i, 0], rects[i, 1])].append(i)
        nsk = 0
        for i in texs:
            x0, y0, x1, y1 = rects[i]
            for vert, nkey in ((True, (x1, y0)), (False, (x0, y1))):
                best = None
                for j in by_cell.get(nkey, ()):
                    gi, gj = geo[i], geo[j]
                    if vert:
                        if gi[2] != x1 or gj[0] != x1:
                            continue
                        lo_, hi_, a0, a1 = max(gi[1], gj[1]), min(gi[3], gj[3]), y0, y1
                        ci_, cj_ = (cdisp[i, 1], cdisp[i, 3]), (cdisp[j, 0], cdisp[j, 2])
                    else:
                        if gi[3] != y1 or gj[1] != y1:
                            continue
                        lo_, hi_, a0, a1 = max(gi[0], gj[0]), min(gi[2], gj[2]), x0, x1
                        ci_, cj_ = (cdisp[i, 2], cdisp[i, 3]), (cdisp[j, 0], cdisp[j, 1])
                    if hi_ <= lo_:
                        continue
                    at = lambda c, v: c[0] + (v - a0) / (a1 - a0) * (c[1] - c[0])
                    di = np.array([at(ci_, lo_), at(ci_, hi_)]); dj = np.array([at(cj_, lo_), at(cj_, hi_)])
                    diff = np.abs(di - dj).max()
                    if diff < 1e-7 or diff * fg * MOVE > SURF_SKIRT:
                        continue
                    if best is None or diff < best[0]:
                        best = (diff, j, lo_, hi_, di, dj)
                if best is None:
                    continue
                _, j, lo_, hi_, di, dj = best
                X = x1 if vert else y1
                if di.mean() <= dj.mean():            # i is farther: its right / bottom edge
                    skirts.setdefault(i, []).append((vert, X, lo_, hi_, di, dj, "end"))
                else:
                    skirts.setdefault(j, []).append((vert, X, lo_, hi_, dj, di, "start"))
                nsk += 1
        print(f"  skirts: {nsk} between unglued neighbouring pieces within {SURF_SKIRT:g} px", flush=True)

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
        if skirts:                                    # skirt quads, each drawn right after its farther piece
            pos = {i: n_ for n_, i in enumerate(ids)}
            sV, sU, sK = [], [], []
            for i in ids:
                for vert, X, lo_, hi_, dF, dO, side in skirts.get(i, ()):
                    gx0, gy0, gx1, gy1 = geo[i]
                    tw_i, th_i = texs[i].shape[1] - 2 * pad[i], texs[i].shape[0] - 2 * pad[i]
                    sp_i = (gx1 - gx0) / max(tw_i, 1)
                    px_i, py_i = pl[i]
                    if vert:                          # edge x = X, from y = lo_ to hi_
                        pts = np.array([[X, lo_], [X, hi_], [X, hi_], [X, lo_]], np.float64)
                        u_ = px_i + (tw_i - 0.5 if side == "end" else 0.5)
                        vs = py_i + (np.array([lo_, hi_, hi_, lo_]) - gy0) / sp_i
                        uvq = np.stack([np.full(4, u_), vs], 1)
                    else:                             # edge y = X, from x = lo_ to hi_
                        pts = np.array([[lo_, X], [hi_, X], [hi_, X], [lo_, X]], np.float64)
                        v_ = py_i + (th_i - 0.5 if side == "end" else 0.5)
                        us = px_i + (np.array([lo_, hi_, hi_, lo_]) - gx0) / sp_i
                        uvq = np.stack([us, np.full(4, v_)], 1)
                    dq = np.maximum(np.array([dF[0], dF[1], dO[1], dO[0]]), 1e-3)
                    qu, qv = (pts[:, 0] - MARGIN) / SS, (pts[:, 1] - MARGIN) / SS
                    sV.append(np.stack([(qu - GW / 2) / fg, (qv - Hg / 2) / fg, np.ones(4)], -1) / dq[:, None])
                    sU.append(uvq / page); sK.append(pos[i] + 0.5)
            if sV:
                Vq = np.concatenate([V, np.stack(sV)]); UVq = np.concatenate([np.stack([uu, vv], -1), np.stack(sU)])
                o_ = np.argsort(np.r_[np.arange(len(ids)), np.array(sK)], kind="stable")
                V, uu, vv = Vq[o_], UVq[o_][..., 0], UVq[o_][..., 1]
        extra = None
        if lq:                                        # LCOL: one quad per plane: colour-page uv, alpha tile, bin
            V_l, U_l, V2_l, C_l = [], [], [], []
            cw_, ch_p = LCOL_DATA["page"].shape[1], LCOL_DATA["page"].shape[0]
            for n_, i in enumerate(ids):
                g_ = pad[i]; nt = max(t for _, t, _ in lq[i]) + 1
                tw_ = texs[i].shape[1] // nt; twi = tw_ - 2 * g_; thi = texs[i].shape[0] - 2 * g_
                px_i, py_i = pl[i]
                x0, y0, x1, y1 = (float(v) for v in rects[i])
                qu = (np.array([x0, x1, x1, x0]) - MARGIN) / SS; qv = (np.array([y0, y0, y1, y1]) - MARGIN) / SS
                uvc = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) / LCOL_DATA["ds"] / [cw_, ch_p]
                for m_, t, ch in lq[i]:
                    c00, c10, c01, c11 = cdisp[m_]
                    dq = np.maximum(np.array([c00, c10, c11, c01]), 1e-3)
                    V_l.append((np.stack([(qu - GW / 2) / fg, (qv - Hg / 2) / fg, np.ones(4)], -1) / dq[:, None])[None])
                    uva = (np.array([[px_i, py_i], [px_i + twi, py_i], [px_i + twi, py_i + thi], [px_i, py_i + thi]])
                           + [t * tw_, 0]) / page
                    U_l.append(uvc[None]); V2_l.append(uva[None])
                    C_l.append(np.stack([np.full(4, float(ch)), np.full(4, float(LCOL_DATA["nbin"][m_]))], -1)[None])
            V = np.concatenate(V_l); UV_ = np.concatenate(U_l); uu, vv = UV_[..., 0], UV_[..., 1]
            extra = (np.concatenate(V2_l).reshape(-1, 2).astype(np.float32),
                     np.concatenate(C_l).reshape(-1, 2).astype(np.float32))
        elif lw:                                        # LCW: one quad per plane: colour tiles, alpha/weight, layers
            V_l, U_l, V2_l, C_l, L_l = [], [], [], [], []
            for n_, i in enumerate(ids):
                if i not in lw:
                    V_l.append(V[n_][None]); U_l.append(np.stack([uu[n_], vv[n_]], -1)[None])
                    V2_l.append(np.zeros((1, 4, 2))); C_l.append(np.tile([-1.0, 0.0], (1, 4, 1))); L_l.append(np.zeros((1, 4, 2)))
                    continue
                nl_, mem_ = lw[i]
                x0, y0, x1, y1 = (float(v) for v in geo[i])
                fx0, fy0, fx1, fy1 = (float(v) for v in rects[i])
                g_ = pad[i]
                tw_ = texs[i].shape[1] // (nl_ + -(-len(mem_) // 2)); twc = tw_ - 2 * g_
                px_i, py_i = pl[i]
                gx0_, gy0_, gsp_ = x0, y0, (x1 - x0) / twc
                for m_, s_, mb_, iA, iB in mem_:
                    if mb_ is None:
                        continue
                    c00, c10, c01, c11 = cdisp[m_]
                    x0, y0, x1, y1 = (gx0_ + mb_[2] * gsp_, gy0_ + mb_[0] * gsp_, gx0_ + (mb_[3] + 1) * gsp_,
                                      gy0_ + (mb_[1] + 1) * gsp_)
                    uvc = np.array([[px_i + mb_[2], py_i + mb_[0]], [px_i + mb_[3] + 1, py_i + mb_[0]],
                                    [px_i + mb_[3] + 1, py_i + mb_[1] + 1], [px_i + mb_[2], py_i + mb_[1] + 1]]) / page
                    qu = (np.array([x0, x1, x1, x0]) - MARGIN) / SS; qv = (np.array([y0, y0, y1, y1]) - MARGIN) / SS
                    sxq = (np.array([x0, x1, x1, x0]) - fx0) / (fx1 - fx0)
                    tyq = (np.array([y0, y0, y1, y1]) - fy0) / (fy1 - fy0)
                    dq = np.where(tyq <= sxq, c00 + sxq * (c10 - c00) + tyq * (c11 - c10),
                                  c00 + tyq * (c01 - c00) + sxq * (c11 - c01))
                    dq = np.maximum(dq, 1e-3)
                    V_l.append((np.stack([(qu - GW / 2) / fg, (qv - Hg / 2) / fg, np.ones(4)], -1) / dq[:, None])[None])
                    U_l.append(uvc[None]); V2_l.append((uvc + np.array([(nl_ + s_ // 2) * tw_, 0]) / page)[None])
                    C_l.append(np.tile([float(s_ % 2), float(iA)], (1, 4, 1)))
                    L_l.append(np.tile([float(iB), tw_ / page], (1, 4, 1)))
            V = np.concatenate(V_l); UV_ = np.concatenate(U_l); uu, vv = UV_[..., 0], UV_[..., 1]
            extra = tuple(np.concatenate(x_).reshape(-1, 2).astype(np.float32) for x_ in (V2_l, C_l, L_l))
        elif shared:                                    # ADAPT_SHARE: one quad per sub-bin of a shared tile
            V_l, U_l, V2_l, C_l, K_l, D_l = [], [], [], [], [], []   # K: the quad's plane's place in the far-to-near order
            pdk = PAR_DATA["dk"] if PAR_DATA else None                   # D: its plane's thickness (PARALLAX; else 0)
            for n_, i in enumerate(ids):
                if i not in shared:
                    V_l.append(V[n_][None]); U_l.append(np.stack([uu[n_], vv[n_]], -1)[None])
                    V2_l.append(np.zeros((1, 4, 2))); C_l.append(np.full((1, 4), -1.0)); K_l.append(rank[i]); D_l.append(0.0)
                    continue
                x0, y0, x1, y1 = (float(v) for v in geo[i])
                fx0, fy0, fx1, fy1 = (float(v) for v in rects[i])
                g_ = pad[i]
                nc_ = shared_nc.get(i, 1)
                twc = texs[i].shape[1] // (nc_ + shared_na[i]) - 2 * g_; thc = texs[i].shape[0] - 2 * g_
                px_i, py_i = pl[i]
                gx0_, gy0_, gsp_ = x0, y0, (x1 - x0) / twc  # the tile's content origin, render px per texel
                for m_, ch_, mb_, ci_ in sorted(shared[i], key=lambda t: t[1]):   # farther sub-bins first
                    if mb_ is None:
                        continue
                    c00, c10, c01, c11 = cdisp[m_]
                    x0, y0, x1, y1 = (gx0_ + mb_[2] * gsp_, gy0_ + mb_[0] * gsp_, gx0_ + (mb_[3] + 1) * gsp_,
                                      gy0_ + (mb_[1] + 1) * gsp_)
                    uvc = np.array([[px_i + mb_[2], py_i + mb_[0]], [px_i + mb_[3] + 1, py_i + mb_[0]],
                                    [px_i + mb_[3] + 1, py_i + mb_[1] + 1], [px_i + mb_[2], py_i + mb_[1] + 1]]) / page
                    qu = (np.array([x0, x1, x1, x0]) - MARGIN) / SS; qv = (np.array([y0, y0, y1, y1]) - MARGIN) / SS
                    sxq = (np.array([x0, x1, x1, x0]) - fx0) / (fx1 - fx0)
                    tyq = (np.array([y0, y0, y1, y1]) - fy0) / (fy1 - fy0)
                    dq = np.where(tyq <= sxq, c00 + sxq * (c10 - c00) + tyq * (c11 - c10),
                                  c00 + tyq * (c01 - c00) + sxq * (c11 - c01))
                    dq = np.maximum(dq, 1e-3)
                    V_l.append((np.stack([(qu - GW / 2) / fg, (qv - Hg / 2) / fg, np.ones(4)], -1) / dq[:, None])[None])
                    uva = uvc + np.array([(nc_ + ch_ // 4) * (twc + 2 * g_), 0]) / page
                    uvq = uvc + np.array([ci_ * (twc + 2 * g_), 0]) / page   # its colour tile (SEPA: its own)
                    U_l.append(uvq[None]); V2_l.append(uva[None]); C_l.append(np.full((1, 4), float(ch_ % 4)))
                    K_l.append(rank[m_]); D_l.append(float(pdk[m_]) if pdk is not None else 0.0)
            # each quad at its own plane's place (a group's planes span depths: drawn together at the group's place,
            # nearer ones would come before other cells' planes between them, which then cover them at moved views)
            o_ = np.argsort(np.array(K_l), kind="stable")
            V_l, U_l, V2_l, C_l, D_l = ([x_[j] for j in o_] for x_ in (V_l, U_l, V2_l, C_l, D_l))
            V = np.concatenate(V_l); UV_ = np.concatenate(U_l); uu, vv = UV_[..., 0], UV_[..., 1]
            extra = (np.concatenate(V2_l).reshape(-1, 2).astype(np.float32),
                     np.stack([np.concatenate(C_l).reshape(-1), np.repeat(np.array(D_l), 4)], 1).astype(np.float32))
        if GAPCOPY_DATA and extra is None:            # GAPCOPY: a copy of the tile on the connected farther plane's sheet
            conn = GAPCOPY_DATA["conn"]
            cid = np.array([n_ for n_, i in enumerate(ids) if conn[i] >= 0], np.int64)
            if len(cid):
                ii = np.array(ids)[cid]; jj = conn[ii]
                fr = rects[ii].astype(np.float64); rr = r[cid]
                sx = (np.stack([rr[:, 0], rr[:, 2], rr[:, 2], rr[:, 0]], 1) - fr[:, :1]) / (fr[:, 2:3] - fr[:, :1])
                ty = (np.stack([rr[:, 1], rr[:, 1], rr[:, 3], rr[:, 3]], 1) - fr[:, 1:2]) / (fr[:, 3:4] - fr[:, 1:2])
                c00, c10, c01, c11 = (cdisp[jj, j][:, None] for j in range(4))
                dC = np.maximum(np.where(ty <= sx, c00 + sx * (c10 - c00) + ty * (c11 - c10),
                                         c00 + ty * (c01 - c00) + sx * (c11 - c01)), 1e-3)
                VC = np.stack([(cu[cid] - GW / 2) / fg, (cv[cid] - Hg / 2) / fg, np.ones_like(dC)], -1) / dC[..., None]
                o_ = np.argsort(np.r_[rank[np.array(ids)], rank[jj] + 0.5], kind="stable")   # right after plane k
                V = np.concatenate([V, VC])[o_]
                uu = np.concatenate([uu, uu[cid]])[o_]; vv = np.concatenate([vv, vv[cid]])[o_]
        faces = (np.arange(len(V))[:, None, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]])[None]).reshape(-1, 3)
        # glTF uv: origin at the top-left of the image (no flip)
        uv = np.stack([uu.reshape(-1), vv.reshape(-1)], 1).astype(np.float32)
        # The viewer draws surface_<k> with larger k first (k = 0 is the front).
        meshes.append((f"surface_{len(pages) - 1 - pg}" + ("_u" if UNPREM else ""), V.reshape(-1, 3).astype(np.float32),
                       faces.astype(np.uint32), uv, img) + ((extra,) if extra is not None else ()))
    if LDI_FRONT or lq or (LCT and shared) or lw:
        # LDI / LCOL / LCT: the front and the MPI drawn in one far-to-near order (no depth test in the viewer): both split into
        # LDI_GROUPS depth groups (quantiles of the triangles' distance), drawn group by group, far first, the
        # MPI's part of a group before the front's. Meshes of one page share its texture (an int image ref).
        tris = []                                     # (distance, kind, mesh source, triangle)
        groups_src = []
        for mi, (n_, V_, F_, UV_, im_, *ex_) in enumerate(meshes):
            groups_src.append(("mpi", V_, F_, UV_, im_, ex_[0] if ex_ else None))
        for V_, F_, UV_, tex_ in LDI_FRONT:
            hh, ww = tex_.shape[:2]                   # pad the texture to a multiple of 4 (block compression)
            tex_ = np.pad(tex_, ((0, -hh % 4), (0, -ww % 4), (0, 0)))
            UV_ = UV_ * np.array([ww / tex_.shape[1], hh / tex_.shape[0]], np.float32)
            groups_src.append(("front", V_, F_, UV_, tex_, None))
        dist = [np.linalg.norm(V_[F_].mean(1), axis=1) for _, V_, F_, _, _, _ in groups_src]
        alld = np.concatenate(dist)
        edges_g = np.quantile(1 / alld, np.linspace(0, 1, LDI_GROUPS + 1)[1:-1])   # in disparity: near groups finer
        out_m = []
        img_of = {}
        for gk in range(LDI_GROUPS):                  # far (small disparity) first
            for kind in ("mpi", "front"):
                for si, (kd, V_, F_, UV_, im_, ex_) in enumerate(groups_src):
                    if kd != kind:
                        continue
                    gsel = np.searchsorted(edges_g, 1 / dist[si]) == gk
                    if not gsel.any():
                        continue
                    Fs = F_[gsel]
                    Fs = Fs[np.argsort(-dist[si][gsel], kind="stable")]
                    used, inv = np.unique(Fs, return_inverse=True)
                    ref = img_of.setdefault(si, len(out_m))
                    out_m.append([V_[used], inv.reshape(-1, 3).astype(np.uint32), UV_[used],
                                  im_ if ref == len(out_m) else ref]
                                 + ([tuple(e_[used] for e_ in ex_)] if ex_ is not None else []))
        meshes = [(f"surface_{len(out_m) - 1 - i}" + ("_u" if UNPREM else ""), *m) for i, m in enumerate(out_m)]
    if lq:                                            # the colour page: a hidden mesh carrying the texture
        pg_ = LCOL_DATA["page"]
        meshes.append((f"ldicolor_{LCOL_DATA['w']}_{LCOL_DATA['h']}", np.zeros((3, 3), np.float32),
                       np.array([[0, 1, 2]], np.uint32), np.zeros((3, 2), np.float32), pg_))
    tick("export: atlas pages, quads")
    path = (Path(EXPORT_DIR) if EXPORT_DIR else scene.parent) / f"{scene.stem}{EXPORT}.glb"
    quad_px = sum((geo[i][2] - geo[i][0]) * (geo[i][3] - geo[i][1]) for i in texs)
    n_tex, t0_ = len(texs), T0

    def finish():                                     # CPU only: PNG / KTX2 encoding and the files
        write_layer_glbs(path, meshes, ktx2=EXPORT_KTX2)
        print(f"  quads cover {quad_px / (GW * SS * Hg * SS):.1f}x the frame (fill before blending/depth test)", flush=True)
        print(f"  [{time.time() - t0_:.0f}s] wrote {path} ({path.stat().st_size / 2**20:.0f} MB, {len(pages)} atlas pages of {page}^2, "
              f"{n_tex} patches, {texels / 1e6:.0f}M texels = {texels / (len(pages) * page * page):.0%} of the pages)",
              flush=True)
    if WRITER is None:
        finish()
    else:                                             # --serve: written in the background, next scene meanwhile
        WRITER.submit(finish)


def write_layer_glbs(path, meshes, ktx2=False):
    """Write the layer meshes [(name, V, faces, uv, page RGBA uint8)] as <path> (PNG pages, fast compression) and,
    with ktx2, also <stem>_ktx2.glb (Basis UASTC KTX2 pages with mipmaps, KHR_texture_basisu; see glb_ktx2.py),
    encoding the pages in parallel."""
    import io, tempfile
    from concurrent.futures import ThreadPoolExecutor
    from glb_ktx2 import write_glb, encode_pages

    def png(img, level):
        b = io.BytesIO(); Image.fromarray(img, "RGBA").save(b, format="PNG", compress_level=level); return b.getvalue()
    # a mesh's image may be an int: the index of an earlier mesh whose image it shares
    owner = [i if not isinstance(m[4], (int, np.integer)) else int(m[4]) for i, m in enumerate(meshes)]
    uniq = sorted(set(owner))
    slot = {o: k for k, o in enumerate(uniq)}
    with ThreadPoolExecutor(len(uniq) or 1) as ex:
        pngs = list(ex.map(lambda o: png(meshes[o][4], 1), uniq))
        tick("export: PNG pages")

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
        for k_, im in enumerate(images):
            g["images"].append({"bufferView": view(im), "mimeType": mime})
            g["textures"].append({"extensions": {"KHR_texture_basisu": {"source": k_}}} if basisu else {"source": k_})
        for k, (name, V, F, uv, _, *ext) in enumerate(meshes):
            pos = accessor(V, 5126, "VEC3", 34962, minmax=True)
            tex = accessor(uv, 5126, "VEC2", 34962)
            idx = accessor(F.reshape(-1), 5125, "SCALAR", 34963)
            attrs = {"POSITION": pos, "TEXCOORD_0": tex}
            if ext:                                   # ADAPT_SHARE / LCT: alpha tile uv, channel; LCW: + layers
                for j_, e_ in enumerate(ext[0]):
                    attrs[f"TEXCOORD_{j_ + 1}"] = accessor(e_, 5126, "VEC2", 34962)
            ti = slot[owner[k]]
            g["materials"].append({"name": f"patches_{k}", "doubleSided": True, "alphaMode": "BLEND",
                                   "pbrMetallicRoughness": {"baseColorTexture": {"index": ti},
                                                            "metallicFactor": 0.0, "roughnessFactor": 1.0}})
            g["meshes"].append({"name": name, "primitives": [{"attributes": attrs,
                                                               "indices": idx, "material": k}]})
            g["nodes"].append({"name": name, "mesh": k})
        if basisu:
            g["extensionsUsed"] = g["extensionsRequired"] = ["KHR_texture_basisu"]
        return g, views
    g, views = build(pngs, "image/png", False)
    write_glb(path, g, views)
    if ktx2:
        raw = pngs                                                    # the PNG pages just written, as ktx input
        for suffix, on, kw in (("_astc", KTX2_ASTC not in ("", "0"), dict(astc=KTX2_ASTC)), ("_ktx2", KTX2_UASTC, {})):
            if not on:
                continue
            with tempfile.TemporaryDirectory() as td:
                kt = encode_pages(raw, td, levels=KTX2_LEVELS, zstd=KTX2_ZSTD, **kw)
            tick(f"export: KTX2 encode ({suffix[1:]})")
            g, views = build(kt, "image/ktx2", True)
            write_glb(Path(path).with_name(Path(path).stem + suffix + ".glb"), g, views)


class _ThreadOut:
    """sys.stdout / sys.stderr for --serve: each thread prints to its own target (the scene it works on)."""
    def __init__(self, default):
        self.default, self.local = default, __import__("threading").local()

    def set(self, f):
        self.local.f = f

    def write(self, x):
        return getattr(self.local, "f", self.default).write(x)

    def flush(self):
        getattr(self.local, "f", self.default).flush()


class _Writer:
    """--serve: runs each scene's file writing (finish() of export_glb) on a background thread; results in order."""
    def __init__(self, out):
        from concurrent.futures import ThreadPoolExecutor
        self.pool, self.out, self.job = ThreadPoolExecutor(1), out, None

    def submit(self, fn):
        f = self.out.local.f                          # this scene's log
        def run_():
            self.out.set(f)
            fn()
        self.job = self.pool.submit(run_)


WRITER = None                                     # --serve: a _Writer (export_glb hands the file writing to it)


def serve():
    """--serve: bake scenes read from stdin, one per line ("<scene.ply>\t<log file>"), in this one process (no
    start-up per scene: Python, torch, CUDA and gsplat load once). Each scene's output goes to its log. Per scene,
    stdout gets "gpu-done <ply>" once its GPU work is done (the next scene can start: its files are written in the
    background meanwhile), then "ok <ply>" when its files are written, or "fail <ply> <error>". Settings come from
    the environment, as for a single run."""
    global WRITER
    import threading, traceback
    real = sys.stdout
    sys.stdout = sys.stderr = tl = _ThreadOut(real)
    WRITER = _Writer(tl)
    lock = threading.Lock()

    def say(msg):
        with lock:
            real.write(msg + "\n"); real.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        ply, _, log = line.rstrip("\n").partition("\t")
        lf = open(log or os.devnull, "a")
        tl.set(lf)
        WRITER.job = None
        try:
            run(Path(ply))
        except Exception as e:
            traceback.print_exc()
            tl.set(real); lf.close()
            say(f"fail {ply} {type(e).__name__}: {str(e)[:200]}")
            continue
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        tl.set(real)
        job = WRITER.job
        say(f"gpu-done {ply}")

        def done_(j, ply=ply, lf=lf):
            e = j.exception() if j is not None else None
            if e is not None:
                lf.write("".join(traceback.format_exception(e)))
            lf.close()
            say(f"ok {ply}" if e is None else f"fail {ply} {type(e).__name__}: {str(e)[:200]}")
        if job is None:
            done_(None)
        else:
            job.add_done_callback(done_)
    WRITER.pool.shutdown(wait=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--serve"]:
        serve()
    else:
        for s in sys.argv[1:]:
            run(Path(s))
