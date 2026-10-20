# 360 panoramas → SHARP splats (towards an MPI-only 360 view)

Goal: an MPI-only-style headset view for equirectangular 360 photos. Step 1
(this doc) is getting a consistent 360 Gaussian scene out of SHARP, which
only takes perspective photos.

## Pipeline (`scripts/pano_predict.py`)

1. Split the panorama into overlapping perspective views at SHARP's 1536 px
   input: `--layout ico` (default) = 20 views on the icosahedron face
   centres at 75° FOV; `cube` = 6 views at 100°.
2. **Pass 1:** SHARP monodepth (front layer) per view → one 360 distance
   map. Per-view log-scale solved by least squares from pairwise weighted
   medians on the overlaps, weights = disparity (sky has no usable depth),
   plus a weak prior towards scale 1. Then a feathered blend in log-distance.
3. **Pass 2:** full SHARP per view with the stitched depth (as z-depth) passed
   as `depth` to the predictor. The checkpoint ships trained
   `depth_alignment` weights (a UNet that turns monodepth + target depth into
   a smooth scale map), so every view's Gaussians follow the shared depth.
4. Gaussians are rotated to the world frame (OpenCV axes, z = pano centre
   column, camera at origin) and each is kept only by the view whose axis is
   nearest its direction (Voronoi ownership; no blending).

Outputs in `-o`: `<name>.ply` (plain 3DGS vertex ply, sRGB like SHARP's
export), `<name>.json`, `<name>_dist.npy` / `_disp.png` (stitched distance).
~4–5 min for `ico`, ~80 s for `cube` on one RTX 3090.

Evaluation: `scripts/pano_render.py <ply> --pano <jpg>` renders equirects
and seam-straddling perspective views at the centre and moved 0.3 m
(right/forward/up), and reports PSNR vs the panorama.

## Results (Poly Haven CC0 test panos in `pano/inputs/`)

| Scene | Layout | Splats | PSNR equirect 2048 | PSNR 60° views (~17 px/deg) |
|---|---|---|---|---|
| hotel_room (indoor) | cube | 4.97M | 29.6 | 34.6 |
| hotel_room | ico | 7.56M | 28.6 | **39.4** |
| abandoned_parking (outdoor) | cube | 4.97M | 30.8 | 30.2 |
| abandoned_parking | ico | 7.59M | 29.6 | **32.5** |

- **Pass-2 alignment is required.** Without it, views disagree in depth by
  ~5–15% and cracks open at view seams as soon as the eye moves (black gap
  along the bed at 0.3 m); with it, there are no visible seams.
- **ico beats cube on detail** (~2× angular resolution). The coarse equirect
  PSNR prefers cube only because tiny splats alias at that render size; use
  the 60° high-res metric.
- A constant scale per view leaves ~4–7% (median) log-depth disagreement on
  overlaps; the blend + learned alignment hide it, but depth maps show faint
  arcs on ground planes at view boundaries with `cube`.
- Outdoor: sky-only overlaps made the unweighted solve give a view 5× scale;
  disparity weighting + prior fixed it (scales 0.74–1.46).
- Known weak spot: content straight below (nadir floor/carpet) looks streaky
  in all variants.

## Multi-sphere image (MSI) — MPI-only for 360 (`scripts/pano_msi.py`)

Splat counts (5–7.5M) are far beyond the Quest budget (~390k for 72 fps), so
the headset representation is the 360 analogue of MPI-only, built like the
sparse MPI (findings §4b):

- 128 concentric spheres uniform in disparity (0.1–99.9 percentile of the
  splats' 1/distance). Seen from inside the innermost sphere (~0.9–1 m on the
  test scenes), far-to-near is always the right order: static draw order.
- Each bin's splats rendered together from the centre into a cube map (1024
  px faces at 2× supersampling), cut into 16-px cells, empty cells dropped;
  corners on the sphere, refined to the alpha-weighted splat distance
  (clamped to the bin; face-border corners stay at the bin centre so faces
  meet without cracks).
- Same anti-aliasing setup as the MPI: half-resolution cells unless they lose
  detail (within 8 pages of 4096²), 4-texel borders, 4-texel-aligned square
  cells (mips never mix cells), KTX2 UASTC with 3 box mips (`glb_ktx2.py`),
  viewer: trilinear + LOD bias −0.5 and a 1.4× XR eye buffer.
- Output `<name>_msi.glb` / `_msi_ktx2.glb` in the viewer's layer layout
  (`surface_<k>`, larger k first). ~5–7 min per scene incl. eval.

| Scene | Cells | Full-res cells | KTX2 | GPU (ASTC) | PSNR vs splats at 0 / 0.1 / 0.3 m |
|---|---|---|---|---|---|
| hotel_room | 203k | 6.2k | 48 MB | 171 MB | 32.8 / 32.7 / 33.0 |
| abandoned_parking | 131k | 30k | 58 MB | 171 MB | 30.1 / 29.9 / 29.9 |

(`--eval`: ray-sphere compositing at bin-centre radii, no mips, 512 px
90° views — sampling-limited; quality is flat with movement, and side-by-side
renders at 0.3 m are visually indistinguishable from the splats.)

Pitfall: bins are uniform in *disparity*, so bin 0 is the farthest; the
first version composited from bin B−1 and drew the far wall over the lamps.

### Sharper MSIs: texture budget and photo detail

The splats keep ~half of the panorama's fine detail (Laplacian variance 0.52 of the photo at Quest
density); the default MSI (8 pages, half-res cells unless they lose > 0.003) only 0.14. Options:
- **High-res** (`--maxpages 16 --texfull-thr 0.0015`): every cell that needs it at full res (2048 px
  per 90°, the panorama's own density). 13–15 pages, ~280–320 MB ASTC.
- **Photo detail** (`--photo-detail <pano>`): composite the bins from the centre, put the residual
  pano − composite on each pixel's dominant bin (÷ transmittance in front, clamped to a valid
  premultiplied colour). Exact from the centre; v1 showed **bright halos** at depth edges (splat edges
  softer than the photo → "brighten" painted on the foreground's edge, visible from an eye 3 cm
  off-centre). v2 skips the detail within `--edge-px` 6 of a dominant-bin disparity jump > 3%
  (`--edge-jump`) and fades it back in: no halos, but 35% (Heng Heng) / 17% (pano_9000) of pixels get
  none and the gain mostly goes.

PSNR as the viewer draws them (`scripts/pano_eval_viewer.py`: headless Chromium + the real page, 18
views at 30° / 640 px ≈ 21 px/deg; centre vs the panorama, and eye moved 0 / 3 / 10 / 30 cm along each
view's right vs gsplat renders of the splats):

| Version | Heng Heng: pano @0 | splats @0 / 3 / 10 / 30 cm | pano_9000: pano @0 | splats @0 / 3 / 10 / 30 cm |
|---|---|---|---|---|
| Splats (gsplat) | 31.7 | — | 36.5 | — |
| Standard | 29.4 | 34.2 / 34.2 / 34.0 / 33.1 | 33.3 | 36.8 / 36.8 / 36.5 / 34.7 |
| High-res | 30.0 | 36.1 / 36.1 / 35.8 / 34.4 | 34.9 | 40.8 / 40.7 / 39.8 / 35.9 |
| Photo v1 (halos) | **34.8** | 32.6 / 32.6 / 32.5 / 31.8 | **39.9** | 37.3 / 37.5 / 37.4 / 34.9 |
| Photo v2 (edge mask) | 30.5 | 35.7 / 35.7 / 35.5 / 34.1 | 35.5 | 40.5 / 40.3 / 39.6 / 35.8 |
| Photo v3 (narrow mask, darkening kept) | 32.0 | 34.3 / 34.2 / 34.1 / 33.1 | 36.9 | 39.2 / 39.2 / 38.7 / 35.5 |
| **Photo v4 (opaque surfaces)** | **32.9** | 33.7 / 33.7 / 33.6 / 32.7 | **37.0** | 38.9 / 38.9 / 38.5 / 35.4 |

v3 (`--edge-px 3 --edge-jump 0.05 --edge-darken`, 13% / 3% of pixels masked) keeps ~1.4 of v1's 4.8 dB
and has no bright halos, but the darkening it keeps at edges shows as **dark specks** along edges from
3 cm off-centre (the mirror image of v1's halos: a bright front object's blurred rim darkened). The real
fix is probably a matting step (the splats' edges are too wide: lower the front layer's alpha there
instead of recolouring it). Default stays v2; v3 is selectable in the viewer.

**v4** (`--detail-alpha 0.95 --detail-erode 2`, default variant): detail only where the pixel's
dominant *surface* is opaque: the bins within ±5% disparity (≥ 1.5 bins) of the dominant bin, since a
surface is sliced over neighbouring spheres and a single bin is rarely opaque (a per-bin alpha mask left
54–67% of pixels without detail). Erode that mask by 2 px; spread the residual over the surface's bins in
proportion to their alpha (the surface's own colour shifts by D / its weight from the centre). Applied on
96–97% of pixels, 87% of the residual used, no halos or specks at the traffic light from 3 and 15 cm.
Costs more texture (more cells need full resolution: 15–16 pages, 320–341 MB ASTC).

The "vs splats" columns penalise photo detail by design (it adds what the splats lack) and can't show
halos well (the reference has no such edges either); judge those in the headset. The eval harness needs
playwright + Chromium (system libs via conda-forge `nss nspr atk-1.0 ...` in `LD_LIBRARY_PATH`).

## Viewer (`web/pano.html`, `node scripts/pano_server.js`)

Separate from `web.js`: HTTP 3100, HTTPS 3543 (same `certs/`), lists
`pano/<dir>/` scenes (`.spz` from `scripts/pano_spz.py`, `_msi[_ktx2].glb`).
MSI / Splats switch. Desktop: drag = look around (no orbit), WASD/QE or
right-drag = move within a limit (0.5 m) for parallax, wheel = FOV, wiggle.
XR (AR via stock `ARButton`, or VR; `local` space): head tracking gives the
parallax; trigger re-centres on the head, right stick snap-turns 30°, A / B
next / previous scene. `?xrscale` (1.4 MSI / 1.0 splats), `?lodbias` (−0.5),
`?foveation` (0), `?tex=png`; initial view `?yaw=&pitch=&fov=&pos=x,y,z`; `window.panoSetView()`
for scripted captures.

## In the main gallery and viewer (`web.js`, `web/index.html`)

Panoramas appear in the main gallery (`/list`) next to the photo scenes, sorted by date, with a
**360°** badge; the options bar has a separate **360°: MSI / Splats** switch (photo view modes don't
grey them out; they can't be deleted from the gallery). Scene id `pano:<name>` for
`pano/inputs/<name>.jpg`; `/scenes` and `/scene-modes` include them, `/pano-info/<name>` lists the
MSI variants (server priority list `PANO_VARIANTS`: photo detail v4, v2, v3, high-res, standard) and the
`.spz`; `/pano/...` serves the files; `/thumb/pano:<name>` thumbnails the equirect.

The viewer (`web/js/panoView.js`) switches by scene kind:
- Desktop: look-around controls (`PanoLook`) instead of orbiting, its own options panel (360 view,
  MSI variant, move limit, FOV, wiggle, back to centre) and a hint bar; `?pmode=msi|splats`,
  `?pvariant=<dir>`. Splats of a panorama are sorted 360° by distance (`sort360`).
- XR: the panorama is centred on your head, main thumbstick left/right snap-turns 30°
  (right = look right), the secondary stick doesn't scale it, eye buffer 1.4× for MSI; the in-XR
  gallery shows panoramas (badged) and its mode buttons become 360 MSI / 360 splats on a panorama;
  stepping between photo scenes and panoramas re-centres.

The standalone viewer (`node scripts/pano_server.js`, port 3100/3543) still works for comparisons
of all variants.

**Uploads**: the gallery's uploads (and Google Photos imports) land in `<collection>/renderings/uploads`
first; `web.js` moves an equirectangular image (2:1 within 1%, ≥ 2000 px wide, or tagged
`equirectangular` in its XMP) to `pano/inputs/` (name sanitised to `[\w.-]`) and anything else to
`images/`. `scripts/pano_watch.py` (started by `scripts/start_server.sh`, log `logs/pano.log`) then runs
`pano_predict.py` → `pano_spz.py` → `pano_msi.py --tex astc` (photo detail v4) for each input without
`out_ico2_photo4/<name>_msi_astc.glb` (or an older `_msi_ktx2.glb`), ~60 s per panorama on the GB10; the gallery
shows it as
"Processing 360°…" until the splats exist. `python scripts/pano_watch.py --root data/public --once [names]`
does the same by hand.

## Processing speed (2026-10-20, GB10)

One panorama went from ~240 s to ~100 s (`R0011227`, 20 ico views, splats bit-identical except where noted):
- **Pass 2 reuses pass 1's monodepth output** (the predictor ran the monodepth network again on the same view):
  only the fields the predictor reads, features as float16 (0.7 GB a view; splats within 1e-5, +14 GB peak).
- **Ownership before the covariance transform**: SHARP's `apply_transform` decomposes every covariance with a fp64
  SVD on the CPU (2.5 s for a view's 1.18M splats); only the view's own third is transformed now, on a thread that
  overlaps the next view's GPU pass. (A GPU Jacobi 3x3 solver was slower than the CPU SVD at 400k splats, and a
  batched `torch.linalg.eigh` on the GPU crashed the machine's session twice — don't.)
- **bf16 autocast for the monodepth pass** (`--bf16 mono`, default): 0.5 vs 1.6 s a view, and *better* vs the
  panorama (+0.6 dB R0011227, +1.2 dB hotel_room at the centre; renders vs fp32 34 dB, no visible change).
  `--bf16 all` saves 2 s more at the same quality. Batching views doesn't help.
- **MSI**: pages written by `eval_alpha_patches.write_layer_glbs` (parallel PNG, fast level; trimesh's export
  took 26 s), and `--tex astc` (ASTC 4x4, 5 s vs UASTC 26 s; the viewer loads `_msi_astc.glb` where the GPU has
  ASTC, else `_msi_ktx2.glb`, else the PNG file).
- **MSI render**: splats bin-sorted once and sliced per bin (a boolean mask over 7.8M splats per bin and pass cost
  ~7 s); cell lists built per bin, not per cell; photo detail's second pass reuses the first pass's alpha (~13 GB)
  instead of rendering every bin again; Zstandard 9 (as the photo bake). Output identical.
- **Cell culling** (`--cull 0.05`, 15 eyes within +-0.3 m, as the photo bake's CULL): 23-25% fewer cells, a page
  less; renders vs unculled from 8 moved eyes 61-83 dB (<= 0.014% of pixels differ by > 8 levels). +4 s.
- **Persistent watcher** (`pano_watch.py`): steps in-process (imports from the NAS and loading the model cost ~15 s
  per panorama), the MSI files written on a background thread while the next panorama is predicted (status lists
  them under `writing`).
- **Round 3**: each MSI bin rendered once (nearest first; photo detail accumulates from those frames, the bins'
  occupied cells are kept as raw crops and get the detail afterwards: textures bit-identical, vertices within
  float rounding); corner refinement vectorized; the ply written with one `tofile` and read with
  `eval_alpha_patches.read_ply_vertex` (plyfile: element by element); the `.spz` written by pano_predict from the
  same data (gzip level 1: half the time, +3% size); the equirect lookup of pass 2 on the GPU; `--bf16 all` the
  default (hotel_room +0.2 / +0.1 / +0.3 dB vs mono, R0011227 within 0.08 dB, ~2 s faster).
Timings (GB10): predict + spz 38 s (was 138 + 6), MSI 19-21 s (was 91), files written alongside the next
panorama: ~60 s per panorama in a batch (was ~240), ~75 s for the first (model load, imports).
