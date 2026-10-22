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
`pano_predict.py --layout icoup --depth dap_ratio` (below) → `pano_spz.py` → `pano_msi.py --tex astc` (photo
detail v4) for each input without `out_dr_photo4/<name>_msi_astc.glb` (splats in `out_dr/`), ~75 s per panorama on
the GB10; the gallery
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

## 360 depth instead of stitching (UniK3D, experiment, 2026-10-21)

The stitched pass-1 depth fixes only one scale per view, but perspective monodepth gets each view's *shape* wrong
in its own way (tilts, bends), so straight or flat things come out piecewise: on `R0010963_20160720193830_78855`
a straight railing ~1 m away became three pieces at different angles, with a ~30 cm jump and an S-kink at the
view 13 -> 11/17 boundary (top-down plot of the stitched distance along the rail).

`pano_predict.py --depth unik3d` replaces the stitched map with **UniK3D** (Piccinelli et al., CVPR 2025; ViT-L;
**CC BY-NC 4.0**): one prediction for the whole equirect (`Spherical` camera `[1, 1, 1, 1, W, H, pi, pi/2]`,
2048x1024), 2.4 s (+29 s model load once per process) vs ~10 s for the 20 views. Pass 1 still runs SHARP's
monodepth (pass 2 reuses its features); pass 2 aligns to the UniK3D distance. Result on the railing panorama: the
depth map is coherent (table a smooth gradient instead of per-view lobes, the near forearm one shape, the ceiling
fan resolved) and the rail in the final splats is straight, with two ~10 cm steps left at SHARP view boundaries
(pass-2 alignment, Voronoi ownership).

Setup (outside the repo): `git clone https://github.com/lpiccinelli-eth/UniK3D ~/.local/opt/UniK3D` (no
`pip install -e`: it would change torch); `pip install einops tabulate termcolor` in `sharp_dgx`; its training code
imports `wandb` and `pkg_resources` (gone from recent setuptools, which gsplat's build needs): empty stand-ins in
`~/.local/opt/unik3d_stubs/`. `UNIK3D_HOME` overrides the location. Weights: `lpiccinelli/unik3d-vitl` (Hugging
Face). Depth Any Camera (CVPR 2025, MIT) was the alternative: smaller indoor/outdoor models, torch 1.13 pins.

Not the watcher's default yet: the railing panorama has the experiment as MSI variant `out_uk_photo4` ("UniK3D
depth (photo detail v4)", `?pvariant=out_uk_photo4`) next to the default `out_ico2_photo4`.

## Cracks along view boundaries (overlap, 2026-10-21)

From an eye moved off-centre (20 cm), thin black cracks opened on ceilings in straight lines forming triangles:
the boundaries of the 20 ico views' ownership regions (spherical triangles). Each splat was kept by exactly one
view, cut exactly at the boundary, so any depth disagreement between neighbouring views opened a crack there with
nothing behind it (the ceiling is the farthest surface). Worse with UniK3D depth (SHARP's pass 2 follows it less
exactly): `pano_9000`, six straight-up views, black 0-0.20% (stitched), 0-0.96% (UniK3D).

`pano_predict.py --overlap-deg 5` (default): a view keeps its splats up to 5 degrees past its boundary (angle
to its axis <= the nearest axis's + 5 degrees). Measured in the real viewer (`window.__dbg.panoLook` set to look up
from eyes 20-25 cm off-centre, 8 views, near-black pixels): `pano_9000` hard boundaries + stitched depth 15356,
UniK3D 2 degrees 943, UniK3D 5 degrees 0. (An offline check with `glb_render.py` first suggested 2 degrees was enough:
it had not reproduced the viewer's eye positions; three.js `(x, y, z)` = OpenCV `(x, -y, -z)`.) Cost: +37% splats
(7.6M -> 10.4M; the "360 splats" file), MSI cells about the same (131k vs 133k at 2 degrees: culling drops the
hidden duplicates). The UniK3D versions (`out_uk*`) were rebuilt with it; the stitched versions (`out_ico2*`,
still the viewer's default) keep hard boundaries for comparison; the first UniK3D build (hard) is in
`pano/_old_uk_hard/`.

## Ceiling flatness and 360 depth models (2026-10-21/22)

Measured instead of eyeballed (scripts in the session scratchpad, numbers here): `pano_9000`'s ceiling (points above
45 deg elevation, robust plane fit, |distance to the plane|) and `R0010963`'s straight railing (nearest surface per
column in its image band, distance from a straight line seen from above).

- **The layout split the ceiling.** With `ico` no view looks up: the two highest (69 deg) own 64% of the ceiling,
  their boundary runs across its middle, four 35-deg views own the rest; each sees it obliquely. `--layout icoup`
  (the same 20 views rotated so one looks straight up and one down; every direction >= 13 px inside its owner's
  image): stitched depth's ceiling p90 off-plane 16.6 -> 3.0 cm, the splats follow (patches and steps gone).
- **Gradient-domain stitching** (`--depth poisson`: the views' log-depth gradients blended by feather weight,
  integrated by conjugate gradient on the 2048x1024 map, cos(lat)-weighted, u wraps; ~1 s): removes steps between
  views (no per-view scales at all) but only averages slope disagreements. With `icoup`: ceiling p90 2.6 cm (splats),
  railing rms 6.8 cm (current 7.3): the rail's hump is a bend inside one view's own prediction. Pulling towards
  UniK3D (`poisson_uk`, also faded towards the poles, `--uk-fade`) straightens the rail (5.8-6.0 cm) but bends the
  ceiling again (13.7-18.5 cm): UniK3D's overall room shape conflicts with the views' globally.
- **360 depth models**, depth maps evaluated directly (scale-free: % of the distance; all 15 panoramas inspected):

  | Method | ceiling off-plane median / p90 | railing off straight rms / max |
  |---|---|---|
  | stitched, `ico` (current) | 1.6% / 15.1% | 6.2% / 14.5% |
  | Poisson, `icoup` | 1.1% / 2.1% | 5.5% / 10.6% |
  | UniK3D (CC BY-NC) | 3.6% / 19.6% | 4.5% / 12.1% |
  | **DAP** (Insta360, "Depth Any Panoramas", arXiv 2512.16913, CC BY-NC) | **0.7% / 1.3%** | **2.4% / 8.3%** |
  | PanDA-Large (CVPR 2025, Apache-2.0) | 63% / 566% | 29% / 74% (relative depth with an unknown offset: not distance) |
  | Depth Anywhere, UniFuse student (NeurIPS 2024, Apache-2.0) | 11% / 40% | 27% / 56% (sky nearest outdoors, blotches) |

  `pano_predict.py --depth dap` (`dap_distance`): DAP at 1024x512, metres, masked sky = 100 m; with `--layout icoup`
  the final splats' ceiling is p90 1.6 cm, tilt 0.4 deg (`pano_9000`). Setup outside the repo: `git clone
  https://github.com/Insta360-Research-Team/DAP ~/.local/opt/DAP`, `weights/model.pth` from Hugging Face
  `Insta360-Research/DAP-weights`, `pip install --no-deps torchmetrics lightning-utilities` (its bundled DINOv3 code
  imports torchmetrics); `DAP_HOME` overrides the location. PanDA / Depth Anywhere are in `~/.local/opt` too.
- Viewer variants for comparison (`?pvariant=`): `out_dr_photo4` (default since 2026-10-22, below), `out_ico2_photo4`
  (the previous default, stitched), `out_uk_photo4` (UniK3D),
  `out_gd_photo4` (Poisson + `icoup`), `out_dap_photo4` (DAP + `icoup`), `out_pdap_photo4` (Poisson + DAP +
  `icoup`, below); all with the 5-degree overlap except the current one. Batch: `logs/pano_variant_batch.py TAG [pano_predict options]`.
- DAP alone is too smooth for cluttered scenes (seen in the headset): `hotel_room`'s desk chair merges into the
  desk, the armchair's edges and the table corner are pulled towards the wall, `yaowarat`'s traffic-light pole and
  lamp arm are lost; pass 2 bends SHARP's sharp per-view geometry towards that target. `--depth poisson_dap`:
  the gradient-domain solve (detail from the views) with DAP as the reference at `--dap-weight 1e-3` (only shapes
  wider than ~30 px of the 2048 map follow DAP). DAP's sky (a flat 100 m) is replaced in the reference by the
  views' own (views-only Poisson + the median offset to DAP): with 100 m the sky's jump disagreed with the views'
  and dragged building edges next to it towards the eye. Ceiling p90 1.5 cm (`pano_9000`), railing rms 4.8 cm
  (DAP 2.4, current 7.3); depth crops keep the chair, desk corner, pole and signal as the current version.
  Weight 1e-2: railing 3.4 cm but small objects (the signal) already flattened towards DAP.

## Default since 2026-10-22: DAP-corrected views (`--depth dap_ratio`, `out_dr*`)

`dap_ratio_targets()`: each view's own SHARP depth times a smooth correction towards DAP: the log ratio DAP / view,
blurred over 10 degrees (`--ratio-deg`) with weights (normalized convolution): 0 in DAP's sky, low on depth edges
of either map (DAP's are blurred), and robust (Cauchy, 3 rounds: where the two disagree by far more than ~10%
beyond the smooth correction, DAP is taken to be wrong). The corrected views are feather-blended into one shared
360 map, which every view is aligned to (as the stitched map before). Why each piece:
- Detail stays SHARP's: Poisson + DAP and this keep `hotel_room`'s desk chair, `yaowarat`'s traffic-light pole;
  DAP alone loses them. Simpler than Poisson + DAP (no stitch, no Poisson solve, no sky special case).
- Blur 10 deg, robust: at 5 deg and without robust weights a sky opening (`R0011338`, the roof edge) got a halo pulled
  towards the eye and mirrors / glass doors (`hotel_room`, `R0011338`) were half-flattened onto the wall; 15 deg no
  longer straightened the railing. Depth maps: ceiling p90 1.4 cm (`pano_9000`; stitched 16.6), railing rms
  4.9 cm (`R0010963`; stitched 7.6, DAP alone 2.4); final splats: ceiling 1.4-1.8 cm, railing 4.9 cm (stitched
  7.3; its worst point 16.9 vs 13.1 cm).
- Shared map, not per-view targets: views aligned to their own corrected depth disagreed on thin objects
  (`pano_9000`'s lamp heads, ~30%) and showed them twice (overlap pixels > 5% apart in the final splats: 4.4%,
  shared 0.55%, stitched 0.49%).

Fixes found on the way (all panoramas):
- **bf16 depth was quantized**: under `--bf16` the monodepth head's output had an 8-bit mantissa: 90% of
  neighbouring depth pixels equal (fp32 25%), 10% off on average; contour stripes across buildings (`yaowarat`).
  `load_model()` runs the disparity head in fp32 (+0.04 s a view). The earlier bf16 check scored renders from the
  centre, which hide depth errors. Photos (`sharp predict`) were never affected (fp32).
- **Overlap only on smooth surfaces** (`--overlap-tol 0.05`, `--overlap-edge 0.1`): past its boundary a view keeps
  only splats within 5% of the shared map and away from its depth edges; the full 5-degree overlap gave
  `pano_9000`'s lamp head a blurred black double from the neighbouring view (every eye, even the centre). Ceiling
  cracks still 0 in the viewer (8 eyes).
- **Views far off before alignment**: the alignment net only corrects moderate factors; a street panorama's
  downward views had SHARP's road at 0.33x / 0.42x the target, stayed at 0.59x / 0.77x and floated as a big
  triangle. Pass 2 aligns each view to the target divided by its median factor and scales its splats by it after.
- Not fixed: a very near object (a selfie head at 0.5 m, `R0011166`) is only a front shell; moving the eye 10 cm
  or more shows its edge (same in the stitched version).
