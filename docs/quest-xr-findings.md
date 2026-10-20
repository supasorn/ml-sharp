# Quest 3 XR viewing of SHARP scenes — findings and experiment log

Everything learned while making SHARP's Gaussian-splat scenes viewable (and
fast) on a Meta Quest 3 in AR/VR, including what did **not** work. Numbers
are from the headset unless marked "offline".

Main test scene for performance: `20260104_122019_7039` (hybrid: 512k splats,
original: 1.18M). Quality test scene: `20251023_155037` (palm trees, people).

---

## 1. System overview

| Piece | Where | Notes |
|---|---|---|
| Prediction | `sharp predict-with-info -i images -o output2 --loop` | Watches `images/`, writes `output2/<scene>.ply` + `.json` (focal, size, depths). |
| Hybrid bake | `sharp bake-hybrid -i output2` (`src/sharp/cli/bake_hybrid.py`) | Writes `<scene>_hybrid.glb` (textured mesh) + `<scene>_hybrid.ply` (remaining splats). ~10–20 s/scene on an RTX 3090. All 79 scenes baked with current defaults. |
| Other bakes | `sharp bake-layers`, `sharp quest-lod` | Older approaches (see §5); `bake_layers.py` also hosts shared helpers (gsplat loading, splat rendering, edge detection). |
| Whole stack | `scripts/start_server.sh` | Web server + prediction watcher (GPU 0) + `bake-hybrid --watch` (GPU 1); photos dropped on the gallery become hybrids automatically. Logs in `logs/`. |
| Server | `node web.js` | HTTP 3000, HTTPS 3443 (WebXR needs HTTPS; cert/key in `certs/`, git-ignored). `/` → gallery `/list`; `/scenes` JSON; `/fps` logs XR reports to `logs/xr_perf.jsonl` (git-ignored); `/delete` removes a scene and its derived files. |
| Hybrid-MPI bake | `python scripts/bake_mpi.py` (all scenes, one per GPU; `--reexport`, `--overwrite`) | Per scene: `bake-hybrid --keep-move 0.3 --max-depth-ratio 0.05` into `renderings/mpi_work/`, then the remaining splats as a sparse MPI (`scripts/eval_alpha_patches.py` MPI mode). Writes `<scene>_mpi_mesh.glb` + `<scene>_mpi.glb` to `output2/`. ~50 s/scene/GPU. See §4b. Not run automatically for new scenes. |
| Viewer | `web/index.html` | Spark 0.1.10 + three.js r178. `?lod=mpionly` (default)`|mpi|hybrid|full|layers|quest` (`patches`, `surfaces`: experimental); compressed layer textures by default (`?tex=png` off); `?hz=120` asks for 120 Hz; view modes switch in XR from the gallery (A/X). FPS reports carry device info (`device`, `xrMode`, `refreshHz`, `fb`, `ua`). |
| XR controls | `web/js/XRSceneController.js` | Ported from `/projects/img3dviewer`'s `XRMeshController`. Secondary X opens the in-XR gallery (`web/js/XRGallery.js`: thumbnails from `/thumb/<scene>`, view-mode buttons, scenes without the mode greyed out via `/scene-modes`). |
| Splat drawing | `web/js/splatFlat.js` | Non-instanced, one triangle per splat (default). See §3. |
| Benchmarks | `web/js/xrBench.js` | Self-running in-headset benchmarks (`?bench=<name>`). |
| Gallery | `web/list.html` | Original / Hybrid / Hybrid-MPI switch (scenes without the mode greyed out), XR options, presets, Performance tests panel (hidden when no test is pending). |

**GPU environment quirk:** the cached gsplat CUDA extension was built with the
system CUDA 13 toolkit while PyTorch is cu128. `load_gsplat()` in
`bake_layers.py` preloads `/usr/local/cuda/lib64/libcudart.so.13` and puts the
env's `bin/` (ninja) on PATH. Use the conda env `/conda_envs/sharp`.

### Viewer URL parameters

| Param | Default | Meaning |
|---|---|---|
| `scene` | — | Scene name (bare `/` redirects to the gallery). |
| `lod` | `full` | `full` original splats, `hybrid` mesh + splats, `layers`, `quest`. |
| `variant` | `_hybrid` | Load a hybrid baked with another `--name-suffix`. |
| `xrscale` | 0.8 | Eye-buffer resolution scale (fixed per session in WebXR). |
| `foveation` | 1 | Quest fixed foveation (0 off … 1 max; blurs the periphery). |
| `xrstddev` | 2.24 (√5) | Splat extent in sigmas in XR (Spark default √8 = 2.83). |
| `xrmaxpx` | 512 | Max splat radius on screen (Spark `maxPixelRadius`). |
| `adaptive` | 0 | Lower splat extent / size cap / foveation when frames drop. |
| `splatdraw` | (triangles) | `instanced` = Spark's own drawing, `quads` = non-instanced quads. |
| `bench` | — | Self-running benchmark (see §6). |
| `autoorbit` | on | `false` = old meta-based desktop orbit centre. |
| `test` | — | Label stored with FPS reports. |

### XR controls (main hand = right by default)

Thumbstick: move along gaze · Trigger: re-centre · Grip: drag · Stick-click
(hold): rotate with hand · A tap: next scene, **A hold 1 s: exit XR** (works in
any state) · B tap: previous, B hold 1 s: reload. Secondary: thumbstick scales
the scene; B/Y opens the settings menu (swap hands, toggles, delete scene with
confirm). The scene auto-centres on the headset 0.5 s after entering XR.

Moving / dragging / re-centring is done by offsetting the XR reference space
(moving the viewer), not by moving the scene: Spark regenerates and re-sorts
every splat whenever a splat object's world transform changes (GPU readback
stall, visible as stutter). Only scaling and hand rotation still move the
scene. Thumbsticks have a 0.15 dead zone.

---

## 2. Key findings (read this first)

1. **Quest 3 splat rendering was limited by Spark's *instanced* drawing**, not
   by pixels, CPU, sorting, splat maths or data reads. Each splat is an
   instance of a tiny quad; with instancing, the GPU's vertex batches are
   mostly empty. Drawing the same splats **non-instanced** (+39%) and as
   **one triangle each** (+16% more) took the test scene from **34.5 → 55 fps**
   with an identical image (`web/js/splatFlat.js`, the default).
2. After that fix, throughput is ~28M splats/s (was ~18M). FPS ≈ 28M /
   splat count, so ~390k splats is roughly the budget for 72 fps.
3. **The hybrid representation** (big textured triangles for view-consistent
   planar regions + the remaining splats, mesh drawn first and splats
   depth-tested against it) removes a median 65% of splats (22–83% across 79
   scenes) at ~30–32 dB vs the original splats for head moves up to 0.45 m
   (offline). It degrades gracefully with movement, unlike layered depth meshes.
4. **Hidden splats must be kept for the full range users actually move.**
   Removing splats only revealed beyond 0.4 m caused black holes at mesh edges
   when moving further; keeping those within 1 m (`--keep-move 1.0`) costs
   +1.4% splats and cuts far-move holes from ~2% to ~0.1% of pixels.
5. **Quest AR stationary mode:** a custom `immersive-ar` request with only
   `optionalFeatures: ['local']` made the Quest switch to its roomscale
   boundary. three's stock `ARButton` (adds `dom-overlay`) keeps stationary
   mode. A page cannot choose the boundary type itself.
6. **Never move the splat objects for navigation.** Spark compares each
   SplatMesh's world matrix every frame; any change regenerates all splats and
   re-sorts them (seen as 20–117 rebuilds per 2 s and 72 → 58 fps while
   moving the scene). Move the viewer via `XRReferenceSpace.getOffsetReferenceSpace`.
7. **Quick proxy estimates were repeatedly far too optimistic** (prepass,
   occluder, visibility removal). Measure the actual payoff (count splats
   removed with the real test, or run a self-running headset benchmark)
   before building.

---

## 3. Performance investigation (headset, scene `20260104_122019_7039`)

All runs at `xrscale=1, foveation=0, xrstddev=2.24` unless noted. "Steady"
excludes the loading seconds. Raw logs: `logs/xr_perf_run*.jsonl` (local only).

| Run | Question | Result | Conclusion |
|---|---|---|---|
| 0 | Resolution (pre-logging) | hybrid 30.7 fps @1.0 vs 33.5 @0.5 | 4× fewer pixels ≈ +9%: **not fill-bound** |
| 1 | Resolution, full vs hybrid | hybrid 36.0 / 33.3 (1.0 / 0.5); full 13.6 / 15.8 | FPS × splats ≈ 16–18M/s: **splat-count bound**; Spark rebuild/sort counters 0 while still (sortOnce works) |
| 2 | Back-layer pruning (strict/loose) | 36.3 → 40.0 → 44.6 fps (505k → 469k → 417k) | Speed follows splat count, **but visible artifacts** (see §5) |
| 3 | Splats in left eye only | 36.1 → **67.8 fps** | Cost is per splat **per eye** |
| 4 | CPU time per frame | 0.7 ms of 27 ms; Spark prep 0.09 ms | **GPU-bound** |
| 5 | Skip Spark's covariance/eigen maths | 36.8 → 39.7 (+6%) | Not the maths |
| 6 | 2-px points vs 2-px quads | 39.7 → **90 fps** (display cap) | Per-vertex/primitive overhead (later: instancing) |
| 7 | Real point-sprite splats | quads 35.5, points **22.9** | Square points shade ~1.9× pixels + extra per-pixel cost: **dead end** |
| 8 | Skip splat data texel reads (synthetic grid) | normal 34.2; reads 38.5; no reads 38.9 | Reads ≈ 1%: not the data fetch |
| 9 | Instanced triangle vs quad | 33.9 vs 34.4 | 3 vs 4 vertices irrelevant **with instancing** |
| 10 | Instanced vs non-instanced quads | 34.5 → **48.0 fps** | **Instancing was the bottleneck** |
| 11 | Non-instanced quads vs triangles | 47.5 → **55.0 fps** | Vertex count matters once batches are full; triangles default |

Other headset observations:
- Close-up scenes cost more at the input view (wider-angle photos fill more
  of the view; more large splats) and when leaning in (~1.35× pixel work vs
  ~1.09× for far scenes). Not a depth-test effect.
- `xrscale` and foveation barely change FPS (not fill-bound); `xrstddev`
  2.24 vs 2.83 mainly changes look (2.24 looks blotchy/blocky).

---

## 4. Hybrid bake: design and offline results

**Algorithm** (`bake_hybrid.py`, details in its module docstring):
1. Render the splats from the input view (gsplat, RGB + expected depth).
2. A pixel is meshable if opaque, not on a depth edge (slope-aware disparity
   test on a 3×3 median-filtered depth), and *view-consistent*: reprojected as
   a textured point into 29 "teacher" renders of the splats within ±0.4 m, its
   colour matches where it's visible.
3. Adaptive quadtree (64 px → `--min-block` 2 px); a block is accepted if it
   has no hard-bad pixels, ≤30% colour-inconsistent pixels, and a plane fits
   its disparity within 0.5 px parallax at 0.4 m. Crack-free fan triangulation.
4. Texture = render of the splats (matches them; `--texture photo` is sharper).
5. Remove splats on the meshed surface (footprint 2σ + 1 px inside the mesh),
   splats behind it that no move within `--keep-move` reveals, and splats
   hidden behind the mesh from every viewpoint within `--keep-move` over their
   whole quad (`--visibility`, lossless; ~4% more).
6. Mesh sits 1% in front of its surface: the depth test hides the splats on it
   (the texture already contains them) while border splats draw past its edge.

**Offline quality** (exact GPU depth-test emulation, see §7), test scene:

| Min block | Splats removed | Triangles | PSNR vs original splats, 0 / 0.15 / 0.35 / 0.45 m |
|---|---|---|---|
| 4 px | 51% | 37k | 32.1 / 31.8 / 30.5 / 29.9 dB |
| **2 px (default)** | 59% | 94k | 31.7 / 31.4 / 30.0 / 29.5 dB |
| 1 px | 63% | 210k | 31.3 / 31.0 / 29.6 / 29.1 dB |

Ideal (every non-edge pixel meshed): 65% removable, so finer triangulation
than 1–2 px blocks isn't worth it. A quadtree bug (blocks overhanging the
right/bottom edge were dropped, not split) cost 11 points of coverage; fixed.

Across all 79 scenes: mesh covers median 87% of the view; splats removed
median 65% (22–83%); kept 198k–917k (median 415k).

Where the remaining splats are (test scene): ~41% of the image is unmeshed —
mostly real depth edges (95% of edge pixels show a colour change) and
fragmented foliage. Only ~6% is low-texture, so texture-aware "rough
geometry" relaxation would gain little on this kind of scene.

**Black holes when moving far** (scene `20260104_122019_7039`, % of pixels
left uncovered): with `--keep-move` 0.4 → 0.01% @0.45 m, 1.4% @0.6 m, 2.2%
@0.8 m; with 1.0 → ≤0.13% up to 1 m (+1.4% splats). Default is 1.0.

---

## 4b. Hybrid-MPI: the remaining splats as a sparse multi-plane image

Idea: keep the hybrid mesh and pre-render *all* remaining splats into
semi-transparent textured quads, so the Quest draws ~100k triangles instead of
~500k splats. What made it work (each step fixed a failure, offline numbers on
`20260104_122019_7039`, dB vs the original splats at 10 / 30 cm head moves;
hybrid 32.5 / 26.4):

1. **No splats may be left** between layers: layers mixed with leftover
   splats can't be drawn in a correct order (best ~1 dB loss + grainy edges).
2. **Overlapping spans, not cut surfaces**: heightfield meshes split at depth
   jumps fall apart into floating squares (−3.6 dB even with 6.8M triangles).
   Patches that cover their splats' full footprints and fade out like them
   (per-tile flat depth slices) were within ~0.3 dB.
3. **Flat patches only**: RANSAC tilted planes mixed foreground and
   background splats in one patch (input view 23 dB).
4. **Sparse MPI** (128 global bins uniform in disparity; each bin's splats
   rendered together and cut into 16-px cells, empty cells dropped): cells of
   a bin abut on one plane (no margins), bins are disjoint in depth, so mesh →
   bins back to front is the right order and overdraw halves (26× vs 53×) for
   32.0 / 25.9 dB. 64 bins: 30.7 / 24.5 at 16×; 256 bins: 32.3 / 26.2 at 43×.
5. **`--max-depth-ratio 0.05`** in the hybrid bake: blocks whose depth varies
   >5% are mostly foliage edges taken for steep slopes and show as slivers
   pointing away once the splats that hid them are gone.
6. **Export**: cells keep a 2-texel border of *real* neighbour texels and the
   viewer disables mipmaps (else seams show as short lines); per-cell adaptive
   texture resolution (half resolution unless the cell loses > 0.003 of
   detail, ≤ 4 pages of 4096²) keeps foliage sharp.
7. **Viewer must not treat the premultiplied textures as sRGB** (see §7).
8. **Snapping must not cross the mesh**: a splat whose bin is the mesh's bin at
   its centre pixel moves one bin farther (behind the mesh) or nearer (in
   front). Otherwise splats kept just behind the mesh for disocclusions (often
   SHARP's large, blurry fill) were snapped in front of it and smudged the
   grass (34% of the remaining splats on `20260914_113455_56657`; input view
   35.0 → 43.7 dB, 10 cm 30.8 → 32.1 dB vs hybrid 33.0).
10. **Per-cell depth** (`REFINE=1`, default in `bake_mpi.py`): each cell's
    corners take the alpha-weighted mean disparity of the bin's splats in the
    neighbouring cells (shared corners, so a bin stays crack-free), instead of
    the bin centre. Free at render time; +0.1 to +0.4 dB at 10–30 cm
    (`20260914_113455_56657`: 32.1 / 27.2 → 32.4 / 27.6 dB at 10 / 30 cm).
    **K-means bins** (`BINMODE=kmeans`, bins where the disparities are) did not
    help: same quality as uniform + refine at 2.2× the overdraw (content split
    over more layers).
11. **Fewer bins don't come free with per-cell depth**: 96 / 64 bins cut
    overdraw ~20% / ~40% but cost 0.2–0.9 / 0.8–1.8 dB at 10–30 cm (worst on
    the heavy `20200927_161631`). Keep 128.
12. **Cell culling** (`CULL=0.05`, default in `bake_mpi.py`): render the MPI +
    mesh from 18 viewpoints (bake_mpi.py: 9, at 0.25 resolution, see §4c) spanning ±0.3 m, composite in depth order, and drop
    cells whose alpha × transmittance in front stays < 0.05 everywhere.
    Drops 35% of cells (triangles, texels) and 11–21% of the input view's layer
    fragments at no measurable cost (≤ 0.04 dB, held-out viewpoints included;
    0.1 drops 43% for ≤ 0.2 dB). Most culled cells sit behind the mesh, whose
    depth test already rejected their pixels, so the fill saving is smaller
    than the cell count. (The export emulator shows extra single-pixel holes
    after culling: its triangle edge test cracks the mesh, a GPU's doesn't.)
13. **Bake-time occluders for a depth pre-pass** (`OCC=1`, off): fully opaque
    8×8-px blocks of a cell as depth-only quads (behind their own cell by
    0.2%). Lossless but only 3–7% fewer layer fragments (17–28k quads, 0.35×
    the frame): pixels get opaque through several stacked semi-transparent
    layers, rarely one opaque layer (same lesson as the splat opaque-core
    pre-pass, §5). Not worth an extra pass.
14. **Texture aliasing on the Quest**: full-resolution cells are drawn 2–3×
    smaller than their texels at xrscale 1 (photo FOV ~56–74° over a ~104°
    eye buffer), which aliases without mipmaps (sharp on desktop). Offline at
    Quest density vs a 4× supersampled ideal: no mips sharpness 1.3–1.4
    (aliasing), plain box mips 0.72–0.80 (blurry), mips with LOD bias −0.5
    1.0 and +6–7 dB — but in the headset −0.5 still crawled when moving, plain
    mips looked soft, and **xrscale 1.4 with bias −0.5 looked best** (Quest 3
    grants 2352×2464 per eye and holds 120 Hz). Defaults now: xrscale 1.4 for
    MPI views (1.0 for splat views), no foveation, KTX2 with 3 box mip levels,
    4-texel cell borders (`CELL_BORDER`), trilinear + bias −0.5 (`?lodbias=`).
    Still-image metrics can't separate motion crawl from blur: judge in the
    headset.
9. SHARP's own two layers are not a front/back pair (92% of same-index pairs
   lie within 1% depth, each layer alone is porous), so two full-resolution
   projective textured layers fail (17.8 dB at 10 cm alone, 19.7 with the mesh).

Headset (`20200927_161631`, xrscale 1, foveation 0): hybrid 33 fps; per-tile
patches 46–54 fps (varies with view direction → fill-bound, 71× overdraw).

**MPI-only** (`?lod=mpionly`, now the default view; `bake_mpi.py --mpionly`):
the same sparse MPI built from *all* splats, no hybrid mesh. Offline it beats
Hybrid-MPI by 4–6 dB at 10 cm and even the splat hybrid at every move (the
hybrid mesh was the largest remaining error), at 2–2.7× the blended layer
fragments but fewer triangles. **Compressed textures** (`scripts/glb_ktx2.py`:
Basis UASTC KTX2 + Zstd, `<x>_ktx2.glb`, loaded by default; `?tex=png` for the
uncompressed files): 4× less GPU texture memory, ~2.4× smaller files, −0.5 dB at
the input view, ~0 when moving.

Headset benchmark (`?bench=views`, `web/js/xrBench.js`; Quest 3, AR, 120 Hz via
`&hz=120`, 1680×1760 per eye, fps):

| Mode | 20260914_113455_56657 | 20200927_161631 | 20250101_140017 |
|---|---|---|---|
| Original splats | 32.5 | 28.9 | 26.0 |
| Hybrid | 79.5 | 42.7 | 36.8 |
| Hybrid-MPI | 119.7 | 76.2 | 57.6 |
| Hybrid-MPI, compressed | 119.5 | 109.8 | 79.7 |
| MPI-only | 119.6 | 89.6 | 62.1 |
| MPI-only, compressed | 119.7 | 119.0 | 94.8 |

Texture reads were a real bottleneck (compression +31–53% on heavy scenes;
hidden at 90 Hz, where every MPI mode sat at the cap), and MPI-only is faster
than Hybrid-MPI (no mesh triangles, no uncompressed mesh texture).

### 4c. MPI-only follow-ups (2026-10-19/20, DGX Spark GB10; `scripts/eval_alpha_patches.py` options)

Scored with `scripts/glb_render.py` (vectorized offline renderer: plain / shared-tile / LCOL / LCW / parallax quads,
KTX2 + ASTC pages with the viewer's trilinear mips and LOD bias, look-at cameras `x,y,z@tx,ty,tz`) against
gsplat renders of the splats, me_small + `20260102_153205_63202`, 5 moved eyes, **on the compressed textures**.
- **Adaptive planes `ad12sep`** (`ADAPT=4 ADAPT_T=12`, 4x finer bins in cells with textured strip edges): +2.5 dB
  over MPI-only on me_small (37.0 vs 34.6), +2 dB on the wide scene, removes forehead/hairline stacking. Cost
  1.6x (me_small) to 2x fill and 2.2x quads; on a cluttered interior (`20260914_155935_32199`, 64% of cells
  flagged, 8 atlas pages) the Quest fell to ~22 fps (MPI-only 90): needs a budget on the flagged cells.
  `bake_mpi.py --mpionly --variant ad12sep`; gallery "Adaptive". 96 base bins x4 matched 128x4 with 13% less fill;
  128x2 / 64x4 cost 0.2 / 0.4 dB for -22% / -30% fill. 8-px cells: 2.6x quads, Quest 103 fps (quads, not fill,
  were the limit up to 21x fill on me_small). 32-px cells: 21.6x fill, still 120 Hz on me_small.
- **Compression artifacts**: premultiplied RGBA of ~10 stacked semi-transparent planes compresses 3.5x worse than
  opaque texels; the errors add up and show as 4x4 blocks on magnified half-res cells. `UNPREM=1` (un-premultiplied,
  spread colour) cut the KTX2 loss from 2.5 to 0.9 dB; `SEPA=1` (each plane's colour tile alpha 255 + alphas
  packed 4 per tile, viewer's shared shader) removes the blocks. SEPA tiles are grouped by runs of 4 depth-adjacent
  planes of a cell inside page-sized depth bands (`SEPA_BAND`), and every quad is drawn at its own plane's place:
  otherwise holes / thin lines at moved views (planes drawn a page too early).
- **Colour layers** (`LCOL` 3 frame-wide layers, `LCT` per-cell tiles, `LCW` weights between 2 layers): remove the
  LCOL ghost face, but one colour per visibility layer costs 2-6 dB vs per-plane colour; cutting planes by layer
  (LCT) leaves dark seams (two filtered alpha edges cover < the plane). Not worth it unless texture memory limits.
- **Gap copies** (`GAPCOPY`): fill the gaps between strips of a surface crossing planes, +0.2 dB, +17% fill; they
  don't fix the shear of features (each strip still at its plane's depth). **Parallax** (`PARALLAX=1` with SEPA:
  per-texel depth offset in the colour tile's alpha, lookup shifted in the shader): 128 planes + parallax = 37.1 dB
  on me_small at plain MPI-only fill, but offsets are bounded by one plane and the shift by the 2-texel border, so
  it breaks up close (0.5 m from a face 2 m away) like plain MPI. Viewer: must apply the per-photo depth warp.
- **ASTC instead of UASTC**: plain ASTC 4x4 KTX2 (`<x>_astc.glb`, `ktx create --format ASTC_4x4_UNORM_BLOCK
  --astc-quality fast`): 6x faster to encode (UASTC: 152 CPU-s for 3 pages, 18 of 20 cores busy) and more accurate
  (+0.25 dB rendered). Quest/Apple/phones sample it natively; the viewer prefers it where supported, else UASTC
  (`_ktx2.glb`), else PNG; gallery "Textures".
- **Bake speed** (plain MPI-only, me_small): 45 s -> ~6 s per scene in a persistent worker (`bake_mpi.py`;
  ~11 s cold): bordered tiles gathered per occupied cell (not unfolded per frame) and kept on the GPU; per-bin work
  on the window of cells with content (`FASTWIN`) and bin-sorted splat slices (byte-identical); culling from 9 eyes
  at 0.25 resolution and Zstandard 9 (within 0.01 dB, +4% download); ASTC; no `sharp.cli`/`trimesh` imports
  (~5 s). `BAKE_TIMING=1|2` prints stage / section times.

---

## 5. Approaches that did not work (and why)

| Approach | Outcome | Why |
|---|---|---|
| Layered depth meshes (`bake-layers`) | Broke quickly when moving; ghost copies, smears | SHARP's splat grid isn't the photo's pixel grid (centres drift), so splat-centre meshes take the wrong photo colours; rebuilt in image space from splat renders it improved but stayed fragile. Superseded by the hybrid. |
| `quest-lod` (prune co-located back layer + merge flat 2×2 blocks) | Only −33% splats on the test scene | SHARP's opacity is ~0.99 almost everywhere, merges are limited by texture |
| Opaque-core depth prepass | Removed | Only 0.04% of splat pixels have alpha ≥ 0.996 — a pixel becomes opaque through many overlapping tails, not single cores |
| Depth-only occluder mesh (verified lossless) | Removed | Lossless but culled only 2–7% of pixel work; the "51–72% behind the surface" proxy counted visible volumetric splats (sky, foliage) |
| Back-layer pruning (strict / loose) | Removed | −7% / −18% splats but visible flicker at mesh edges / see-through squares, despite ≤0.06% "holes" offline |
| Point-sprite splats | Removed | Slower with real splat sizes (see run 7) |
| Multiview (both eyes in one pass) | Not attempted | three r178's WebGLRenderer has no multiview (only its WebGPU renderer); Spark needs WebGLRenderer |
| Lowering resolution / foveation for speed | Little gain | Not fill-bound |
| Semi-transparent textured layers for the remaining hybrid splats (`scripts/eval_alpha_layers.py`, offline only) | Not built | Geometrically promising: per 16-px tile, ≤4 tilted planes within 0.5 px parallax at 1 m take a median 37% of the remaining splats (26–67% over 8 scenes; 3–8× extra blended overdraw). But ~60% of the remaining splats lie *between* layers in depth (mostly hidden splats kept for reveals), and no viewer draw order fixes that: best (mesh → sorted layers writing depth where α > 0.5 → splats) costs ~1 dB at 0.15 m and grainy edges at the input view (36 vs 52 dB). Layers only behind every remaining splat of their tile keep 0.2%. The highest-payoff scene ghosts even with exact per-pixel order (−4 dB, smeared people) |
| Mesh + *all* remaining splats as K non-flat alpha surfaces (`scripts/eval_alpha_surfaces.py`, `eval_alpha_surfaces2.py`, offline, 0.3 m range) | Not built | Depth peeling into K heightfield meshes (depth + RGBA from a render of each surface's splats). Test scene vs hybrid at 0.1 / 0.3 m: −3.6 / −3.0 dB even at 2-px vertex spacing (6.8M tris); K = 8: −2.7 / −2.3 dB (14M tris). Splitting cells at depth jumps, depth smoothing and multi-view texture optimisation (24 views) didn't help (≤ +0.6 dB at 0.3 m, worse at small moves). Diagnostic: draw order costs < 1 dB; flattening the edge/volumetric splats onto one depth per pixel costs the rest. Superseded by overlapping patches → sparse MPI (§4b) |

---

## 6. How to measure on the headset

- Prefer **self-running benchmarks**: `?bench=<name>` (`web/js/xrBench.js`)
  runs all variants in one XR session (10 s per phase, first 2 s skipped),
  logs each phase to `logs/xr_perf.jsonl` and exits XR by itself. Add the
  link to `PERF_TESTS` in `web/list.html` (the panel shows only pending tests).
- Otherwise the viewer posts a report every 2 s while in XR with fps, splat
  count, Spark rebuild/sort counts, CPU ms per frame and settings.
- Ask the user to test as rarely as possible: one link, one session.
- The user cares about what they see in the headset; offline metrics missed
  artifacts that were obvious in XR (pruning flicker), so confirm visually.

## 7. Offline evaluation pitfalls (cost many wrong conclusions)

- **Score the compressed textures** (`<x>_astc.glb` / `_ktx2.glb` through `scripts/glb_read.py`), not the PNG
  pages: compression cost per-plane premultiplied colour 2.5 dB and reversed rankings (§4c).
- **Viewer-only effects**: the per-photo depth warp (`depthWarp.js`) changes geometry in the viewer only;
  check viewer changes headless (Playwright Chromium + SwiftShader; needs nss/fonts libs, see `chromelibs` env).

The harness renders splats with gsplat and meshes with pyvista/VTK:
- **Don't flip** VTK textures or screenshots: with camera up = (0,−1,0) and
  focal point +z the image is already in OpenCV orientation. An earlier
  double flip mirrored vertical parallax and invalidated results.
- `p.get_image_depth()` needs a prior `screenshot()` and
  `reset_camera_clipping_range=False`; background reads back as the far plane
  (treat > 0.95·far as "no geometry").
- gsplat's anti-aliasing blur is resolution dependent: render reference and
  test at the same (supersampled) resolution.
- Emulate the GPU depth test exactly by turning the rendered mesh into opaque
  pixel-sized splats composited with the kept splats (pass 1: kept colour,
  mesh black; pass 2: kept black, mesh white; image = C1 + C2·mesh). Fill the
  spill around the mesh border with the nearest mesh colour, not black.
- PSNR vs the photo is ~18–22 dB even when perfect (splat renders are softer);
  compare against splat renders instead.
- **Premultiplied RGBA textures must not be sRGB textures**: the GPU decodes
  colour to linear before filtering, so filtered texels at soft edges get more
  colour than their alpha (bright contours along every MPI bin boundary). The
  viewer samples them raw (`NoColorSpace`) in a minimal ShaderMaterial with no
  output conversion. An offline emulator of the viewer has to model this (and
  the draw order), or it misses such artifacts.
- Vertices exactly on pixel centres plus an inclusive inside test count each
  pixel ~3× in a software rasteriser (too opaque); use half-open tests and
  nudge vertices. Sort keys packing pixel·1e7 + order must keep |order| < 1e7.
- `scripts/eval_alpha_layers.py` is a gsplat-only alternative: the mesh (and
  any planar layers) are ray-cast per eye pixel and composited as tiny
  per-pixel Gaussians (`eps2d` ≈ 0 so they don't leak), giving exact
  per-pixel depth order; colour/black pass variants emulate viewer draw
  orders. Its numbers run slightly harsher than the VTK harness (hybrid at
  0.15 m: 29.8 vs 31.4 dB), so compare within one harness only.

## 8. Open ideas

- A per-eye pre-pass computing each splat's projected ellipse once, so each
  of its 3 vertices only reads the result (vertex work still matters).
- Fewer splats at a visible cost (careful: the user notices artifacts).
- Photo-textured hybrid meshes (`--texture photo`): sharper than the splats.
- Holes revealed beyond SHARP's own second layer could be filled with a
  blurred far backdrop.

## 9. Operations (stack, batches, machine)

- **Stack**: `scripts/start_server.sh` runs exactly three services, each in
  its own process group (setsid): `node web.js`, the SHARP prediction watcher
  (`predict-with-info --loop`, GPU 0) and `bake_mpi.py --watch --mpionly`
  (GPU 1). pids are in `logs/stack.pids`; `scripts/stop_server.sh` stops them;
  starting again stops an old stack first. New photos get MPI-only only (not
  Hybrid / Hybrid-MPI). Logs: `logs/{web,predict,mpi}.log`.
- **Batch MPI bakes**: `python scripts/bake_mpi.py -i output2 --mpionly
  --gpus auto --jobs 2` (`--reexport` re-exports from cached bins,
  `--overwrite` redoes everything, `--no-ktx2`). The KTX2 step
  (`scripts/glb_ktx2.py`, KTX-Software 4.4.2 under `KTX_HOME`) encodes pages in
  parallel; export itself downsamples/trims on the GPU (~2× faster than the
  first version). Verify a file's mip levels by reading its KTX2 headers.
- **Shared GPUs**: other users hold ~7–11 GB per GPU, so bakes can run out of
  memory. `--gpus auto` picks the GPUs with the most free memory and each job
  retries OOM 3×; re-run leftovers with `--jobs 1`. Don't run experiments on
  the GPUs while a batch runs (several earlier OOMs were self-inflicted).
- **Shell (zsh)**: unquoted `$VAR` is not word-split (use `${=VAR}` or
  `xargs`); scene names can contain spaces, so quote lists (`xargs -d '\n'`);
  `pkill -f <pattern>` can match the zsh wrapper's own command line (kill
  exact pids instead); `bc` isn't installed.
- **Google Photos import** (`google_photos.js`, Photos Picker API): OAuth
  client in `secrets/google_oauth.json`, refresh token in
  `secrets/google_token.json` (both git-ignored; never commit or print).
  Redirect URI `http://localhost:3000/google/callback` must be registered and
  the user must be a test user of the OAuth app (else `403 access_denied`).
  Downloads land in `images/` written in one go, so the prediction watcher
  picks them up; `/pending` drives the gallery's spinner cards.
- **Thumbnails**: `/thumb/<scene>` serves cached 512-px JPEGs
  (`renderings/thumbs/`, Pillow + pillow_heif), used by both galleries.
- **Hybrid-MPI files** are only partly refreshed with mipmaps (~23 scenes);
  MPI-only (all 85 scenes as of 2026-10-17) is the maintained mode.
