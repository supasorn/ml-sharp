# Baking hybrid scenes (`sharp bake-hybrid`)

The hybrid bake turns a SHARP scene into a fast-to-render pair of files:

- `<scene>_hybrid.glb`: large textured triangles for the parts of the scene a
  flat surface reproduces from nearby viewpoints;
- `<scene>_hybrid.ply`: the Gaussians that the mesh doesn't replace.

The viewer (`?lod=hybrid`) draws the mesh first and the remaining splats on
top with depth testing. On the test scenes this removes a median 65% of the
splats at ~30–32 dB against the original splats for head moves up to
~0.5 m. How it works and what was measured: `docs/quest-xr-findings.md` §4.

## Requirements

- A CUDA GPU (the bake renders the splats with gsplat).
- The conda env with the project installed:

  ```bash
  conda activate /conda_envs/sharp   # or call /conda_envs/sharp/bin/sharp directly
  ```

  On this machine the gsplat extension needs the CUDA 13 runtime; `load_gsplat()`
  preloads it and puts the env's `ninja` on PATH automatically. If gsplat fails
  to load, see "Troubleshooting" below.

## Inputs

For each scene, two files from SHARP's prediction step in the same folder
(normally `output2/`):

- `<scene>.ply`: the predicted Gaussians;
- `<scene>.json`: camera metadata (focal length, image size, depths).

These are written by the prediction watcher:

```bash
sharp predict-with-info -i images -o output2 --loop
```

Scenes without a `.json` are skipped. The source photo isn't needed unless
you use `--texture photo`.

## Baking

All scenes in a folder (skips scenes whose hybrid files already exist):

```bash
sharp bake-hybrid -i output2
```

One scene:

```bash
sharp bake-hybrid -i output2/<scene>.ply
```

Rebuild existing outputs (after changing options or the code):

```bash
sharp bake-hybrid -i output2 --overwrite
```

In folder mode, files ending in `_hybrid`, `_quest`, `_pruned` or `_mesh`
are not treated as scenes, and scenes without a `.json` are skipped with a
message.

**Time and size:** ~10–20 s per scene on an RTX 3090 (all 79 current scenes:
~15 minutes). Output per scene: 2–14 MB `.glb` (median 7) + 11–51 MB `.ply`
(median 23).

**New scenes:** the watcher doesn't bake automatically. After new photos are
processed, run `sharp bake-hybrid -i output2` again; only the new scenes are
baked.

## What the log line means

One line per scene, for example:

```
20251023_155037.ply: meshable 80% of pixels, 28150 blocks (2px:16160, 4px:8119, ...),
94320 tris covering 78%; Gaussians 1179648 -> 468671 (60% removed; 20682 hidden
behind the mesh from every viewpoint) -> 20251023_155037_hybrid.glb, 20251023_155037_hybrid.ply
```

- **meshable**: share of pixels that pass the "looks like a textured surface
  from nearby viewpoints" test.
- **blocks / tris covering**: the quadtree blocks meshed and how much of the
  view the mesh covers.
- **Gaussians A -> B (x% removed)**: splats in the original scene and kept in
  the hybrid. Rendering speed on the Quest scales with the kept count
  (~28M splats/s, so ~390k for 72 fps).
- **hidden behind the mesh from every viewpoint**: extra splats removed by
  the lossless visibility check (`--visibility`).

## Options worth knowing

Defaults are what all current scenes were baked with.

| Option | Default | Effect |
|---|---|---|
| `--move` | 0.4 | Head movement (m) the mesh must look right for (teacher views within ±move). |
| `--keep-move` | 1.0 | Hidden splats that moving this far could reveal are kept. Lower = fewer splats but black holes at mesh edges when moving further (0.4 → ~2% of pixels at 0.8 m). |
| `--min-block` | 2 | Smallest mesh block (grid px). 1: more splats removed (63% vs 59% on the test scene) for ~2× triangles and −0.4 dB; 4: fewer removed (51%), +0.4 dB. |
| `--color-tol` | 0.08 | How different (0–1) a textured surface may look from the splats in nearby views. Lower = stricter, fewer splats replaced. |
| `--edge-tol` | 0.05 | Disparity jump that counts as a depth edge (edges are never meshed). |
| `--texture` | `splats` | `photo` textures the mesh with the original photo (sharper than the splats, may not match neighbouring splats exactly). Needs the photo in `--images-dir`. |
| `--texture-width` | 2048 | Texture size. With `--texture photo`, 4096 keeps the photo's detail. |
| `--visibility` / `--no-visibility` | on | Lossless removal of splats the mesh hides from every viewpoint within `--keep-move`. |
| `--name-suffix` | `_hybrid` | Write `<scene><suffix>.glb/.ply` instead, to compare variants side by side. |
| `--overwrite` | off | Rebuild existing outputs. |

Run `sharp bake-hybrid --help` for the full list.

## Viewing the result

Start the server (`node web.js`; HTTPS on port 3443 for the headset) and:

- gallery: `https://<server>:3443/` → choose **Hybrid** at the top, then a scene;
- direct: `https://<server>:3443/?scene=<scene>&lod=hybrid`;
- a variant baked with `--name-suffix _hybrid_test`:
  `...&lod=hybrid&variant=_hybrid_test`.

In the viewer's GUI, **showMesh** / **showSplat** toggle the two parts, which
shows what the mesh replaced. If a scene has no hybrid files, the viewer falls
back to the original splats.

## Comparing variants

```bash
sharp bake-hybrid -i output2/<scene>.ply --min-block 1 --name-suffix _hybrid_mb1
```

Then open `...&lod=hybrid&variant=_hybrid_mb1` next to the default. For a
speed comparison on the headset, add a self-running benchmark
(`web/js/xrBench.js`, see `docs/quest-xr-findings.md` §6) rather than
measuring by hand. Delete variant files when done: a later folder bake
treats `<scene>_hybrid_mb1.ply` as a scene name, finds no matching `.json`
and skips it with a message, so it's harmless but noisy and takes disk space.

## Troubleshooting

- **`No output2/<scene>.json, skipping.`**: the scene has no metadata;
  re-run prediction for it (`sharp predict-with-info`).
- **gsplat import / `libcudart.so.13` / `Ninja is required` errors**: run with
  the env's own interpreter (`/conda_envs/sharp/bin/sharp`) so `load_gsplat()`
  finds `ninja`; the CUDA 13 runtime must exist at
  `/usr/local/cuda/lib64/libcudart.so.13`.
- **Out of GPU memory**: bake scenes one at a time (`-i output2/<scene>.ply`)
  or close other GPU jobs; a 1.2M-splat scene needs a few GB.
- **Black holes at mesh edges when moving**: the scene was probably baked with
  a small `--keep-move`; rebake with the default 1.0.
