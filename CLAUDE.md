# CLAUDE.md

This fork of Apple's SHARP adds a Quest 3 AR/VR viewer and a hybrid
mesh + Gaussian representation for fast headset rendering.

**Read `docs/quest-xr-findings.md` before working on the viewer, the bakes or
performance** — it records the system layout, every experiment (including the
ones that failed and why) and the evaluation pitfalls.

Quick facts:
- Python env: `/conda_envs/sharp` (`sharp ...` CLI). gsplat needs the CUDA 13
  runtime preload handled by `load_gsplat()` in `src/sharp/cli/bake_layers.py`.
- Server: `node web.js` (HTTPS on 3443 for WebXR; `certs/` holds the key —
  never commit it). Gallery at `/list`, viewer at `/?scene=<name>&lod=hybrid`.
- Bake hybrids: `sharp bake-hybrid -i output2` (outputs next to each scene);
  how-to with options and troubleshooting: `docs/hybrid-baking.md`.
- Quest 3 splat rendering is GPU-bound by per-splat vertex work; the viewer
  draws splats non-instanced as triangles (`web/js/splatFlat.js`). Measure
  speed with a self-running headset benchmark (`web/js/xrBench.js`) and the
  server's `logs/xr_perf.jsonl` before building optimizations.
- Keep AR on three's stock `ARButton` (dom-overlay) or the Quest switches to
  roomscale.
- Work on the `quest-xr-hybrid` branch; commit/push when asked.
