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
  `scripts/start_server.sh` runs it together with the SHARP prediction watcher
  and the MPI-only watcher (`scripts/bake_mpi.py --watch --mpionly`; new scenes
  get only MPI-only, not Hybrid / Hybrid-MPI); stop it with Ctrl+C or
  `scripts/stop_server.sh` (pids in `logs/stack.pids`).
- Bake hybrids: `sharp bake-hybrid -i output2` (outputs next to each scene);
  how-to with options and troubleshooting: `docs/hybrid-baking.md`.
- Default view is **MPI-only** (`?lod=mpionly`): all splats as a sparse
  multi-plane image with compressed (KTX2/UASTC) textures — fastest and best
  quality on the Quest 3. Hybrid-MPI (`?lod=mpi`) = hybrid mesh + MPI of the
  remaining splats. Bake with `python scripts/bake_mpi.py` (Hybrid-MPI) and
  `--mpionly`; both also write `_ktx2.glb` (needs KTX-Software in
  `~/.local/opt/ktx-4.4.2`). Not automatic for new scenes. Design, headset
  numbers and pitfalls (premultiplied textures must not be sRGB): findings doc
  §4b, §7. Benchmark: `?bench=views&hz=120` (`web/js/xrBench.js`).
- Quest 3 splat rendering is GPU-bound by per-splat vertex work; the viewer
  draws splats non-instanced as triangles (`web/js/splatFlat.js`). Measure
  speed with a self-running headset benchmark (`web/js/xrBench.js`) and the
  server's `logs/xr_perf.jsonl` before building optimizations.
- Keep AR on three's stock `ARButton` (dom-overlay) or the Quest switches to
  roomscale.
- Work on `main` (the `quest-xr-hybrid` work was merged into it); commit/push when asked.
- 360 panoramas (equirect -> splats -> multi-sphere image; in the main gallery/viewer as
  `pano:<name>`, also a separate comparison viewer
  `node scripts/pano_server.js` on 3100/3543, data in git-ignored `pano/`): `docs/pano-360.md`.
