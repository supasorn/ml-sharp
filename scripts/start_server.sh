#!/usr/bin/env bash
# Start the ml-sharp viewer stack and keep it running until Ctrl+C:
#   1. web server (node web.js): gallery + viewer, HTTP 3000 / HTTPS 3443,
#      uploads dropped on the gallery go to images/;
#   2. prediction watcher: images/ -> output2/<scene>.ply + .json;
#   3. hybrid-bake watcher: output2/<scene>.ply + .json -> <scene>_hybrid.glb/.ply;
#   4. MPI watcher (scripts/bake_mpi.py --watch): output2/<scene>.ply -> MPI-only
#      (the default view) and Hybrid-MPI, with compressed textures.
# A dropped photo shows up in the gallery once predicted (original splats); the
# other view modes become available as their bakes finish.
#
# Logs: logs/web.log, logs/predict.log, logs/bake.log, logs/mpi.log.
# Env overrides: SHARP_ENV (conda env, default /conda_envs/sharp),
#                PREDICT_GPU (default 0), BAKE_GPU (default 1), MPI_GPU (default 2).
set -u
cd "$(dirname "$0")/.."
ENV="${SHARP_ENV:-/conda_envs/sharp}"
mkdir -p logs images output2

pids=()
cleanup() {
  trap - INT TERM EXIT
  echo "Stopping..."
  kill "${pids[@]}" 2>/dev/null
  wait
}
trap cleanup INT TERM EXIT

node web.js >> logs/web.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES="${PREDICT_GPU:-0}" "$ENV/bin/sharp" predict-with-info -i images -o output2 --loop >> logs/predict.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES="${BAKE_GPU:-1}" "$ENV/bin/sharp" bake-hybrid -i output2 --watch >> logs/bake.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES="${MPI_GPU:-2}" "$ENV/bin/python" scripts/bake_mpi.py -i output2 --watch >> logs/mpi.log 2>&1 &
pids+=($!)

echo "Started web server, prediction watcher, hybrid-bake watcher and MPI watcher (pids ${pids[*]})."
echo "Gallery: https://$(hostname -I | awk '{print $1}'):3443/   Logs: logs/{web,predict,bake,mpi}.log"
# Stop everything if any one of them exits.
wait -n
echo "A process exited; see logs/. Stopping the rest."
