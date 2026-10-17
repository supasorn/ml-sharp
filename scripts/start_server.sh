#!/usr/bin/env bash
# Start the ml-sharp viewer stack and keep it running until Ctrl+C (or scripts/stop_server.sh):
#   1. web server (node web.js): gallery + viewer, HTTP 3000 / HTTPS 3443,
#      uploads dropped on the gallery go to images/;
#   2. prediction watcher: images/ -> output2/<scene>.ply + .json;
#   3. hybrid-bake watcher: output2/<scene>.ply + .json -> <scene>_hybrid.glb/.ply;
#   4. MPI watcher (scripts/bake_mpi.py --watch): output2/<scene>.ply -> MPI-only
#      (the default view) and Hybrid-MPI, with compressed textures.
# A dropped photo shows up in the gallery once predicted (original splats); the
# other view modes become available as their bakes finish.
#
# Each service runs in its own process group, so stopping it also stops the
# processes it started. The stack's pids are kept in logs/stack.pids; starting
# again while an old stack is running stops the old one first.
#
# Logs: logs/web.log, logs/predict.log, logs/bake.log, logs/mpi.log.
# Env overrides: SHARP_ENV (conda env, default /conda_envs/sharp),
#                PREDICT_GPU (default 0), BAKE_GPU (default 1), MPI_GPU (default 2).
set -u
cd "$(dirname "$0")/.."
ENV="${SHARP_ENV:-/conda_envs/sharp}"
PIDFILE=logs/stack.pids
mkdir -p logs images output2

# A previous stack still running (e.g. its terminal was closed)? Stop it first.
if [ -f "$PIDFILE" ]; then
  echo "Stopping a previous stack..."
  bash scripts/stop_server.sh
fi

pids=()
stopping=0
cleanup() {
  [ "$stopping" = 1 ] && return
  stopping=1
  trap - INT TERM HUP EXIT
  echo "Stopping..."
  for p in "${pids[@]}"; do kill -TERM -- "-$p" 2>/dev/null; done     # whole process group of each service
  for _ in $(seq 1 20); do                                             # up to 10 s to exit cleanly
    alive=0
    for p in "${pids[@]}"; do kill -0 -- "-$p" 2>/dev/null && alive=1; done
    [ "$alive" = 0 ] && break
    sleep 0.5
  done
  for p in "${pids[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done
  rm -f "$PIDFILE"
}
trap cleanup INT TERM HUP EXIT

# Start a service in its own session/process group (setsid execs it, so $! is the service and its group id).
start() {
  local log="$1"; shift
  setsid "$@" >> "logs/$log" 2>&1 &
  pids+=($!)
}
start web.log node web.js
start predict.log env CUDA_VISIBLE_DEVICES="${PREDICT_GPU:-0}" "$ENV/bin/sharp" predict-with-info -i images -o output2 --loop
start bake.log env CUDA_VISIBLE_DEVICES="${BAKE_GPU:-1}" "$ENV/bin/sharp" bake-hybrid -i output2 --watch
start mpi.log env CUDA_VISIBLE_DEVICES="${MPI_GPU:-2}" "$ENV/bin/python" scripts/bake_mpi.py -i output2 --watch
echo "$$ ${pids[*]}" > "$PIDFILE"

echo "Started web server, prediction watcher, hybrid-bake watcher and MPI watcher (pids ${pids[*]})."
echo "Gallery: https://$(hostname -I | awk '{print $1}'):3443/   Logs: logs/{web,predict,bake,mpi}.log"
echo "Stop with Ctrl+C or scripts/stop_server.sh."
# Stop everything if any one of them exits.
wait -n
echo "A process exited; see logs/. Stopping the rest."
