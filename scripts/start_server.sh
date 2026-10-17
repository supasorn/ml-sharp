#!/usr/bin/env bash
# Start the ml-sharp viewer stack and keep it running until Ctrl+C (or scripts/stop_server.sh):
#   1. web server (node web.js): gallery + viewer, HTTP 3000 / HTTPS 3443,
#      uploads dropped on the gallery go to images/;
#   2. prediction watcher (SHARP): images/ -> output2/<scene>.ply + .json;
#   3. MPI watcher (scripts/bake_mpi.py --watch --mpionly): output2/<scene>.ply ->
#      <scene>_mpionly.glb + _mpionly_ktx2.glb (the default view; compressed textures, mipmaps).
# A dropped photo shows up in the gallery once predicted (original splats) and in
# MPI-only once baked. Hybrid / Hybrid-MPI are no longer baked automatically
# (sharp bake-hybrid, scripts/bake_mpi.py by hand).
#
# Each service runs in its own process group, so stopping it also stops the
# processes it started. The stack's host and pids are kept in logs/stack.pids; starting
# again while an old stack is running on this machine stops the old one first.
#
# Logs: logs/web.log, logs/predict.log, logs/mpi.log.
# Env overrides: SHARP_ENV (conda env, default /conda_envs/sharp),
#                PREDICT_GPU (default 0), MPI_GPU (default 1).
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
pidline=
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
  # Only our own pid file: a stack started meanwhile (restart) may already have written its own.
  [ "$(cat "$PIDFILE" 2>/dev/null)" = "$pidline" ] && rm -f "$PIDFILE"
}
trap cleanup INT TERM HUP EXIT

# Start a service in its own session/process group (setsid execs it, so $! is the service and its group id).
start() {
  local log="$1"; shift
  setsid "$@" >> "logs/$log" 2>&1 &
  pids+=($!)
}
start web.log node web.js
start predict.log env CUDA_VISIBLE_DEVICES="${PREDICT_GPU:-0}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$ENV/bin/sharp" predict-with-info -i images -o output2 --loop
start mpi.log env CUDA_VISIBLE_DEVICES="${MPI_GPU:-1}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$ENV/bin/python" scripts/bake_mpi.py -i output2 --watch --mpionly
pidline="$(hostname) $$ ${pids[*]}"      # host first: logs/ may be on a disk shared between machines
echo "$pidline" > "$PIDFILE"

echo "Started web server, prediction watcher and MPI-only watcher (pids ${pids[*]})."
echo "Gallery: https://$(hostname -I | awk '{print $1}'):3443/   Logs: logs/{web,predict,mpi}.log"
echo "Stop with Ctrl+C or scripts/stop_server.sh."
# Stop everything if any one of them exits.
wait -n
echo "A process exited; see logs/. Stopping the rest."
