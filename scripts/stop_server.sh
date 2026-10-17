#!/usr/bin/env bash
# Stop the stack started by scripts/start_server.sh on this machine (host and pids in logs/stack.pids): ask the start script to stop
# its services, then make sure every service's process group is gone.
set -u
cd "$(dirname "$0")/.."
PIDFILE=logs/stack.pids
[ -f "$PIDFILE" ] || { echo "No running stack ($PIDFILE not found)."; exit 0; }
read -r host main services < "$PIDFILE"
if [[ "$host" =~ ^[0-9]+$ ]]; then                               # old format without the host: "main services..."
  services="$main${services:+ $services}"; main="$host"; host="$(hostname)"
fi
if [ "$host" != "$(hostname)" ]; then                            # logs/ is shared between machines
  echo "$PIDFILE is from $host, not this machine ($(hostname)); not stopping anything."
  exit 0
fi
kill -TERM "$main" 2>/dev/null                                   # the start script's cleanup stops the services
for _ in $(seq 1 24); do                                         # wait up to 12 s
  alive=0
  for p in $services; do kill -0 -- "-$p" 2>/dev/null && alive=1; done
  [ "$alive" = 0 ] && break
  sleep 0.5
done
for p in $services; do kill -KILL -- "-$p" 2>/dev/null; done      # anything left (e.g. start script already gone)
rm -f "$PIDFILE"
echo "Stopped."
