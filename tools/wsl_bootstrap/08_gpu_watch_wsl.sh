#!/usr/bin/env bash
# Start (or keep) the dual-GPU supervisor. Does not touch training queues.
set -euo pipefail
ROOT=/mnt/d/LuSNAR_Seg
LOGDIR="$ROOT/logs"
PY="$ROOT/tools/wsl_bootstrap/08_gpu_watch_wsl.py"
mkdir -p "$LOGDIR"
if pgrep -f '08_gpu_watch_wsl.py' >/dev/null 2>&1; then
  echo "gpu watch already running: $(pgrep -af '08_gpu_watch_wsl.py' | grep -v pgrep)"
  exit 0
fi
setsid nohup python3 -u "$PY" >> "$LOGDIR/gpu_watch_daemon.log" 2>&1 < /dev/null &
sleep 1
pgrep -af '08_gpu_watch_wsl.py' | grep -v pgrep || { echo "failed to start watch"; exit 1; }
