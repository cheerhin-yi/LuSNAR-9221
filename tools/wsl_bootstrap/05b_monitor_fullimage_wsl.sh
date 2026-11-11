#!/usr/bin/env bash
# Hardware monitor for the WSL fullimage queue. Does not relaunch training.
set -u
LOG=/mnt/d/LuSNAR_Seg/fullimage_compare/logs/fullimage_hw_monitor.log
mkdir -p "$(dirname "$LOG")"
echo "$(date +'%Y-%m-%d %H:%M:%S') monitor start" >>"$LOG"
while true; do
  {
    echo "===== $(date +'%Y-%m-%d %H:%M:%S') ====="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
    grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo
    echo -n "loadavg "; cat /proc/loadavg
    echo -n "shm "; df -h /dev/shm | awk 'NR==2{print $3"/"$2" used"}'
    pgrep -af 'fullimage_smoke.py|05_fullimage_queue_wsl' || echo "NO_TRAIN_PROC"
  } >>"$LOG" 2>&1
  sleep 60
done
