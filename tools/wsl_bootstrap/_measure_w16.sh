#!/usr/bin/env bash
set -euo pipefail
OUT=/mnt/d/LuSNAR_Seg/logs/wsl_workers16_measure.txt
{
  echo "=== sample $(date -Is) ==="
  tail -8 /mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/train_log.jsonl
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader
  free -h
  df -h /dev/shm
  uptime
  # WSL cpu approx from /proc/stat
  grep '^cpu ' /proc/stat
  # worker count
  echo "lusnar_procs=$(pgrep -c -f lusnar_row_blv_ddp.py || true)"
  pgrep -af 'torch.distributed.run.*29601' | head -1
  pgrep -af 'run_opensource_followup|run_error_attrib_after' || true
} | tee -a "$OUT"