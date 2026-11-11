#!/usr/bin/env bash
set -euo pipefail
LOG=/mnt/d/LuSNAR_Seg/logs/wsl_workers16_switch.log
echo "=== watch $(date -Is) ===" | tee -a "$LOG"
ps -ef | grep -E 'torch.distributed.run.*29601|lusnar_row_blv_ddp.py' | grep -v grep | head -5 | tee -a "$LOG" || echo "NO TRAIN" | tee -a "$LOG"
WORKER_N=$(ps -ef | grep 'lusnar_row_blv_ddp.py' | grep -v grep | wc -l)
echo "proc_count=$WORKER_N" | tee -a "$LOG"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | tee -a "$LOG"
free -h | tee -a "$LOG"
df -h /dev/shm | tee -a "$LOG"
tail -30 /mnt/d/LuSNAR_Seg/logs/train_ROW_BLV_20k.log | tee -a "$LOG"
# check for OOM keywords in recent log
if tail -80 /mnt/d/LuSNAR_Seg/logs/train_ROW_BLV_20k.log | grep -Ei 'out of memory|CUDA out of memory|RuntimeError|Bus error|Killed|DataLoader worker|shm|Cannot allocate' >/tmp/oomhits.txt; then
  echo "OOM_OR_CRASH_HITS:" | tee -a "$LOG"
  cat /tmp/oomhits.txt | tee -a "$LOG"
else
  echo "NO_OOM_HITS" | tee -a "$LOG"
fi
# last jsonl
tail -3 /mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/train_log.jsonl | tee -a "$LOG"