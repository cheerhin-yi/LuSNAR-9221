#!/usr/bin/env bash
set -euo pipefail
LOG=/mnt/d/LuSNAR_Seg/logs/wsl_workers16_switch.log
echo "=== stop $(date -Is) ===" | tee -a "$LOG"
for p in $(pgrep -f 'run_opensource_followup_wsl.py|run_error_attrib_after_wsl.py' || true); do
  echo "TERM waiter $p" | tee -a "$LOG"
  kill -TERM "$p" 2>/dev/null || true
done
for p in $(pgrep -f 'torch.distributed.run.*lusnar_row_blv_ddp|lusnar_row_blv_ddp.py' || true); do
  echo "TERM train $p" | tee -a "$LOG"
  kill -TERM "$p" 2>/dev/null || true
done
sleep 5
LEFT=$(pgrep -f 'lusnar_row_blv_ddp|torch.distributed.run.*29601|run_opensource_followup_wsl|run_error_attrib_after_wsl' || true)
if [ -n "${LEFT:-}" ]; then
  echo "still alive, KILL: $LEFT" | tee -a "$LOG"
  kill -KILL $LEFT 2>/dev/null || true
  sleep 2
fi
echo "after stop:" | tee -a "$LOG"
(ps -ef | grep -E 'lusnar_row_blv|torch.distributed|run_opensource|run_error_attrib' | grep -v grep || echo "(none)") | tee -a "$LOG"
(ss -ltn | grep 29601 || echo "port 29601 free") | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "$LOG"