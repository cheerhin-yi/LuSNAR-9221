cd /mnt/d/LuSNAR_Seg
grep -n '=== ' logs/train_ROW_BLV_20k.log | tail -1
if ! pgrep -f 'run_opensource_followup_wsl|run_error_attrib_after_wsl' >/dev/null; then setsid bash tools/wsl_bootstrap/04_relaunch_waiters_wsl.sh </dev/null 2>&1 | tail -3; else echo waiters_alive; pgrep -af 'run_opensource_followup_wsl|run_error_attrib_after_wsl'; fi
for i in $(seq 1 12); do
  echo "$(date +%T) $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ') cpu_idle=$(top -bn1 | awk '/Cpu/{print $8}') alive=$(pgrep -fc 'torch.distributed.run --nproc') oom=$(grep -c OutOfMemoryError logs/train_ROW_BLV_20k.log)"
  sleep 20
done
tail -5 work_dirs/train_ROW_BLV_20k/train_log.jsonl