#!/bin/bash
set -u
date
echo "==== PRECHECK ===="
if pgrep -f '/opt/miniconda3/envs/lusnar/bin/python -u tools/portable/lusnar_row_blv_ddp.py' >/dev/null; then
  echo ALREADY_ALIVE
  pgrep -af 'torch.distributed.run|lusnar_row_blv_ddp.py'
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
  exit 0
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ':29601$'; then
  echo PORT_BUSY
  ss -ltnp | grep 29601
  exit 1
fi
echo "==== WAITERS ===="
pgrep -af 'run_opensource_followup_wsl|run_error_attrib_after_wsl' || echo waiters_missing
echo "==== GPU PRE ===="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
echo "==== LAUNCH ===="
export WORKERS=8
bash /mnt/d/LuSNAR_Seg/tools/wsl_bootstrap/03_resume_train_wsl.sh
echo "==== WAIT 180s ===="
sleep 180
date
echo "==== POST PROCS ===="
pgrep -af 'torch.distributed.run|lusnar_row_blv_ddp.py' || echo NO_TRAIN
echo "==== POST GPU ===="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
echo "==== JSONL ===="
tail -n 8 /mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/train_log.jsonl
echo "==== MARKS ===="
grep -n 'WSL resume\|OutOfMemoryError\|ChildFailedError\|ERROR Traceback' /mnt/d/LuSNAR_Seg/logs/train_ROW_BLV_20k.log | tail -n 25
echo "==== LOG TAIL ===="
tail -n 40 /mnt/d/LuSNAR_Seg/logs/train_ROW_BLV_20k.log | grep -v 'mmengine\|PretrainedInit\|torch.Size\|^ *$' | cut -c1-240
echo "==== WAITERS POST ===="
pgrep -af 'run_opensource_followup_wsl|run_error_attrib_after_wsl' || echo waiters_missing
