#!/bin/bash
set -u
date
echo "==== ENV ===="
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lusnar
python -c "import torch,ftfy,mmseg; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count()); print(torch.cuda.get_device_name(0), torch.cuda.get_device_name(1))"
echo "==== DATA ===="
test -d /root/lusnar_data/train/Moon_5/color && echo DATA_OK || echo DATA_MISSING
ls /root/lusnar_data/lists
echo "==== GPU ===="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
echo "==== SHM ===="
df -h /dev/shm
free -h
echo "==== PORT ===="
ss -ltn 2>/dev/null | grep 29601 || echo port_29601_free
if pgrep -f '/opt/miniconda3/envs/lusnar/bin/python -u tools/portable/lusnar_row_blv_ddp.py' >/dev/null; then
  echo TRAIN_ALREADY_ALIVE
  pgrep -af lusnar_row_blv_ddp
  exit 0
fi
echo "==== LAUNCH TRAIN ===="
export WORKERS=8
bash /mnt/d/LuSNAR_Seg/tools/wsl_bootstrap/03_resume_train_wsl.sh
echo "==== LAUNCH WAITERS ===="
bash /mnt/d/LuSNAR_Seg/tools/wsl_bootstrap/04_relaunch_waiters_wsl.sh
echo "==== WAIT 180s ===="
sleep 180
date
echo "==== POST PROCS ===="
pgrep -af 'torch.distributed.run|lusnar_row_blv_ddp.py|run_opensource_followup_wsl|run_error_attrib_after_wsl' || echo NO_PROCS
echo "==== POST GPU ===="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
echo "==== JSONL ===="
tail -n 8 /mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/train_log.jsonl
echo "==== MARKS ===="
grep -n 'WSL resume\|OutOfMemoryError\|ChildFailedError' /mnt/d/LuSNAR_Seg/logs/train_ROW_BLV_20k.log | tail -n 12
echo "==== LOG TAIL ===="
tail -n 25 /mnt/d/LuSNAR_Seg/logs/train_ROW_BLV_20k.log | grep -v 'mmengine\|PretrainedInit\|torch.Size\|^ *$' | cut -c1-240
