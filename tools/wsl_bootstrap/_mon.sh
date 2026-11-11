#!/usr/bin/env bash
date -Is
pgrep -af 'lusnar_row_blv_ddp|torch.distributed.run' | cut -c1-90
echo "procs: $(pgrep -fc lusnar_row_blv_ddp)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader
for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr '\n' ' '; echo; sleep 1; done
top -bn1 | head -5
free -g | head -2; df -h /dev/shm | tail -1
tail -n 4 /mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/train_log.jsonl
grep -v 'Grad strides\|grad.sizes\|bucket_view\|return Variable' /mnt/d/LuSNAR_Seg/logs/train_ROW_BLV_20k.log | tail -n 8
dmesg 2>/dev/null | grep -i 'dxg\|oom\|killed' | tail -n 5