#!/usr/bin/env bash
W=/mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k
for i in $(seq 1 ${1:-50}); do
  if grep -q '"iter": 5000, "val_mIoU' $W/train_log.jsonl && [ -f $W/iter_5000.pth ]; then echo VAL_DONE $(date); break; fi
  sleep 30
done
tail -n 2 $W/train_log.jsonl
ls -l --time-style=+%T $W/*.pth