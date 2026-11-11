#!/usr/bin/env bash
kill -TERM -- -500 2>/dev/null
kill -TERM 966 967 2>/dev/null
for i in $(seq 1 30); do
  pgrep -f 'lusnar_row_blv_ddp|distributed.run|run_opensource|run_error_attrib' >/dev/null || break
  sleep 2
done
pkill -KILL -f 'lusnar_row_blv_ddp|distributed.run|run_opensource|run_error_attrib' 2>/dev/null
sleep 1
pgrep -af 'lusnar_row_blv|distributed.run|run_opensource|run_error' | cut -c1-80
echo STOPPED $(date)
tail -n 5 /mnt/d/LuSNAR_Seg/logs/train_ROW_BLV_20k.log