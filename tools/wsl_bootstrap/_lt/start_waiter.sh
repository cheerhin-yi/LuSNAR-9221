#!/usr/bin/env bash
cd /mnt/d/LuSNAR_Seg
setsid nohup bash tools/wsl_bootstrap/09_fi_longtrain_wsl.sh >> logs/LT_FI_ROW_BLV_waiter_nohup.log 2>&1 < /dev/null &
sleep 3
pgrep -af 09_fi_longtrain_wsl.sh
tail -n 3 logs/LT_FI_ROW_BLV_waiter_wsl.log