#!/bin/bash
echo PID_FILE=$(cat /mnt/d/LuSNAR_Seg/logs/scene5_followup_wrapper.pid)
ps -p $(cat /mnt/d/LuSNAR_Seg/logs/scene5_followup_wrapper.pid) -o pid,etime,cmd
ps -p 1144 -o pid,etime,cmd
echo ---
# also any followup bash
ps -eo pid,etime,cmd | grep followup | grep -v grep
echo ---
tail -3 /mnt/d/LuSNAR_Seg/logs/scene5_cache_diag.log
ls /root/lusnar_data/cache/scene5_tta_probs | wc -l
