#!/bin/bash
ps -p 1144 -o pid,etime,cmd
echo ===PROCS===
ps aux | grep -E 'followup|cache_diag' | grep -v grep
echo ===CACHE_TAIL===
tail -4 /mnt/d/LuSNAR_Seg/logs/scene5_cache_diag.log
echo ===PROBS===
ls /root/lusnar_data/cache/scene5_tta_probs | wc -l
echo ===GPU===
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo ===FOLLOW_LOG===
tail -8 /mnt/d/LuSNAR_Seg/logs/scene5_infer_phase_followup.log
