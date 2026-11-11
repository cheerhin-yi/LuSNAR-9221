#!/bin/bash
ps -p 1144,3426 -o pid,etime,cmd 2>/dev/null
echo ===PROCS===
ps -eo pid,etime,pcpu,cmd | grep -E 'lusnar_scene5|followup|cache_diag' | grep -v grep
echo ===CACHE_TAIL===
tail -8 /mnt/d/LuSNAR_Seg/logs/scene5_cache_diag.log
echo ===FOLLOW_TAIL===
tail -15 /mnt/d/LuSNAR_Seg/logs/scene5_infer_phase_followup.log
echo ===PROBS===
ls /root/lusnar_data/cache/scene5_tta_probs | wc -l
ls /mnt/d/LuSNAR_Seg/results/inference_phase/
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
