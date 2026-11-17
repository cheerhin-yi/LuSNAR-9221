#!/bin/bash
export OUT=/mnt/d/LuSNAR_Seg/results/sc_diag/smoke
setsid nohup bash /mnt/d/LuSNAR_Seg/tools/sc_diag/run_sc_diag.sh --iters 30 --max-images 12 --workers 4 > /mnt/d/LuSNAR_Seg/logs/sc_diag/smoke.log 2>&1 < /dev/null & disown
echo launched $!
