#!/bin/bash
export OUT=/mnt/d/LuSNAR_Seg/results/sc_diag/main
nohup bash /mnt/d/LuSNAR_Seg/tools/sc_diag/run_sc_diag.sh --iters 1500 --workers 6 > /mnt/d/LuSNAR_Seg/logs/sc_diag/main.log 2>&1 < /dev/null
