#!/bin/bash
cd /mnt/d/LuSNAR_Seg || exit 1
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_VISIBLE_DEVICES=1 USE_LIBUV=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
mkdir -p /mnt/d/LuSNAR_Seg/logs/sc_diag
OUT=${OUT:-/mnt/d/LuSNAR_Seg/results/sc_diag/main}
if [ -e "$OUT/sc_diag.json" ]; then echo "refuse: $OUT exists"; exit 2; fi
exec /opt/miniconda3/envs/lusnar/bin/python -u /mnt/d/LuSNAR_Seg/tools/sc_diag/sc_diag.py --out-dir "$OUT" "$@"
