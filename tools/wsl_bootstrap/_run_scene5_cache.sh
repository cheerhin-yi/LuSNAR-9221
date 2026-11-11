#!/bin/bash
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lusnar
cd /mnt/d/LuSNAR_Seg
mkdir -p logs results/inference_phase
export PYTHONUNBUFFERED=1
nohup python -u tools/portable/lusnar_scene5_cache_diag.py \
  --stage cache \
  --out-json /mnt/d/LuSNAR_Seg/results/inference_phase/cache_diag.json \
  > logs/scene5_cache_diag.log 2>&1 &
echo $! > logs/scene5_cache_diag.pid
echo STARTED $(cat logs/scene5_cache_diag.pid)
