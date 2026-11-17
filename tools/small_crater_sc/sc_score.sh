#!/usr/bin/env bash
# usage: sc_score.sh GPU CKPT TAG
set -euo pipefail
GPU=$1; CKPT=$2; TAG=$3
cd /mnt/d/LuSNAR_Seg
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_VISIBLE_DEVICES=$GPU USE_LIBUV=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH=/mnt/d/LuSNAR_Seg/tools/portable
exec /opt/miniconda3/envs/lusnar/bin/python -u tools/portable/lusnar_small_crater_score_sc.py \
  --checkpoint "$CKPT" --tag "$TAG" --project-root /mnt/d/LuSNAR_Seg --data-root /root/lusnar_data/train \
  --out-dir /mnt/d/LuSNAR_Seg/results/small_crater_sc
