#!/usr/bin/env bash
# usage: sc_train.sh GPU RUN_NAME ITERS BATCH WORKERS [extra args...]
set -euo pipefail
GPU=$1; RUN=$2; ITERS=$3; BATCH=$4; WORKERS=$5; shift 5
cd /mnt/d/LuSNAR_Seg
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_VISIBLE_DEVICES=$GPU USE_LIBUV=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH=/mnt/d/LuSNAR_Seg/tools/portable
test ! -e work_dirs/$RUN/last.pth || { echo "refuse: work_dirs/$RUN has last.pth"; exit 1; }
mkdir -p logs/small_crater_sc
exec /opt/miniconda3/envs/lusnar/bin/python -u tools/portable/lusnar_row_blv_ddp_sc.py \
  --recipe M_ROW_BLV --run-name "$RUN" --iters "$ITERS" --batch "$BATCH" --workers "$WORKERS" \
  --amp bf16 --crop 512 --lr 6e-5 --warmup 100 --cosine --min-lr-ratio 0.1 \
  --project-root /mnt/d/LuSNAR_Seg --data-root /root/lusnar_data --list-dir /root/lusnar_data/lists \
  --train-subdir train --cc-dir /root/lusnar_data/cc_bank \
  --init-from /mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/best.pth "$@"
