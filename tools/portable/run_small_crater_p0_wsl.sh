#!/bin/bash
cd /mnt/d/LuSNAR_Seg || exit 1
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_VISIBLE_DEVICES=1 USE_LIBUV=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
exec /opt/miniconda3/envs/lusnar/bin/python -u tools/portable/lusnar_small_crater_p0_probe_wsl.py \
  --checkpoint /mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/best.pth \
  --project-root /mnt/d/LuSNAR_Seg --data-root /root/lusnar_data/train \
  --out-dir /mnt/d/LuSNAR_Seg/results/small_crater_E0 "$@"
