#!/bin/bash
cd /mnt/d/LuSNAR_Seg
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
/opt/miniconda3/envs/lusnar/bin/python -u /mnt/d/LuSNAR_Seg/tools/sc_diag/check_pred.py 2>&1 | grep -v -i "mmengine\|init_weights\|torch.Size\|^$" | tail -20
