#!/usr/bin/env bash
# dry runs for 06d: 5-iter train + 2-image TTA, separate run names (smoke_DRY_M2F_*)
set -euo pipefail
export USE_LIBUV=0 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
unset PYTORCH_CUDA_ALLOC_CONF || true
ROOT=/mnt/d/LuSNAR_Seg
PY=/home/lusnar/.conda/envs/lusnar_bb/bin/python
SCRIPT_DIR="$ROOT/tools/portable_m2f"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
cd "$ROOT"
BATCH="${M2F_BATCH:-16}"; ACCUM="${M2F_ACCUM:-1}"
C=("$PY" -u "$SCRIPT_DIR/lusnar_smoke_compare.py" --project-root "$ROOT" --data-root /root/lusnar_data
   --train-subdir train --smoke-dir /root/lusnar_data/lists_backbone --skip-prepare --iters 5
   --batch "$BATCH" --accum "$ACCUM" --lr 6e-5 --tta-scales 0.75,1.0,1.25 --tta-crop 512 --tta-stride 256 --backbone m2f --dry-run)
if [[ "${1:-train}" == train ]]; then
  "${C[@]}" --exp M_ROW_BLV --workers 6 --run-name DRY_M2F_ROW_BLV
else
  "${C[@]}" --exp M_TTA --workers 0 --run-name DRY_M2F_TTA --ckpt "$ROOT/work_dirs/smoke_DRY_M2F_ROW_BLV/dry.pth"
fi
