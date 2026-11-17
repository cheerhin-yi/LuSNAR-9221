#!/usr/bin/env bash
# B5 crop dry run: usage _dry.sh BATCH WORKERS ITERS   (scratch work_dir LT_B5_dry_b<BATCH>, --no-val)
set -uo pipefail
export USE_LIBUV=0 MASTER_ADDR=127.0.0.1 PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF || true
ROOT=/mnt/d/LuSNAR_Seg
export PYTHONPATH="$ROOT/tools/portable"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lusnar
cd "$ROOT"
B=$1; W=$2; IT=$3
RUN=LT_B5_dry_b${B}
LIST=/root/lusnar_data/lists_b5_crop
mkdir -p "$LIST"; [ -f "$LIST/full_crater_weights.json" ] || cp -p /root/lusnar_data/lists/full_crater_weights.json "$LIST/"
[ -e "$ROOT/work_dirs/$RUN" ] && { echo "REFUSE $RUN exists"; exit 2; }
LOG=$ROOT/logs/${RUN}_wsl.log
echo "=== $RUN start $(date '+%F %T') batch/gpu=$B workers=$W iters=$IT ===" >>"$LOG"
python -u -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 29690 \
  tools/portable/lusnar_row_blv_ddp_b5.py \
  --recipe M_ROW_BLV --run-name "$RUN" --iters "$IT" --batch "$B" --workers "$W" \
  --amp bf16 --crop 512 --lr 6e-5 --val-every 1000 --log-every 10 --no-val \
  --pretrain "$ROOT/pretrained/mit_b5_20220624-658746d9.pth" \
  --project-root "$ROOT" --data-root /root/lusnar_data --list-dir "$LIST" --train-subdir train \
  >>"$LOG" 2>&1 < /dev/null
echo "=== $RUN exit $? $(date '+%F %T') ===" >>"$LOG"