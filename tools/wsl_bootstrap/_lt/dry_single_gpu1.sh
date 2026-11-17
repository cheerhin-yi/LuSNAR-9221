#!/usr/bin/env bash
# Single-GPU (GPU1) memory/speed probe for the full-image longtrain. Never touches GPU0.
# usage: dry_single_gpu1.sh <batch> <iters> [workers]
set -uo pipefail
export USE_LIBUV=0 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1
unset PYTORCH_CUDA_ALLOC_CONF || true
ROOT=/mnt/d/LuSNAR_Seg
export PYTHONPATH="$ROOT/tools/portable${PYTHONPATH:+:$PYTHONPATH}"
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate lusnar
cd "$ROOT"
B=$1; N=$2; W=${3:-12}
DATA_ROOT=/root/lusnar_data; LIST1=$DATA_ROOT/lists_fi_lt
mkdir -p "$LIST1"
[ -f "$LIST1/full_crater_weights.json" ] || cp -p "$DATA_ROOT/lists/full_crater_weights.json" "$LIST1/"
RUN=LT_FI_probe_gpu1_b${B}
if [ -e "work_dirs/$RUN" ]; then RUN=${RUN}_$(date +%H%M%S); fi
LOG=$ROOT/logs/${RUN}_wsl.log
echo "=== probe $(date '+%F %T') batch=$B iters=$N workers=$W run=$RUN ===" | tee -a "$LOG"
python -u tools/portable/lusnar_row_blv_ddp.py --recipe M_ROW_BLV --run-name "$RUN" --iters "$N" \
  --batch "$B" --workers "$W" --amp bf16 --crop 1024 --lr 6e-5 --val-every 100000 --log-every 5 \
  --no-val --project-root "$ROOT" --data-root "$DATA_ROOT" --list-dir "$LIST1" --train-subdir train \
  >>"$LOG" 2>&1 < /dev/null
echo "=== probe exit $? $(date '+%F %T') ===" | tee -a "$LOG"