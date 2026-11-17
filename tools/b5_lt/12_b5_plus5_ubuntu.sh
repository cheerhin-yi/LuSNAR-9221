#!/usr/bin/env bash
# SegFormer-B5 crop plus5 20k, stage 2 — NATIVE UBUNTU, fresh run (no resume).
# Same hyperparameters as LT_B5_crop_20k (batch/gpu 24, crop 512, lr 6e-5, bf16, seed 42),
# train scenes 1/3/5/6/7/8, --no-val (saves last.pth at final iter; no Scene5 selection).
# New work dir + new list dir. Never touches LT_B5_crop_20k or B2 plus5 dirs.
set -uo pipefail

ROOT=/run/media/cui/数据/LuSNAR_Seg
export PYTHONPATH="$ROOT/tools/portable"
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF || true
PY=$HOME/miniconda3/envs/lusnar/bin/python
cd "$ROOT" || exit 90

RUN=LT_B5_crop_20k_plus5
W="$ROOT/work_dirs/$RUN"
BATCH="${LT_BATCH:-24}"
WORKERS="${LT_WORKERS:-12}"
ITERS=20000
PORT=29701
LIST="$ROOT/data/lists_b5_plus5"
LOG="$ROOT/logs/${RUN}_ubuntu.log"
B5PRE="$ROOT/pretrained/mit_b5_20220624-658746d9.pth"

if [ -e "$W" ] && [ -n "$(ls -A "$W" 2>/dev/null)" ]; then
  echo "REFUSE: $W exists and is not empty" | tee -a "$LOG"; exit 8
fi
if pgrep -f 'lusnar_row_blv_ddp|lusnar_small_crater_score_sc|torch.distributed.run' >/dev/null; then
  echo "REFUSE: training/scoring process running" | tee -a "$LOG"; exit 4
fi
[ -f "$B5PRE" ] || { echo "REFUSE: missing $B5PRE" | tee -a "$LOG"; exit 3; }

mkdir -p "$LIST"
echo "=== $RUN start $(date '+%F %T') batch/gpu=$BATCH x2 crop=512 lr=6e-5 iters=$ITERS no-val scenes=1,3,5,6,7,8 ===" >>"$LOG"
"$PY" -u -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.0.1 --master_port "$PORT" \
  tools/portable/lusnar_row_blv_ddp_b5.py \
  --recipe M_ROW_BLV --run-name "$RUN" --iters "$ITERS" --batch "$BATCH" --workers "$WORKERS" \
  --amp bf16 --crop 512 --lr 6e-5 --val-every 1000 --log-every 50 --no-val \
  --train-scenes "1,3,5,6,7,8" \
  --pretrain "$B5PRE" \
  --project-root "$ROOT" --data-root "$ROOT/data" --list-dir "$LIST" --train-subdir train \
  >>"$LOG" 2>&1 < /dev/null
rc=$?
echo "=== $RUN exit $rc $(date '+%F %T') ===" >>"$LOG"
exit $rc