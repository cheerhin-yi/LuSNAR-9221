#!/usr/bin/env bash
# Short dual-GPU NCCL / dataloader / power stress trial on native Ubuntu.
# Writes ONLY to temp dirs (work_dirs/LT_B5_ubuntu_dry*, data/lists_b5_ubuntu_dry*).
# Does NOT touch work_dirs/LT_B5_crop_20k or its checkpoints. --no-val keeps it short
# (no 1599-image Scene5 pass); the it==1 DDP all_gather sync check still runs.
set -uo pipefail

ROOT=/run/media/cui/数据/LuSNAR_Seg
export PYTHONPATH="$ROOT/tools/portable"
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF || true
PY=$HOME/miniconda3/envs/lusnar/bin/python
cd "$ROOT" || exit 90

RUN=LT_B5_ubuntu_dry
ITERS="${ITERS:-30}"
BATCH="${LT_BATCH:-24}"
WORKERS="${LT_WORKERS:-12}"
PORT=29701
LIST="$ROOT/data/lists_b5_ubuntu_dry"
LOG="$ROOT/logs/${RUN}_ubuntu.log"
B5PRE="$ROOT/pretrained/mit_b5_20220624-658746d9.pth"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "ubuntu dryrun start pid=$$ run=$RUN iters=$ITERS batch/gpu=$BATCH workers=$WORKERS"
[ -f "$B5PRE" ] || { say "REFUSE: missing $B5PRE"; exit 3; }
[ -x "$PY" ] || { say "REFUSE: missing python $PY"; exit 7; }
mkdir -p "$LIST"
say "POWER before dryrun: $(nvidia-smi --query-gpu=index,power.limit,temperature.gpu --format=csv,noheader | tr '\n' ' | ')"

"$PY" -u -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.0.1 --master_port "$PORT" \
  tools/portable/lusnar_row_blv_ddp_b5.py \
  --recipe M_ROW_BLV --run-name "$RUN" --iters "$ITERS" --batch "$BATCH" --workers "$WORKERS" \
  --amp bf16 --crop 512 --lr 6e-5 --val-every 1000 --log-every 5 \
  --pretrain "$B5PRE" \
  --project-root "$ROOT" --data-root "$ROOT/data" --list-dir "$LIST" --train-subdir train \
  --no-val \
  >>"$LOG" 2>&1 < /dev/null
rc=$?
say "END dryrun exit=$rc"
exit $rc
