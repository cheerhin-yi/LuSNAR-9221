#!/usr/bin/env bash
set -euo pipefail
export USE_LIBUV=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29621
export PYTHONUNBUFFERED=1
export PYTHONPATH="/mnt/d/LuSNAR_Seg/tools/portable${PYTHONPATH:+:$PYTHONPATH}"
unset PYTORCH_CUDA_ALLOC_CONF

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lusnar

ROOT=/mnt/d/LuSNAR_Seg
cd "$ROOT"
DATA_ROOT=/root/lusnar_data
LIST_DIR=$DATA_ROOT/lists_plus5
CKPT=$ROOT/work_dirs/train_ROW_BLV_20k_plus5/iter_2000.pth
LOG=$ROOT/logs/train_ROW_BLV_20k_plus5_wsl.log
WORKERS=14

test -f "$CKPT"
test -f "$LIST_DIR/train_scenes.txt"
test -d "$DATA_ROOT/train/Moon_5/color"
wc -l "$LIST_DIR/train_scenes.txt"
head -n 1 "$LIST_DIR/train_scenes.txt"

if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${MASTER_PORT}$"; then
  echo "port $MASTER_PORT busy" >&2
  ss -ltn | grep 29621 >&2
  exit 1
fi
if pgrep -af 'lusnar_row_blv_ddp.py' | grep -v pgrep >/dev/null; then
  echo "train already running" >&2
  pgrep -af 'lusnar_row_blv_ddp.py' >&2
  exit 1
fi

echo "=== WSL plus5 resume $(date '+%Y-%m-%dT%H:%M:%S%z') port=$MASTER_PORT workers=$WORKERS data=$DATA_ROOT lists=$LIST_DIR ckpt=$CKPT ===" | tee -a "$LOG"

setsid python -u -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_addr 127.0.0.1 \
  --master_port "$MASTER_PORT" \
  tools/portable/lusnar_row_blv_ddp.py \
  --recipe M_ROW_BLV \
  --run-name train_ROW_BLV_20k_plus5 \
  --iters 20000 \
  --batch 48 \
  --workers "$WORKERS" \
  --amp bf16 \
  --crop 512 \
  --lr 6e-5 \
  --val-every 1000 \
  --train-scenes 1,3,5,6,7,8 \
  --no-val \
  --list-dir "$LIST_DIR" \
  --project-root "$ROOT" \
  --data-root "$DATA_ROOT" \
  --train-subdir train \
  --resume "$CKPT" \
  >>"$LOG" 2>&1 < /dev/null &

echo "TRAIN_LAUNCH_PID=$!"
sleep 3
pgrep -af 'lusnar_row_blv_ddp.py' || echo 'NO PROC'
sleep 8
tail -n 30 "$LOG"
