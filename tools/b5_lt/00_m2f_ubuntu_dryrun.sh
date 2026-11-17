#!/usr/bin/env bash
# M2F (HF Mask2Former Swin-B) dual-GPU 30-iter dry run on native Ubuntu.
# Writes ONLY to temp dirs (work_dirs/LT_M2F_ubuntu_dry, data/lists_m2f_ubuntu_dry).
# Validates: DDP/NCCL, M2F wrapper forward/backward under bf16, HF weight load,
# batch 16/卡 (smoke-measured ~37GB). Port 29703 (B5 used 29700-29702).
# Power guard intentionally REMOVED: user authorized 450W full load 2026-11-16.
set -uo pipefail

ROOT=/run/media/cui/数据/LuSNAR_Seg
export PYTHONPATH="$ROOT/tools/portable_m2f"
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF || true
PY=$HOME/miniconda3/envs/lusnar_bb/bin/python   # lusnar_bb: transformers 4.49.0 + full mmcv 2.1.0
cd "$ROOT" || exit 90

RUN=LT_M2F_ubuntu_dry
ITERS=30
BATCH=16
WORKERS=12
PORT=29703
LIST="$ROOT/data/lists_m2f_ubuntu_dry"
LOG="$ROOT/logs/${RUN}.log"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "M2F dryrun start pid=$$ run=$RUN iters=$ITERS batch/gpu=$BATCH workers/rank=$WORKERS env=lusnar_bb"
[ -x "$PY" ] || { say "REFUSE: missing python $PY"; exit 7; }
[ -d "$ROOT/pretrained/hf/mask2former-swin-base-IN21k-ade-semantic" ] || { say "REFUSE: missing HF weights dir"; exit 3; }
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l)
[ "$NGPU" -eq 2 ] || { say "REFUSE: expected 2 GPUs, nvidia-smi reports $NGPU"; exit 9; }
nvidia-smi | grep -qi "gpu is lost" && { say "REFUSE: GPU is lost present in nvidia-smi"; exit 11; } || true
mkdir -p "$LIST"
say "POWER at dryrun: $(nvidia-smi --query-gpu=index,power.limit,power.draw --format=csv,noheader | tr '\n' ' | ')"

"$PY" -u -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.0.1 --master_port "$PORT" \
  tools/portable_m2f/lusnar_row_blv_ddp_m2f.py \
  --recipe M_ROW_BLV --run-name "$RUN" --iters "$ITERS" --batch "$BATCH" --workers "$WORKERS" \
  --amp bf16 --crop 512 --lr 6e-5 --val-every 1000 --log-every 5 --no-val \
  --project-root "$ROOT" --data-root "$ROOT/data" --list-dir "$LIST" --train-subdir train \
  >>"$LOG" 2>&1 < /dev/null
rc=$?
say "END dryrun exit=$rc"
exit $rc
