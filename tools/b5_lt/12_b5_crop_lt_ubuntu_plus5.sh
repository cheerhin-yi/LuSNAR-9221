#!/usr/bin/env bash
# SegFormer-B5 crop plus5 20k, stage 2 — NATIVE UBUNTU, fresh run (no resume).
# Handoff 2026-11-13 §8-2 口径: port 29700, list-dir data/lists_b5_crop_plus5 (new dir,
# never reuses data/lists_b5_crop so the live lists are not overwritten).
# Same hyperparameters as LT_B5_crop_20k (batch/gpu 24, crop 512, lr 6e-5, bf16, seed 42),
# train scenes 1/3/5/6/7/8, --no-val (no Scene5 selection; trainer writes last.pth at final iter).
# No --tta-final, no --resume. Never touches LT_B5_crop_20k or B2 plus5 dirs.
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
PORT=29700
LIST="$ROOT/data/lists_b5_crop_plus5"
LOG="$ROOT/logs/${RUN}_ubuntu.log"
B5PRE="$ROOT/pretrained/mit_b5_20220624-658746d9.pth"

refuse() { echo "[$(date '+%F %T')] REFUSE: $*" | tee -a "$LOG"; exit "$1"; }

# --- preflight: nothing is overwritten, any doubt => refuse ---
[ -x "$PY" ] || refuse 7 "missing python $PY"
[ -f "$B5PRE" ] || refuse 3 "missing $B5PRE"
if [ -e "$W" ] && [ -n "$(ls -A "$W" 2>/dev/null)" ]; then
  refuse 8 "work dir $W exists and is not empty (move it away first)"
fi
if pgrep -f 'lusnar_row_blv_ddp|lusnar_small_crater_score_sc|torch.distributed.run' >/dev/null; then
  refuse 4 "training/scoring process already running"
fi
# both 4090s must be visible and healthy (catches the 41:00.0 drop-off)
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l)
[ "$NGPU" -eq 2 ] || refuse 9 "expected 2 GPUs, nvidia-smi reports $NGPU"
if ! nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
     | awk 'BEGIN{x=0}{if($1+0>4000)x=1}END{exit x}'; then
  refuse 5 "GPU busy: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')"
fi
# power limit 300-350W on both cards (handoff allows 300-350W; user locked 300W on 2026-11-15
# after the second bus drop, per handoff §2.2 fallback — do NOT raise it back without user approval)
if ! nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null \
     | awk 'BEGIN{x=0}{if($1+0<299.0 || $1+0>350.5)x=1}END{exit x}'; then
  refuse 10 "power limit not in 300-350W: $(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader 2>/dev/null | tr '\n' ' | ')"
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$PORT$"; then
  refuse 6 "port $PORT busy"
fi

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "POWER before launch: $(nvidia-smi --query-gpu=index,power.limit,power.draw --format=csv,noheader | tr '\n' ' | ')"
mkdir -p "$LIST"
say "plus5 start pid=$$ batch/gpu=$BATCH workers/rank=$WORKERS iters=$ITERS port=$PORT list=$LIST"

echo "=== $RUN UBUNTU start $(date '+%F %T') trainer=lusnar_row_blv_ddp_b5.py batch/gpu=$BATCH x2 crop=512 lr=6e-5 workers=$WORKERS iters=$ITERS no-val tta-final=off scenes=1,3,5,6,7,8 ===" >>"$LOG"
"$PY" -u -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.0.1 --master_port "$PORT" \
  tools/portable/lusnar_row_blv_ddp_b5.py \
  --recipe M_ROW_BLV --run-name "$RUN" --iters "$ITERS" --batch "$BATCH" --workers "$WORKERS" \
  --amp bf16 --crop 512 --lr 6e-5 --val-every 1000 --log-every 50 --no-val \
  --train-scenes "1,3,5,6,7,8" \
  --pretrain "$B5PRE" \
  --project-root "$ROOT" --data-root "$ROOT/data" --list-dir "$LIST" --train-subdir train \
  >>"$LOG" 2>&1 < /dev/null
rc=$?
echo "=== $RUN UBUNTU exit $rc $(date '+%F %T') ===" >>"$LOG"
say "END $RUN exit=$rc"
exit $rc