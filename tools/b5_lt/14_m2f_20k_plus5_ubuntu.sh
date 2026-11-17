#!/usr/bin/env bash
# M2F (HF Mask2Former Swin-B) crop plus5 20k — NATIVE UBUNTU, fresh run (no resume).
# 2026-11-16 evening: M2F 20k straight run finished (Scene5 best.pth official I1a 0.88985
# vs B2 0.88381 => gate passed) so plus5 is auto-launched by the B5 precedent.
# Same recipe/hyperparameters as LT_M2F_crop_20k (M_ROW_BLV, batch 16/GPU, crop 512, lr 6e-5,
# bf16, seed 42); difference is exactly the B5 plus5 delta: train scenes 1,3,5,6,7,8,
# --no-val (trainer writes last.pth at the final iter), NO --tta-final, straight to 20000
# in ONE continuous run (no segmenting / no rests —用户 2026-11-16 决定：让卡一次性跑完).
# New work dir + new list dir (never overwrites data/lists_m2f_crop or any B2/B5 dir).
# Power guard intentionally REMOVED (user authorized 450W full load 2026-11-16).
# Safety: any doubt => REFUSE, keep scene, NO auto-resume (值守 reports immediately).
set -uo pipefail

ROOT=/run/media/cui/数据/LuSNAR_Seg
export PYTHONPATH="$ROOT/tools/portable_m2f"
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF || true
PY=$HOME/miniconda3/envs/lusnar_bb/bin/python
cd "$ROOT" || exit 90

RUN=LT_M2F_crop_20k_plus5
W="$ROOT/work_dirs/$RUN"
PORT=29704
LIST="$ROOT/data/lists_m2f_plus5"
LOG="$ROOT/logs/${RUN}_ubuntu.log"
HFW="$ROOT/pretrained/hf/mask2former-swin-base-IN21k-ade-semantic"

refuse() { echo "[$(date '+%F %T')] REFUSE: $*" | tee -a "$LOG"; exit "$1"; }
say()    { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# --- preflight: nothing is overwritten, any doubt => refuse ---
[ -x "$PY" ] || refuse 7 "missing python $PY"
[ -d "$HFW" ] || refuse 3 "missing HF weights $HFW"
if [ -e "$W" ] && [ -n "$(ls -A "$W" 2>/dev/null)" ]; then
  refuse 8 "work dir $W exists and is not empty (move it away first)"
fi
if pgrep -f 'lusnar_row_blv_ddp|lusnar_small_crater_score_sc|torch.distributed.run' >/dev/null; then
  refuse 4 "training/scoring process already running"
fi
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l)
[ "$NGPU" -eq 2 ] || refuse 9 "expected 2 GPUs, nvidia-smi reports $NGPU"
nvidia-smi 2>/dev/null | grep -qi "gpu is lost" && refuse 11 "GPU is lost — do not start"
if ! nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
     | awk 'BEGIN{x=0}{if($1+0>4000)x=1}END{exit x}'; then
  refuse 5 "GPU busy: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')"
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$PORT$"; then
  refuse 6 "port $PORT busy"
fi

mkdir -p "$LIST"
say "POWER before launch: $(nvidia-smi --query-gpu=index,uuid,power.limit,power.draw --format=csv,noheader | tr '\n' ' | ')"
say "M2F plus5 STRAIGHT start pid=$$ iters=20000 batch/gpu=16 workers/rank=12 port=$PORT list=$LIST no-val tta-final=off scenes=1,3,5,6,7,8 USER-AUTHORIZED-450W goal=card-stability-test-continuous"

echo "=== $RUN UBUNTU start $(date '+%F %T') trainer=lusnar_row_blv_ddp_m2f.py recipe=M_ROW_BLV batch/gpu=16 x2 crop=512 lr=6e-5 iters=20000 no-val tta-final=off scenes=1,3,5,6,7,8 ===" >>"$LOG"
"$PY" -u -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.0.1 --master_port "$PORT" \
  tools/portable_m2f/lusnar_row_blv_ddp_m2f.py \
  --recipe M_ROW_BLV --run-name "$RUN" --iters 20000 --batch 16 --workers 12 \
  --amp bf16 --crop 512 --lr 6e-5 --val-every 1000 --log-every 50 --no-val \
  --train-scenes 1,3,5,6,7,8 \
  --project-root "$ROOT" --data-root "$ROOT/data" --list-dir "$LIST" --train-subdir train \
  >>"$LOG" 2>&1 < /dev/null
rc=$?
say "END straight run exit=$rc"
if [ "$rc" -eq 0 ]; then
  say "ALL DONE $RUN exit=0 straight-to-20000"
else
  say "ABNORMAL exit=$rc — scene kept, NO auto-resume; report immediately (stability-test evidence)"
fi
exit "$rc"