#!/usr/bin/env bash
# M2F (HF Mask2Former Swin-B) crop 20k, formal stage — NATIVE UBUNTU, segmented run.
# [M2F COPY 先例] trainer = tools/portable_m2f/lusnar_row_blv_ddp_m2f.py (B5 DDP trainer,
#   only backbone swapped). Hyperparams = S2 smoke validated: M_ROW_BLV, lr 6e-5, wd 0.01,
#   crop 512, bf16, seed 42, batch 16/gpu (smoke-measured 37GB/48GB), workers 12/rank.
# Segmentation rule (USER.md 2026-11-16): ~1h train -> stop at 1000-iter checkpoint ->
#   10 min rest -> resume. Stops: 3000,6000,...,18000,20000 (3000 iters ≈ 49min + 3 vals).
# --tta-final ONLY on the final segment. Abnormal exit (rc != 0) => STOP the loop,
#   keep the scene, NO auto-resume (值守任务 will report).
# Power guard intentionally REMOVED: user authorized 450W full load on 2026-11-16
#   ("不锁功率，直接满载运行，GPU 检测不到及时通知我"). GPU-lost guard stays.
# Writes ONLY to work_dirs/LT_M2F_crop_20k + data/lists_m2f_crop. Never touches B2/B5 dirs.
set -uo pipefail

ROOT=/run/media/cui/数据/LuSNAR_Seg
export PYTHONPATH="$ROOT/tools/portable_m2f"
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF || true
PY=$HOME/miniconda3/envs/lusnar_bb/bin/python
cd "$ROOT" || exit 90

RUN=LT_M2F_crop_20k
W="$ROOT/work_dirs/$RUN"
PORT=29703
LIST="$ROOT/data/lists_m2f_crop"
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
if pgrep -f 'lusnar_row_blv_ddp|lusnar_small_crater_score|torch.distributed.run' >/dev/null; then
  refuse 4 "training/scoring process already running"
fi
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l)
[ "$NGPU" -eq 2 ] || refuse 9 "expected 2 GPUs, nvidia-smi reports $NGPU"
if nvidia-smi 2>/dev/null | grep -qi "gpu is lost"; then
  refuse 11 "GPU is lost present in nvidia-smi — do not start"
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$PORT$"; then
  refuse 6 "port $PORT busy"
fi

mkdir -p "$LIST"
say "M2F 20k SEGMENTED start pid=$$ batch/gpu=16 workers/rank=12 iters=20000 port=$PORT power=$(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader | tr '\n' ' | ') USER-AUTHORIZED-450W"

STOPS=(3000 6000 9000 12000 15000 18000 20000)
PREV=0
for STOP in "${STOPS[@]}"; do
  RESUME=""
  [ "$PREV" -gt 0 ] && RESUME="--resume $W/iter_${PREV}.pth"
  TTA=""
  [ "$STOP" -eq 20000 ] && TTA="--tta-final"
  say "SEGMENT start ${PREV}->${STOP} ${RESUME:+resume=iter_${PREV}} ${TTA:+tta_final}"
  "$PY" -u -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.0.1 --master_port "$PORT" \
    tools/portable_m2f/lusnar_row_blv_ddp_m2f.py \
    --recipe M_ROW_BLV --run-name "$RUN" --iters "$STOP" --batch 16 --workers 12 \
    --amp bf16 --crop 512 --lr 6e-5 --val-every 1000 --log-every 50 \
    --train-scenes 1,3,6,7,8 $TTA $RESUME \
    --project-root "$ROOT" --data-root "$ROOT/data" --list-dir "$LIST" --train-subdir train \
    >>"$LOG" 2>&1 < /dev/null
  rc=$?
  say "SEGMENT end ${PREV}->${STOP} exit=$rc"
  if [ "$rc" -ne 0 ]; then
    say "ABNORMAL exit=$rc at stop ${STOP} — segmentation loop STOPPED, scene kept, NO auto-resume"
    exit "$rc"
  fi
  PREV=$STOP
  if [ "$STOP" -lt 20000 ]; then
    say "REST 10 minutes (2026-11-16 rule: ~1h train -> 10 min rest)"
    sleep 600
  fi
done
say "ALL DONE $RUN exit=0"
exit 0
