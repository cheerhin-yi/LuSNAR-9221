#!/usr/bin/env bash
# SegFormer-B5 crop plus5 20k, stage 2 — NATIVE UBUNTU RESUME from iter_17000.pth.
# User-approved option C (2026-11-16): the 2026-11-15 13:58 restart of LT_B5_crop_20k_plus5
# died 2026-11-15 19:20:40 (GPU1 Xid 79 fell off bus, rank1 SIGABRT) at iter 17900/20000;
# last intact archive iter_17000.pth (19:06:13). This script resumes that SAME run in place:
#   - python argv identical to the 13:58 restart (M_ROW_BLV, 20000 iters, batch/gpu 24,
#     crop 512, lr 6e-5, bf16, workers 12, scenes 1/3/5/6/7/8, --no-val, tta-final off)
#   - plus --resume work_dirs/LT_B5_crop_20k_plus5/iter_17000.pth  -> remaining 3000 iters
#   - checkpoints every 1000 iters (18000/19000/20000) + last.pth at iter 20000
# Appends to the SAME run log so duty monitoring stays on logs/LT_B5_crop_20k_plus5_ubuntu.log.
# Never: new work dir, single-GPU fallback, Moon_2/4/9, official submit, PYTORCH_CUDA_ALLOC_CONF,
# touching LT_B5_crop_20k / B2 / other protected dirs; refuses if power limit outside 300-350W.
set -uo pipefail

ROOT=/run/media/cui/数据/LuSNAR_Seg
export PYTHONPATH="$ROOT/tools/portable"
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF || true
PY=$HOME/miniconda3/envs/lusnar/bin/python
cd "$ROOT" || exit 90

RUN=LT_B5_crop_20k_plus5
W="$ROOT/work_dirs/$RUN"
CKPT="$W/iter_17000.pth"
BATCH="${LT_BATCH:-24}"
WORKERS="${LT_WORKERS:-12}"
ITERS=20000
PORT=29702
LIST="$ROOT/data/lists_b5_plus5"
LOG="$ROOT/logs/${RUN}_ubuntu.log"
B5PRE="$ROOT/pretrained/mit_b5_20220624-658746d9.pth"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "ubuntu plus5 RESUME start pid=$$ run=$RUN batch/gpu=$BATCH workers/rank=$WORKERS iters=$ITERS"

# --- preflight (REFUSE, never degrade) ---
[ -f "$CKPT" ] || { say "REFUSE: missing $CKPT"; exit 2; }
[ -f "$B5PRE" ] || { say "REFUSE: missing $B5PRE"; exit 3; }
[ -x "$PY" ] || { say "REFUSE: missing python $PY"; exit 7; }
if [ -f "$W/last.pth" ] || [ -f "$W/iter_20000.pth" ]; then
  say "REFUSE: run already finished (last.pth / iter_20000.pth exists)"
  exit 8
fi
if pgrep -f 'lusnar_row_blv_ddp|lusnar_small_crater_score_sc|torch.distributed.run' >/dev/null; then
  say "REFUSE: training/scoring process running"
  pgrep -af 'lusnar_row_blv_ddp|lusnar_small_crater_score_sc|torch.distributed.run' | tee -a "$LOG"
  exit 4
fi
if ! nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{x=0}{if($1+0>4000)x=1}END{exit x}'; then
  say "REFUSE: GPU busy: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
  exit 5
fi
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c 'GeForce')
if [ "$NGPU" != "2" ]; then
  say "REFUSE: expected 2 healthy GPUs, nvidia-smi -L reports $NGPU"
  exit 6
fi
read -ra PLIMS <<< "$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')"
if [ "${#PLIMS[@]}" != "2" ]; then
  say "REFUSE: cannot read both GPU power limits"
  exit 9
fi
for pl in "${PLIMS[@]}"; do
  if [ "$(awk -v v="$pl" 'BEGIN{print (v>=300 && v<=350)?1:0}')" != "1" ]; then
    say "REFUSE: GPU power limit ${pl}W outside 300-350W (operator must set: sudo nvidia-smi -pl 300)"
    exit 9
  fi
done
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$PORT$"; then
  say "REFUSE: port $PORT busy"
  exit 10
fi
[ -f "$LIST/train_scenes.txt" ] || { say "REFUSE: missing $LIST/train_scenes.txt"; exit 3; }
# checkpoint integrity (CPU load; catches truncated write before torchrun spins up)
CKOK=$("$PY" - "$CKPT" <<'PYEOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert int(ck.get("iter", -1)) == 17000, f"bad iter: {ck.get('iter')}"
assert "model" in ck and "optimizer" in ck, "missing model/optimizer state"
print("iter_17000.pth ok: iter=17000 model+optimizer present", flush=True)
PYEOF
) || { say "REFUSE: checkpoint integrity check failed"; exit 11; }
say "PRECHECK $CKOK"

say "POWER before launch: $(nvidia-smi --query-gpu=index,power.limit,power.draw --format=csv,noheader | tr '\n' ' | ')"
say "RESUME from $CKPT (remaining $((ITERS-17000)) iters)"
echo "=== $RUN UBUNTU RESUME start $(date '+%F %T') trainer=lusnar_row_blv_ddp_b5.py batch/gpu=$BATCH x2 crop=512 lr=6e-5 workers=$WORKERS iters=$ITERS no-val tta-final=off scenes=1,3,5,6,7,8 resume=$CKPT ===" >>"$LOG"
"$PY" -u -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.0.1 --master_port "$PORT" \
  tools/portable/lusnar_row_blv_ddp_b5.py \
  --recipe M_ROW_BLV --run-name "$RUN" --iters "$ITERS" --batch "$BATCH" --workers "$WORKERS" \
  --amp bf16 --crop 512 --lr 6e-5 --val-every 1000 --log-every 50 --no-val \
  --train-scenes "1,3,5,6,7,8" \
  --pretrain "$B5PRE" \
  --project-root "$ROOT" --data-root "$ROOT/data" --list-dir "$LIST" --train-subdir train \
  --resume "$CKPT" \
  >>"$LOG" 2>&1 < /dev/null
rc=$?
echo "=== $RUN UBUNTU RESUME exit $rc $(date '+%F %T') ===" >>"$LOG"
say "END $RUN exit=$rc"
exit $rc
