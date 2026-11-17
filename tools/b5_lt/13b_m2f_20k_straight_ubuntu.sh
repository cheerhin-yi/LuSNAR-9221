#!/usr/bin/env bash
# M2F (HF Mask2Former Swin-B) crop 20k — STRAIGHT single run, NATIVE UBUNTU.
# 2026-11-16 16:15 user decision: 「取消10分钟限制，让卡一次性跑完，
# 我想借这个机会重新测试卡的稳定性」. The segmented wrapper 13_m2f_20k_ubuntu.sh
# was killed WITHOUT touching its running child (segment 1 runs to its target
# normally — zero progress lost, no torn-checkpoint risk). This script waits for
# that torchrun to exit cleanly at its --iters target (default segment-1 = 3000),
# verifies iter_<target>.pth integrity, then runs ONE torchrun <target> -> 20000
# with --tta-final. No rests, no stops, continuous full load (stability test).
# Safety: any abnormal state (child crash/hang, missing or torn checkpoint,
# ABNORMAL marker in the old segment log, GPU lost) => REFUSE, keep scene,
# NO auto-resume (值守 reports immediately — this is the evidence the user wants).
# Power guard intentionally REMOVED (user authorized 450W full load 2026-11-16).
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
LOG="$ROOT/logs/${RUN}_ubuntu_straight.log"
OLDLOG="$ROOT/logs/${RUN}_ubuntu.log"
HFW="$ROOT/pretrained/hf/mask2former-swin-base-IN21k-ade-semantic"
STOPS="3000 6000 9000 12000 15000 18000"   # completed-segment targets of the old wrapper

refuse() { echo "[$(date '+%F %T')] REFUSE: $*" | tee -a "$LOG"; exit "$1"; }
say()    { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# --- preflight ---
[ -x "$PY" ] || refuse 7 "missing python $PY"
[ -d "$HFW" ] || refuse 3 "missing HF weights $HFW"
[ -d "$W" ] || refuse 8 "work dir $W missing"
pgrep -f '13_m2f_20k_ubuntu\.sh' >/dev/null && refuse 13 "old segment wrapper still alive — kill it first"
if grep -q "ABNORMAL" "$OLDLOG" 2>/dev/null; then
  refuse 14 "ABNORMAL marker found in $OLDLOG — abnormal state, no auto-resume (report to user)"
fi
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l)
[ "$NGPU" -eq 2 ] || refuse 9 "expected 2 GPUs, nvidia-smi reports $NGPU"
nvidia-smi 2>/dev/null | grep -qi "gpu is lost" && refuse 11 "GPU is lost — do not start"
[ -f "$W/iter_20000.pth" ] || [ -f "$W/last.pth" ] && refuse 18 "run already complete (iter_20000/last.pth present)"

# --- wait for the orphaned segment to finish cleanly at its --iters target ---
TARGET=""
if pgrep -f 'lusnar_row_blv_ddp_m2f' >/dev/null; then
  TPID=$(pgrep -f 'torch\.distributed\.run' | head -1)
  if [ -n "$TPID" ]; then
    TARGET=$(tr '\0' ' ' < "/proc/$TPID/cmdline" 2>/dev/null | grep -o '\-\-iters [0-9]*' | awk '{print $2}')
    [ -z "$TARGET" ] && refuse 19 "could not read --iters from running torchrun cmdline ($TPID)"
  fi
  say "segment run still active (target=${TARGET}) — waiting for clean exit (stability test continues unbroken)"
  WAITED=0
  while pgrep -f 'lusnar_row_blv_ddp_m2f' >/dev/null; do
    sleep 10; WAITED=$((WAITED+10))
    [ $WAITED -ge 3600 ] && refuse 15 "waited 60 min, segment still running (hang?) — report, no auto-action"
  done
  say "segment run exited after ${WAITED}s of waiting"
fi
# no segment running: use the latest completed-segment checkpoint
if [ -z "$TARGET" ]; then
  LATEST=$(ls -1 "$W"/iter_*.pth 2>/dev/null | sed 's/.*iter_\([0-9]*\)\.pth/\1/' | sort -n | tail -1)
  case " $STOPS " in
    *" $LATEST "*) TARGET=$LATEST ;;
    *) refuse 16 "latest ckpt iter_${LATEST:-none} is not a completed segment target — abnormal, no auto-resume" ;;
  esac
fi

CKPT="$W/iter_${TARGET}.pth"
[ -f "$CKPT" ] || refuse 16 "iter_${TARGET}.pth missing — segment did not finish normally, no auto-resume"
if ! "$PY" -c "import sys,torch; d=torch.load(sys.argv[1],map_location='cpu',weights_only=False); assert 'model' in d and 'optimizer' in d and isinstance(d.get('iter'),int); print('[ckpt-ok] iter', d['iter'])" "$CKPT" >>"$LOG" 2>&1; then
  refuse 17 "iter_${TARGET}.pth failed integrity check (torn?) — no auto-resume"
fi
# wait up to 120s for GPU memory release after the segment exit
BUSY=1
for i in $(seq 1 24); do
  BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk 'BEGIN{x=0}{if($1+0>4000)x=1}END{print x}')
  [ "$BUSY" = "0" ] && break
  sleep 5
done
[ "$BUSY" = "0" ] || refuse 5 "GPU still busy after 120s — investigate before start"
# port frees with the exited segment — check only NOW (after the wait loop)
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$PORT$"; then
  refuse 6 "port $PORT busy"
fi

say "M2F 20k STRAIGHT start pid=$$ resume=iter_${TARGET} iters=20000 batch/gpu=16 workers/rank=12 port=$PORT power=$(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader | tr '\n' ' | ') USER-AUTHORIZED-450W goal=card-stability-test-continuous"

"$PY" -u -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.0.1 --master_port "$PORT" \
  tools/portable_m2f/lusnar_row_blv_ddp_m2f.py \
  --recipe M_ROW_BLV --run-name "$RUN" --iters 20000 --batch 16 --workers 12 \
  --amp bf16 --crop 512 --lr 6e-5 --val-every 1000 --log-every 50 \
  --train-scenes 1,3,6,7,8 --tta-final --resume "$CKPT" \
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
