#!/usr/bin/env bash
# WSL fullimage 1024 smoke queue on DESKTOP-ORQLB8F. Single visible GPU (the idle 4090).
# Does not launch Windows plus5, does not touch train_B1 or train_ROW_BLV_20k, does not submit.
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lusnar

ROOT=/mnt/d/LuSNAR_Seg
FI="$ROOT/fullimage_compare"
PY=/opt/miniconda3/envs/lusnar/bin/python
SCRIPT="$FI/scripts/fullimage_smoke.py"
LISTS=/root/lusnar_data/lists_fullimage
LOGS="$FI/logs"
STOP="$FI/STOP"
HOST=DESKTOP-ORQLB8F-WSL
WORKERS="${WORKERS:-12}"
MEM_CAP_GB="${MEM_CAP_GB:-40}"
HEADROOM_GB="${HEADROOM_GB:-4}"
ITERS="${ITERS:-2000}"
GPU="${GPU:-1}"
PROBE_BATCHES="${PROBE_BATCHES:-1,2,4,8,12,16}"
PROBE_EXPS="FI_V_B1C,FI_M_ROW_BLV,FI_M_DYSAMPLE,FI_M_CTRAST,FI_M_SAPA"
EXPS="${EXPS:-FI_V_B1C,FI_M_ROW_BLV,FI_M_ROW,FI_M_BLV,FI_M_SHAD,FI_M_ABL,FI_M_INVFORM,FI_V_B1E,FI_M_HARD,FI_M_DYSAMPLE,FI_M_CTRAST,FI_M_SAPA,FI_M_FADE}"

cd "$ROOT"
mkdir -p "$LOGS" "$FI/results" "$FI/work_dirs"
QLOG="$LOGS/queue_${HOST}.log"
log() { echo "$(date +'%Y-%m-%d %H:%M:%S') $*" | tee -a "$QLOG"; }

export COMPUTERNAME="$HOST"
export CUDA_VISIBLE_DEVICES="$GPU"
export USE_LIBUV=0
export FI_ALLOW_MNT=1
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
unset PYTORCH_CUDA_ALLOC_CONF || true

if pgrep -af 'fullimage_smoke.py' | grep -v "pgrep" >/dev/null; then
  log "fullimage_smoke.py already running; exit"
  exit 4
fi

# User asked to resume the fullimage line in WSL; drop the leftover STOP/lock from the Windows abort.
if [ -f "$STOP" ]; then
  log "removing leftover STOP so the WSL queue can start"
  rm -f "$STOP"
fi
rm -f "$LOGS/queue.lock"

log "=== fullimage WSL queue start host=$HOST gpu=$GPU workers=$WORKERS iters=$ITERS mem_cap=${MEM_CAP_GB}G headroom=${HEADROOM_GB}G ==="
log "prepare lists"
"$PY" -u /mnt/d/LuSNAR_Seg/tools/wsl_bootstrap/prepare_fullimage_lists_wsl.py | tee -a "$QLOG"

PROBE_JSON="$FI/results/vram_probe_${HOST}.json"
need_probe=1
if [ -f "$PROBE_JSON" ]; then
  need_probe=$("$PY" -c "import json,sys; p=json.load(open(sys.argv[1],encoding='utf-8')); ex=set(p.get('exps',{})); need=['FI_V_B1C','FI_M_ROW_BLV','FI_M_DYSAMPLE','FI_M_CTRAST','FI_M_SAPA']; print(int((not p.get('recommended_common_micro_batch')) or any(x not in ex for x in need)))" "$PROBE_JSON")
fi
if [ "$need_probe" = "1" ]; then
  log "start vram_probe batches=$PROBE_BATCHES"
  set +e
  "$PY" -u "$SCRIPT" --vram-probe --exps "$PROBE_EXPS" --workers "$WORKERS" \
    --mem-cap-gb "$MEM_CAP_GB" --headroom-gb "$HEADROOM_GB" --probe-iters 30 \
    --probe-batches "$PROBE_BATCHES" --project-root "$ROOT" --lists-dir "$LISTS" \
    >>"$LOGS/vram_probe_wsl.log" 2>>"$LOGS/vram_probe_wsl.err.log"
  rc=$?
  set -e
  log "end vram_probe exit=$rc"
  if [ "$rc" = "3" ]; then log "STOP during probe; queue ends"; exit 3; fi
  if [ "$rc" != "0" ]; then log "probe failed; see $LOGS/vram_probe_wsl.err.log"; exit "$rc"; fi
else
  log "probe already done: $PROBE_JSON"
fi

MICRO=$("$PY" -c "import json,sys; print(int(json.load(open(sys.argv[1],encoding='utf-8'))['recommended_common_micro_batch']))" "$PROBE_JSON")
if [ -z "$MICRO" ] || [ "$MICRO" -le 0 ]; then
  log "probe found no micro batch; inspect $PROBE_JSON"
  exit 1
fi
log "using micro=$MICRO grad_accum=1 effective=$MICRO lr=6e-5 (never rescaled) for ALL exps"

IFS=',' read -r -a EXP_ARR <<< "$EXPS"
for e in "${EXP_ARR[@]}"; do
  if [ -f "$STOP" ]; then log "STOP file seen between experiments; queue ends"; exit 3; fi
  summ="$FI/work_dirs/$e/summary.json"
  if [ -f "$summ" ]; then
    st=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8-sig')).get('status',''))" "$summ")
    if [ "$st" = "done" ]; then log "skip $e (summary.json done)"; continue; fi
  fi
  log "start $e micro=$MICRO workers=$WORKERS"
  set +e
  "$PY" -u "$SCRIPT" --exps "$e" --batch "$MICRO" --grad-accum 1 --iters "$ITERS" \
    --workers "$WORKERS" --lr 6e-5 --seed 42 --val-every 500 --ckpt-every 500 \
    --project-root "$ROOT" --lists-dir "$LISTS" \
    >>"$LOGS/${e}_wsl.log" 2>>"$LOGS/${e}_wsl.err.log"
  rc=$?
  set -e
  log "end $e exit=$rc"
  if [ "$rc" = "3" ]; then log "STOP during $e; queue ends"; exit 3; fi
  if [ "$rc" != "0" ]; then log "WARN $e exit=$rc; continuing"; fi
done
log "=== fullimage WSL queue finished ==="
