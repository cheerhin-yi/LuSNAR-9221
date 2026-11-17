#!/usr/bin/env bash
# Native-Ubuntu follow-up for LT_B5_crop_20k (adapted from 11b_b5_post_wsl.sh).
# Waits for the torchrun PID (arg1) to exit, then:
#   lt_postprocess (iter_20000.pth -> last.pth, summary_full.json)
#   -> TTA sanity check -> official I1a (B5 scorer, unchanged protocol)
#   -> last.pth on GPU0 + best.pth on GPU1 in parallel. No plus5 chained.
set -uo pipefail
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF || true
ROOT=/run/media/cui/数据/LuSNAR_Seg
export PYTHONPATH="$ROOT/tools/portable"
PY=$HOME/miniconda3/envs/lusnar/bin/python
cd "$ROOT" || exit 90

TPID=$1
RUN=LT_B5_crop_20k
W="$ROOT/work_dirs/$RUN"
LOG="$ROOT/logs/${RUN}_ubuntu_resume.log"
QLOG="$ROOT/logs/${RUN}_post_ubuntu.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }

say "post-waiter start pid=$$ waiting for torchrun pid=$TPID"
while kill -0 "$TPID" 2>/dev/null; do sleep 20; done

st=$("$PY" - "$W/summary.json" <<'PY'
import json,sys
try:
    s=json.load(open(sys.argv[1],encoding="utf-8")); print("ok" if (s.get("error") is None and s.get("iters")==s.get("target_iters")) else "bad")
except Exception: print("missing")
PY
)
echo "=== $RUN torchrun ended $(date '+%F %T') summary=$st ===" >>"$LOG"
say "torchrun gone summary=$st"

"$PY" "$ROOT/tools/wsl_bootstrap/_lt/lt_postprocess.py" "$W" 20000 >>"$QLOG" 2>&1 || say "postprocess failed"

"$PY" - "$W/tta_final.json" <<'PY' 2>&1 | tee -a "$QLOG"
import json,sys
try:
    t=json.load(open(sys.argv[1],encoding="utf-8"))
    m=t.get("mIoU")
    print("TTA_CHECK","mIoU=",m,"best_iter=",t.get("best_iter"),"OK" if (m is not None and m>0.6) else "WARN: TTA error or implausible mIoU",(t.get("error","")[:300] if t.get("error") else ""))
except Exception as e: print("TTA_CHECK WARN missing/unreadable", e)
PY

sleep 10

score() {  # GPU CKPT TAG
  CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4 "$PY" -u tools/portable/lusnar_small_crater_score_sc_b5.py \
    --checkpoint "$2" --tag "$3" --project-root "$ROOT" --data-root "$ROOT/data/train" \
    --out-dir "$ROOT/results/small_crater_sc" > "$ROOT/logs/score_$3_ubuntu.log" 2>&1 < /dev/null
}

p0=""; p1=""
if [ -f "$W/last.pth" ]; then score 0 "work_dirs/$RUN/last.pth" "${RUN}_last" & p0=$!; say "SCORE last.pth started pid=$p0"; else say "no last.pth"; fi
if [ -f "$W/best.pth" ]; then score 1 "work_dirs/$RUN/best.pth" "${RUN}_best" & p1=$!; say "SCORE best.pth started pid=$p1"; else say "no best.pth"; fi
[ -n "$p0" ] && { wait $p0; say "SCORE ${RUN}_last exit=$?"; }
[ -n "$p1" ] && { wait $p1; say "SCORE ${RUN}_best exit=$?"; }
say "ALL $RUN DONE summary=$st (no plus5 chained)"