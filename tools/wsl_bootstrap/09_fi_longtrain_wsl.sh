#!/usr/bin/env bash
# Full-image (native 1024x1024 whole image, crop=1024 -> no crop) ROW_BLV longtrain, B2 protocol, WSL 2-GPU DDP.
#   Stage 1: LT_FI_ROW_BLV_20k        train scenes 1/3/6/7/8, val = Scene5 whole image every 1000, best.pth + iter_*.pth,
#            iter_20000.pth copied to last.pth, --tta-final (same sliding 512/256 x 0.75/1.0/1.25 TTA as B2) on best.pth.
#   Stage 2: LT_FI_ROW_BLV_20k_plus5  same hyperparams, train scenes 1/3/5/6/7/8, --no-val, last.pth = deliverable.
#            Chained automatically only if stage 1 exits 0 (CHAIN_PLUS5=1, default). No Scene5 selection in stage 2.
# Trainer: tools/portable/lusnar_row_blv_ddp.py (the B2 train_ROW_BLV_20k DDP trainer, unmodified) with --crop 1024.
# Recipe M_ROW_BLV == FI_M_ROW_BLV smoke recipe (same recipe_flags, RowHeightAttn, BLV sigma 0.5, CE+Dice, ACW+crater boost,
# RCS crater weights, mosaic, CC paste, hflip, bf16, AdamW lr 6e-5 wd 0.01). Data = full scene lists (like B2), not the 1600 subset.
# Waits for: GPU0 Mask2Former queue finished (ALL M2F 5k DONE, or its PID gone) AND no fullimage_smoke.py AND both GPUs < 4 GB used.
# Never: PYTORCH_CUDA_ALLOC_CONF, Moon_2/4/9, overwrite of existing work dirs, official submit.
# Open-source Moon_2/4/9 eval after stage 2 is a separate step (see 09_fi_longtrain_README.md).
set -uo pipefail
export USE_LIBUV=0 MASTER_ADDR=127.0.0.1 PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=0
unset PYTORCH_CUDA_ALLOC_CONF || true
ROOT=/mnt/d/LuSNAR_Seg
export PYTHONPATH="$ROOT/tools/portable${PYTHONPATH:+:$PYTHONPATH}"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lusnar
cd "$ROOT"

DATA_ROOT=/root/lusnar_data
BATCH="${LT_BATCH:-8}"
WORKERS="${LT_WORKERS:-12}"
ITERS=20000
VAL_EVERY=1000
CHAIN_PLUS5="${CHAIN_PLUS5:-1}"
M2F_QPID="${M2F_QPID:-27518}"
RUN1=LT_FI_ROW_BLV_20k
RUN2=LT_FI_ROW_BLV_20k_plus5
LIST1=$DATA_ROOT/lists_fi_lt
LIST2=$DATA_ROOT/lists_fi_lt_plus5
WLOG=$ROOT/logs/LT_FI_ROW_BLV_waiter_wsl.log
LOG1=$ROOT/logs/LT_FI_ROW_BLV_20k_wsl.log
LOG2=$ROOT/logs/LT_FI_ROW_BLV_20k_plus5_wsl.log
M2F_LOG=$ROOT/logs/backbone_m2f_5k_wsl.log
PP=$ROOT/tools/wsl_bootstrap/_lt/lt_postprocess.py

say() { echo "[$(date '+%F %T')] $*" | tee -a "$WLOG"; }

say "waiter start pid=$$ batch/gpu=$BATCH workers/rank=$WORKERS iters=$ITERS val_every=$VAL_EVERY chain_plus5=$CHAIN_PLUS5 m2f_qpid=$M2F_QPID"
for r in "$RUN1" "$RUN2"; do
  if [ -e "$ROOT/work_dirs/$r" ]; then say "REFUSE: work_dirs/$r exists (never overwrite)"; exit 2; fi
done

m2f_done() {
  grep -q "ALL M2F 5k DONE" "$M2F_LOG" 2>/dev/null && return 0
  kill -0 "$M2F_QPID" 2>/dev/null && return 1
  return 0
}
gpus_free() {
  local m
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk 'BEGIN{x=0}{if($1+0>4000)x=1}END{print x}')
  [ "$m" = "0" ]
}
last_note=0
while true; do
  a=0; b=0; c=0
  m2f_done && a=1
  pgrep -f 'fullimage_smoke.py' >/dev/null 2>&1 || b=1
  if [ $a = 1 ] && [ $b = 1 ] && ! pgrep -f 'lusnar_smoke_compare.py|lusnar_row_blv_ddp.py' >/dev/null 2>&1 && gpus_free; then c=1; fi
  [ $c = 1 ] && break
  now=$(date +%s)
  if [ $((now - last_note)) -ge 1800 ]; then
    say "waiting: m2f_done=$a no_fullimage_smoke=$b gpus=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    last_note=$now
  fi
  sleep 60
done
grep -q "ALL M2F 5k DONE" "$M2F_LOG" 2>/dev/null || say "WARN: M2F queue gone without 'ALL M2F 5k DONE' (crash?) -> launching anyway, GPU0 free"
sleep 20

prep_list() {  # $1 dst list dir, $2 src list dir with RCS cache for the same scenes (trainer verifies pairs)
  mkdir -p "$1"
  if [ ! -f "$1/full_crater_weights.json" ] && [ -f "$2/full_crater_weights.json" ]; then cp -p "$2/full_crater_weights.json" "$1/"; fi
}

port_free() { ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$1$"; }

train() {  # $1 run $2 log $3 port $4 list ; rest = extra args
  local run=$1 log=$2 port=$3 list=$4; shift 4
  if [ -e "$ROOT/work_dirs/$run" ]; then say "REFUSE: work_dirs/$run exists"; return 2; fi
  port_free "$port" || { say "port $port busy"; return 3; }
  say "LAUNCH $run port=$port list=$list batch/gpu=$BATCH workers=$WORKERS extra=$*"
  echo "=== $run start $(date '+%F %T') batch/gpu=$BATCH x2 GPUs crop=1024(whole image) lr=6e-5 workers=$WORKERS iters=$ITERS val_every=$VAL_EVERY $* ===" >>"$log"
  python -u -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.0.1 --master_port "$port" \
    tools/portable/lusnar_row_blv_ddp.py \
    --recipe M_ROW_BLV --run-name "$run" --iters "$ITERS" --batch "$BATCH" --workers "$WORKERS" \
    --amp bf16 --crop 1024 --lr 6e-5 --val-every "$VAL_EVERY" --log-every 50 \
    --project-root "$ROOT" --data-root "$DATA_ROOT" --list-dir "$list" --train-subdir train "$@" \
    >>"$log" 2>&1 < /dev/null
  local rc=$?
  echo "=== $run exit $rc $(date '+%F %T') ===" >>"$log"
  say "END $run exit=$rc"
  python "$PP" "$ROOT/work_dirs/$run" "$ITERS" >>"$WLOG" 2>&1 || say "postprocess failed for $run"
  return $rc
}

# ---- Stage 1 ----
prep_list "$LIST1" "$DATA_ROOT/lists"
train "$RUN1" "$LOG1" 29641 "$LIST1" --tta-final
rc1=$?
if [ $rc1 -ne 0 ]; then say "STAGE1 FAILED exit=$rc1 -> not chaining plus5"; exit $rc1; fi
say "STAGE1 DONE $RUN1"

# ---- Stage 2 (plus5) ----
# Manual launch if not chained: CHAIN_PLUS5=1 and stage 1 skipped is not supported here; run the train() call below by hand:
#   same command with --train-scenes 1,3,5,6,7,8 --no-val, run LT_FI_ROW_BLV_20k_plus5, list dir lists_fi_lt_plus5, port 29651.
if [ "$CHAIN_PLUS5" != "1" ]; then say "CHAIN_PLUS5=$CHAIN_PLUS5 -> stop after stage 1"; exit 0; fi
sleep 30
prep_list "$LIST2" "$DATA_ROOT/lists_plus5"
train "$RUN2" "$LOG2" 29651 "$LIST2" --train-scenes 1,3,5,6,7,8 --no-val
rc2=$?
if [ $rc2 -ne 0 ]; then say "STAGE2 FAILED exit=$rc2"; exit $rc2; fi
say "ALL LT_FI DONE (stage1 $RUN1 + stage2 $RUN2). Next: open-source Moon_2/4/9 eval of $RUN2/last.pth (separate step, not automatic)"