#!/usr/bin/env bash
# GPU0 only: Mask2Former (Swin-B, HF) 5000-step crop smoke + TTA, same protocol as 06c (B2/B5).
# Uses side env lusnar_bb and tools/portable_m2f copies. Does not touch GPU1 / full-image queue,
# train_B1, train_ROW_BLV_20k*, or completed smoke dirs. Scene5 val only; no Moon_2/4/9.
set -euo pipefail
export USE_LIBUV=0
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset PYTORCH_CUDA_ALLOC_CONF || true

ROOT=/mnt/d/LuSNAR_Seg
PY=/home/lusnar/.conda/envs/lusnar_bb/bin/python
SCRIPT_DIR="$ROOT/tools/portable_m2f"
LISTS=/root/lusnar_data/lists_backbone
LOGDIR="$ROOT/logs"
QLOG="$LOGDIR/backbone_m2f_5k_wsl.log"
BATCH="${M2F_BATCH:-16}"
ACCUM="${M2F_ACCUM:-1}"
mkdir -p "$LOGDIR"
cd "$ROOT"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

for d in smoke_S2_M2F_ROW_BLV_5k smoke_S2_M2F_TTA_5k; do
  if [[ -e "$ROOT/work_dirs/$d" ]]; then
    echo "[$(date '+%F %T')] REFUSE: $ROOT/work_dirs/$d exists (never overwrite)" | tee -a "$QLOG"
    exit 2
  fi
done

COMMON=(
  "$PY" -u "$SCRIPT_DIR/lusnar_smoke_compare.py"
  --project-root "$ROOT"
  --data-root /root/lusnar_data
  --train-subdir train
  --smoke-dir "$LISTS"
  --skip-prepare
  --iters 5000
  --batch "$BATCH"
  --accum "$ACCUM"
  --lr 6e-5
  --tta-scales 0.75,1.0,1.25
  --tta-crop 512
  --tta-stride 256
  --backbone m2f
)

run_one() {
  local tag="$1"
  shift
  local log="$LOGDIR/${tag}_wsl.log"
  echo "[$(date '+%F %T')] START $tag $*" | tee -a "$QLOG"
  if "${COMMON[@]}" "$@" 2>&1 | tee -a "$log"; then
    echo "[$(date '+%F %T')] DONE $tag" | tee -a "$QLOG"
    return 0
  fi
  echo "[$(date '+%F %T')] FAIL $tag exit $?" | tee -a "$QLOG"
  return 1
}

pick_ckpt() {
  local dir="$1"
  for n in best.pth iter_5000.pth last.pth; do
    if [[ -f "$dir/$n" ]]; then
      echo "$dir/$n"
      return 0
    fi
  done
  return 1
}

echo "[$(date '+%F %T')] M2F 5k from scratch on GPU0 (batch=$BATCH accum=$ACCUM); GPU1 fullimage queue not touched" | tee -a "$QLOG"
run_one S2_M2F_ROW_BLV_5k --exp M_ROW_BLV --workers 6 --run-name S2_M2F_ROW_BLV_5k
CKPT2=$(pick_ckpt "$ROOT/work_dirs/smoke_S2_M2F_ROW_BLV_5k")
echo "[$(date '+%F %T')] M2F 5k ckpt=$CKPT2" | tee -a "$QLOG"
run_one S2_M2F_TTA_5k --exp M_TTA --workers 4 --run-name S2_M2F_TTA_5k --ckpt "$CKPT2"

echo "[$(date '+%F %T')] ALL M2F 5k DONE" | tee -a "$QLOG"
