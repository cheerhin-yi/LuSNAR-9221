#!/usr/bin/env bash
# GPU0 only: rerun today's B2 control and SegFormer-B5 at 5000 steps + TTA.
# Does not touch the full-image queue, STOP, 2k work dirs, train_B1, or plus5 weights.
set -euo pipefail
export USE_LIBUV=0
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset PYTORCH_CUDA_ALLOC_CONF || true

ROOT=/mnt/d/LuSNAR_Seg
PY=/opt/miniconda3/envs/lusnar/bin/python
LISTS=/root/lusnar_data/lists_backbone
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"
cd "$ROOT"
export PYTHONPATH="$ROOT/tools/portable:${PYTHONPATH:-}"

COMMON=(
  "$PY" -u "$ROOT/tools/portable/lusnar_smoke_compare.py"
  --project-root "$ROOT"
  --data-root /root/lusnar_data
  --train-subdir train
  --smoke-dir "$LISTS"
  --skip-prepare
  --iters 5000
  --batch 16
  --lr 6e-5
  --tta-scales 0.75,1.0,1.25
  --tta-crop 512
  --tta-stride 256
)

run_one() {
  local tag="$1"
  shift
  local log="$LOGDIR/${tag}_wsl.log"
  echo "[$(date '+%F %T')] START $tag $*" | tee -a "$LOGDIR/backbone_5k_wsl.log"
  if "${COMMON[@]}" "$@" 2>&1 | tee -a "$log"; then
    echo "[$(date '+%F %T')] DONE $tag" | tee -a "$LOGDIR/backbone_5k_wsl.log"
    return 0
  fi
  echo "[$(date '+%F %T')] FAIL $tag exit $?" | tee -a "$LOGDIR/backbone_5k_wsl.log"
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

echo "[$(date '+%F %T')] B2+B5 5k from scratch on GPU0; GPU1 fullimage queue not touched" | tee -a "$LOGDIR/backbone_5k_wsl.log"
run_one S0_B2_ROW_BLV_5k --exp M_ROW_BLV --backbone b2 --workers 6 --run-name S0_B2_ROW_BLV_5k
CKPT0=$(pick_ckpt "$ROOT/work_dirs/smoke_S0_B2_ROW_BLV_5k")
echo "[$(date '+%F %T')] B2 5k ckpt=$CKPT0" | tee -a "$LOGDIR/backbone_5k_wsl.log"
run_one S0_B2_TTA_5k --exp M_TTA --backbone b2 --workers 4 --run-name S0_B2_TTA_5k --ckpt "$CKPT0"

run_one S1_B5_ROW_BLV_5k --exp M_ROW_BLV --backbone b5 --workers 6 --run-name S1_B5_ROW_BLV_5k
CKPT1=$(pick_ckpt "$ROOT/work_dirs/smoke_S1_B5_ROW_BLV_5k")
echo "[$(date '+%F %T')] B5 5k ckpt=$CKPT1" | tee -a "$LOGDIR/backbone_5k_wsl.log"
run_one S1_B5_TTA_5k --exp M_TTA --backbone b5 --workers 4 --run-name S1_B5_TTA_5k --ckpt "$CKPT1"

echo "[$(date '+%F %T')] ALL B2/B5 5k DONE" | tee -a "$LOGDIR/backbone_5k_wsl.log"
