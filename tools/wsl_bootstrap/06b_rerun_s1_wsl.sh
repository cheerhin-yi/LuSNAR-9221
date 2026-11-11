#!/usr/bin/env bash
# GPU0 only: rerun S1 SegFormer-B5 2k from scratch + S1 TTA after cuDNN crash.
# Does not touch the full-image queue, STOP file, S0, train_B1, or first-round / plus5 weights.
# Never sets PYTORCH_CUDA_ALLOC_CONF. Single visible GPU. No DDP. No official submit.
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
  --iters 2000
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
  echo "[$(date '+%F %T')] START $tag $*" | tee -a "$LOGDIR/backbone_smoke_wsl.log"
  if "${COMMON[@]}" "$@" 2>&1 | tee -a "$log"; then
    echo "[$(date '+%F %T')] DONE $tag" | tee -a "$LOGDIR/backbone_smoke_wsl.log"
    return 0
  fi
  echo "[$(date '+%F %T')] FAIL $tag exit $?" | tee -a "$LOGDIR/backbone_smoke_wsl.log"
  return 1
}

pick_ckpt() {
  local dir="$1"
  for n in best.pth iter_2000.pth last.pth; do
    if [[ -f "$dir/$n" ]]; then
      echo "$dir/$n"
      return 0
    fi
  done
  return 1
}

echo "[$(date '+%F %T')] S1 rerun from scratch on GPU0 (batch=16 workers=6); GPU1 not touched" | tee -a "$LOGDIR/backbone_smoke_wsl.log"
run_one S1_B5_ROW_BLV --exp M_ROW_BLV --backbone b5 --workers 6 --run-name S1_B5_ROW_BLV
CKPT1=$(pick_ckpt "$ROOT/work_dirs/smoke_S1_B5_ROW_BLV")
echo "[$(date '+%F %T')] S1 ckpt=$CKPT1" | tee -a "$LOGDIR/backbone_smoke_wsl.log"
run_one S1_B5_TTA --exp M_TTA --backbone b5 --workers 4 --run-name S1_B5_TTA --ckpt "$CKPT1"
echo "[$(date '+%F %T')] ALL S1 RERUN DONE; Mask2Former not started (needs a side env, lusnar left unchanged)" | tee -a "$LOGDIR/backbone_smoke_wsl.log"
