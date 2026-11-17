#!/usr/bin/env bash
# M2F (HF Mask2Former Swin-B) open-source Moon_2/4/9 eval — NATIVE UBUNTU.
# 2026-11-16 23:00 user approved: 「现在开始跑」 → score LT_M2F_crop_20k/best.pth
# (iter 4000, official I1a 0.88985) on Moon_2/4/9 with the SAME frozen protocol
# stack used for opensource_b5_plus5_last (single / C0 / I1a / P1 / P1P2,
# scales 0.75/1.0/1.25, sliding 512/256, bf16).
# One-shot eval only: reads Moon_2/4/9 labels AFTER prediction; never trains,
# never tunes. Power: no lock, 300W per-GPU cap as configured.
set -uo pipefail

ROOT=/run/media/cui/数据/LuSNAR_Seg
export PYTHONPATH="$ROOT/tools/portable:$ROOT/tools/portable_m2f"
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF || true
PY=$HOME/miniconda3/envs/lusnar_bb/bin/python
cd "$ROOT" || exit 90

RUN=LT_M2F_crop_20k
CKPT="$ROOT/work_dirs/$RUN/best.pth"
OUT="$ROOT/results/opensource_m2f_20k_best"
LOG="$ROOT/logs/opensource_m2f_20k_best.log"
HFW="$ROOT/pretrained/hf/mask2former-swin-base-IN21k-ade-semantic"

refuse() { echo "[$(date '+%F %T')] REFUSE: $*" | tee -a "$LOG"; exit "$1"; }
say()    { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# --- preflight ---
[ -x "$PY" ] || refuse 7 "missing python $PY"
[ -d "$HFW" ] || refuse 3 "missing HF weights $HFW"
[ -s "$CKPT" ] || refuse 8 "checkpoint missing $CKPT"
[ -f "$OUT/summary.json" ] && refuse 18 "already evaluated ($OUT/summary.json exists)"
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l)
[ "$NGPU" -eq 2 ] || refuse 9 "expected 2 GPUs, nvidia-smi reports $NGPU"
nvidia-smi 2>/dev/null | grep -qi "gpu is lost" && refuse 11 "GPU is lost — do not start"
pgrep -f 'lusnar_row_blv_ddp|lusnar_small_crater_score|lusnar_opensource_final_eval' >/dev/null \
  && refuse 13 "another lusnar train/score job is alive — wait for it"

say "preflight ok: ckpt=$CKPT out=$OUT"
nvidia-smi --query-gpu=index,pci.bus_id,power.limit,temperature.gpu --format=csv,noheader | tee -a "$LOG"

"$PY" tools/portable/lusnar_opensource_final_eval_m2f.py \
  --project-root "$ROOT" \
  --checkpoint "$CKPT" \
  --out-dir "$OUT" \
  --data-root "$ROOT/data" \
  --gpus 2 \
  --ref-pred-root "" 2>&1 | tee -a "$LOG"

RC=${PIPESTATUS[0]}
say "END opensource_m2f_20k_best exit=$RC"
exit "$RC"