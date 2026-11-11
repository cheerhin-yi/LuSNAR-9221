#!/usr/bin/env bash
# Create /opt/miniconda3/envs/lusnar_bb as a clone of lusnar, then install
# transformers==4.49.0 without touching torch or the lusnar env.
# CPU/disk only. Never sets PYTORCH_CUDA_ALLOC_CONF. Never uses the GPUs.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
unset PYTORCH_CUDA_ALLOC_CONF || true

ROOT=/mnt/d/LuSNAR_Seg
LOGDIR="$ROOT/logs"
ENV=/opt/miniconda3/envs/lusnar_bb
CONDA=/opt/miniconda3/bin/conda
PY_BB="$ENV/bin/python"
PIP_BB="$ENV/bin/pip"
WEIGHTS="$ROOT/pretrained/hf/mask2former-swin-base-IN21k-ade-semantic"
mkdir -p "$LOGDIR"
cd "$ROOT"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/lusnar_bb_setup.log"; }

if [[ -x "$PY_BB" ]]; then
  log "lusnar_bb already exists, skip clone"
else
  log "cloning lusnar -> lusnar_bb (nice/ionice idle)"
  nice -n 19 ionice -c 3 "$CONDA" create -y -n lusnar_bb --clone lusnar
  log "clone done"
fi

# Record torch before pip, refuse if it changes.
BEFORE=$("$PY_BB" -c "import torch,mmseg,sys; print(torch.__version__); print(mmseg.__version__); print(sys.version.split()[0])")
log "pre-pip"$'\n'"$BEFORE"

log "pip install transformers==4.49.0 (no torch upgrade)"
# Prefer a wheel cache copied from Windows if present.
WHEELDIR="$ROOT/pretrained/hf/wheels"
PIP_ARGS=(install --disable-pip-version-check "transformers==4.49.0" "timm==1.0.15" --upgrade-strategy only-if-needed)
if [[ -d "$WHEELDIR" ]] && ls "$WHEELDIR"/*.whl >/dev/null 2>&1; then
  log "using local wheels in $WHEELDIR"
  "$PIP_BB" "${PIP_ARGS[@]}" --no-index --find-links "$WHEELDIR" || "$PIP_BB" "${PIP_ARGS[@]}" --find-links "$WHEELDIR"
else
  "$PIP_BB" "${PIP_ARGS[@]}"
fi

AFTER=$("$PY_BB" -c "import torch,mmseg,transformers,sys; print('torch', torch.__version__); print('mmseg', mmseg.__version__); print('transformers', transformers.__version__); print('cuda_build', torch.version.cuda); print('py', sys.version.split()[0])")
log "post-pip"$'\n'"$AFTER"
echo "$AFTER" | grep -q "torch 2.7.1+cu128" || { log "ERROR: torch changed"; exit 3; }

log "CPU import test"
CUDA_VISIBLE_DEVICES="" "$PY_BB" - <<'PY'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import transformers
from transformers import Mask2FormerConfig, Mask2FormerImageProcessor
print("transformers", transformers.__version__, "torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("Mask2FormerConfig ok", Mask2FormerConfig().model_type)
print("processor class ok", Mask2FormerImageProcessor.__name__)
PY

if [[ -f "$WEIGHTS/config.json" ]]; then
  log "CPU load weights from $WEIGHTS"
  CUDA_VISIBLE_DEVICES="" "$PY_BB" - <<PY
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
p = r"$WEIGHTS"
proc = Mask2FormerImageProcessor.from_pretrained(p, local_files_only=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(p, local_files_only=True, torch_dtype=torch.float32)
model.eval()
n = sum(x.numel() for x in model.parameters())
print("loaded", p, "params", n, "device", next(model.parameters()).device)
# tiny random forward on CPU
x = torch.randn(1, 3, 64, 64)
with torch.no_grad():
    out = model(pixel_values=x)
print("forward ok", type(out).__name__)
PY
else
  log "weights not yet at $WEIGHTS; skip load test (download may still be running)"
fi

log "lusnar_bb setup ALL DONE"
