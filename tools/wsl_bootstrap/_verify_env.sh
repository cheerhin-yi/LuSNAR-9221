#!/usr/bin/env bash
ps -ef | grep -E 'setup_mmcv|pip install' | grep -v grep | head
echo ---
/opt/miniconda3/envs/lusnar/bin/python - <<'PY'
import sys
sys.path.insert(0, "/mnt/d/LuSNAR_Seg/tools/portable")
import mmcv_stub
import torch, mmseg, mmcv
print(sys.version.split()[0], torch.__version__, torch.cuda.device_count(), mmseg.__version__, mmcv.__version__, torch.cuda.is_available())
print("OK" if torch.cuda.is_available() else "NOCUDA")
PY