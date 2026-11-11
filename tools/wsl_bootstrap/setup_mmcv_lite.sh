#!/usr/bin/env bash
set -euo pipefail
LOG=/mnt/d/LuSNAR_Seg/logs/wsl_setup_lusnar.log
exec > >(tee -a "$LOG") 2>&1
echo "=== mmcv-lite + mmseg CONTINUE $(date -Is) ==="
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lusnar
python - <<'PY'
import torch, sys
print("py", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(), "avail", torch.cuda.is_available())
PY
# Windows uses mmcv-lite (no CUDA ops); project has mmcv_stub for _ext
pip install "setuptools<81" "mmcv-lite==2.1.0" || pip install "mmcv-lite==2.1.0"
pip install "mmsegmentation==1.2.2" "mmengine>=0.10.0" opencv-python-headless pillow numpy scipy tqdm
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "/mnt/d/LuSNAR_Seg/tools/portable")
import mmcv_stub  # noqa: F401
import torch, mmseg, mmcv
print("py", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("mmseg", mmseg.__version__, "mmcv", mmcv.__version__)
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 2
print("cuda0", torch.zeros(1, device="cuda:0").device)
print("cuda1", torch.zeros(1, device="cuda:1").device)
print("OK_LUSNAR_ENV")
PY
# user + symlink
if ! id lusnar >/dev/null 2>&1; then useradd -m -s /bin/bash lusnar; fi
usermod -aG sudo lusnar || true
echo 'lusnar ALL=(ALL) NOPASSWD:ALL' >/etc/sudoers.d/lusnar
chmod 440 /etc/sudoers.d/lusnar
cat >/etc/wsl.conf <<'EOF'
[boot]
systemd=true
[user]
default=lusnar
[interop]
appendWindowsPath=true
EOF
mkdir -p /home/lusnar_work
ln -sfn /mnt/d/LuSNAR_Seg /home/lusnar_work/LuSNAR_Seg
chown -R lusnar:lusnar /home/lusnar /home/lusnar_work || true
test -f /mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/iter_4000.pth
echo "OK: env lusnar ready (mmcv-lite)"
echo "=== setup end $(date -Is) ==="