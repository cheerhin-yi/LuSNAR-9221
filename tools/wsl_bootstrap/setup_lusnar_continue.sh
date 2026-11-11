#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
LOG=/mnt/d/LuSNAR_Seg/logs/wsl_setup_lusnar.log
exec > >(tee -a "$LOG") 2>&1
echo "=== setup CONTINUE $(date -Is) ==="
MC=/opt/miniconda3
source "$MC/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
conda config --set always_yes yes
conda config --set channel_priority flexible
# prefer conda-forge to avoid ToS friction for python if needed
if ! conda env list | grep -qE '^lusnar\s'; then
  echo "[3b] create lusnar python=3.10"
  conda create -n lusnar python=3.10 -y || conda create -n lusnar python=3.10 -y -c conda-forge
fi
conda activate lusnar
echo "[4/7] torch + mmcv/mmseg"
pip install -U pip
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -U openmim
mim install "mmcv==2.1.0" || pip install "mmcv==2.1.0"
pip install "mmsegmentation==1.2.2" "mmengine>=0.10.0" opencv-python-headless pillow numpy scipy tqdm
echo "[5/7] verify"
python - <<'PY'
import torch, mmseg, mmcv, sys
print("py", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("mmseg", mmseg.__version__, "mmcv", mmcv.__version__)
assert torch.cuda.is_available(), "CUDA not visible in WSL"
assert torch.cuda.device_count() >= 2, "need 2 GPUs"
print("cuda0", torch.zeros(1, device="cuda:0").device)
print("cuda1", torch.zeros(1, device="cuda:1").device)
PY
echo "[6/7] symlink + user"
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
ls /mnt/d/LuSNAR_Seg/data/train/Moon_1/color | head
test -f /mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/iter_4000.pth
echo "OK: env lusnar ready"
echo "=== setup end $(date -Is) ==="