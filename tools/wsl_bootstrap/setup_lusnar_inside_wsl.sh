#!/usr/bin/env bash
# Inside Ubuntu WSL: miniconda + lusnar matching Windows F:\Conda\envs\lusnar
# torch 2.7.1+cu128, mmseg 1.2.2, mmcv 2.1.0, py3.10
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

LOG=/mnt/d/LuSNAR_Seg/logs/wsl_setup_lusnar.log
mkdir -p /mnt/d/LuSNAR_Seg/logs
exec > >(tee -a "$LOG") 2>&1
echo "=== setup start $(date -Is) ==="

echo "[0/7] create user lusnar + sudo"
if ! id lusnar >/dev/null 2>&1; then
  useradd -m -s /bin/bash lusnar
fi
usermod -aG sudo lusnar || true
echo 'lusnar ALL=(ALL) NOPASSWD:ALL' >/etc/sudoers.d/lusnar
chmod 440 /etc/sudoers.d/lusnar
# default user for future interactive
mkdir -p /etc
cat >/etc/wsl.conf <<'EOF'
[boot]
systemd=true
[user]
default=lusnar
[interop]
appendWindowsPath=true
EOF

echo "[1/7] apt basics"
apt-get update -y
apt-get install -y curl wget git build-essential ca-certificates libgl1 libglib2.0-0

echo "[2/7] remount /dev/shm"
mount -o remount,size=32G /dev/shm || true
df -h /dev/shm || true

echo "[3/7] miniconda"
MC=/opt/miniconda3
if [ ! -x "$MC/bin/conda" ]; then
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$MC"
fi
# shellcheck disable=SC1091
source "$MC/etc/profile.d/conda.sh"
conda config --set always_yes yes
conda config --set channel_priority flexible
if ! conda env list | grep -q '^lusnar '; then
  conda create -n lusnar python=3.10 -y
fi
conda activate lusnar

echo "[4/7] torch 2.7.1+cu128 + mmcv/mmseg"
pip install -U pip
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -U openmim
pip install "setuptools<81" "mmcv-lite==2.1.0" || pip install "mmcv-lite==2.1.0"
pip install "mmsegmentation==1.2.2" "mmengine>=0.10.0" opencv-python-headless pillow numpy scipy tqdm

echo "[5/7] verify"
python - <<'PY'
import torch, mmseg, mmcv, sys
print("py", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("mmseg", mmseg.__version__, "mmcv", mmcv.__version__)
assert torch.cuda.is_available(), "CUDA not visible in WSL"
assert torch.cuda.device_count() >= 2, "need 2 GPUs"
x = torch.zeros(1, device="cuda:0")
print("cuda0 ok", x.device)
PY

echo "[6/7] project symlink"
mkdir -p /home/lusnar_work
ln -sfn /mnt/d/LuSNAR_Seg /home/lusnar_work/LuSNAR_Seg
chown -R lusnar:lusnar /home/lusnar /home/lusnar_work || true
ls -la /mnt/d/LuSNAR_Seg/data/train/Moon_1/color | head
ls -la /mnt/d/LuSNAR_Seg/data/lists_portable | head
test -f /mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/iter_4000.pth

echo "[7/7] done"
echo "OK: env lusnar ready"
echo "Activate: source /opt/miniconda3/etc/profile.d/conda.sh && conda activate lusnar"
echo "=== setup end $(date -Is) ==="