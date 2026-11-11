#!/bin/bash
# Wait for cache done, then attrib + cpu + I4 + gpu I2/I3 + report
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lusnar
cd /mnt/d/LuSNAR_Seg
export PYTHONUNBUFFERED=1
LOG=logs/scene5_infer_phase_followup.log
exec > >(tee -a "$LOG") 2>&1

echo "[followup] waiting for CACHE_DONE ..."
while ! grep -q CACHE_DONE logs/scene5_cache_diag.log 2>/dev/null; do
  sleep 30
  n=$(ls /root/lusnar_data/cache/scene5_tta_probs 2>/dev/null | wc -l)
  echo "[followup] probs=$n $(date)"
done
echo "[followup] cache finished at $(date)"

echo "[followup] attrib"
python -u tools/portable/lusnar_scene5_cache_diag.py --stage attrib \
  --val-list /root/lusnar_data/lists/val_scene5.txt \
  --preds-dir /root/lusnar_data/cache/scene5_preds \
  --out-json /mnt/d/LuSNAR_Seg/results/inference_phase/scene5_error_attrib.json

echo "[followup] cpu protocols"
python -u tools/portable/lusnar_scene5_protocols.py --stage cpu

echo "[followup] I4 hybrid full-image 1024 + tiles (A select, B once)"
mkdir -p /root/lusnar_data/cache/scene5_full1024_logits
python -u tools/portable/lusnar_scene5_i4.py

echo "[followup] gpu protocols I2/I3"
python -u tools/portable/lusnar_scene5_protocols.py --stage gpu

echo "[followup] write report"
python -u tools/portable/lusnar_scene5_write_report.py

echo FOLLOWUP_DONE $(date)
