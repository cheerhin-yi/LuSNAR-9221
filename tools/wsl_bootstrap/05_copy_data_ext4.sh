#!/usr/bin/env bash
# Copy train+Scene5 val data (Moon_1,3,5,6,7,8 color/label only) to WSL ext4
set -euo pipefail
SRC=/mnt/d/LuSNAR_Seg/data/train
DST=/root/lusnar_data/train
LOG=/mnt/d/LuSNAR_Seg/logs/wsl_copy_data_ext4.log
mkdir -p "$DST"
echo "=== copy start $(date -Is)" >> "$LOG"
T0=$(date +%s)
for m in Moon_1 Moon_3 Moon_5 Moon_6 Moon_7 Moon_8; do
  mkdir -p "$DST/$m"
  ( cd "$SRC/$m" && tar cf - color label ) | ( cd "$DST/$m" && tar xf - )
  echo "$m done $(date -Is) color=$(ls $DST/$m/color | wc -l)/$(ls $SRC/$m/color | wc -l) label=$(ls $DST/$m/label | wc -l)/$(ls $SRC/$m/label | wc -l)" >> "$LOG"
done
echo "=== copy done $(date -Is) secs=$(( $(date +%s)-T0 )) size=$(du -sh $DST | cut -f1)" >> "$LOG"