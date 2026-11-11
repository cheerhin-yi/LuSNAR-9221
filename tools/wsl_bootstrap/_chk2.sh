#!/bin/bash
echo "cwd 3176=$(readlink /proc/3176/cwd)"
echo "cwd 1144=$(readlink /proc/1144/cwd)"
ls -la /mnt/d/LuSNAR_Seg/logs/ | sed -n '1,40p'
echo ===
# show if followup log has CR name
ls -lb /mnt/d/LuSNAR_Seg/logs/ | grep follow
# convert followup script to LF
sed -i 's/\r$//' /mnt/d/LuSNAR_Seg/tools/wsl_bootstrap/_run_scene5_followup_i4.sh
sed -i 's/\r$//' /mnt/d/LuSNAR_Seg/tools/portable/lusnar_scene5_i4.py
file /mnt/d/LuSNAR_Seg/tools/wsl_bootstrap/_run_scene5_followup_i4.sh
