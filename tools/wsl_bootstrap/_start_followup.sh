#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lusnar
cd /mnt/d/LuSNAR_Seg
nohup bash tools/wsl_bootstrap/_run_scene5_followup.sh > /dev/null 2>&1 &
echo $! > logs/scene5_followup_wrapper.pid
echo WRAPPER=$!
