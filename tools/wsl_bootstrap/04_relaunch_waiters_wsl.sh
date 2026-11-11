#!/usr/bin/env bash
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lusnar
ROOT=/mnt/d/LuSNAR_Seg
cd "$ROOT"
mkdir -p logs
export PYTHONUNBUFFERED=1
export USE_LIBUV=0
pkill -f 'run_opensource_followup' 2>/dev/null || true
pkill -f 'run_error_attrib_after' 2>/dev/null || true
sleep 1
nohup python -u tools/wsl_bootstrap/run_opensource_followup_wsl.py >>logs/opensource_followup.log 2>&1 &
echo "FOLLOWUP_PID=$!"
nohup python -u tools/wsl_bootstrap/run_error_attrib_after_wsl.py >>logs/error_attrib_waiter.log 2>&1 &
echo "ATTRIB_PID=$!"
sleep 2
pgrep -af 'run_opensource_followup|run_error_attrib_after' || true