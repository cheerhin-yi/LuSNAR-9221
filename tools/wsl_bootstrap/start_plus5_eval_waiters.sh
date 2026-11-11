#!/usr/bin/env bash
# Arm post-plus5 eval waiters only. Do not use 04_relaunch_waiters_wsl.sh.
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lusnar
ROOT=/mnt/d/LuSNAR_Seg
cd "$ROOT"
mkdir -p logs
export PYTHONUNBUFFERED=1
export USE_LIBUV=0
pkill -f 'run_opensource_followup_wsl.py' 2>/dev/null || true
pkill -f 'run_opensource_followup_wsl_stage1.py' 2>/dev/null || true
pkill -f 'run_opensource_followup_wsl_plus5.py' 2>/dev/null || true
pkill -f 'run_error_attrib_after' 2>/dev/null || true
sleep 1
# never start the old followup that calls run_plus5()
nohup python -u tools/wsl_bootstrap/run_opensource_followup_wsl_plus5.py >>logs/opensource_followup_plus5.log 2>&1 &
echo "PLUS5_EVAL_WAITER_PID=$!"
nohup python -u tools/wsl_bootstrap/run_error_attrib_after_wsl.py >>logs/error_attrib_waiter.log 2>&1 &
echo "ATTRIB_PID=$!"
sleep 2
pgrep -af 'run_opensource_followup_wsl_plus5|run_error_attrib_after' || true
tail -n 8 logs/opensource_followup_plus5.log || true
