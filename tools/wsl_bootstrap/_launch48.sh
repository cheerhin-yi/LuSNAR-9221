#!/usr/bin/env bash
cd /mnt/d/LuSNAR_Seg
WORKERS=8 bash tools/wsl_bootstrap/03_resume_train_wsl.sh 2>&1 | grep -v '^[0-9]* .*lusnar_row_blv_ddp.py --recipe' | tail -n 30