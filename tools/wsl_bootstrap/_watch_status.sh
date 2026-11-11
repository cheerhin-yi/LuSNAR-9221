#!/bin/bash
set -u
echo "=== TIME ==="
date
echo "=== TRAIN / WAITER PROCS ==="
ps -ef | grep -E 'lusnar_row_blv_ddp|torch.distributed.run|run_opensource_followup_wsl|run_error_attrib_after_wsl' | grep -v grep || echo "NO_MATCHING_PROCS"
echo "=== PORTS ==="
ss -lptn 2>/dev/null | grep -E '29601|29621' || true
echo "=== DATALOADER CHILDREN ==="
pgrep -af 'lusnar_row_blv_ddp' || true
echo "=== GPU ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm --format=csv
echo "=== SHM / MEM ==="
df -h /dev/shm
free -h
echo "=== DMESG TAIL INTERESTING ==="
dmesg -T 2>/dev/null | tail -80 | grep -iE 'oom|killed process|out of memory|dxgk|nvrm|oom-kill' || echo "no oom/kill keywords"
echo "=== WSL UPTIME ==="
uptime
