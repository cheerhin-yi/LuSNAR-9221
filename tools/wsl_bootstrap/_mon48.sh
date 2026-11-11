#!/usr/bin/env bash
read a b c d e f g h rest < <(grep '^cpu ' /proc/stat); t1=$((b+c+d+e+f+g+h)); i1=$((e+f))
sleep 10
read a b c d e f g h rest < <(grep '^cpu ' /proc/stat); t2=$((b+c+d+e+f+g+h)); i2=$((e+f))
echo "WSL CPU busy% (10s): $(( 100*( (t2-t1)-(i2-i1) )/(t2-t1) )) iowait-incl idle; loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
free -g | sed -n 2p; df -h /dev/shm | tail -1
grep -v '"event"' /mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/train_log.jsonl | tail -n 6 | cut -c1-150
dmesg | grep -iE 'oom|killed' | tail -3
grep -iE 'error|Traceback|killed' /mnt/d/LuSNAR_Seg/logs/train_ROW_BLV_20k.log | tail -n 3