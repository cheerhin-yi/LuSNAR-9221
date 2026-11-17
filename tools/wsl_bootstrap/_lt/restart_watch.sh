#!/usr/bin/env bash
cd /mnt/d/LuSNAR_Seg
kill 27699 2>/dev/null; sleep 2
pgrep -af 08_gpu_watch_wsl.py && { echo still running; exit 1; }
setsid nohup python3 -u tools/wsl_bootstrap/08_gpu_watch_wsl.py >> logs/gpu_watch_daemon.log 2>&1 < /dev/null &
sleep 8
pgrep -af 08_gpu_watch_wsl.py
tail -n 3 logs/gpu_watch_events.jsonl
python3 -c "import json;d=json.load(open('logs/gpu_watch_status.json'));print(d['ts']);[print(k,d[k]['state'],d[k]['stage'],d[k]['last_iter']) for k in ('gpu0','gpu1','lt')]"