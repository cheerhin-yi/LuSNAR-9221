#!/usr/bin/env bash
uptime; free -g; df -h /dev/shm /root; nproc
ls -la /root/lusnar_data/lists
for f in /root/lusnar_data/lists/*.txt; do echo == $f; wc -l < $f; head -n2 $f; done
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate lusnar
python - <<'PY'
import json,os
d=json.load(open('/root/lusnar_data/lists/full_crater_weights.json'))
print('keys',list(d.keys())); p=d['pairs']; print(len(p), p[0])
print('exists', all(os.path.exists(x) for x in p[0]))
bad=[x for x in p if not x[0].startswith('/root/lusnar_data')]; print('bad',len(bad))
PY
pgrep -af python || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
ls -la /mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/ | tail -n 12
grep -n 'crater_weights\|list.dir\|pairs' /mnt/d/LuSNAR_Seg/tools/portable/lusnar_row_blv_ddp.py /mnt/d/LuSNAR_Seg/tools/portable/_lusnar_common.py | head -40