import json
from pathlib import Path
p = Path(r"D:\LuSNAR_Seg\work_dirs\train_ROW_BLV_20k\train_log.jsonl")
rows = []
for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        o = json.loads(line)
    except Exception:
        continue
    if "iter" in o and "elapsed_s" in o:
        rows.append(o)
segs = []
cur = []
for o in rows:
    if cur and o["iter"] < cur[-1]["iter"]:
        segs.append(cur)
        cur = [o]
    else:
        cur.append(o)
if cur:
    segs.append(cur)
for i, s in enumerate(segs):
    a, b = s[0], s[-1]
    di = b["iter"] - a["iter"]
    de = b["elapsed_s"] - a["elapsed_s"]
    if di > 0:
        print(f"seg{i}: iter {a['iter']}->{b['iter']} ({di}) elapsed {a['elapsed_s']:.1f}->{b['elapsed_s']:.1f} ({de:.1f}s) => {de/di:.3f} s/iter n={len(s)}")
# mid-segment rates excluding first 50 after resume
for i, s in enumerate(segs):
    if len(s) < 4:
        continue
    a, b = s[1], s[-1]
    di = b["iter"] - a["iter"]
    de = b["elapsed_s"] - a["elapsed_s"]
    if di > 0:
        print(f"seg{i}_steady: {a['iter']}->{b['iter']} => {de/di:.3f} s/iter")
