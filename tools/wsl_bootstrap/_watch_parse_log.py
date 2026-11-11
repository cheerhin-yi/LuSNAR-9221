import json, os, glob, time
p = r"D:\LuSNAR_Seg\work_dirs\train_ROW_BLV_20k\train_log.jsonl"
print("jsonl_mtime", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p))))
print("jsonl_size", os.path.getsize(p))
rows = []
with open(p, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
print("n_rows", len(rows))
for r in rows[-20:]:
    keys = ["ts","time","event","iter","loss","elapsed_s","mem_mb","miou","mIoU","best","from_iter","resume_miou","lr","msg"]
    print({k: r.get(k) for k in keys if k in r})
iters = [r for r in rows if isinstance(r.get("iter"), int)]
if iters:
    last = iters[-1]
    print("LAST_ITER", last.get("iter"), "loss", last.get("loss"), "elapsed_s", last.get("elapsed_s"), "mem_mb", last.get("mem_mb"))
ckpts = sorted(glob.glob(r"D:\LuSNAR_Seg\work_dirs\train_ROW_BLV_20k\iter_*.pth"), key=os.path.getmtime, reverse=True)
print("NEWEST_CKPTS")
for c in ckpts[:8]:
    print(os.path.basename(c), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(c))), os.path.getsize(c))
