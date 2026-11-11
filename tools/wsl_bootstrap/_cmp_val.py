import json, glob, os
root = r"D:\LuSNAR_Seg\work_dirs"
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d, "train_log.jsonl")
    if not os.path.exists(p):
        continue
    vals = []
    for line in open(p, encoding="utf-8", errors="ignore"):
        try:
            o = json.loads(line)
        except Exception:
            continue
        keys = [k for k in o if "miou" in k.lower()]
        if keys and "iter" in o:
            vals.append((o["iter"], {k: o[k] for k in keys}, {k: o[k] for k in o if k.startswith("iou") or k in ("per_class","class_iou")}))
    if vals:
        print("==", d)
        seen = {}
        for it, m, pc in vals:
            seen[it] = (m, pc)
        for it in sorted(seen):
            print(it, seen[it][0], seen[it][1] if seen[it][1] else "")
