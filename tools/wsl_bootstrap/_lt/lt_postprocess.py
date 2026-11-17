#!/usr/bin/env python3
"""Post-process a lusnar_row_blv_ddp.py longtrain work dir (read-only on ckpts except last.pth copy).
Writes <work>/summary_full.json: full Scene5 val curve (per-class, English keys), best/last,
trainer summary.json, tta_final.json, module_effect.json. Copies iter_<N>.pth -> last.pth if missing."""
import json, shutil, sys
from pathlib import Path
ZH2EN = {"\u6708\u58e4": "regolith", "\u5ca9\u77f3": "rock", "\u649e\u51fb\u5751": "crater", "\u5c71\u4f53": "mountain", "\u5929\u7a7a": "sky"}
work = Path(sys.argv[1]); target = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
def rj(p):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
curve, train_pts = [], []
lp = work / "train_log.jsonl"
if lp.exists():
    for line in lp.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "val_mIoU_scene5" in r:
            curve.append({"iter": r["iter"], "mIoU": r["val_mIoU_scene5"],
                          "per_class_IoU": {ZH2EN.get(k, k): v for k, v in (r.get("per_class_IoU") or {}).items()}})
        elif "loss" in r:
            train_pts.append({k: r.get(k) for k in ("iter", "loss", "ce", "dice", "elapsed_s", "mem_mb")})
last_src = work / f"iter_{target}.pth"
if not (work / "last.pth").exists() and last_src.exists():
    shutil.copy2(last_src, work / "last.pth")
best = max(curve, key=lambda r: r["mIoU"]) if curve else None
out = {"work_dir": str(work), "target_iters": target,
       "best": best, "last": curve[-1] if curve else None, "val_curve_scene5": curve,
       "train_curve": train_pts,
       "checkpoints": sorted(p.name for p in work.glob("*.pth")),
       "trainer_summary": rj(work / "summary.json"), "tta_final": rj(work / "tta_final.json"),
       "module_effect": rj(work / "module_effect.json"),
       "note": "Local Scene5 whole-image validation (stage 1) / no val (plus5). Not an official score. Moon_2/4/9 never used in training."}
(work / "summary_full.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("summary_full written", work, "best", best and best["iter"], best and best["mIoU"])