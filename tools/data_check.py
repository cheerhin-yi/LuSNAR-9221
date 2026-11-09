"""LuSNAR Phase1 data check: mapping, integrity, stats for scenes 1/3/5/6/7/8 only.
Official layout: data/raw/LuSNAR_train/Moon_{s}/{color,label}/*.png (no image0/).
"""
from __future__ import annotations
import json, os, csv, hashlib
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import cv2

ROOT = Path(r"D:\LuSNAR_Seg\data\raw\LuSNAR_train")
OUT = Path(r"D:\LuSNAR_Seg\results")
OUT.mkdir(parents=True, exist_ok=True)
SCENES = [1, 3, 5, 6, 7, 8]
CLASS_NAMES = ["月壤", "岩石", "撞击坑", "山体", "天空"]  # official submission order

# Official submission IDs (RGB hex); OpenCV loads BGR
COLOR_TO_ID = {
    (187, 70, 156): 0,   # #BB469C 月壤
    (232, 250, 80): 1,   # #E8FA50 岩石
    (120, 0, 200): 2,    # #7800C8 撞击坑
    (173, 69, 31): 3,    # #AD451F 山体
    (34, 201, 248): 4,   # #22C9F8 天空
}
BGR_TO_ID = { (c[2], c[1], c[0]): i for c, i in COLOR_TO_ID.items() }

def scene_dirs(s: int):
    base = ROOT / f"Moon_{s}"
    return base / "color", base / "label"

def color_to_index(lab_bgr: np.ndarray) -> np.ndarray:
    h, w = lab_bgr.shape[:2]
    out = np.full((h, w), 255, dtype=np.uint8)  # 255=ignore/invalid
    for bgr, cid in BGR_TO_ID.items():
        mask = np.all(lab_bgr == np.array(bgr, dtype=np.uint8), axis=-1)
        out[mask] = cid
    return out

def phash_gray(img, hash_size=8):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (hash_size+1, hash_size), interpolation=cv2.INTER_AREA)
    diff = g[:, 1:] > g[:, :-1]
    return sum([2**i for i, v in enumerate(diff.flatten()) if v])

def hamming(a, b):
    return bin(a ^ b).count("1")

def main():
    report = {"scenes": {}, "mapping_note": {}, "issues": []}
    report["mapping_note"] = {
        "official_submission_order": "0月壤 1岩石 2撞击坑 3山体 4天空",
        "used_in_this_pipeline": "official_submission_order",
        "data_root": str(ROOT),
        "layout": "Moon_{s}/color|label/*.png (no image0)",
        "palette_RGB": {str(k): v for k, v in COLOR_TO_ID.items()},
    }
    stats_rows = []
    all_invalid = 0
    near_dups = []

    for s in SCENES:
        cdir, ldir = scene_dirs(s)
        if not cdir.exists() or not ldir.exists():
            report["issues"].append(f"Moon_{s}: missing color/label dirs")
            continue
        colors = sorted(cdir.glob("*.png"))
        labels = sorted(ldir.glob("*.png"))
        cset = {p.stem for p in colors}
        lset = {p.stem for p in labels}
        missing_lab = sorted(cset - lset)
        missing_col = sorted(lset - cset)
        paired = sorted(cset & lset)
        class_pix = np.zeros(5, dtype=np.int64)
        invalid_pix = 0
        res_counter = Counter()
        for stem in paired:
            lab = cv2.imread(str(ldir / f"{stem}.png"), cv2.IMREAD_COLOR)
            if lab is None:
                report["issues"].append(f"Moon_{s}: unreadable label {stem}")
                continue
            res_counter[f"{lab.shape[1]}x{lab.shape[0]}"] += 1
            idx = color_to_index(lab)
            invalid = int((idx == 255).sum())
            invalid_pix += invalid
            for cid in range(5):
                class_pix[cid] += int((idx == cid).sum())
        size_mismatch = 0
        for stem in paired[::max(1, len(paired)//50)][:50]:
            im = cv2.imread(str(cdir / f"{stem}.png"), cv2.IMREAD_COLOR)
            lab = cv2.imread(str(ldir / f"{stem}.png"), cv2.IMREAD_COLOR)
            if im is None or lab is None or im.shape[:2] != lab.shape[:2]:
                size_mismatch += 1
        step = max(1, len(paired)//200)
        hashes = []
        for stem in paired[::step]:
            im = cv2.imread(str(cdir / f"{stem}.png"), cv2.IMREAD_COLOR)
            if im is None: continue
            hashes.append((stem, phash_gray(im)))
        dup_pairs = 0
        for i in range(len(hashes)-1):
            if hamming(hashes[i][1], hashes[i+1][1]) <= 5:
                dup_pairs += 1
                if len(near_dups) < 20:
                    near_dups.append({"scene": s, "a": hashes[i][0], "b": hashes[i+1][0]})
        total = int(class_pix.sum() + invalid_pix)
        frac = (class_pix / total * 100).tolist() if total else [0]*5
        row = {
            "scene": s,
            "n_color": len(colors),
            "n_label": len(labels),
            "n_paired": len(paired),
            "missing_label": len(missing_lab),
            "missing_color": len(missing_col),
            "resolutions": dict(res_counter),
            "size_mismatch_sampled": size_mismatch,
            "invalid_pixels": invalid_pix,
            "class_pixels": class_pix.tolist(),
            "class_pct": [round(x, 3) for x in frac],
            "near_dup_adjacent_sampled": dup_pairs,
            "sample_step": step,
        }
        report["scenes"][str(s)] = row
        stats_rows.append(row)
        all_invalid += invalid_pix
        print(f"Moon_{s}: paired={len(paired)} invalid_pix={invalid_pix} pct={row['class_pct']}", flush=True)

    report["near_duplicate_examples"] = near_dups
    report["total_invalid_pixels"] = all_invalid

    c5, l5 = scene_dirs(5)
    stems = sorted({p.stem for p in c5.glob("*.png")} & {p.stem for p in l5.glob("*.png")})
    ok = 0
    for stem in stems[::max(1,len(stems)//5)][:5]:
        im = cv2.imread(str(c5/f"{stem}.png"))
        lab = cv2.imread(str(l5/f"{stem}.png"))
        idx = color_to_index(lab)
        assert im is not None and lab is not None and im.shape[:2]==lab.shape[:2]
        assert idx.min() >= 0 and (idx[idx<255].max() <= 4 if (idx<255).any() else True)
        ok += 1
    report["scene5_load_check"] = {"checked": ok, "ok": True}

    (OUT / "data_check_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(OUT / "class_pixel_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene","n_paired","res"] + [f"pix_{n}" for n in CLASS_NAMES] + [f"pct_{n}" for n in CLASS_NAMES] + ["invalid_pix"])
        for r in stats_rows:
            w.writerow([r["scene"], r["n_paired"], r["resolutions"]] + r["class_pixels"] + r["class_pct"] + [r["invalid_pixels"]])
    lines = ["| 场景 | 样本数 | 分辨率 | 月壤% | 岩石% | 撞击坑% | 山体% | 天空% | 无效像素 |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in stats_rows:
        res = ",".join(r["resolutions"].keys())
        p = r["class_pct"]
        lines.append(f"| Moon_{r['scene']} | {r['n_paired']} | {res} | {p[0]} | {p[1]} | {p[2]} | {p[3]} | {p[4]} | {r['invalid_pixels']} |")
    (OUT / "class_pixel_stats.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("WROTE", OUT/"data_check_report.json")
    print("ISSUES", report["issues"])

if __name__ == "__main__":
    main()
