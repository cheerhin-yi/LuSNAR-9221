"""Build fixed smoke subsets, rarity scores, and rare-class CC bank (train scenes only)."""
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import cv2
from collections import defaultdict

ROOT = Path(r"D:\LuSNAR_Seg")
OUT = ROOT / "data" / "smoke"
OUT.mkdir(parents=True, exist_ok=True)

COLOR_TO_ID = {
    (187, 70, 156): 0,
    (232, 250, 80): 1,
    (120, 0, 200): 2,
    (173, 69, 31): 3,
    (34, 201, 248): 4,
}
BGR_TO_ID = {(c[2], c[1], c[0]): i for c, i in COLOR_TO_ID.items()}
IGNORE = 255
SEED = 42
TRAIN_N = 1600
VAL_N = 350
RARE = {1, 2}  # rock, crater
MAX_CC_PER_CLASS = 400
MAX_CC_SIDE = 256
MIN_CC_AREA = 64


def rgb_label_to_index(lab_bgr: np.ndarray) -> np.ndarray:
    out = np.full(lab_bgr.shape[:2], IGNORE, np.uint8)
    for bgr, cid in BGR_TO_ID.items():
        m = np.all(lab_bgr == np.asarray(bgr, np.uint8), axis=-1)
        out[m] = cid
    return out


def load_pairs(list_file: Path):
    pairs = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        a, b = line.split("\t") if "\t" in line else line.split()
        pairs.append((a, b))
    return pairs


def main():
    rng = random.Random(SEED)
    train_all = load_pairs(ROOT / "data" / "train_scenes.txt")
    val_all = load_pairs(ROOT / "data" / "val_scene5.txt")
    assert len(train_all) == 6403, len(train_all)
    assert len(val_all) == 1599, len(val_all)

    train_idx = list(range(len(train_all)))
    val_idx = list(range(len(val_all)))
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    train_sub = [train_all[i] for i in train_idx[:TRAIN_N]]
    val_sub = [val_all[i] for i in val_idx[:VAL_N]]

    def write_list(path, pairs):
        path.write_text("\n".join(f"{a}\t{b}" for a, b in pairs) + "\n", encoding="utf-8")

    write_list(OUT / "train_sub.txt", train_sub)
    write_list(OUT / "val_sub.txt", val_sub)

    # rarity + class hist on train_sub; also CC bank
    rarity = []
    class_pix = np.zeros(5, np.float64)
    bank = {1: [], 2: []}  # list of dicts with paths later saved as npz index

    bank_dir = OUT / "cc_bank"
    bank_dir.mkdir(exist_ok=True)
    counts = {1: 0, 2: 0}

    for i, (ip, lp) in enumerate(train_sub):
        lab = cv2.imread(lp, cv2.IMREAD_COLOR)
        if lab is None:
            raise RuntimeError(lp)
        mask = rgb_label_to_index(lab)
        valid = mask != IGNORE
        n_valid = int(valid.sum()) or 1
        rare_frac = float(((mask == 1) | (mask == 2)).sum()) / n_valid
        rarity.append({"img": ip, "lab": lp, "rare_frac": rare_frac, "idx": i})
        for c in range(5):
            class_pix[c] += int((mask == c).sum())

        img = None
        for cid in (1, 2):
            if counts[cid] >= MAX_CC_PER_CLASS:
                continue
            binm = (mask == cid).astype(np.uint8)
            if binm.sum() < MIN_CC_AREA:
                continue
            nlab, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
            for lab_id in range(1, nlab):
                if counts[cid] >= MAX_CC_PER_CLASS:
                    break
                x, y, w, h, area = stats[lab_id]
                if area < MIN_CC_AREA:
                    continue
                if w > MAX_CC_SIDE or h > MAX_CC_SIDE:
                    # center crop
                    cx, cy = x + w // 2, y + h // 2
                    x0 = max(0, cx - MAX_CC_SIDE // 2)
                    y0 = max(0, cy - MAX_CC_SIDE // 2)
                    x1 = min(mask.shape[1], x0 + MAX_CC_SIDE)
                    y1 = min(mask.shape[0], y0 + MAX_CC_SIDE)
                    x, y = x0, y0
                    w, h = x1 - x0, y1 - y0
                if img is None:
                    img = cv2.imread(ip, cv2.IMREAD_COLOR)
                comp = (labels[y:y+h, x:x+w] == lab_id)
                if comp.sum() < MIN_CC_AREA:
                    continue
                patch_img = img[y:y+h, x:x+w].copy()
                patch_msk = np.full((h, w), IGNORE, np.uint8)
                patch_msk[comp] = cid
                # only keep component pixels in image (optional zero elsewhere already)
                fn = bank_dir / f"c{cid}_{counts[cid]:04d}.npz"
                np.savez_compressed(fn, img=patch_img, mask=patch_msk, cid=cid)
                bank[cid].append(str(fn))
                counts[cid] += 1

        if (i + 1) % 200 == 0:
            print(f"scanned {i+1}/{len(train_sub)} bank={counts}", flush=True)

    # sampling weights ~ rare_frac + eps, then inverse-sqrt style boost
    fracs = np.array([r["rare_frac"] for r in rarity], np.float64)
    # weight = (rare_frac + 0.02)^(-0.5) inverted: higher rare -> higher weight
    weights = (fracs + 0.02) ** 0.5  # proportional to sqrt rarity
    weights = weights / weights.sum()

    meta = {
        "seed": SEED,
        "train_n": TRAIN_N,
        "val_n": VAL_N,
        "class_pix": class_pix.tolist(),
        "class_freq": (class_pix / class_pix.sum()).tolist(),
        "bank_counts": counts,
        "note": "Scene5 val_sub only; no Moon_2/4/9",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "rarity.json").write_text(json.dumps({"items": rarity, "weights": weights.tolist()}, ensure_ascii=False), encoding="utf-8")
    (OUT / "cc_bank_index.json").write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", meta, flush=True)


if __name__ == "__main__":
    main()
