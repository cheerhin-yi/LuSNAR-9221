#!/usr/bin/env python3
"""Remap Windows fullimage smoke lists onto WSL ext4. Never touch lists/DESKTOP-ORQLB8F."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

WIN_LISTS = Path("/mnt/d/LuSNAR_Seg/fullimage_compare/lists/DESKTOP-ORQLB8F")
WIN_BANK = Path("/mnt/d/LuSNAR_Seg/data/smoke/cc_bank")
OUT_LISTS = Path("/root/lusnar_data/lists_fullimage")
OUT_BANK = Path("/root/lusnar_data/cc_bank")
OUT_TRAIN = Path("/root/lusnar_data/train")
LIST_FILES = ("train_sub.txt", "val_sub.txt", "meta.json", "rarity.json", "cc_bank_index.json")
TRAIN_N, VAL_N, SEED = 1600, 350, 42
FORBIDDEN = {2, 4, 9}


def remap_text(s: str) -> str:
    s = s.replace("D:\\\\LuSNAR_Seg\\\\data\\\\train", "/root/lusnar_data/train")
    s = s.replace("D:\\LuSNAR_Seg\\data\\train", "/root/lusnar_data/train")
    s = s.replace("D:/LuSNAR_Seg/data/train", "/root/lusnar_data/train")
    s = s.replace("D:\\\\LuSNAR_Seg\\\\data\\\\smoke\\\\cc_bank", "/root/lusnar_data/cc_bank")
    s = s.replace("D:\\LuSNAR_Seg\\data\\smoke\\cc_bank", "/root/lusnar_data/cc_bank")
    s = s.replace("D:/LuSNAR_Seg/data/smoke/cc_bank", "/root/lusnar_data/cc_bank")
    s = s.replace("\\", "/")
    return s


def scene_of(p: str):
    m = re.search(r"Moon_(\d+)", p.replace("\\", "/"))
    return int(m.group(1)) if m else None


def copy_bank() -> int:
    OUT_BANK.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(WIN_BANK.glob("*.npz")):
        dst = OUT_BANK / src.name
        if (not dst.exists()) or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        n += 1
    return n


def main() -> int:
    if not WIN_LISTS.is_dir():
        raise SystemExit(f"missing Windows lists {WIN_LISTS}")
    if not OUT_TRAIN.is_dir():
        raise SystemExit(f"missing ext4 train {OUT_TRAIN}")
    copied = copy_bank()
    print(f"cc_bank copied/verified {copied} npz -> {OUT_BANK}", flush=True)
    OUT_LISTS.mkdir(parents=True, exist_ok=True)
    for name in LIST_FILES:
        src = WIN_LISTS / name
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        text = remap_text(src.read_text(encoding="utf-8"))
        if "D:" in text or "d:" in text:
            raise SystemExit(f"unmapped Windows path still in {name}")
        (OUT_LISTS / name).write_text(text, encoding="utf-8")
    meta = json.loads((OUT_LISTS / "meta.json").read_text(encoding="utf-8"))
    meta["scene_root"] = str(OUT_TRAIN)
    meta["note"] = "Scene5 val_sub only; no Moon_2/4/9; remapped to WSL ext4"
    (OUT_LISTS / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def load_pairs(path: Path):
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[\t ]+", line, maxsplit=1)
            if len(parts) != 2:
                raise SystemExit(f"bad list line in {path}: {line}")
            rows.append((parts[0], parts[1]))
        return rows

    train = load_pairs(OUT_LISTS / "train_sub.txt")
    val = load_pairs(OUT_LISTS / "val_sub.txt")
    if len(train) != TRAIN_N or len(val) != VAL_N or int(meta.get("seed", -1)) != SEED:
        raise SystemExit(f"protocol mismatch train={len(train)} val={len(val)} seed={meta.get('seed')}")
    bad_val = [ip for ip, _ in val if scene_of(ip) != 5]
    bad_tr = [ip for ip, _ in train if scene_of(ip) not in (1, 3, 6, 7, 8)]
    leak = [p for p, _ in train + val if scene_of(p) in FORBIDDEN]
    if bad_val or bad_tr or leak:
        raise SystemExit(f"scene leak val={bad_val[:3]} train={bad_tr[:3]} forbidden={leak[:3]}")

    bank = json.loads((OUT_LISTS / "cc_bank_index.json").read_text(encoding="utf-8"))
    bank_files = [p for v in bank.values() for p in (v if isinstance(v, list) else [v])]
    missing = []
    for ip, lp in train + val:
        if not os.path.exists(ip):
            missing.append(ip)
        if not os.path.exists(lp):
            missing.append(lp)
    for p in bank_files:
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        raise SystemExit(f"{len(missing)} files missing, e.g. {missing[:3]}")
    print(
        f"lists ready at {OUT_LISTS}: train={len(train)} val={len(val)} "
        f"bank={len(bank_files)} missing=0",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
