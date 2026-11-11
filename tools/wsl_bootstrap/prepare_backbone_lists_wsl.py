#!/usr/bin/env python3
"""Remap historical 1600/350 smoke lists onto ext4 for WSL backbone smokes.

Never overwrites D:\\LuSNAR_Seg\\data\\smoke. Writes /root/lusnar_data/lists_backbone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path("/mnt/d/LuSNAR_Seg/data/smoke")
DST = Path("/root/lusnar_data/lists_backbone")
TRAIN_DST = Path("/root/lusnar_data/train")
BANK_DST = Path("/root/lusnar_data/cc_bank")
FORBIDDEN = ("Moon_2", "Moon_4", "Moon_9")


def remap_path(p: str) -> str:
    s = p.replace("\\", "/")
    for old, new in (
        ("D:/LuSNAR_Seg/data/train/", str(TRAIN_DST) + "/"),
        ("D:/LuSNAR_Seg/data/smoke/cc_bank/", str(BANK_DST) + "/"),
        ("/mnt/d/LuSNAR_Seg/data/train/", str(TRAIN_DST) + "/"),
        ("/mnt/d/LuSNAR_Seg/data/smoke/cc_bank/", str(BANK_DST) + "/"),
    ):
        if s.lower().startswith(old.lower()):
            s = new + s[len(old):]
            break
    return s


def scene_of(path: str) -> str | None:
    parts = path.replace("\\", "/").split("/")
    for tok in parts:
        if tok.startswith("Moon_"):
            return tok
    return None


def remap_txt(src: Path, dst: Path) -> list[tuple[str, str]]:
    rows = []
    missing = 0
    scenes = {}
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        a, b = line.split("\t") if "\t" in line else line.split()
        a2, b2 = remap_path(a), remap_path(b)
        if not Path(a2).is_file() or not Path(b2).is_file():
            missing += 1
        sc = scene_of(a2)
        scenes[sc] = scenes.get(sc, 0) + 1
        rows.append((a2, b2))
    dst.write_text("".join(f"{a}\t{b}\n" for a, b in rows), encoding="utf-8")
    return rows, missing, scenes


def main() -> int:
    if not SRC.is_dir():
        print(f"missing source {SRC}", file=sys.stderr)
        return 2
    DST.mkdir(parents=True, exist_ok=True)
    train_rows, train_miss, train_sc = remap_txt(SRC / "train_sub.txt", DST / "train_sub.txt")
    val_rows, val_miss, val_sc = remap_txt(SRC / "val_sub.txt", DST / "val_sub.txt")

    rarity = json.loads((SRC / "rarity.json").read_text(encoding="utf-8"))
    for item in rarity.get("items", []):
        for k in ("img", "lab"):
            if k in item:
                item[k] = remap_path(item[k])
    (DST / "rarity.json").write_text(json.dumps(rarity, ensure_ascii=False), encoding="utf-8")

    meta = json.loads((SRC / "meta.json").read_text(encoding="utf-8"))
    (DST / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    bank = json.loads((SRC / "cc_bank_index.json").read_text(encoding="utf-8"))
    bank_miss = 0
    n_bank = 0
    for _k, files in bank.items():
        if not isinstance(files, list):
            continue
        for i, p in enumerate(files):
            files[i] = remap_path(p)
            n_bank += 1
            if not Path(files[i]).is_file():
                bank_miss += 1
    (DST / "cc_bank_index.json").write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")

    bad_train = [s for s in train_sc if s in FORBIDDEN]
    bad_val = [s for s in val_sc if s != "Moon_5"]
    report = {
        "dst": str(DST),
        "train_n": len(train_rows),
        "val_n": len(val_rows),
        "train_scenes": train_sc,
        "val_scenes": val_sc,
        "train_missing": train_miss,
        "val_missing": val_miss,
        "bank_n": n_bank,
        "bank_missing": bank_miss,
    }
    (DST / "remap_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if train_miss or val_miss or bank_miss:
        print("ERROR: missing files after remap", file=sys.stderr)
        return 3
    if len(train_rows) != 1600 or len(val_rows) != 350:
        print("ERROR: unexpected list sizes", file=sys.stderr)
        return 4
    if bad_train or bad_val:
        print(f"ERROR: forbidden/unexpected scenes train={bad_train} val={bad_val}", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
