"""Rebuild train/val pair lists for official LuSNAR_train layout.
Layout: D:\LuSNAR_Seg\data\raw\LuSNAR_train\Moon_{s}\{color,label}\*.png
Train scenes 1/3/6/7/8; val Scene 5. Tab-separated paths.
(Old zip/image0 extractor retired — official packages use flat color/label layout.)
"""
from pathlib import Path

ROOT = Path(r"D:\LuSNAR_Seg")
DATA_ROOT = ROOT / "data" / "raw" / "LuSNAR_train"
TRAIN_SCENES = [1, 3, 6, 7, 8]
VAL_SCENES = [5]


def build_list(scenes):
    pairs = []
    for s in scenes:
        cdir = DATA_ROOT / f"Moon_{s}" / "color"
        ldir = DATA_ROOT / f"Moon_{s}" / "label"
        if not cdir.exists() or not ldir.exists():
            raise FileNotFoundError(f"missing dirs for Moon_{s}: {cdir} / {ldir}")
        cset = {p.stem for p in cdir.glob("*.png")}
        lset = {p.stem for p in ldir.glob("*.png")}
        for stem in sorted(cset & lset):
            pairs.append((str(cdir / f"{stem}.png"), str(ldir / f"{stem}.png")))
        missing = sorted(cset - lset) + sorted(lset - cset)
        if missing:
            print(f"WARN Moon_{s} unpaired stems={len(missing)} e.g. {missing[:3]}")
    return pairs


def write_list(path: Path, pairs):
    path.write_text("\n".join(f"{a}\t{b}" for a, b in pairs) + "\n", encoding="utf-8")
    print(f"WROTE {path} lines={len(pairs)}")


def main():
    train = build_list(TRAIN_SCENES)
    val = build_list(VAL_SCENES)
    write_list(ROOT / "data" / "train_scenes.txt", train)
    write_list(ROOT / "data" / "val_scene5.txt", val)
    print(f"TOTAL train+val={len(train)+len(val)} (expect 8002)")


if __name__ == "__main__":
    main()
