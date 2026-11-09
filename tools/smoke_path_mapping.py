# -*- coding: utf-8 -*-
"""Minimal path+mapping smoke: read 1 train + 1 val pair, verify RGB->ID."""
from pathlib import Path
import sys
sys.path.insert(0, r"D:\LuSNAR_Seg\tools")
import numpy as np
import cv2

# Import mapping from train script by exec of constants only
ns = {}
src = Path(r"D:\LuSNAR_Seg\tools\train_segformer_b2_phase1.py").read_text(encoding="utf-8")
# pull COLOR_TO_ID / CLASS_NAMES / rgb_label_to_index by running until functions defined
# Safer: duplicate expected mapping and also import from data_check
from data_check import COLOR_TO_ID, CLASS_NAMES, color_to_index, BGR_TO_ID

assert CLASS_NAMES == ["月壤", "岩石", "撞击坑", "山体", "天空"], CLASS_NAMES
assert COLOR_TO_ID[(232, 250, 80)] == 1
assert COLOR_TO_ID[(120, 0, 200)] == 2

train_line = Path(r"D:\LuSNAR_Seg\data\train_scenes.txt").read_text(encoding="utf-8").splitlines()[0]
val_line = Path(r"D:\LuSNAR_Seg\data\val_scene5.txt").read_text(encoding="utf-8").splitlines()[0]
for tag, line in [("train", train_line), ("val", val_line)]:
    ip, lp = line.split("\t")
    assert "LuSNAR_train" in ip and "image0" not in ip, ip
    assert "LuSNAR_train" in lp and "image0" not in lp, lp
    img = cv2.imread(ip, cv2.IMREAD_COLOR)
    lab = cv2.imread(lp, cv2.IMREAD_COLOR)
    assert img is not None and lab is not None, (ip, lp)
    assert img.shape[:2] == lab.shape[:2] == (1024, 1024), (img.shape, lab.shape)
    idx = color_to_index(lab)
    present = sorted({int(x) for x in np.unique(idx) if x != 255})
    print(f"{tag}: ok path={Path(ip).name} classes_present={present} names={[CLASS_NAMES[i] for i in present]}")
    # synthetic color check
    for rgb, cid in COLOR_TO_ID.items():
        pix = np.zeros((1,1,3), np.uint8)
        pix[0,0] = (rgb[2], rgb[1], rgb[0])  # BGR
        got = int(color_to_index(pix)[0,0])
        assert got == cid, (rgb, cid, got)
print("SMOKE_OK")
