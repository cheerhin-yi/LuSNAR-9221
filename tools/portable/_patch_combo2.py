# -*- coding: utf-8 -*-
"""Add M_BLV_SHAD and M_ROW_BLV recipes (pairwise train combos)."""
from pathlib import Path

COMMON = Path(r"D:\LuSNAR_Seg\tools\portable\_lusnar_common.py")
SMOKE = Path(r"D:\LuSNAR_Seg\tools\portable\lusnar_smoke_compare.py")

c = COMMON.read_text(encoding="utf-8")
old_train = '''    train_method = (
        "M_ROW", "M_BLV", "M_SHAD", "M_ROW_SHAD", "M_ABL", "M_INVFORM", "M_FADE",
    )'''
new_train = '''    train_method = (
        "M_ROW", "M_BLV", "M_SHAD", "M_ROW_SHAD", "M_BLV_SHAD", "M_ROW_BLV",
        "M_ABL", "M_INVFORM", "M_FADE",
    )'''
assert old_train in c, "train_method missing"
c = c.replace(old_train, new_train, 1)

if "M_BLV_SHAD:" not in c:
    needle = "      M_ROW_SHAD: V_B1C + RowHeightAttn + shadow_augment (combo train)\n"
    assert needle in c
    c = c.replace(
        needle,
        needle
        + "      M_BLV_SHAD: V_B1C + BLV + shadow_augment (combo train)\n"
        + "      M_ROW_BLV: V_B1C + RowHeightAttn + BLV (combo train)\n",
        1,
    )

# row_attn / blv / shadow_aug flags
c = c.replace(
    '        "row_attn": r in ("M_ROW", "M_ROW_SHAD"),\n',
    '        "row_attn": r in ("M_ROW", "M_ROW_SHAD", "M_ROW_BLV"),\n',
    1,
)
c = c.replace(
    '        "blv": r == "M_BLV",\n',
    '        "blv": r in ("M_BLV", "M_BLV_SHAD", "M_ROW_BLV"),\n',
    1,
)
c = c.replace(
    '        "shadow_aug": r in ("M_SHAD", "M_ROW_SHAD"),\n',
    '        "shadow_aug": r in ("M_SHAD", "M_ROW_SHAD", "M_BLV_SHAD"),\n',
    1,
)
COMMON.write_text(c, encoding="utf-8")
print("[ok]", COMMON)

s = SMOKE.read_text(encoding="utf-8")
old_note = '''        "M_ROW_SHAD": "V_B1C + RowHeightAttn + shadow_augment (combo)",
        "M_ABL": "V_B1C + ABL-lite boundary BCE/Dice (train-only)",'''
new_note = '''        "M_ROW_SHAD": "V_B1C + RowHeightAttn + shadow_augment (combo)",
        "M_BLV_SHAD": "V_B1C + BLV + shadow_augment (combo)",
        "M_ROW_BLV": "V_B1C + RowHeightAttn + BLV (combo)",
        "M_ABL": "V_B1C + ABL-lite boundary BCE/Dice (train-only)",'''
assert old_note in s
s = s.replace(old_note, new_note, 1)

old_mex = '''METHOD_EXPS = (
    "V_B1C",
    "M_ROW",
    "M_BLV",
    "M_SHAD",
    "M_ROW_SHAD",
    "M_ABL",
    "M_INVFORM",
    "M_FADE",
    "M_TTA",
    "M_SEGFIX",
    "T_B1_TTA",
    "M_HRDA",
)'''
new_mex = '''METHOD_EXPS = (
    "V_B1C",
    "M_ROW",
    "M_BLV",
    "M_SHAD",
    "M_ROW_SHAD",
    "M_BLV_SHAD",
    "M_ROW_BLV",
    "M_ABL",
    "M_INVFORM",
    "M_FADE",
    "M_TTA",
    "M_SEGFIX",
    "T_B1_TTA",
    "M_HRDA",
)'''
assert old_mex in s
s = s.replace(old_mex, new_mex, 1)

s = s.replace(
    '"Recipes: A0/A1/B1/C1 + V_B1C/V_B1E + M_ROW/M_BLV/M_SHAD/M_ROW_SHAD/M_ABL/"',
    '"Recipes: A0/A1/B1/C1 + V_B1C/V_B1E + M_ROW/M_BLV/M_SHAD/M_ROW_SHAD/M_BLV_SHAD/M_ROW_BLV/M_ABL/"',
    1,
)
SMOKE.write_text(s, encoding="utf-8")
print("[ok]", SMOKE)

import py_compile
py_compile.compile(str(COMMON), doraise=True)
py_compile.compile(str(SMOKE), doraise=True)
import sys
sys.path.insert(0, r"D:\LuSNAR_Seg\tools\portable")
# fresh import
import importlib
import _lusnar_common as lc
importlib.reload(lc)
for name in ("M_ROW_SHAD", "M_BLV_SHAD", "M_ROW_BLV", "M_ROW", "M_BLV", "M_SHAD"):
    f = lc.recipe_flags(name)
    print(name, {k: f[k] for k in ("row_attn", "blv", "shadow_aug", "crater_boost")})
print("py_compile OK")
