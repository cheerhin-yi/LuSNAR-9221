from pathlib import Path
import json
b3=Path(r"D:/LuSNAR_Seg/results/smoke_method_eval_batch3.md").read_text(encoding="utf-8", errors="replace")
print(b3[:6000].encode("unicode_escape").decode("ascii"))
print("---FADE---")
for p in Path(r"D:/LuSNAR_Seg/work_dirs").glob("smoke_M_FADE*"):
    print(p)
for p in Path(r"D:/LuSNAR_Seg/work_dirs").glob("smoke_*"):
    if p.is_dir():
        print(p.name)
