from pathlib import Path
import base64
files = [
 Path(r"D:/LuSNAR_Seg/PROJECT_STATUS.md"),
 Path(r"D:/LuSNAR_Seg/results/smoke_method_eval.md"),
 Path(r"D:/LuSNAR_Seg/results/smoke_method_eval_batch3.md"),
 Path(r"D:/LuSNAR_Seg/tools/portable/README_METHOD_SMOKE.md"),
 Path(r"D:/LuSNAR_Seg/tools/portable_b3/README_METHOD_SMOKE.md"),
]
parts = []
for f in files:
    parts.append("===== FILE " + str(f) + " =====\n")
    parts.append(f.read_text(encoding="utf-8", errors="replace")[:25000])
    parts.append("\n\n")
raw = "".join(parts).encode("utf-8")
Path(r"D:/LuSNAR_Seg/logs/_modules_b64.txt").write_text(base64.b64encode(raw).decode("ascii"), encoding="ascii")
print(len(raw))
