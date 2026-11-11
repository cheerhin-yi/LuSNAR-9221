from pathlib import Path
import re, json
files = {
 "status": Path(r"D:/LuSNAR_Seg/PROJECT_STATUS.md").read_text(encoding="utf-8", errors="replace"),
 "eval": Path(r"D:/LuSNAR_Seg/results/smoke_method_eval.md").read_text(encoding="utf-8", errors="replace"),
 "b3": Path(r"D:/LuSNAR_Seg/results/smoke_method_eval_batch3.md").read_text(encoding="utf-8", errors="replace"),
 "r1": Path(r"D:/LuSNAR_Seg/tools/portable/README_METHOD_SMOKE.md").read_text(encoding="utf-8", errors="replace"),
 "r3": Path(r"D:/LuSNAR_Seg/tools/portable_b3/README_METHOD_SMOKE.md").read_text(encoding="utf-8", errors="replace"),
}
# dump key sections mentioning modules
keys = ["ROW","BLV","SHAD","TTA","DYSAMPLE","DySample","HARD","CTRAST","Contextrast","SAPA","FADE","INVFORM","ABL","SEGFIX","MSINFER","HANet","Albumentations","BlindNet"]
out = {}
for name, text in files.items():
    hits = []
    for line in text.splitlines():
        if any(k.lower() in line.lower() for k in keys):
            hits.append(line.strip())
    out[name] = hits[:120]
Path(r"D:/LuSNAR_Seg/logs/_modules_hits.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
# also print length
print({k: len(v) for k,v in out.items()})
# print README sections fully ascii-escaped
for name in ["r1","r3","b3","eval"]:
    print("====", name, "====")
    print(files[name][:8000].encode("unicode_escape").decode("ascii"))
    print()
