from pathlib import Path
p = Path(r"D:/LuSNAR_Seg/tools/wsl_bootstrap/03_resume_train_wsl.sh")
if p.exists():
    t = p.read_text(encoding="utf-8", errors="replace")
    print(t.encode("unicode_escape").decode("ascii")[:4000])
else:
    print("MISSING")
