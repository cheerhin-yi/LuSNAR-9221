from pathlib import Path
import re
root = Path(r"D:/LuSNAR_Seg/tools")
pats = re.compile(r"(arxiv|doi\.org|github\.com|HANet|DySample|SegFix|InverseForm|FADE|SAPA|Contextrast|ABL|BLV|HRDA|Albumentations)", re.I)
for p in list(root.rglob("*.py"))[:200] + list(root.rglob("*.md")):
    try:
        t = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue
    hits = [ln.strip() for ln in t.splitlines() if pats.search(ln)]
    if hits:
        print("FILE", p)
        for h in hits[:25]:
            print(" ", h.encode("unicode_escape").decode("ascii")[:240])
        print()
