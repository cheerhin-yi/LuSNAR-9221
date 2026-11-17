from pathlib import Path
p = Path("/mnt/d/LuSNAR_Seg/tools/portable/lusnar_scene5_protocols.py")
text = p.read_text(encoding="utf-8")
# show PREREGISTERED and followup-relevant bits
lines = text.splitlines()
for i, l in enumerate(lines[:80], 1):
    print(f"{i}:{l}")
print("----LEN----", len(lines))
