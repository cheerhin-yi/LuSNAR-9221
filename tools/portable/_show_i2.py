from pathlib import Path
lines = Path("/mnt/d/LuSNAR_Seg/tools/portable/lusnar_scene5_protocols.py").read_text(encoding="utf-8").splitlines()
for i, l in enumerate(lines[540:820], start=541):
    print(f"{i}:{l}")
