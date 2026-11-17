from pathlib import Path
import runpy
root = Path("/mnt/d/LuSNAR_Seg/tools/portable")
for p in root.glob("lusnar_scene5*.py"):
    b = p.read_bytes()
    if b.startswith(b"\xef\xbb\xbf"):
        p.write_bytes(b[3:])
        print("stripped BOM", p.name)
    else:
        print("ok", p.name)
# also strip patch script BOM if any
pp = root / "_patch_protocols_mem.py"
b = pp.read_bytes()
if b.startswith(b"\xef\xbb\xbf"):
    pp.write_bytes(b[3:])
runpy.run_path(str(pp))
