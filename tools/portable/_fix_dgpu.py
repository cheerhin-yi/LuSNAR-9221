from pathlib import Path
p = Path("/mnt/d/LuSNAR_Seg/tools/portable/lusnar_scene5_protocols.py")
text = p.read_text(encoding="utf-8")
# Remove the dummy if False block through just before "results = {}"
start = text.find("    # keep a dummy to satisfy any leftover references")
end = text.find("    results = {}")
if start < 0 or end < 0 or end < start:
    raise SystemExit(f"markers missing start={start} end={end}")
text = text[:start] + text[end:]
p.write_text(text, encoding="utf-8")
compile(text, str(p), "exec")
print("COMPILE_OK, lines", text.count("\n")+1)
