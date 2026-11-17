from pathlib import Path
p = Path("/mnt/d/LuSNAR_Seg/tools/portable/lusnar_row_blv_ddp_sc.py")
t = p.read_text(encoding="utf-8")
old = "    if cc_bank is not None:\n        cc_dir ="
new = ("    if cc_bank is not None and args.cc_off:\n"
       "        cc_bank = None\n"
       "        if is_main(rank):\n"
       "            print(\"[cc_bank] OFF (matches effective WSL training of train_ROW_BLV_20k: Windows-path index never loaded)\", flush=True)\n"
       "    if cc_bank is not None:\n        cc_dir =")
assert t.count(old) == 1; t = t.replace(old, new)
old = '    ap.add_argument("--cc-dir", default=None)\n'
assert t.count(old) == 1
t = t.replace(old, old + '    ap.add_argument("--cc-off", action="store_true")\n')
p.write_text(t, encoding="utf-8"); print("ok")
