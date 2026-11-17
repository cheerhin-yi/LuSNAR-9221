from pathlib import Path
P = Path("/mnt/d/LuSNAR_Seg/tools/portable")
B5 = "pretrained/mit_b5_20220624-658746d9.pth"
# trainer copy
src = (P / "lusnar_row_blv_ddp.py").read_text(encoding="utf-8")
dst = P / "lusnar_row_blv_ddp_b5.py"
assert not dst.exists()
n0 = src.count("core = build_segformer_b2(pretrain).to(device)")
assert n0 == 1, n0
t = src.replace("core = build_segformer_b2(pretrain).to(device)", 'core = build_segformer(pretrain, "b5").to(device)')
t = t.replace("    build_segformer_b2,\n", "    build_segformer,\n    build_segformer_b2,\n", 1)
n1 = t.count("pretrain = find_pretrain(project, args.pretrain)")
assert n1 == 1, n1
t = t.replace("pretrain = find_pretrain(project, args.pretrain)", 'pretrain = find_pretrain(project, args.pretrain or "%s")' % B5)
t = t.replace('            "pretrain": str(pretrain),\n        }\n        summary', '            "pretrain": str(pretrain),\n            "backbone": "mit_b5",\n        }\n        summary', 1)
assert '"backbone": "mit_b5"' in t
t = t.replace('"""LuSNAR SegFormer-B2 formal training', '"""[B5 COPY 2026-11-13: only change = SegFormer-B5 (MiT-B5) backbone/head via build_segformer(...,"b5"), default pretrain mit_b5]\nLuSNAR SegFormer-B2 formal training', 1)
dst.write_text(t, encoding="utf-8")
# scorer copy
src = (P / "lusnar_small_crater_score_sc.py").read_text(encoding="utf-8")
dst = P / "lusnar_small_crater_score_sc_b5.py"
assert not dst.exists()
old = "net = build_segformer_b2(find_pretrain(project, None)).to(device)"
assert src.count(old) == 1
t = src.replace(old, 'net = build_segformer(find_pretrain(project, "%s"), "b5").to(device)' % B5)
t = t.replace("    build_segformer_b2,\n", "    build_segformer,\n    build_segformer_b2,\n", 1)
dst.write_text(t, encoding="utf-8")
print("ok")