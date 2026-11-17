from pathlib import Path
src = Path("/mnt/d/LuSNAR_Seg/tools/portable/lusnar_small_crater_e0_wsl.py")
dst = Path("/mnt/d/LuSNAR_Seg/tools/portable/lusnar_small_crater_score_sc.py")
assert not dst.exists()
t = src.read_text(encoding="utf-8")
old = """        residual.eval()
    net.eval()"""
new = """        residual.eval()
    elif ckpt.get("detail_residual"):
        from lusnar_row_blv_ddp_sc import DetailResidual12

        residual = DetailResidual12().to(device)
        residual.load_state_dict(ckpt["detail_residual"])
        residual.attach(net)
        residual.eval()
        print("[score] detail_residual alpha=", float(residual.alpha), flush=True)
    net.eval()"""
assert t.count(old) == 1
t = t.replace(old, new)
t = t.replace('"""WSL copy of lusnar_small_crater_e0.py', '"""[SC copy: adds detail_residual (stage1+2) loading; scoring identical to lusnar_small_crater_e0_wsl.py]\nWSL copy of lusnar_small_crater_e0.py', 1)
dst.write_text(t, encoding="utf-8")
print("wrote", dst)
