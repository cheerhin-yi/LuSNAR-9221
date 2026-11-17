import re, sys
from pathlib import Path
P = Path("/mnt/d/LuSNAR_Seg/tools/portable")
for src, dst in (("lusnar_row_blv_ddp.py", "lusnar_row_blv_ddp_fi.py"), ("lusnar_row_blv_ddp_sc.py", "lusnar_row_blv_ddp_sc_fi.py")):
    t = P.joinpath(src).read_text(encoding="utf-8")
    a = "probe = torch.stack([p.detach().double().sum() for p in model.parameters()]).cpu()"
    b = "probe = torch.stack([p.detach().double().sum() for p in model.parameters()]).to(device)  # FI copy: NCCL cannot all_gather CPU tensors"
    c = "[row_grad_l1, blv_ce_delta, moved], dtype=torch.float64\n"
    d = "[row_grad_l1, blv_ce_delta, moved], dtype=torch.float64, device=device  # FI copy: NCCL needs CUDA tensors\n"
    assert t.count(a) == 1 and t.count(c) == 1, (src, t.count(a), t.count(c))
    t = t.replace(a, b).replace(c, d)
    hdr = '# FI copy of %s (2026-11-12): only change = the two it==1 DDP check all_gather tensors moved to the CUDA device\n# (NCCL on WSL rejects CPU tensors; the original only ran on Windows gloo or on resume where it==1 is skipped).\n' % src
    P.joinpath(dst).write_text(hdr + t, encoding="utf-8")
    print("wrote", dst)