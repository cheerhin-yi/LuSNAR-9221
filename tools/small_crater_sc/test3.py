import sys, torch
sys.path.insert(0, "/mnt/d/LuSNAR_Seg/tools/portable")
from lusnar_row_blv_ddp_sc import DetailResidual12 as D
for s in ("12", "1", "2"):
    m = D(stages=s); m._f2 = torch.randn(1, 128, 16, 16)
    out = m(torch.randn(1, 64, 32, 32), torch.zeros(1, 256, 32, 32))
    print(s, sorted(m.state_dict()), tuple(out.shape), float(out.abs().sum()))
