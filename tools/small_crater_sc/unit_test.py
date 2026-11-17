import sys, torch, numpy as np
sys.path.insert(0, "/mnt/d/LuSNAR_Seg/tools/portable")
from lusnar_row_blv_ddp_sc import local_margin_loss_inst, DetailResidual12
m = torch.zeros(2, 64, 64, dtype=torch.long)
m[0, 10:14, 10:14] = 2          # small crater
m[0, 30:62, 2:60] = 2           # large crater (band only)
m[0, 0:3, :] = 3                # other class
m[1, 5:8, 5:8] = 255
logits = torch.zeros(2, 5, 64, 64, requires_grad=True)
l, n = local_margin_loss_inst(logits, m)
print("loss", float(l), "n_inst", n, "expect softplus(0.5)=", float(torch.nn.functional.softplus(torch.tensor(0.5))))
l.backward(); print("grad nonzero px", int((logits.grad.abs() > 0).sum()))
lg = torch.zeros(2, 5, 64, 64); lg[:, 2] = (m == 2).float() * 3
print("separated loss", float(local_margin_loss_inst(lg, m)[0]))
print("empty", local_margin_loss_inst(torch.zeros(1, 5, 8, 8), torch.zeros(1, 8, 8, dtype=torch.long)))
