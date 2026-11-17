from pathlib import Path
p = Path("/mnt/d/LuSNAR_Seg/tools/portable/lusnar_row_blv_ddp_b5.py")
t = p.read_text(encoding="utf-8")
a = "probe = torch.stack([p.detach().double().sum() for p in model.parameters()]).cpu()"
b = "probe = torch.stack([p.detach().double().sum() for p in model.parameters()]).to(device)  # FI/B5 copy: NCCL cannot all_gather CPU tensors"
c = "[row_grad_l1, blv_ce_delta, moved], dtype=torch.float64\n"
d = "[row_grad_l1, blv_ce_delta, moved], dtype=torch.float64, device=device  # FI/B5 copy: NCCL needs CUDA tensors\n"
assert t.count(a) == 1 and t.count(c) == 1
t = t.replace(a, b).replace(c, d)
t = t.replace('backbone/head via build_segformer(...,"b5"), default pretrain mit_b5]', 'backbone/head via build_segformer(...,"b5"), default pretrain mit_b5; plus the same NCCL CUDA-tensor fix as lusnar_row_blv_ddp_fi.py]', 1)
p.write_text(t, encoding="utf-8")
print("ok")