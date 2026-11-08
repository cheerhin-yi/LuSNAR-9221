import io
f=r"D:\LuSNAR_Seg\tools\portable\_lusnar_common.py"
t=io.open(f,encoding="utf-8").read()
old='    bce_map = F.binary_cross_entropy(pred_e.clamp(1e-4, 1 - 1e-4), gt, reduction="none")\n'
new=('    # BCE is autocast-unsafe: compute in fp32 with autocast disabled (math unchanged)\n'
     '    with torch.autocast(device_type=pred_e.device.type, enabled=False):\n'
     '        bce_map = F.binary_cross_entropy(pred_e.float().clamp(1e-4, 1 - 1e-4), gt.float(), reduction="none")\n')
assert t.count(old)==1
io.open(f,"w",encoding="utf-8",newline="").write(t.replace(old,new))
import sys,torch;sys.path.insert(0,r"D:\LuSNAR_Seg\tools\portable");import _lusnar_common as c
print(c.CLASS_NAMES)
lg=torch.randn(2,5,64,64,device="cuda",requires_grad=True);m=torch.randint(0,5,(2,64,64),device="cuda");e=(torch.rand(2,64,64,device="cuda")>0.8).float()
with torch.autocast("cuda",dtype=torch.bfloat16):
    l=c.abl_lite_loss(lg,m,e)
l.backward();print("ok",float(l))
