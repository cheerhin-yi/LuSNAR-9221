import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, "/mnt/d/LuSNAR_Seg/tools/portable"); sys.path.insert(0, "/mnt/d/LuSNAR_Seg/tools/sc_diag")
import lusnar_small_crater_e0_wsl as E
import sc_diag as D
dev = torch.device("cuda:0")
net = D.build_segformer_b2(D.find_pretrain(Path("/mnt/d/LuSNAR_Seg"), None)).to(dev).eval()
row = D.RowHeightAttn().to(dev).eval()
ck = torch.load("/mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/best.pth", map_location=dev, weights_only=False)
net.load_state_dict(ck["model"]); row.load_state_dict(ck["row_attn"])
va = D.load_list("/root/lusnar_data/lists/val_scene5.txt")
for i in [0, 100, 199, 800]:
    cp, lp = va[i]
    img = D.imread_bgr(Path(cp)); gt = D.load_gt_official(Path(lp))
    with torch.no_grad():
        pe = E.predict(net, img, dev, row)
        L = [D.sliding(net, row, img, dev, s)[0] for s in D.SCALES]
    fused = torch.zeros_like(L[0])
    for c in range(5):
        for s in range(3):
            if D.I1A_W[c, s]: fused[c] += L[s][c]
    pm = fused.argmax(0).cpu().numpy()
    g = gt == 2
    print(i, "gt crater px", g.sum(), "E0 pred crater", (pe == 2).sum(), "mine", (pm == 2).sum(), "agree", (pe == pm).mean(),
          "E0 IoU", ((pe == 2) & g).sum() / max(((pe == 2) | g).sum(), 1), flush=True)
