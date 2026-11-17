#!/usr/bin/env python3
"""P0 linear probe (reduced budget) for small-crater vs regolith separability.

Frozen first-round best.pth. Features: MiT stages 1-4 and decoder fused map (fusion_conv out,
before cls_seg), plus the model's own logit margin z_crater - z_regolith ("cls").
Fit: binary logistic (crater vs regolith) per feature level on sampled pixels from scenes 1/3/6/7/8.
Eval: Scene5 sampled pixels; AUC and balanced acc of crater instances (by area / short-side bucket)
vs regolith. Single scale 1.0, non-overlapping 512 tiles. Never reads Moon_2/4/9.
"""
from __future__ import annotations

import json, sys, time, random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
from _lusnar_common import RowHeightAttn, build_segformer_b2, find_pretrain, scan_scene_pairs, _preprocess_bgr  # noqa
from lusnar_oneshot_opensource import imread_bgr, load_gt_official  # noqa

AREA_BUCKETS = [(0, 16), (16, 64), (64, 256), (256, 1024), (1024, 4096), (4096, 1 << 62)]
LEVELS = ["stage1", "stage2", "stage3", "stage4", "fused", "cls"]


def bucket(v, bs):
    for i, (lo, hi) in enumerate(bs):
        if lo <= v < hi:
            return i
    return len(bs) - 1


@torch.no_grad()
def feats_at(net, row, tile_bgr, pts, device):
    """pts: (N,2) int (y,x) in tile coords. Returns dict level -> (N,C) float32 cpu."""
    t = _preprocess_bgr(tile_bgr).unsqueeze(0).to(device)
    S = tile_bgr.shape[0]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        x = net.extract_feat(t)
        head = net.decode_head
        from mmseg.models.utils import resize
        inputs = head._transform_inputs(x)
        outs = [resize(head.convs[i](xi), size=inputs[0].shape[2:], mode=head.interpolate_mode,
                       align_corners=head.align_corners) for i, xi in enumerate(inputs)]
        h0 = head.fusion_conv(torch.cat(outs, 1))
        logits = row(head.cls_seg(h0))
    grid = torch.from_numpy(pts[:, ::-1].copy()).float().to(device)  # x,y
    grid = (grid + 0.5) / S * 2 - 1
    grid = grid.view(1, 1, -1, 2)
    res = {}
    maps = {"stage1": x[0], "stage2": x[1], "stage3": x[2], "stage4": x[3], "fused": h0}
    for k, m in maps.items():
        res[k] = F.grid_sample(m.float(), grid, mode="bilinear", align_corners=False)[0, :, 0].t().cpu()
    lg = F.grid_sample(logits.float(), grid, mode="bilinear", align_corners=False)[0, :, 0].t()
    res["cls"] = (lg[:, 2] - lg[:, 0]).unsqueeze(1).cpu()
    return res


def sample_image(img, gt, rng, n_pos, n_neg, tile=512):
    """Yield (tile_img, pts, y(1 crater/0 reg), area, side) per tile."""
    crater = (gt == 2).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(crater, connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    side = np.minimum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    H, W = gt.shape
    for y0 in range(0, H - tile + 1, tile):
        for x0 in range(0, W - tile + 1, tile):
            g = gt[y0:y0 + tile, x0:x0 + tile]
            L = lab[y0:y0 + tile, x0:x0 + tile]
            pos = np.argwhere(g == 2)
            neg = np.argwhere(g == 0)
            out_pts, ys, ar, sd = [], [], [], []
            if len(pos):
                # stratify: up to half from small (side<32) instances
                ids = L[pos[:, 0], pos[:, 1]]
                small = side[ids] < 32
                sp, lp = pos[small], pos[~small]
                k1 = min(len(sp), n_pos // 2 if len(lp) else n_pos)
                k2 = min(len(lp), n_pos - k1)
                sel = []
                if k1: sel.append(sp[rng.choice(len(sp), k1, replace=False)])
                if k2: sel.append(lp[rng.choice(len(lp), k2, replace=False)])
                p = np.concatenate(sel)
                ii = L[p[:, 0], p[:, 1]]
                out_pts.append(p); ys.append(np.ones(len(p))); ar.append(area[ii]); sd.append(side[ii])
            if len(neg):
                k = min(len(neg), n_neg)
                p = neg[rng.choice(len(neg), k, replace=False)]
                out_pts.append(p); ys.append(np.zeros(len(p))); ar.append(np.full(len(p), -1)); sd.append(np.full(len(p), -1))
            if not out_pts:
                continue
            yield img[y0:y0 + tile, x0:x0 + tile], np.concatenate(out_pts), np.concatenate(ys), np.concatenate(ar), np.concatenate(sd)


def collect(net, row, pairs, device, rng, n_pos, n_neg, tag, deadline):
    F_ = {k: [] for k in LEVELS}
    Y, A, S = [], [], []
    t0 = time.time()
    for i, (cp, lp) in enumerate(pairs, 1):
        if time.time() > deadline:
            print(f"[P0] {tag} deadline hit at {i}", flush=True)
            break
        img = imread_bgr(Path(cp)); gt = load_gt_official(Path(lp))
        for tile, pts, y, a, s in sample_image(img, gt, rng, n_pos, n_neg):
            r = feats_at(net, row, tile, pts, device)
            for k in LEVELS:
                F_[k].append(r[k])
            Y.append(y); A.append(a); S.append(s)
        if i % 25 == 0:
            print(f"[P0] {tag} {i}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    return {k: torch.cat(v) for k, v in F_.items()}, np.concatenate(Y), np.concatenate(A), np.concatenate(S)


def auc(score, y):
    order = np.argsort(score)
    r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
    npos = (y == 1).sum(); nneg = (y == 0).sum()
    if npos == 0 or nneg == 0:
        return None
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def fit_probe(X, y, device, steps=400):
    X = X.to(device); yt = torch.from_numpy(y).float().to(device)
    mu = X.mean(0, keepdim=True); sd = X.std(0, keepdim=True) + 1e-5
    Xn = (X - mu) / sd
    w = torch.zeros(X.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    pw = torch.tensor([(yt == 0).sum() / max((yt == 1).sum(), 1)], device=device)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xn @ w + b, yt, pos_weight=pw) + 1e-4 * (w * w).sum()
        loss.backward(); opt.step()
    return lambda Z: ((Z.to(device) - mu) / sd @ w + b).detach().cpu().numpy()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-per-scene", type=int, default=30)
    ap.add_argument("--eval-images", type=int, default=150)
    ap.add_argument("--n-pos", type=int, default=200)
    ap.add_argument("--n-neg", type=int, default=200)
    ap.add_argument("--budget-min", type=float, default=30)
    args = ap.parse_args()
    t_start = time.time()
    deadline_fit = t_start + args.budget_min * 60 * 0.5
    deadline_eval = t_start + args.budget_min * 60 * 0.95
    project = Path(args.project_root)
    device = torch.device("cuda:0")
    net = build_segformer_b2(find_pretrain(project, None)).to(device).eval()
    row = RowHeightAttn().to(device).eval()
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(ck["model"]); row.load_state_dict(ck["row_attn"])
    rng = np.random.default_rng(42); random.seed(42)
    data = Path(args.data_root)
    tr = []
    for s in (1, 3, 6, 7, 8):
        p = scan_scene_pairs(data, s)
        tr += [p[i] for i in sorted(rng.choice(len(p), min(args.train_per_scene, len(p)), replace=False))]
    random.shuffle(tr)
    p5 = scan_scene_pairs(data, 5)
    ev = [p5[i] for i in sorted(rng.choice(len(p5), min(args.eval_images, len(p5)), replace=False))]
    Ftr, ytr, _, _ = collect(net, row, tr, device, rng, args.n_pos, args.n_neg, "fit", deadline_fit)
    Fev, yev, aev, sev = collect(net, row, ev, device, rng, 600, 300, "eval", deadline_eval)
    print(f"[P0] fit px pos={int(ytr.sum())} neg={int((ytr == 0).sum())}; eval px pos={int(yev.sum())} neg={int((yev == 0).sum())}", flush=True)
    neg = yev == 0
    res = {"n_fit_px_pos": int(ytr.sum()), "n_fit_px_neg": int((ytr == 0).sum()),
           "n_eval_px_pos": int(yev.sum()), "n_eval_px_neg": int(neg.sum()), "levels": {}}
    for k in LEVELS:
        if k == "cls":
            sc = Fev[k][:, 0].numpy()
        else:
            sc = fit_probe(Ftr[k], ytr, device)(Fev[k])
        lv = {"auc_all": auc(sc, yev), "bal_acc_all": float(((sc[yev == 1] > 0).mean() + (sc[neg] <= 0).mean()) / 2),
              "by_area": [], "small_side_lt32": None}
        for bi, (lo, hi) in enumerate(AREA_BUCKETS):
            m = (yev == 1) & (aev >= lo) & (aev < hi)
            if m.sum() == 0:
                lv["by_area"].append({"lo": lo, "hi": hi, "n_px": 0}); continue
            sel = m | neg
            lv["by_area"].append({"lo": lo, "hi": hi, "n_px": int(m.sum()), "auc_vs_reg": auc(sc[sel], yev[sel]),
                                  "recall_at0": float((sc[m] > 0).mean())})
        m = (yev == 1) & (sev >= 0) & (sev < 32)
        if m.sum():
            sel = m | neg
            lv["small_side_lt32"] = {"n_px": int(m.sum()), "auc_vs_reg": auc(sc[sel], yev[sel]), "recall_at0": float((sc[m] > 0).mean())}
        lv["regolith_fpr_at0"] = float((sc[neg] > 0).mean())
        res["levels"][k] = lv
        print(f"[P0] {k}: auc_all={lv['auc_all']:.4f} small<32 {lv['small_side_lt32']}", flush=True)
    res["runtime_s"] = time.time() - t_start
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "P0_probe.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("P0_DONE", flush=True)


if __name__ == "__main__":
    main()
