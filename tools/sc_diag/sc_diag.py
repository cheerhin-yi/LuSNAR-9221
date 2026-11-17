#!/usr/bin/env python3
"""Frozen-weight small-crater diagnostics (D1 candidate scorer under FP budget, D2 GT-center local refine UB).

Frozen M_ROW_BLV best.pth. Heads trained on TRAIN scenes 1/3/6/7/8 only (list train_scenes.txt);
evaluation on Scene5 (val_scene5.txt, 1599 imgs). Never reads Moon_2/4/9. Writes only to --out-dir.
"""
from __future__ import annotations
import json, sys, time, random, argparse, math
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, "/mnt/d/LuSNAR_Seg/tools/portable")
from _lusnar_common import RowHeightAttn, build_segformer_b2, find_pretrain, MEAN, STD  # noqa
from lusnar_oneshot_opensource import imread_bgr, load_gt_official  # noqa
from mmseg.models.utils import resize

NC = 5
I1A_W = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
SCALES = (0.75, 1.0, 1.25)
BINS = [(0, 16), (16, 64), (64, 256), (256, 1024)]
WIN = 64
HALF = WIN // 2
FORBID = ("Moon_2/", "Moon_4/", "Moon_9/")


def load_list(p):
    out = []
    for ln in Path(p).read_text().splitlines():
        if not ln.strip():
            continue
        a, b = ln.split("\t")[:2]
        assert not any(f in a or f in b for f in FORBID), a
        out.append((a, b))
    return out


def binof(a):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= a < hi:
            return i
    return -1


# ---------------- frozen model features ----------------
@torch.no_grad()
def frozen_feats(net, row, t):
    """t: B3SS normalized. Returns Q (B,C,S/4,S/4) float32 [h0, s1, up(s2), lg], lg (B,5,S/4,S/4)."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        x = net.extract_feat(t)
        head = net.decode_head
        inputs = head._transform_inputs(x)
        outs = [resize(head.convs[i](xi), size=inputs[0].shape[2:], mode=head.interpolate_mode,
                       align_corners=head.align_corners) for i, xi in enumerate(inputs)]
        h0 = head.fusion_conv(torch.cat(outs, 1))
        lg = row(head.cls_seg(h0))
    q = x[0].shape[-2:]
    s2 = F.interpolate(x[1].float(), size=q, mode="bilinear", align_corners=False)
    Q = torch.cat([h0.float(), x[0].float(), s2, lg.float()], 1)
    return Q, lg.float()


@torch.no_grad()
def logits_only(net, row, t):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        x = net.extract_feat(t)
        lg = row(net.decode_head.forward(x))
    return lg.float()


def prep(img_bgr):
    return torch.from_numpy(((img_bgr.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1).copy())


def win_starts(n, crop=512, stride=256):
    ys = list(range(0, max(n - crop, 0) + 1, stride)) or [0]
    if ys[-1] != max(0, n - crop):
        ys.append(max(0, n - crop))
    return ys


# ---------------- heads ----------------
class CandHead(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm2d(cin), nn.Conv2d(cin, 64, 1), nn.ReLU(True),
                                 nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
                                 nn.Conv2d(64, 64, 3, padding=2, dilation=2), nn.ReLU(True), nn.Conv2d(64, 1, 1))
        nn.init.constant_(self.net[-1].bias, -2.19)

    def forward(self, q):
        return self.net(q)


class RefineHead(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm2d(cin), nn.Conv2d(cin, 64, 1), nn.ReLU(True),
                                 nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(True),
                                 nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(True),
                                 nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.ReLU(True), nn.Conv2d(32, 1, 1))

    def forward(self, x):
        return self.net(x)


def sample_windows(Qb, imgb, centers, S):
    """Qb (C,h,w) quarter map of an S-sized canvas, imgb (3,S,S). centers (N,2) int (y,x) full-res.
    Returns (N, 3+C, WIN, WIN) via exact bilinear (align_corners=False)."""
    dev = Qb.device
    c = torch.as_tensor(centers, device=dev, dtype=torch.float32)
    off = torch.arange(-HALF, HALF, device=dev, dtype=torch.float32)
    py = c[:, 0:1] + off.view(1, -1)  # N,WIN
    px = c[:, 1:2] + off.view(1, -1)
    gy = (py + 0.5) / S * 2 - 1
    gx = (px + 0.5) / S * 2 - 1
    N = len(centers)
    grid = torch.stack([gx[:, None, :].expand(N, WIN, WIN), gy[:, :, None].expand(N, WIN, WIN)], -1)
    grid = grid.reshape(1, N * WIN, WIN, 2)
    a = F.grid_sample(Qb[None], grid, mode="bilinear", align_corners=False)[0]
    b = F.grid_sample(imgb[None], grid, mode="bilinear", align_corners=False)[0]
    o = torch.cat([b, a], 0).view(-1, N, WIN, WIN).permute(1, 0, 2, 3)
    return o.contiguous()


def crop_np(arr, cy, cx, fill):
    H, W = arr.shape
    out = np.full((WIN, WIN), fill, arr.dtype)
    y0, x0 = cy - HALF, cx - HALF
    ya, xa = max(0, y0), max(0, x0)
    yb, xb = min(H, y0 + WIN), min(W, x0 + WIN)
    if yb > ya and xb > xa:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = arr[ya:yb, xa:xb]
    return out


# ---------------- train dataset ----------------
class TrainDS(torch.utils.data.Dataset):
    def __init__(self, pairs, n, seed):
        self.pairs, self.n, self.seed = pairs, n, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 100003 + i)
        cp, lp = self.pairs[rng.integers(len(self.pairs))]
        img = imread_bgr(Path(cp)); gt = load_gt_official(Path(lp))
        H, W = gt.shape
        cr = (gt == 2).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(cr, connectivity=8)
        area = st[:, cv2.CC_STAT_AREA]
        tgt = [k for k in range(1, n) if 16 <= area[k] < 1024]
        S = 512
        if tgt and rng.random() < 0.7:
            k = tgt[rng.integers(len(tgt))]
            cy, cx = cen[k][1], cen[k][0]
            y0 = int(np.clip(cy - rng.integers(64, S - 64), 0, H - S)); x0 = int(np.clip(cx - rng.integers(64, S - 64), 0, W - S))
        else:
            y0 = int(rng.integers(0, H - S + 1)); x0 = int(rng.integers(0, W - S + 1))
        im = img[y0:y0 + S, x0:x0 + S]; g = gt[y0:y0 + S, x0:x0 + S]; L = lab[y0:y0 + S, x0:x0 + S]
        # heatmap target at 1/4
        q = S // 4
        heat = np.zeros((q, q), np.float32)
        yy, xx = np.mgrid[0:q, 0:q].astype(np.float32)
        ign_full = np.zeros((S, S), np.uint8)
        pos_c = []
        for k in np.unique(L):
            if k == 0:
                continue
            a = area[k]
            cy, cx = cen[k][1] - y0, cen[k][0] - x0
            if 16 <= a < 1024:
                if 0 <= cy < S and 0 <= cx < S:
                    qy, qx = (cy + 0.5) / 4 - 0.5, (cx + 0.5) / 4 - 0.5
                    sg = max(0.7, math.sqrt(a / math.pi) / 8)
                    heat = np.maximum(heat, np.exp(-((yy - qy) ** 2 + (xx - qx) ** 2) / (2 * sg * sg)))
                    iy, ix = int(np.clip(round(qy), 0, q - 1)), int(np.clip(round(qx), 0, q - 1))
                    heat[iy, ix] = 1.0
            else:
                ign_full[L == k] = 1
            if 4 <= a < 1024 and HALF <= cy < S - HALF and HALF <= cx < S - HALF:
                pos_c.append((int(np.clip(round(cy + rng.integers(-3, 4)), HALF, S - HALF)), int(np.clip(round(cx + rng.integers(-3, 4)), HALF, S - HALF))))
        ign = cv2.resize(ign_full, (q, q), interpolation=cv2.INTER_AREA) > 0
        ign &= heat < 1.0
        pos_c = pos_c[:16]
        # random regolith negatives far from craters
        dcr = cv2.distanceTransform((g != 2).astype(np.uint8), cv2.DIST_L2, 5)
        cand = np.argwhere((g == 0) & (dcr >= HALF))
        cand = cand[(cand[:, 0] >= HALF) & (cand[:, 0] < S - HALF) & (cand[:, 1] >= HALF) & (cand[:, 1] < S - HALF)]
        neg_c = [tuple(map(int, cand[j])) for j in rng.choice(len(cand), min(len(cand), 4), replace=False)] if len(cand) else []
        P = np.full((16, 2), -1, np.int64); P[:len(pos_c)] = pos_c if pos_c else P[:0]
        Nn = np.full((4, 2), -1, np.int64); Nn[:len(neg_c)] = neg_c if neg_c else Nn[:0]
        farmask = cv2.resize((dcr >= HALF).astype(np.uint8) * (g == 0), (q, q), interpolation=cv2.INTER_NEAREST)
        return prep(im), torch.from_numpy(g.astype(np.int64)), torch.from_numpy(heat), torch.from_numpy(ign), \
            torch.from_numpy(P), torch.from_numpy(Nn), torch.from_numpy(farmask.astype(bool))


def focal_heat(logit, heat, ign):
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = heat >= 1.0
    w = (~ign).float()
    lp = -torch.log(p) * (1 - p) ** 2 * pos
    ln = -torch.log(1 - p) * p ** 2 * (1 - heat) ** 4 * (~pos) * w
    return (lp.sum() + ln.sum()) / pos.sum().clamp(min=1)


def train_heads(net, row, train_pairs, args, dev, log):
    ds = TrainDS(train_pairs, args.iters * args.bs, 7)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, num_workers=args.workers, shuffle=False, drop_last=True,
                                     persistent_workers=False, prefetch_factor=4)
    H1 = H2 = None
    opt = None
    t0 = time.time()
    for it, (img, g, heat, ign, P, Nn, far) in enumerate(dl):
        img = img.to(dev, non_blocking=True); g = g.to(dev); heat = heat.to(dev); ign = ign.to(dev)
        Q, lg = frozen_feats(net, row, img)
        if H1 is None:
            C = Q.shape[1]
            H1 = CandHead(C).to(dev); H2 = RefineHead(C + 3).to(dev)
            opt = torch.optim.AdamW(list(H1.parameters()) + list(H2.parameters()), lr=args.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.1)
            log(f"[train] Q channels={C}")
        l1 = focal_heat(H1(Q)[:, 0], heat, ign)
        # refine windows
        X, Y = [], []
        cm = torch.softmax(lg, 1)[:, 2]
        for b in range(img.shape[0]):
            cs = [tuple(c) for c in P[b].tolist() if c[0] >= 0] + [tuple(c) for c in Nn[b].tolist() if c[0] >= 0]
            # hard negatives: top crater-prob quarter cells on far regolith
            sc = torch.where(far[b].to(dev), cm[b], torch.full_like(cm[b], -1))
            v, ix = sc.flatten().topk(2)
            for vv, ii in zip(v.tolist(), ix.tolist()):
                if vv > 0:
                    qy, qx = divmod(ii, sc.shape[1]); cy, cx = 4 * qy + 1, 4 * qx + 1
                    if HALF <= cy < 512 - HALF and HALF <= cx < 512 - HALF:
                        cs.append((cy, cx))
            if not cs:
                continue
            X.append(sample_windows(Q[b], img[b], cs, 512))
            gb = g[b]
            Y.append(torch.stack([gb[cy - HALF:cy + HALF, cx - HALF:cx + HALF] for cy, cx in cs]))
        if X:
            X = torch.cat(X); Y = torch.cat(Y)
            o = H2(X)[:, 0]
            valid = (Y != 255).float(); tgt = (Y == 2).float()
            bce = (F.binary_cross_entropy_with_logits(o, tgt, reduction="none", pos_weight=torch.tensor(2.0, device=dev)) * valid).sum() / valid.sum().clamp(min=1)
            p = torch.sigmoid(o) * valid
            dice = 1 - (2 * (p * tgt).sum() + 1) / (p.sum() + (tgt * valid).sum() + 1)
            l2 = bce + dice
        else:
            l2 = torch.zeros((), device=dev)
        opt.zero_grad(set_to_none=True)
        (l1 + l2).backward()
        opt.step(); sched.step()
        if it % 100 == 0 or it == args.iters - 1:
            log(f"[train] it {it}/{args.iters} l1={l1.item():.4f} l2={l2.item():.4f} {time.time() - t0:.0f}s")
    H1.eval(); H2.eval()
    return H1, H2, time.time() - t0


# ---------------- eval ----------------
class EvalDS(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        cp, lp = self.pairs[i]
        img = imread_bgr(Path(cp)); gt = load_gt_official(Path(lp))
        return img, gt


@torch.no_grad()
def sliding(net, row, img_bgr, dev, scale, H1=None):
    h0, w0 = img_bgr.shape[:2]
    nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
    im = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if scale != 1.0 else img_bgr
    ys, xs = win_starts(nh), win_starts(nw)
    pos = [(y, x) for y in ys for x in xs]
    t = torch.stack([prep(im[y:y + 512, x:x + 512]) for y, x in pos]).to(dev)
    acc = torch.zeros(NC, nh, nw, device=dev); cnt = torch.zeros(1, nh, nw, device=dev)
    extra = None
    if H1 is not None:
        Q, lg = frozen_feats(net, row, t)
        hm = H1(Q)
        qa = torch.zeros(Q.shape[1], nh // 4, nw // 4, device=dev); ha = torch.zeros(1, nh // 4, nw // 4, device=dev)
        qc = torch.zeros(1, nh // 4, nw // 4, device=dev)
        for j, (y, x) in enumerate(pos):
            qa[:, y // 4:y // 4 + 128, x // 4:x // 4 + 128] += Q[j]
            ha[:, y // 4:y // 4 + 128, x // 4:x // 4 + 128] += hm[j]
            qc[:, y // 4:y // 4 + 128, x // 4:x // 4 + 128] += 1
        extra = (qa / qc, (ha / qc)[0])
    else:
        lg = logits_only(net, row, t)
    up = F.interpolate(lg, size=(512, 512), mode="bilinear", align_corners=False)
    for j, (y, x) in enumerate(pos):
        acc[:, y:y + 512, x:x + 512] += up[j]; cnt[:, y:y + 512, x:x + 512] += 1
    out = acc / cnt
    if scale != 1.0:
        out = F.interpolate(out[None], size=(h0, w0), mode="bilinear", align_corners=False)[0]
    return out, extra


def peaks(score, k, stride, off, topn=1000, r=8, keep=50):
    m = F.max_pool2d(score[None, None], k, 1, k // 2)[0, 0]
    s = torch.where(score == m, score, torch.full_like(score, -1e9)).flatten()
    v, ix = s.topk(min(topn, s.numel()))
    v = v.cpu().numpy(); ix = ix.cpu().numpy()
    ok = v > -1e8
    v, ix = v[ok], ix[ok]
    W = score.shape[1]
    yy = (ix // W) * stride + off; xx = (ix % W) * stride + off
    sel = []
    for j in range(len(v)):
        if all((yy[j] - yy[s_]) ** 2 + (xx[j] - xx[s_]) ** 2 > r * r for s_ in sel):
            sel.append(j)
            if len(sel) >= keep:
                break
    sel = np.array(sel, int)
    return np.stack([yy[sel], xx[sel]], 1).astype(int), v[sel]


def assign(cands, lab, H, W):
    res = []
    for y, x in cands:
        y = int(np.clip(y, 0, H - 1)); x = int(np.clip(x, 0, W - 1))
        w = lab[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        nz = w[w > 0]
        res.append(int(np.bincount(nz).argmax()) if len(nz) else 0)
    return res


def conf(gt, pred):
    v = gt < NC
    return np.bincount(gt[v].astype(np.int64) * NC + pred[v], minlength=NC * NC).reshape(NC, NC)


def ious(cm):
    i = np.diag(cm).astype(float); u = cm.sum(0) + cm.sum(1) - np.diag(cm)
    io = i / np.maximum(u, 1)
    return {"mIoU": float(io.mean()), "regolith": float(io[0]), "rock": float(io[1]), "crater": float(io[2]),
            "mountain": float(io[3]), "sky": float(io[4])}


@torch.no_grad()
def _noop():
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/best.pth")
    ap.add_argument("--project-root", default="/mnt/d/LuSNAR_Seg")
    ap.add_argument("--train-list", default="/root/lusnar_data/lists/train_scenes.txt")
    ap.add_argument("--val-list", default="/root/lusnar_data/lists/val_scene5.txt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--budget-min", type=float, default=55)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    T0 = time.time()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    dev = torch.device("cuda:0")
    log(f"[env] device={torch.cuda.get_device_name(0)} args={vars(args)}")
    net = build_segformer_b2(find_pretrain(Path(args.project_root), None)).to(dev).eval()
    row = RowHeightAttn().to(dev).eval()
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    net.load_state_dict(ck["model"]); row.load_state_dict(ck["row_attn"])
    assert not ck.get("plain_residual")
    for p in list(net.parameters()) + list(row.parameters()):
        p.requires_grad_(False)
    tr = load_list(args.train_list)
    assert all(("/Moon_5/" not in a) for a, _ in tr)
    va = load_list(args.val_list)
    assert all("/Moon_5/" in a for a, _ in va)
    if args.max_images:
        va = va[:args.max_images]
    H1, H2, t_train = train_heads(net, row, tr, args, dev, log)
    torch.save({"H1": H1.state_dict(), "H2": H2.state_dict()}, out / "heads.pth")
    log(f"[train] done {t_train:.0f}s")

    # ---- Scene5 pass ----
    dl = torch.utils.data.DataLoader(EvalDS(va), batch_size=None, num_workers=args.workers, prefetch_factor=4)
    inst = []  # dict rows
    cands = {"logit": [], "learned": []}  # per image list of (score, inst_gid or 0/-1)
    VARS = ["base", "pos64", "pos32", "rand", "hard", "pos64+rand+hard"]
    CM = {k: np.zeros((NC, NC), np.int64) for k in VARS}
    hit50 = {k: np.zeros((len(BINS),), int) for k in VARS}
    nb = np.zeros(len(BINS), int)
    touch = {k: dict(rock=0, rock_gtcrater=0, mtn=0, mtn_gtcrater=0, conv_tp=0, conv_fp=0, nwin=0, nwin_fire8=0) for k in VARS[1:5]}
    t_eval0 = time.time(); n_done = 0
    deadline = T0 + args.budget_min * 60
    torch.set_grad_enabled(False)
    for ii, (img, gt) in enumerate(dl):
        if time.time() > deadline:
            log(f"[eval] budget hit at {ii}"); break
        img = img.numpy() if torch.is_tensor(img) else img
        gt = gt.numpy() if torch.is_tensor(gt) else gt
        H, W = gt.shape
        L = []
        extra = None
        for s in SCALES:
            lg, ex = sliding(net, row, img, dev, s, H1 if s == 1.0 else None)
            L.append(lg)
            if ex is not None:
                extra = ex
        fused = torch.zeros_like(L[0])
        for c in range(NC):
            for s in range(3):
                if I1A_W[c, s]:
                    fused[c] += I1A_W[c, s] * L[s][c]
        pred = fused.argmax(0).cpu().numpy().astype(np.uint8)
        oth = torch.cat([fused[:2], fused[3:]], 0).max(0)[0]
        prob = fused[2] - oth  # crater logit margin (I1a fused)
        Qf, hm = extra
        cr = (gt == 2).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(cr, connectivity=8)
        area = st[:, cv2.CC_STAT_AREA]
        base_tp = np.bincount(lab[(pred == 2) & (gt == 2)].ravel(), minlength=n)
        dtp = cv2.distanceTransform((pred != 2).astype(np.uint8), cv2.DIST_L2, 5)
        gid0 = len(inst)
        mind = np.full(n, np.inf)
        if n > 1:
            idx = lab.ravel(); order = np.argsort(idx, kind="stable"); dsrt = dtp.ravel()[order]
            starts = np.searchsorted(idx[order], np.arange(n))
            mind = np.minimum.reduceat(dsrt, starts)
        for k in range(1, n):
            inst.append(dict(img=ii, area=int(area[k]), bin=binof(area[k]), hit50=int(base_tp[k] * 2 >= area[k]),
                             dist=float(mind[k]), cy=float(cen[k][1]), cx=float(cen[k][0])))
        # D1 candidates
        for name, (sm, k_, stride, off) in {"logit": (prob, 9, 1, 0), "learned": (torch.sigmoid(hm), 3, 4, 1)}.items():
            cc, sv = peaks(sm, k_, stride, off)
            a = assign(cc, lab, H, W)
            cands[name].append([(float(v), (gid0 + k - 1) if k > 0 else -1, int(y), int(x)) for v, k, (y, x) in zip(sv, a, cc)])
        # D2 windows
        pos_c = [(int(round(cen[k][1])), int(round(cen[k][0]))) for k in range(1, n) if area[k] < 1024]
        dcr = cv2.distanceTransform((gt != 2).astype(np.uint8), cv2.DIST_L2, 5)
        cand = np.argwhere((gt == 0) & (dcr >= HALF))
        rng = np.random.default_rng(1000 + ii)
        rand_c = [tuple(map(int, cand[j])) for j in rng.choice(len(cand), min(2, len(cand)), replace=False)] if len(cand) else []
        hard_c = [(y, x) for (v, g_, y, x) in cands["learned"][-1] if g_ < 0][:2]
        imgt = prep(img).to(dev)
        R = {}
        for key, cs in (("pos64", pos_c), ("rand", rand_c), ("hard", hard_c)):
            Rm = np.zeros((H, W), bool)
            Rm32 = np.zeros((H, W), bool)
            if cs:
                pr = np.concatenate([(torch.sigmoid(H2(sample_windows(Qf, imgt, cs[j:j + 64], H))[:, 0]) > 0.5).cpu().numpy()
                                     for j in range(0, len(cs), 64)])
                for (cy, cx), m in zip(cs, pr):
                    y0, x0 = cy - HALF, cx - HALF
                    ya, xa, yb, xb = max(0, y0), max(0, x0), min(H, y0 + WIN), min(W, x0 + WIN)
                    if yb <= ya or xb <= xa:
                        continue
                    sub = m[ya - y0:yb - y0, xa - x0:xb - x0]
                    Rm[ya:yb, xa:xb] |= sub
                    if key == "pos64":
                        m32 = np.zeros_like(m); m32[16:48, 16:48] = m[16:48, 16:48]
                        Rm32[ya:yb, xa:xb] |= m32[ya - y0:yb - y0, xa - x0:xb - x0]
                    if key in ("rand", "hard"):
                        newfp = sub & (pred[ya:yb, xa:xb] == 0) & (gt[ya:yb, xa:xb] != 2)
                        touch[key]["nwin"] += 1; touch[key]["nwin_fire8"] += int(newfp.sum() >= 8)
            R[key] = Rm
            if key == "pos64":
                R["pos32"] = Rm32
                touch["pos32"]["nwin"] += len(cs); touch["pos64"]["nwin"] += len(cs)
        R["pos64+rand+hard"] = R["pos64"] | R["rand"] | R["hard"]
        gtc = gt == 2
        for key in VARS:
            if key == "base":
                M = pred
            else:
                Rm = R[key]
                M = pred.copy(); conv = Rm & (pred == 0); M[conv] = 2
                if key in touch:
                    t_ = touch[key]
                    t_["rock"] += int((Rm & (pred == 1)).sum()); t_["rock_gtcrater"] += int((Rm & (pred == 1) & gtc).sum())
                    t_["mtn"] += int((Rm & (pred == 3)).sum()); t_["mtn_gtcrater"] += int((Rm & (pred == 3) & gtc).sum())
                    t_["conv_tp"] += int((conv & gtc).sum()); t_["conv_fp"] += int((conv & ~gtc & (gt < NC)).sum())
            CM[key] += conf(gt, M)
            if n > 1:
                tp = np.bincount(lab[(M == 2) & gtc].ravel(), minlength=n)
                for k in range(1, n):
                    b = binof(area[k])
                    if b >= 0:
                        hit50[key][b] += int(tp[k] * 2 >= area[k])
        for k in range(1, n):
            b = binof(area[k])
            if b >= 0:
                nb[b] += 1
        n_done = ii + 1
        if n_done % 100 == 0:
            el = time.time() - t_eval0
            log(f"[eval] {n_done}/{len(va)} {el:.0f}s eta {el / n_done * (len(va) - n_done):.0f}s")
    t_eval = time.time() - t_eval0

    # ---- D1 metrics ----
    N = n_done
    ib = np.array([r["bin"] for r in inst]); ih = np.array([r["hit50"] for r in inst]); idist = np.array([r["dist"] for r in inst])
    D1 = {}
    for name, per in cands.items():
        allc = [(v, g, i_) for i_, lst in enumerate(per) for (v, g, _, _) in lst]
        allc.sort(key=lambda z: -z[0])
        best = np.full(len(inst), -np.inf)
        rank = np.full(len(inst), 10 ** 9)
        for i_, lst in enumerate(per):
            for r_, (v, g, _, _) in enumerate(lst):
                if g >= 0:
                    best[g] = max(best[g], v); rank[g] = min(rank[g], r_)
        fps = np.array([z[1] < 0 for z in allc]); scs = np.array([z[0] for z in allc])
        cumfp = np.cumsum(fps)
        res = {"total_cands": int(len(allc)), "fp_per_img_at_all_top50": float(fps.sum() / N)}
        for bi in (1, 2):
            for sub, m in (("all", ib == bi), ("frozen_missed", (ib == bi) & (ih == 0))):
                d = {"n": int(m.sum())}
                for K in (2, 5, 10):
                    d[f"R@top{K}"] = float((rank[m] < K).mean()) if m.sum() else None
                for f in (0.5, 1, 2, 5, 10):
                    lim = f * N
                    okk = np.where(cumfp <= lim)[0]
                    thr = scs[okk[-1]] if len(okk) else np.inf
                    d[f"R@{f}FP/img"] = float((best[m] >= thr).mean()) if m.sum() else None
                d["R@any(top50)"] = float((best[m] > -np.inf).mean()) if m.sum() else None
                res[f"{BINS[bi][0]}-{BINS[bi][1]}_{sub}"] = d
        D1[name] = res
    miss = {}
    for bi in range(len(BINS)):
        m = (ib == bi) & (ih == 0)
        dd = idist[m]; nopred = ~(dd < 1e4); dd = dd[~nopred]
        miss[f"{BINS[bi][0]}-{BINS[bi][1]}"] = {"n_missed": int(m.sum()), "n_total": int((ib == bi).sum()),
                                                "frac_dist0(partial)": float((idist[m] == 0).mean()) if m.sum() else None,
                                                "median_px": float(np.median(dd)) if len(dd) else None,
                                                "p90_px": float(np.percentile(dd, 90)) if len(dd) else None,
                                                "frac_no_crater_pred_in_img": float(nopred.mean()) if m.sum() else None,
                                                "p75_px": float(np.percentile(dd, 75)) if len(dd) else None,
                                                "frac_le16px": float((dd <= 16).sum() / max(m.sum(), 1)),
                                                "frac_le32px": float((dd <= 32).sum() / max(m.sum(), 1))}
    # ---- D2 metrics ----
    D2 = {"iou": {k: ious(CM[k]) for k in VARS},
          "recall50": {k: {f"{lo}-{hi}": float(hit50[k][b] / max(nb[b], 1)) for b, (lo, hi) in enumerate(BINS)} for k in VARS},
          "n_inst": {f"{lo}-{hi}": int(nb[b]) for b, (lo, hi) in enumerate(BINS)},
          "touch": touch}
    res = {"n_images": N, "train_iters": args.iters, "bs": args.bs, "t_train_s": t_train, "t_eval_s": t_eval,
           "t_total_s": time.time() - T0, "D1": D1, "D1_missed_dist_to_pred_crater": miss, "D2": D2}
    (out / "sc_diag.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(out / "instances.npy", np.array([[r["img"], r["area"], r["bin"], r["hit50"], r["dist"], r["cy"], r["cx"]] for r in inst], np.float64))
    for name, per in cands.items():
        np.save(out / f"cands_{name}.npy", np.array([[i_, v, g, y, x] for i_, lst in enumerate(per) for (v, g, y, x) in lst], np.float64))
    log(json.dumps(res, ensure_ascii=False))
    log("SC_DIAG_DONE")


if __name__ == "__main__":
    main()
