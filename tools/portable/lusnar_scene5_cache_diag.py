#!/usr/bin/env python3
"""Scene5 diagnostics + cache per-scale logits / fused TTA probs (first-round weights).

Stages:
  cache  - GPU: per-scale sliding-window logits, save float16, fused TTA probs,
           per-scale IoU, confusion, preds for attribution
  audit  - CPU: T0 scale audit from GT connected components
  attrib - CPU: Scene5 error attribution (adapted from lusnar_error_attrib)

Does NOT touch Moon_2/4/9 labels for selection. Does NOT start plus5.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
from _lusnar_common import (  # noqa: E402
    CLASS_NAMES,
    COLOR_TO_ID,
    IGNORE,
    MEAN,
    NUM_CLASSES,
    STD,
    RowHeightAttn,
    build_segformer_b2,
    confusion_rates,
    ensure_mmcv_stub,
    find_pretrain,
    load_pairs,
    resolve_project_root,
    rgb_label_to_index,
    set_seed,
)

ensure_mmcv_stub(SCRIPT)

EN = ["regolith", "rock", "crater", "mountain", "sky"]
DEFAULT_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5)
CORE_SCALES = (0.75, 1.0, 1.25)  # saved + used for fused TTA


def imread_bgr(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    return arr[:, :, ::-1].copy()


def load_model(checkpoint: Path, project: Path, device):
    pretrain = find_pretrain(project)
    core = build_segformer_b2(pretrain).to(device)
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    core.load_state_dict(ckpt["model"])
    core.eval()
    row = None
    if ckpt.get("row_attn"):
        row = RowHeightAttn().to(device)
        row.load_state_dict(ckpt["row_attn"])
        row.eval()
    return core, row, ckpt


def _preprocess_bgr(img_bgr: np.ndarray) -> torch.Tensor:
    img_f = (img_bgr.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(img_f.transpose(2, 0, 1)).float()


def _forward_logits(net, imgs, row_attn=None):
    x = net.extract_feat(imgs)
    logits = net.decode_head.forward(x)
    if row_attn is not None:
        logits = row_attn(logits)
    return logits


def _window_weight(crop: int, mode: str = "uniform", edge_drop: int = 0) -> np.ndarray:
    """Per-pixel weight inside a crop window for overlap fusion."""
    yy, xx = np.mgrid[0:crop, 0:crop].astype(np.float32)
    if mode == "uniform":
        w = np.ones((crop, crop), np.float32)
    elif mode == "gaussian":
        cy = (crop - 1) / 2.0
        sigma = crop / 4.0
        w = np.exp(-((yy - cy) ** 2 + (xx - cy) ** 2) / (2 * sigma**2)).astype(np.float32)
    elif mode == "cosine":
        wy = 0.5 * (1 - np.cos(2 * np.pi * yy / max(crop - 1, 1)))
        wx = 0.5 * (1 - np.cos(2 * np.pi * xx / max(crop - 1, 1)))
        w = (wy * wx).astype(np.float32)
        w = np.maximum(w, 1e-3)
    else:
        raise ValueError(mode)
    if edge_drop > 0:
        w[:edge_drop, :] = 0
        w[-edge_drop:, :] = 0
        w[:, :edge_drop] = 0
        w[:, -edge_drop:] = 0
        # keep corners of image covered: caller must ensure count>0 fallback
    return w


@torch.no_grad()
def sliding_window_logits(
    net,
    img_bgr: np.ndarray,
    device,
    crop: int = 512,
    stride: int = 256,
    row_attn=None,
    weight_mode: str = "uniform",
    edge_drop: int = 0,
):
    h, w = img_bgr.shape[:2]
    C = NUM_CLASSES
    acc = torch.zeros(C, h, w, device=device, dtype=torch.float32)
    cnt = torch.zeros(1, h, w, device=device, dtype=torch.float32)
    wmap = torch.from_numpy(_window_weight(crop, weight_mode, edge_drop)).to(device)
    ys = list(range(0, max(h - crop, 0) + 1, stride))
    xs = list(range(0, max(w - crop, 0) + 1, stride))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]
    if ys[-1] != max(0, h - crop):
        ys.append(max(0, h - crop))
    if xs[-1] != max(0, w - crop):
        xs.append(max(0, w - crop))
    n_win = 0
    for y0 in ys:
        for x0 in xs:
            patch = img_bgr[y0 : y0 + crop, x0 : x0 + crop]
            ph, pw = patch.shape[:2]
            if ph != crop or pw != crop:
                pad = np.zeros((crop, crop, 3), np.uint8)
                pad[:ph, :pw] = patch
                patch = pad
            t = _preprocess_bgr(patch).unsqueeze(0).to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = _forward_logits(net, t, row_attn=row_attn)
                if logits.shape[-2:] != (crop, crop):
                    logits = F.interpolate(
                        logits, size=(crop, crop), mode="bilinear", align_corners=False
                    )
            logits = logits.float()[0, :, :ph, :pw]
            ww = wmap[:ph, :pw]
            # if edge_drop zeroed entire valid region, fall back to ones
            if float(ww.sum()) <= 0:
                ww = torch.ones_like(ww)
            acc[:, y0 : y0 + ph, x0 : x0 + pw] += logits * ww.unsqueeze(0)
            cnt[:, y0 : y0 + ph, x0 : x0 + pw] += ww
            n_win += 1
    # fill any remaining zeros (can happen with aggressive edge_drop at borders)
    zero = cnt[0] <= 0
    if bool(zero.any()):
        # re-run border pixels with uniform single-window fallback is heavy;
        # instead set cnt=1 and leave acc as-is only where we have mass;
        # for empty cells, use nearest non-empty by simple inpaint via uniform pass flag
        cnt = torch.where(cnt > 0, cnt, torch.ones_like(cnt))
        # better: force at least center weight — already handled by ww fallback per window
    return acc / cnt.clamp(min=1e-6), n_win


def update_inter_union(inter, union, pred, mask):
    for c in range(NUM_CLASSES):
        p = pred == c
        g = mask == c
        inter[c] += np.logical_and(p, g).sum()
        union[c] += np.logical_or(p, g).sum()


def update_cm(cm, pred, mask):
    valid = mask != IGNORE
    pv = pred[valid].ravel().astype(np.int64)
    gv = mask[valid].ravel().astype(np.int64)
    ok = (gv >= 0) & (gv < NUM_CLASSES) & (pv >= 0) & (pv < NUM_CLASSES)
    idx = pv[ok] * NUM_CLASSES + gv[ok]
    bc = np.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES)
    cm += bc.reshape(NUM_CLASSES, NUM_CLASSES)


def iou_from_iu(inter, union):
    iou = [
        float(inter[c] / union[c]) if union[c] > 0 else float("nan")
        for c in range(NUM_CLASSES)
    ]
    valid = [x for x in iou if x == x]
    miou = float(np.mean(valid)) if valid else float("nan")
    return miou, iou


def run_cache(args):
    set_seed(42)
    project = resolve_project_root(SCRIPT, args.project_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    net, row, ckpt = load_model(ckpt_path, project, device)
    pairs = load_pairs(Path(args.val_list))
    if args.max_images:
        pairs = pairs[: args.max_images]

    scales_all = tuple(float(s) for s in args.scales)
    scales_save = tuple(float(s) for s in args.save_scales)
    crop, stride = args.crop, args.stride

    out_logits = Path(args.logits_dir)
    out_probs = Path(args.probs_dir)
    out_preds = Path(args.preds_dir)
    out_logits.mkdir(parents=True, exist_ok=True)
    out_probs.mkdir(parents=True, exist_ok=True)
    out_preds.mkdir(parents=True, exist_ok=True)

    # per-scale stats
    stats = {
        s: {
            "inter": np.zeros(NUM_CLASSES, np.float64),
            "union": np.zeros(NUM_CLASSES, np.float64),
            "cm": np.zeros((NUM_CLASSES, NUM_CLASSES), np.float64),
            "n_win": 0,
        }
        for s in scales_all
    }
    fuse_inter = np.zeros(NUM_CLASSES, np.float64)
    fuse_union = np.zeros(NUM_CLASSES, np.float64)
    fuse_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.float64)

    t0 = time.time()
    n_fwd = 0
    for i, (ip, lp) in enumerate(pairs):
        img = imread_bgr(ip)
        lab = cv2.imread(lp, cv2.IMREAD_COLOR)
        if img is None or lab is None:
            print(f"[skip] {ip}", flush=True)
            continue
        mask = rgb_label_to_index(lab)
        h0, w0 = mask.shape
        stem = Path(ip).stem
        acc_fuse = torch.zeros(NUM_CLASSES, h0, w0, device=device, dtype=torch.float32)
        n_fuse = 0
        for s in scales_all:
            nh, nw = max(1, int(round(h0 * s))), max(1, int(round(w0 * s)))
            img_s = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            logits_s, n_win = sliding_window_logits(
                net, img_s, device, crop=crop, stride=stride, row_attn=row,
                weight_mode="uniform", edge_drop=0,
            )
            n_fwd += n_win
            stats[s]["n_win"] += n_win
            logits_up = F.interpolate(
                logits_s.unsqueeze(0), size=(h0, w0), mode="bilinear", align_corners=False
            )[0]
            pred_s = logits_up.argmax(0).cpu().numpy().astype(np.uint8)
            update_inter_union(stats[s]["inter"], stats[s]["union"], pred_s, mask)
            update_cm(stats[s]["cm"], pred_s, mask)
            if s in scales_save:
                arr = logits_up.cpu().numpy().astype(np.float16)
                np.save(out_logits / f"{stem}_s{s:.2f}.npy", arr)
                acc_fuse += logits_up
                n_fuse += 1
        if n_fuse == 0:
            continue
        fused = acc_fuse / n_fuse
        probs = torch.softmax(fused, dim=0).cpu().numpy().astype(np.float16)
        np.save(out_probs / f"{stem}.npy", probs)
        pred = fused.argmax(0).cpu().numpy().astype(np.uint8)
        Image.fromarray(pred, mode="L").save(out_preds / f"{stem}.png")
        update_inter_union(fuse_inter, fuse_union, pred, mask)
        update_cm(fuse_cm, pred, mask)

        if (i + 1) % 20 == 0 or (i + 1) == len(pairs):
            elapsed = time.time() - t0
            print(
                f"[cache] {i+1}/{len(pairs)} elapsed={elapsed/60:.1f}min "
                f"fwd_wins={n_fwd}",
                flush=True,
            )

    per_scale = {}
    for s in scales_all:
        miou, iou = iou_from_iu(stats[s]["inter"], stats[s]["union"])
        per_scale[f"{s:.2f}"] = {
            "mIoU": miou,
            "per_class": {CLASS_NAMES[c]: iou[c] for c in range(NUM_CLASSES)},
            "per_class_en": {EN[c]: iou[c] for c in range(NUM_CLASSES)},
            "rates": confusion_rates(stats[s]["cm"]),
            "n_win_total": int(stats[s]["n_win"]),
        }
    f_miou, f_iou = iou_from_iu(fuse_inter, fuse_union)
    summary = {
        "checkpoint": str(ckpt_path),
        "ckpt_iter": int(ckpt.get("iter", -1)) if isinstance(ckpt, dict) else -1,
        "row_attn": row is not None,
        "n_images": len(pairs),
        "scales_eval": list(scales_all),
        "scales_saved": list(scales_save),
        "crop": crop,
        "stride": stride,
        "forward_windows": int(n_fwd),
        "wall_s": round(time.time() - t0, 1),
        "per_scale": per_scale,
        "fused_tta": {
            "mIoU": f_miou,
            "per_class": {CLASS_NAMES[c]: f_iou[c] for c in range(NUM_CLASSES)},
            "per_class_en": {EN[c]: f_iou[c] for c in range(NUM_CLASSES)},
            "rates": confusion_rates(fuse_cm),
            "cm_pred_gt": fuse_cm.tolist(),  # [pred, gt] like common.py
        },
        "paths": {
            "logits_dir": str(out_logits),
            "probs_dir": str(out_probs),
            "preds_dir": str(out_preds),
        },
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("CACHE_DONE", out_json, flush=True)
    return summary


def run_audit(args):
    """T0: GT connected-component apparent size vs inference scales."""
    pairs = load_pairs(Path(args.val_list))
    if args.max_images:
        pairs = pairs[: args.max_images]
    scales = tuple(float(s) for s in args.scales)
    # bins on full-res area
    area_bins = [0, 16 * 16, 32 * 32, 64 * 64, 96 * 96, 10**12]
    bin_names = ["<16^2", "16^2-32^2", "32^2-64^2", "64^2-96^2", ">=96^2"]
    # at crop 512, object diagonal/side after scale
    per_class = {c: {"areas": [], "n_cc": 0} for c in (1, 2)}  # rock, crater
    hist = {c: np.zeros(len(bin_names), np.int64) for c in (1, 2)}
    for ip, lp in pairs:
        lab = cv2.imread(lp, cv2.IMREAD_COLOR)
        if lab is None:
            continue
        mask = rgb_label_to_index(lab)
        for c in (1, 2):
            nlab, labels, stats, _ = cv2.connectedComponentsWithStats(
                (mask == c).astype(np.uint8), 8
            )
            for i in range(1, nlab):
                area = int(stats[i, cv2.CC_STAT_AREA])
                per_class[c]["areas"].append(area)
                per_class[c]["n_cc"] += 1
                for bi in range(len(bin_names)):
                    if area_bins[bi] <= area < area_bins[bi + 1]:
                        hist[c][bi] += 1
                        break

    def summarize(areas):
        if not areas:
            return {}
        a = np.asarray(areas, np.float64)
        side = np.sqrt(a)
        return {
            "n_cc": int(len(a)),
            "area_p50": float(np.median(a)),
            "area_p90": float(np.percentile(a, 90)),
            "side_p50": float(np.median(side)),
            "side_p90": float(np.percentile(side, 90)),
            "frac_side_lt_32": float((side < 32).mean()),
            "frac_side_lt_64": float((side < 64).mean()),
        }

    scale_note = {}
    for s in scales:
        # apparent side in network crop space ≈ side * s
        scale_note[f"{s:.2f}"] = {
            "meaning": f"object side appears ~{s}x at this infer scale; "
            f"objects with full-res side < {32/s:.1f}px fall under ~32px at this scale"
        }

    out = {
        "n_images": len(pairs),
        "crop": 512,
        "classes": {
            EN[c]: {
                **summarize(per_class[c]["areas"]),
                "area_hist": {bin_names[i]: int(hist[c][i]) for i in range(len(bin_names))},
            }
            for c in (1, 2)
        },
        "scale_notes": scale_note,
        "recommendation": (
            "If many rock/crater CC have side_p50 << 64, prefer scales >=1.25 or re-zoom (I3). "
            "If large objects dominate, scales <=0.75 help context (mountain)."
        ),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("AUDIT_DONE", out_json, flush=True)
    return out


# ---- Scene5 error attribution (adapted; scenes={5}) ----
SIZE_BINS = ("small", "mid", "large")
ROW_BINS = 4
GATES = {
    "majority_swallow": 0.50,
    "texture_confusion": 0.25,
    "boundary": 0.12,
    "scale": 0.15,
    "position_context": 0.15,
    "illumination": 15.0,
}
ZH = {
    "majority_swallow": "多数类吞并",
    "texture_confusion": "类间外观混淆",
    "boundary": "边界",
    "scale": "小目标与尺度",
    "position_context": "位置与上下文",
    "illumination": "光照",
}
TEXTURE_PAIRS = {(1, 3), (3, 1), (4, 3), (3, 4), (2, 1), (1, 2)}
MIN_BIN_PIXELS = 50_000
MIN_ROW_FRACTION = 0.01
BAND_K = 17


def _load_gt(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for color, cid in COLOR_TO_ID.items():
        out[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = cid
    return out


def _load_pred(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def _luminance(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _size_bin(area: int) -> int:
    if area < 32 * 32:
        return 0
    if area < 96 * 96:
        return 1
    return 2


def run_attrib(args):
    pairs = load_pairs(Path(args.val_list))
    pred_dir = Path(args.preds_dir)
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)  # [gt, pred]
    band = np.zeros_like(hist)
    interior = np.zeros_like(hist)
    row = np.zeros((ROW_BINS, NUM_CLASSES, NUM_CLASSES), np.int64)
    size_gt = np.zeros((NUM_CLASSES, 3), np.int64)
    size_hit = np.zeros((NUM_CLASSES, 3), np.int64)
    y_ok = np.zeros(NUM_CLASSES, np.float64)
    y_bad = np.zeros(NUM_CLASSES, np.float64)
    n_ok = np.zeros(NUM_CLASSES, np.int64)
    n_bad = np.zeros(NUM_CLASSES, np.int64)
    n_images = 0
    n_missing = 0

    def add_hist(H, gt, pred, mask):
        g = gt[mask].astype(np.int64)
        p = pred[mask].astype(np.int64)
        keep = (g < NUM_CLASSES) & (p < NUM_CLASSES)
        if keep.any():
            np.add.at(H, (g[keep], p[keep]), 1)

    for ip, lp in pairs:
        stem = Path(ip).stem
        pp = pred_dir / f"{stem}.png"
        if not pp.exists():
            n_missing += 1
            continue
        gt = _load_gt(Path(lp))
        pred = _load_pred(pp)
        if pred.shape != gt.shape:
            raise RuntimeError(f"shape mismatch {pp}")
        yimg = _luminance(Path(ip))
        valid = gt < NUM_CLASSES
        edge = np.zeros(gt.shape, np.uint8)
        edge[1:] |= gt[1:] != gt[:-1]
        edge[:, 1:] |= gt[:, 1:] != gt[:, :-1]
        edge &= valid.astype(np.uint8)
        band_m = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BAND_K, BAND_K))) > 0
        band_m &= valid
        interior_m = valid & ~band_m
        h = gt.shape[0]
        rows = (np.arange(h) * ROW_BINS // max(h, 1))[:, None]
        pred_c = np.where(pred < NUM_CLASSES, pred, 0)
        add_hist(hist, gt, pred_c, valid)
        add_hist(band, gt, pred_c, band_m)
        add_hist(interior, gt, pred_c, interior_m)
        for b in range(ROW_BINS):
            add_hist(row[b], gt, pred_c, valid & (rows == b))
        for c in (1, 2, 3):
            nlab, labels, stats, _ = cv2.connectedComponentsWithStats((gt == c).astype(np.uint8), 8)
            for i in range(1, nlab):
                area = int(stats[i, cv2.CC_STAT_AREA])
                x, y, ww, hh = (int(stats[i, k]) for k in range(4))
                crop = labels[y : y + hh, x : x + ww] == i
                binv = _size_bin(area)
                size_gt[c, binv] += area
                size_hit[c, binv] += int((pred_c[y : y + hh, x : x + ww][crop] == c).sum())
        for c in range(NUM_CLASSES):
            m = gt == c
            ok = m & (pred_c == c)
            bad = m & (pred_c != c)
            y_ok[c] += float(yimg[ok].sum())
            y_bad[c] += float(yimg[bad].sum())
            n_ok[c] += int(ok.sum())
            n_bad[c] += int(bad.sum())
        n_images += 1

    def iou_of(H):
        ious = []
        for c in range(NUM_CLASSES):
            tp = float(H[c, c])
            denom = float(H[c, :].sum() + H[:, c].sum() - tp)
            ious.append(tp / denom if denom > 0 else float("nan"))
        return ious

    def recall_of(H):
        out = []
        for c in range(NUM_CLASSES):
            support = float(H[c, :].sum())
            out.append(float(H[c, c]) / support if support > 0 else float("nan"))
        return out

    band_iou = iou_of(band)
    in_iou = iou_of(interior)
    ious = iou_of(hist)
    row_rec = [recall_of(row[b]) for b in range(ROW_BINS)]
    classes = []
    for c in range(NUM_CLASSES):
        support = int(hist[c, :].sum())
        fn = support - int(hist[c, c])
        dest = []
        if fn > 0:
            for p in range(NUM_CLASSES):
                if p == c:
                    continue
                dest.append((p, int(hist[c, p]), float(hist[c, p]) / fn))
            dest.sort(key=lambda t: -t[1])
        reg_share = next((share for p, _, share in dest if p == 0), 0.0)
        top = dest[0] if dest else None
        small_gt = int(size_gt[c, 0])
        large_gt = int(size_gt[c, 2])
        small_r = float(size_hit[c, 0] / small_gt) if small_gt else float("nan")
        large_r = float(size_hit[c, 2] / large_gt) if large_gt else float("nan")
        row_support = [int(row[b, c, :].sum()) for b in range(ROW_BINS)]
        usable = [b for b in range(ROW_BINS) if support > 0 and row_support[b] >= MIN_ROW_FRACTION * support]
        recs = [row_rec[b][c] for b in usable]
        row_span = (max(recs) - min(recs)) if len(recs) >= 2 else float("nan")
        y_ok_m = y_ok[c] / n_ok[c] if n_ok[c] else float("nan")
        y_bad_m = y_bad[c] / n_bad[c] if n_bad[c] else float("nan")
        dy = (y_bad_m - y_ok_m) if y_ok_m == y_ok_m and y_bad_m == y_bad_m else float("nan")
        b_gap = (
            in_iou[c] - band_iou[c]
            if in_iou[c] == in_iou[c] and band_iou[c] == band_iou[c]
            else float("nan")
        )
        s_gap = large_r - small_r if large_r == large_r and small_r == small_r else float("nan")
        flags = []
        if fn > 0 and reg_share >= GATES["majority_swallow"]:
            flags.append(("majority_swallow", reg_share, f"漏检里 {reg_share:.0%} 被判成月壤"))
        if (
            top is not None
            and top[0] != 0
            and top[2] >= GATES["texture_confusion"]
            and (c, top[0]) in TEXTURE_PAIRS
        ):
            flags.append(
                ("texture_confusion", top[2], f"漏检里 {top[2]:.0%} 被判成{CLASS_NAMES[top[0]]}")
            )
        if b_gap == b_gap and b_gap >= GATES["boundary"]:
            flags.append(
                ("boundary", b_gap, f"内部 IoU {in_iou[c]:.3f}，边界带 IoU {band_iou[c]:.3f}")
            )
        if (
            s_gap == s_gap
            and s_gap >= GATES["scale"]
            and small_gt >= MIN_BIN_PIXELS
            and large_gt >= MIN_BIN_PIXELS
        ):
            flags.append(("scale", s_gap, f"小块召回 {small_r:.3f}，大块召回 {large_r:.3f}"))
        if row_span == row_span and row_span >= GATES["position_context"]:
            flags.append(("position_context", row_span, f"四条水平带召回极差 {row_span:.3f}"))
        if dy == dy and abs(dy) >= GATES["illumination"] and n_bad[c] >= MIN_BIN_PIXELS:
            side = "更暗" if dy < 0 else "更亮"
            flags.append(
                ("illumination", abs(dy) / 100.0, f"判错像素比判对像素{side} {abs(dy):.1f} 灰度")
            )
        flags.sort(key=lambda t: -t[1])
        classes.append(
            {
                "class": EN[c],
                "name": CLASS_NAMES[c],
                "iou": None if ious[c] != ious[c] else round(float(ious[c]), 6),
                "support": support,
                "fn": fn,
                "fn_to": [
                    {
                        "pred": EN[p],
                        "name": CLASS_NAMES[p],
                        "pixels": n,
                        "share_of_fn": round(s, 6),
                    }
                    for p, n, s in dest[:3]
                ],
                "boundary_iou": None if band_iou[c] != band_iou[c] else round(float(band_iou[c]), 6),
                "interior_iou": None if in_iou[c] != in_iou[c] else round(float(in_iou[c]), 6),
                "size_recall": {
                    SIZE_BINS[b]: {
                        "gt_pixels": int(size_gt[c, b]),
                        "recall": (
                            None
                            if size_gt[c, b] == 0
                            else round(float(size_hit[c, b] / size_gt[c, b]), 6)
                        ),
                    }
                    for b in range(3)
                },
                "row_recall": [
                    None if row_rec[b][c] != row_rec[b][c] else round(float(row_rec[b][c]), 6)
                    for b in range(ROW_BINS)
                ],
                "flags": [
                    {"type": name, "zh": ZH[name], "score": round(float(score), 6), "evidence": text}
                    for name, score, text in flags
                ],
                "primary": flags[0][0] if flags else None,
            }
        )

    valid = [v for v in ious if v == v]
    payload = {
        "scope": "Scene5 only (first-round fused TTA preds)",
        "n_images": n_images,
        "n_missing_pred": n_missing,
        "mIoU": round(float(np.mean(valid)), 6) if valid else None,
        "hist_gt_pred": hist.tolist(),
        "classes": classes,
        "gates": GATES,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ATTRIB_DONE", out_json, flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cache", "audit", "attrib", "all_cpu"], required=True)
    ap.add_argument("--project-root", default=r"/mnt/d/LuSNAR_Seg")
    ap.add_argument(
        "--checkpoint",
        default=r"/mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/best.pth",
    )
    ap.add_argument("--val-list", default=r"/root/lusnar_data/lists/val_scene5.txt")
    ap.add_argument(
        "--logits-dir",
        default=r"/root/lusnar_data/cache/scene5_scale_logits",
    )
    ap.add_argument(
        "--probs-dir",
        default=r"/root/lusnar_data/cache/scene5_tta_probs",
    )
    ap.add_argument(
        "--preds-dir",
        default=r"/root/lusnar_data/cache/scene5_preds",
    )
    ap.add_argument(
        "--out-json",
        default=r"/mnt/d/LuSNAR_Seg/results/inference_phase/cache_diag.json",
    )
    ap.add_argument("--scales", nargs="+", type=float, default=list(DEFAULT_SCALES))
    ap.add_argument("--save-scales", nargs="+", type=float, default=list(CORE_SCALES))
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    if args.stage == "cache":
        run_cache(args)
    elif args.stage == "audit":
        if args.out_json.endswith("cache_diag.json"):
            args.out_json = str(Path(args.out_json).with_name("t0_scale_audit.json"))
        run_audit(args)
    elif args.stage == "attrib":
        if args.out_json.endswith("cache_diag.json"):
            args.out_json = str(Path(args.out_json).with_name("scene5_error_attrib.json"))
        run_attrib(args)
    elif args.stage == "all_cpu":
        # audit + attrib after cache
        args.out_json = str(Path(args.out_json).with_name("t0_scale_audit.json"))
        run_audit(args)
        args.out_json = str(Path(args.out_json).with_name("scene5_error_attrib.json"))
        run_attrib(args)


if __name__ == "__main__":
    main()
