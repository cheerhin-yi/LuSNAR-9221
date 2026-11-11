#!/usr/bin/env python3
"""Scene5 inference protocols I1/I2/I3/I4 + postproc on first-round cached logits/probs.

Pre-registered candidates (8+I4), select on A only, report B once.
Moon_2/4/9 labels never used for selection.
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
from scipy import stats as scipy_stats

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
from _lusnar_common import (  # noqa: E402
    CLASS_NAMES,
    IGNORE,
    MEAN,
    NUM_CLASSES,
    STD,
    RowHeightAttn,
    build_segformer_b2,
    ensure_mmcv_stub,
    find_pretrain,
    load_pairs,
    resolve_project_root,
    rgb_label_to_index,
    set_seed,
)
from lusnar_scene5_cache_diag import (  # noqa: E402
    EN,
    CORE_SCALES,
    imread_bgr,
    load_model,
    sliding_window_logits,
    update_cm,
    update_inter_union,
    iou_from_iu,
)

ensure_mmcv_stub(SCRIPT)

# ---- Pre-registered candidates (BEFORE looking at B) ----
# C0 baseline equal-weight TTA 0.75/1.0/1.25
# I1a per-class pick-best-scale (from A scale IoU)
# I1b per-class soft weights (grid on A)
# I2a gaussian overlap on C0 scales
# I2b cosine overlap + edge_drop=32 on best of C0/I1
# I3  small-object rezoom on best so far
# P1  class logit bias on fused probs
# P1P2 P1 + geometric/CC rules
# I4  hybrid full-image 1024 (large classes) + tiled C0 (rock/crater)
#     9th candidate, added before any B scores were used
PREREGISTERED = [
    "C0_equal_tta",
    "I1a_best_scale_per_class",
    "I1b_soft_scale_weights",
    "I2a_gaussian_overlap",
    "I2b_cosine_edge_drop",
    "I3_small_rezoom",
    "P1_logit_bias",
    "P1P2_bias_plus_geo",
    "I4_hybrid_full_tiles",
]


def load_stems(list_file: Path):
    pairs = load_pairs(list_file)
    return [(Path(ip).stem, ip, lp) for ip, lp in pairs]


def load_scale_logits(logits_dir: Path, stem: str, scales=CORE_SCALES):
    outs = []
    for s in scales:
        p = logits_dir / f"{stem}_s{s:.2f}.npy"
        if not p.exists():
            return None
        outs.append(np.load(p).astype(np.float32))  # C,H,W
    return outs


def load_probs(probs_dir: Path, stem: str):
    p = probs_dir / f"{stem}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


def preload_scale_logits(logits_dir, items, scales=CORE_SCALES, max_items=None):
    """Keep float16 in RAM; cast on use. Optional max_items for search subsets."""
    data = []
    for stem, ip, lp in items:
        zs = load_scale_logits(logits_dir, stem, scales)
        if zs is None:
            continue
        # store float16 to cut RAM ~2x
        zs16 = [z.astype(np.float16, copy=False) for z in zs]
        mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
        data.append((stem, zs16, mask))
        if max_items is not None and len(data) >= max_items:
            break
    return data


def preload_probs(probs_dir, items, max_items=None):
    data = []
    for stem, ip, lp in items:
        pr = load_probs(probs_dir, stem)
        if pr is None:
            continue
        mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
        data.append((stem, pr.astype(np.float16, copy=False), mask))
        if max_items is not None and len(data) >= max_items:
            break
    return data



def per_image_iou(pred, mask):
    """Return per-class IoU vector and mIoU for one image (nan if no support)."""
    ious = np.full(NUM_CLASSES, np.nan, np.float64)
    for c in range(NUM_CLASSES):
        p = pred == c
        g = mask == c
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        if union > 0:
            ious[c] = inter / union
    valid = ious[~np.isnan(ious)]
    miou = float(valid.mean()) if len(valid) else float("nan")
    return ious, miou


def fuse_equal(logits_list):
    acc = np.zeros_like(logits_list[0])
    for z in logits_list:
        acc += z
    return acc / len(logits_list)


def fuse_weighted(logits_list, weights_CxS):
    """weights_CxS: (C, S) per-class scale weights (rows sum to 1)."""
    C, H, W = logits_list[0].shape
    S = len(logits_list)
    out = np.zeros((C, H, W), np.float32)
    for c in range(C):
        acc = np.zeros((H, W), np.float32)
        for si, z in enumerate(logits_list):
            acc += weights_CxS[c, si] * z[c]
        out[c] = acc
    return out


def pick_best_scale_weights(per_scale_iou_A):
    """I1a: one-hot on best scale per class from A aggregate IoU dict scale->iou[C]."""
    scales = list(per_scale_iou_A.keys())
    C = NUM_CLASSES
    W = np.zeros((C, len(scales)), np.float32)
    for c in range(C):
        best_i = int(np.nanargmax([per_scale_iou_A[s][c] for s in scales]))
        W[c, best_i] = 1.0
    return W, scales


def search_soft_weights(logits_dir, items_A, scales=CORE_SCALES):
    """I1b: coarse grid; preload A logits once."""
    S = len(scales)
    equal = np.ones(S, np.float32) / S
    candidates = []
    for pref in range(S):
        for alpha in (0.5, 0.7, 0.9):
            w = np.full(S, (1 - alpha) / (S - 1), np.float32)
            w[pref] = alpha
            candidates.append(w)
    candidates.append(equal)

    packed_search = preload_scale_logits(logits_dir, items_A, scales, max_items=240)
    print(f"[I1b] search subset {len(packed_search)} A images", flush=True)
    W = np.stack([equal] * NUM_CLASSES, axis=0)
    for c in (1, 2, 3):
        best_c = equal
        best_c_iou = -1.0
        for w in candidates:
            inter = 0.0
            union = 0.0
            for stem, zs, mask in packed_search:
                zs_f = [z.astype(np.float32) for z in zs]
                Wc = np.stack([equal] * NUM_CLASSES, axis=0)
                Wc[c] = w
                fused = fuse_weighted(zs_f, Wc)
                pred = fused.argmax(0).astype(np.uint8)
                pp = pred == c
                gg = mask == c
                inter += np.logical_and(pp, gg).sum()
                union += np.logical_or(pp, gg).sum()
            iou = inter / union if union > 0 else 0.0
            if iou > best_c_iou:
                best_c_iou = iou
                best_c = w
        W[c] = best_c
        print(f"[I1b] class {EN[c]} best_iou={best_c_iou:.4f} w={best_c.tolist()}", flush=True)
    # final score on full A via streaming (no giant RAM)
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    per_img = []
    for stem, ip, lp in items_A:
        zs = load_scale_logits(logits_dir, stem, scales)
        if zs is None:
            continue
        fused = fuse_weighted(zs, W)
        pred = fused.argmax(0).astype(np.uint8)
        mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
        update_inter_union(inter, union, pred, mask)
        _, mi = per_image_iou(pred, mask)
        per_img.append(mi)
    miou, ious = iou_from_iu(inter, union)
    return W, {"mIoU": miou, "per_class": ious, "per_image_miou": per_img}


def eval_fusion_on(items, logits_dir, W=None, scales=CORE_SCALES):
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    per_img = []
    per_img_cls = []
    t0 = time.time()
    for stem, ip, lp in items:
        zs = load_scale_logits(logits_dir, stem, scales)
        if zs is None:
            continue
        fused = fuse_equal(zs) if W is None else fuse_weighted(zs, W)
        pred = fused.argmax(0).astype(np.uint8)
        mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
        update_inter_union(inter, union, pred, mask)
        ious, mi = per_image_iou(pred, mask)
        per_img.append(mi)
        per_img_cls.append(ious)
    miou, ious = iou_from_iu(inter, union)
    return {
        "mIoU": miou,
        "per_class": {CLASS_NAMES[c]: ious[c] for c in range(NUM_CLASSES)},
        "per_class_en": {EN[c]: ious[c] for c in range(NUM_CLASSES)},
        "per_image_miou": per_img,
        "per_image_cls": [x.tolist() for x in per_img_cls],
        "n": len(per_img),
        "wall_s": round(time.time() - t0, 2),
        "ms_per_img": round(1000 * (time.time() - t0) / max(len(per_img), 1), 2),
        "forward_passes_note": "CPU fusion from cached logits (0 new GPU forwards)",
    }


# ---- Post-processing ----
def apply_logit_bias(probs, bias):
    # probs CxHxW; convert to logits via log, add bias, softmax
    logits = np.log(np.clip(probs, 1e-6, 1.0))
    logits = logits + bias.reshape(-1, 1, 1)
    logits = logits - logits.max(axis=0, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=0, keepdims=True)


def search_logit_bias(probs_dir, items_A, step=0.1, span=0.8):
    """P1: bias search with preloaded probs."""
    grid = np.arange(-span, span + 1e-9, step)
    packed = preload_probs(probs_dir, items_A, max_items=240)
    print(f"[P1] search subset {len(packed)} A probs", flush=True)
    bias = np.zeros(NUM_CLASSES, np.float32)

    def score(b):
        inter = np.zeros(NUM_CLASSES, np.float64)
        union = np.zeros(NUM_CLASSES, np.float64)
        for stem, pr, mask in packed:
            out = apply_logit_bias(pr.astype(np.float32), b)
            pred = out.argmax(0).astype(np.uint8)
            update_inter_union(inter, union, pred, mask)
        miou, _ = iou_from_iu(inter, union)
        return miou

    base = score(bias)
    for c in (3, 1, 2, 4):
        best_v = bias[c]
        best_s = base
        for v in grid:
            bias[c] = v
            s = score(bias)
            if s > best_s:
                best_s = s
                best_v = v
        bias[c] = best_v
        base = best_s
        print(f"[P1] class {EN[c]} bias={best_v:.2f} mIoU={best_s:.4f}", flush=True)
    return bias, base


def geometric_cc_rules(pred, min_sky_row_frac=0.02, rock_embed_thresh=0.55, min_area=64):
    """P2: sky-above, embedded-rock-in-mountain -> mountain; small CC clean (not rock)."""
    out = pred.copy()
    h, w = out.shape
    # (a) sky must not appear below mountain/regolith column-wise dominant band — simplified:
    # if a sky CC has centroid in bottom 40% and touches mountain/regolith above it wrongly,
    # reassign to neighbor majority.
    nlab, labels, stats, cents = cv2.connectedComponentsWithStats((out == 4).astype(np.uint8), 8)
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        cy = cents[i, 1]
        if cy > 0.55 * h and area < 0.05 * h * w:
            x, y, ww, hh = (int(stats[i, k]) for k in range(4))
            region = labels[y : y + hh, x : x + ww] == i
            # neighbor majority from dilated border
            mask_i = (labels == i)
            dil = cv2.dilate(mask_i.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            border = dil & ~mask_i
            vals = out[border]
            vals = vals[vals != 4]
            if len(vals):
                out[mask_i] = np.bincount(vals, minlength=NUM_CLASSES).argmax()

    # (b) rock embedded in mountain
    nlab, labels, stats, cents = cv2.connectedComponentsWithStats((out == 1).astype(np.uint8), 8)
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        mask_i = labels == i
        dil = cv2.dilate(mask_i.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        border = dil & ~mask_i
        if border.sum() == 0:
            continue
        frac_mtn = float((out[border] == 3).mean())
        cy = cents[i, 1]
        if frac_mtn >= rock_embed_thresh and cy < 0.7 * h:
            out[mask_i] = 3

    # (c) small CC cleanup for mountain/sky/crater only (NOT rock)
    for c in (2, 3, 4):
        nlab, labels, stats, _ = cv2.connectedComponentsWithStats((out == c).astype(np.uint8), 8)
        for i in range(1, nlab):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                mask_i = labels == i
                dil = cv2.dilate(mask_i.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                border = dil & ~mask_i
                vals = out[border]
                vals = vals[vals != c]
                if len(vals):
                    out[mask_i] = np.bincount(vals, minlength=NUM_CLASSES).argmax()
    return out


def eval_probs_protocol(items, probs_dir, bias=None, use_geo=False, custom_pred_fn=None):
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    per_img = []
    t0 = time.time()
    for stem, ip, lp in items:
        if custom_pred_fn is not None:
            pred = custom_pred_fn(stem, ip, lp)
            if pred is None:
                continue
        else:
            pr = load_probs(probs_dir, stem)
            if pr is None:
                continue
            if bias is not None:
                pr = apply_logit_bias(pr, bias)
            pred = pr.argmax(0).astype(np.uint8)
            if use_geo:
                pred = geometric_cc_rules(pred)
        mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
        update_inter_union(inter, union, pred, mask)
        _, mi = per_image_iou(pred, mask)
        per_img.append(mi)
    miou, ious = iou_from_iu(inter, union)
    return {
        "mIoU": miou,
        "per_class": {CLASS_NAMES[c]: ious[c] for c in range(NUM_CLASSES)},
        "per_class_en": {EN[c]: ious[c] for c in range(NUM_CLASSES)},
        "per_image_miou": per_img,
        "n": len(per_img),
        "wall_s": round(time.time() - t0, 2),
        "ms_per_img": round(1000 * (time.time() - t0) / max(len(per_img), 1), 2),
    }


def paired_tests(base_per_img, cand_per_img, n_boot=10000, seed=42):
    a = np.asarray(base_per_img, np.float64)
    b = np.asarray(cand_per_img, np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = b - a
    # Wilcoxon
    try:
        w_stat, w_p = scipy_stats.wilcoxon(d)
    except Exception:
        w_stat, w_p = float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(d[idx].mean())
    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "delta_mean": float(d.mean()),
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
        "wilcoxon_p": float(w_p) if w_p == w_p else None,
        "bootstrap_ci95": [float(lo), float(hi)],
        "n": int(n),
    }


def holm_correct(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.zeros(m, np.float64)
    running = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * pvals[i]
        running = max(running, adj)
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist()


def _merge_partial(parts):
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    per_img = []
    n_fwd = 0
    wall = 0.0
    for p in parts:
        inter += np.asarray(p["inter"], np.float64)
        union += np.asarray(p["union"], np.float64)
        per_img.extend(p["per_image_miou"])
        n_fwd += int(p["forward_windows"])
        wall = max(wall, float(p["wall_s"]))
    miou, ious = iou_from_iu(inter, union)
    return {
        "mIoU": miou,
        "per_class": {CLASS_NAMES[c]: ious[c] for c in range(NUM_CLASSES)},
        "per_class_en": {EN[c]: ious[c] for c in range(NUM_CLASSES)},
        "per_image_miou": per_img,
        "n": len(per_img),
        "forward_windows": int(n_fwd),
        "wall_s": round(wall, 2),
        "ms_per_img": round(1000 * wall / max(len(per_img), 1), 2),
    }


def _i2_worker(rank, items, weight_mode, edge_drop, rezoom, W, args_dict, out_path):
    set_seed(42)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    project = resolve_project_root(SCRIPT, args_dict["project_root"])
    net, row, ckpt = load_model(Path(args_dict["checkpoint"]), project, device)
    scales = CORE_SCALES
    crop, stride = args_dict["crop"], args_dict["stride"]
    W = None if W is None else np.asarray(W, np.float32)
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    per_img = []
    n_fwd = 0
    t0 = time.time()
    for j, (stem, ip, lp) in enumerate(items):
        img = imread_bgr(ip)
        mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
        h0, w0 = mask.shape
        scale_logits = []
        for s in scales:
            nh, nw = max(1, int(round(h0 * s))), max(1, int(round(w0 * s)))
            img_s = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            logits_s, n_win = sliding_window_logits(
                net, img_s, device, crop=crop, stride=stride, row_attn=row,
                weight_mode=weight_mode, edge_drop=edge_drop,
            )
            n_fwd += n_win
            logits_up = F.interpolate(
                logits_s.unsqueeze(0), size=(h0, w0), mode="bilinear", align_corners=False
            )[0].cpu().numpy().astype(np.float32)
            scale_logits.append(logits_up)
        fused = fuse_equal(scale_logits) if W is None else fuse_weighted(scale_logits, W)
        pred = fused.argmax(0).astype(np.uint8)
        if rezoom:
            for c in (1, 2):
                nlab, labels, stats, _ = cv2.connectedComponentsWithStats(
                    (pred == c).astype(np.uint8), 8
                )
                for i in range(1, nlab):
                    area = int(stats[i, cv2.CC_STAT_AREA])
                    if area == 0 or area > 64 * 64:
                        continue
                    x, y, ww, hh = (int(stats[i, k]) for k in range(4))
                    pad = 32
                    x0 = max(0, x - pad)
                    y0 = max(0, y - pad)
                    x1 = min(w0, x + ww + pad)
                    y1 = min(h0, y + hh + pad)
                    crop_img = img[y0:y1, x0:x1]
                    if crop_img.size == 0:
                        continue
                    ch, cw = crop_img.shape[:2]
                    img_r = cv2.resize(
                        crop_img,
                        (max(1, int(cw * 1.5)), max(1, int(ch * 1.5))),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    use_crop = crop
                    if min(img_r.shape[:2]) < crop:
                        use_crop = min(crop, img_r.shape[0], img_r.shape[1])
                        use_crop = max(use_crop, 32)
                    logits_r, n_win = sliding_window_logits(
                        net, img_r, device, crop=use_crop, stride=min(stride, use_crop),
                        row_attn=row, weight_mode="gaussian", edge_drop=0,
                    )
                    n_fwd += n_win
                    logits_back = F.interpolate(
                        logits_r.unsqueeze(0), size=(ch, cw), mode="bilinear", align_corners=False
                    )[0].cpu().numpy()
                    local = fused[:, y0:y1, x0:x1].copy()
                    local = 0.5 * local + 0.5 * logits_back
                    pred[y0:y1, x0:x1] = local.argmax(0).astype(np.uint8)
                    fused[:, y0:y1, x0:x1] = local
        update_inter_union(inter, union, pred, mask)
        _, mi = per_image_iou(pred, mask)
        per_img.append(mi)
        if (j + 1) % 25 == 0 or (j + 1) == len(items):
            print(f"[gpu{rank} {weight_mode} ed{edge_drop} rz{int(rezoom)}] {j+1}/{len(items)}", flush=True)
    payload = {
        "inter": inter.tolist(),
        "union": union.tolist(),
        "per_image_miou": per_img,
        "forward_windows": int(n_fwd),
        "wall_s": round(time.time() - t0, 2),
    }
    Path(out_path).write_text(json.dumps(payload), encoding="utf-8")


def predict_custom_multi(items, weight_mode, edge_drop, rezoom=False, W=None, args=None, tag=""):
    import torch.multiprocessing as mp
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_gpu = min(n_gpu, 2)
    if n_gpu <= 1 or len(items) < 8:
        # single process fallback using worker with rank 0
        tmp = Path(args.out_json).with_name(f"_part_{tag}_0.json")
        args_dict = {
            "project_root": args.project_root,
            "checkpoint": args.checkpoint,
            "crop": args.crop,
            "stride": args.stride,
        }
        _i2_worker(0, items, weight_mode, edge_drop, rezoom, None if W is None else W.tolist(), args_dict, str(tmp))
        part = json.loads(tmp.read_text(encoding="utf-8"))
        return _merge_partial([part])
    chunks = [items[i::n_gpu] for i in range(n_gpu)]
    args_dict = {
        "project_root": args.project_root,
        "checkpoint": args.checkpoint,
        "crop": args.crop,
        "stride": args.stride,
    }
    tmp_paths = [str(Path(args.out_json).with_name(f"_part_{tag}_{r}.json")) for r in range(n_gpu)]
    ctx = mp.get_context("spawn")
    procs = []
    Wl = None if W is None else np.asarray(W, np.float32).tolist()
    for r in range(n_gpu):
        pr = ctx.Process(
            target=_i2_worker,
            args=(r, chunks[r], weight_mode, edge_drop, rezoom, Wl, args_dict, tmp_paths[r]),
        )
        pr.start()
        procs.append(pr)
    for pr in procs:
        pr.join()
        if pr.exitcode != 0:
            raise RuntimeError(f"I2 worker failed exit={pr.exitcode}")
    parts = [json.loads(Path(tp).read_text(encoding="utf-8")) for tp in tmp_paths]
    return _merge_partial(parts)


@torch.no_grad()
def run_i2_i3_gpu(args, items_A, items_B, base_name, i1_W):
    """GPU protocols I2 and I3; select on A, score B once. Uses up to 2 GPUs."""
    set_seed(42)
    scales = CORE_SCALES

    def predict_custom(items, weight_mode, edge_drop, rezoom=False, W=None, tag="x"):
        return predict_custom_multi(items, weight_mode, edge_drop, rezoom, W, args, tag=tag)

    results = {}
    # I2a
    print("[I2a] gaussian on A", flush=True)
    results["I2a_gaussian_overlap"] = {
        "A": predict_custom(items_A, "gaussian", 0, False, i1_W, tag="I2aA"),
    }
    # I2b
    print("[I2b] cosine+edge_drop on A", flush=True)
    results["I2b_cosine_edge_drop"] = {
        "A": predict_custom(items_A, "cosine", 32, False, i1_W, tag="I2bA"),
    }
    # pick better I2 for I3 base
    i2_best = "I2a_gaussian_overlap"
    if results["I2b_cosine_edge_drop"]["A"]["mIoU"] > results["I2a_gaussian_overlap"]["A"]["mIoU"]:
        i2_best = "I2b_cosine_edge_drop"
    wm = "gaussian" if i2_best.startswith("I2a") else "cosine"
    ed = 0 if i2_best.startswith("I2a") else 32
    print(f"[I3] rezoom on A (base={i2_best})", flush=True)
    results["I3_small_rezoom"] = {
        "A": predict_custom(items_A, wm, ed, True, i1_W, tag="I3A"),
        "base": i2_best,
    }

    # B once only for A-selected winners (keep cost down)
    print("[I2/I3] select on A then B report-once", flush=True)
    # load C0 A mIoU from cpu full if present
    c0_A = None
    fullp = Path(args.out_json).with_name("protocols_cpu_full.json")
    if fullp.exists():
        import json as _json
        c0_A = _json.loads(fullp.read_text(encoding="utf-8"))["candidates"]["C0_equal_tta"]["A"]["mIoU"]
    scores = {
        "I2a_gaussian_overlap": results["I2a_gaussian_overlap"]["A"]["mIoU"],
        "I2b_cosine_edge_drop": results["I2b_cosine_edge_drop"]["A"]["mIoU"],
        "I3_small_rezoom": results["I3_small_rezoom"]["A"]["mIoU"],
    }
    # always B-report best I2 and I3 if I3 >= best I2 - 0.002; skip clearly worse
    best_i2 = max(scores["I2a_gaussian_overlap"], scores["I2b_cosine_edge_drop"])
    to_b = set()
    to_b.add(i2_best)
    if scores["I3_small_rezoom"] >= best_i2 - 0.002:
        to_b.add("I3_small_rezoom")
    if c0_A is not None:
        to_b = {k for k in to_b if scores[k] >= c0_A - 0.001} or {max(to_b, key=lambda k: scores[k])}
    print(f"[I2/I3] B candidates: {sorted(to_b)}", flush=True)
    if "I2a_gaussian_overlap" in to_b:
        results["I2a_gaussian_overlap"]["B"] = predict_custom(items_B, "gaussian", 0, False, i1_W, tag="I2aB")
    if "I2b_cosine_edge_drop" in to_b:
        results["I2b_cosine_edge_drop"]["B"] = predict_custom(items_B, "cosine", 32, False, i1_W, tag="I2bB")
    if "I3_small_rezoom" in to_b:
        results["I3_small_rezoom"]["B"] = predict_custom(items_B, wm, ed, True, i1_W, tag="I3B")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=r"/mnt/d/LuSNAR_Seg")
    ap.add_argument("--checkpoint", default=r"/mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/best.pth")
    ap.add_argument("--list-A", default=r"/mnt/d/LuSNAR_Seg/data/lists_portable/scene5_ab/scene5_A.txt")
    ap.add_argument("--list-B", default=r"/mnt/d/LuSNAR_Seg/data/lists_portable/scene5_ab/scene5_B.txt")
    ap.add_argument("--logits-dir", default=r"/root/lusnar_data/cache/scene5_scale_logits")
    ap.add_argument("--probs-dir", default=r"/root/lusnar_data/cache/scene5_tta_probs")
    ap.add_argument("--cache-json", default=r"/mnt/d/LuSNAR_Seg/results/inference_phase/cache_diag.json")
    ap.add_argument("--out-json", default=r"/mnt/d/LuSNAR_Seg/results/inference_phase/protocols.json")
    ap.add_argument("--stage", choices=["cpu", "gpu", "all"], default="cpu")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    items_A = load_stems(Path(args.list_A))
    items_B = load_stems(Path(args.list_B))
    if args.max_images:
        items_A = items_A[: args.max_images]
        items_B = items_B[: args.max_images]

    print("PREREGISTERED", PREREGISTERED, flush=True)
    out = {
        "preregistered": PREREGISTERED,
        "n_A": len(items_A),
        "n_B": len(items_B),
        "selection_rule": "maximize A mIoU; require rock/crater drop <=0.01 vs C0 on A; B reported once",
        "candidates": {},
    }

    if args.stage in ("cpu", "all"):
        # Need cache per-scale IoU for I1a
        cache = json.loads(Path(args.cache_json).read_text(encoding="utf-8"))
        # Rebuild per-scale IoU on A only from cached logits
        scales = list(CORE_SCALES)
        per_scale_iou_A = {s: np.zeros(NUM_CLASSES, np.float64) for s in scales}
        per_scale_inter = {s: np.zeros(NUM_CLASSES, np.float64) for s in scales}
        per_scale_union = {s: np.zeros(NUM_CLASSES, np.float64) for s in scales}
        for stem, ip, lp in items_A:
            zs = load_scale_logits(Path(args.logits_dir), stem, scales)
            if zs is None:
                continue
            mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
            for si, s in enumerate(scales):
                pred = zs[si].argmax(0).astype(np.uint8)
                update_inter_union(per_scale_inter[s], per_scale_union[s], pred, mask)
        for s in scales:
            _, iou = iou_from_iu(per_scale_inter[s], per_scale_union[s])
            per_scale_iou_A[s] = np.asarray(iou, np.float64)

        W_i1a, _ = pick_best_scale_weights(per_scale_iou_A)
        print("[I1a] weights", W_i1a.tolist(), flush=True)
        print("[I1b] searching soft weights on A...", flush=True)
        W_i1b, i1b_A_quick = search_soft_weights(Path(args.logits_dir), items_A, scales)

        c0_A = eval_fusion_on(items_A, Path(args.logits_dir), W=None)
        i1a_A = eval_fusion_on(items_A, Path(args.logits_dir), W=W_i1a)
        i1b_A = eval_fusion_on(items_A, Path(args.logits_dir), W=W_i1b)

        # choose best I1 for later stacking
        i1_best_name = "I1a_best_scale_per_class"
        i1_best_W = W_i1a
        if i1b_A["mIoU"] > i1a_A["mIoU"]:
            i1_best_name = "I1b_soft_scale_weights"
            i1_best_W = W_i1b

        # Also build I1-fused probs for postproc on the fly from logits when scoring P1
        # For P1 we search on equal-fused saved probs (C0), as planned.
        print("[P1] searching logit bias on A...", flush=True)
        bias, _ = search_logit_bias(Path(args.probs_dir), items_A, step=0.1, span=0.8)
        p1_A = eval_probs_protocol(items_A, Path(args.probs_dir), bias=bias, use_geo=False)
        p1p2_A = eval_probs_protocol(items_A, Path(args.probs_dir), bias=bias, use_geo=True)
        p2_only_A = eval_probs_protocol(items_A, Path(args.probs_dir), bias=None, use_geo=True)

        # B once for CPU candidates
        c0_B = eval_fusion_on(items_B, Path(args.logits_dir), W=None)
        i1a_B = eval_fusion_on(items_B, Path(args.logits_dir), W=W_i1a)
        i1b_B = eval_fusion_on(items_B, Path(args.logits_dir), W=W_i1b)
        p1_B = eval_probs_protocol(items_B, Path(args.probs_dir), bias=bias, use_geo=False)
        p1p2_B = eval_probs_protocol(items_B, Path(args.probs_dir), bias=bias, use_geo=True)

        out["candidates"]["C0_equal_tta"] = {"A": c0_A, "B": c0_B}
        out["candidates"]["I1a_best_scale_per_class"] = {
            "A": i1a_A,
            "B": i1a_B,
            "weights": W_i1a.tolist(),
            "scales": list(scales),
            "per_scale_iou_A": {f"{s:.2f}": per_scale_iou_A[s].tolist() for s in scales},
        }
        out["candidates"]["I1b_soft_scale_weights"] = {
            "A": i1b_A,
            "B": i1b_B,
            "weights": W_i1b.tolist(),
            "scales": list(scales),
        }
        out["candidates"]["P1_logit_bias"] = {
            "A": p1_A,
            "B": p1_B,
            "bias": bias.tolist(),
        }
        out["candidates"]["P1P2_bias_plus_geo"] = {"A": p1p2_A, "B": p1p2_B, "bias": bias.tolist()}
        out["candidates"]["P2_geo_only_ref"] = {"A": p2_only_A}  # reference, not counting toward 8
        out["i1_best_for_stack"] = i1_best_name
        out["i1_best_W"] = i1_best_W.tolist()

        # paired tests vs C0 on A
        tests = {}
        for name in ["I1a_best_scale_per_class", "I1b_soft_scale_weights", "P1_logit_bias", "P1P2_bias_plus_geo"]:
            tests[name] = paired_tests(
                c0_A["per_image_miou"], out["candidates"][name]["A"]["per_image_miou"]
            )
        pvals = [tests[k]["wilcoxon_p"] or 1.0 for k in tests]
        holm = holm_correct(pvals)
        for i, k in enumerate(tests):
            tests[k]["holm_p"] = holm[i]
        out["paired_tests_vs_C0_on_A"] = tests

        # strip bulky per-image arrays from saved file after tests
        for cand in out["candidates"].values():
            for split in list(cand.keys()):
                if isinstance(cand[split], dict):
                    cand[split].pop("per_image_cls", None)
                    # keep per_image_miou? large; store summary only in final, keep in tests
                    # Actually keep for GPU stage comparison file separately
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        # save a slim + a full
        full_path = Path(args.out_json).with_name("protocols_cpu_full.json")
        full_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        slim = json.loads(json.dumps(out))
        for cand in slim["candidates"].values():
            for split, val in list(cand.items()):
                if isinstance(val, dict) and "per_image_miou" in val:
                    val["per_image_miou_mean"] = float(np.nanmean(val["per_image_miou"]))
                    val.pop("per_image_miou", None)
        Path(args.out_json).write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
        print("CPU_PROTOCOLS_DONE", args.out_json, flush=True)

        # save W for GPU stage
        np.save(Path(args.out_json).with_name("i1_best_W.npy"), np.asarray(i1_best_W, np.float32))

    if args.stage in ("gpu", "all"):
        # load i1 W
        wpath = Path(args.out_json).with_name("i1_best_W.npy")
        if wpath.exists():
            i1_W = np.load(wpath)
        else:
            i1_W = None
        gpu_res = run_i2_i3_gpu(args, items_A, items_B, "stack", i1_W)
        # attach tests
        if args.stage == "all" and "C0_equal_tta" in out.get("candidates", {}):
            base = out["candidates"]["C0_equal_tta"]["A"]["per_image_miou"]
        else:
            # reload
            full = json.loads(Path(args.out_json).with_name("protocols_cpu_full.json").read_text(encoding="utf-8"))
            base = full["candidates"]["C0_equal_tta"]["A"]["per_image_miou"]
            out = full
        for name, res in gpu_res.items():
            out["candidates"][name] = res
            out["paired_tests_vs_C0_on_A"][name] = paired_tests(base, res["A"]["per_image_miou"])
        # holm over all with p values
        keys = list(out["paired_tests_vs_C0_on_A"].keys())
        pvals = [out["paired_tests_vs_C0_on_A"][k]["wilcoxon_p"] or 1.0 for k in keys]
        holm = holm_correct(pvals)
        for i, k in enumerate(keys):
            out["paired_tests_vs_C0_on_A"][k]["holm_p"] = holm[i]

        # decide freeze
        def rock_crater_ok(cand_A, base_A):
            r0 = base_A["per_class_en"]["rock"]
            c0 = base_A["per_class_en"]["crater"]
            r1 = cand_A["per_class_en"]["rock"]
            c1 = cand_A["per_class_en"]["crater"]
            return (r1 >= r0 - 0.01) and (c1 >= c0 - 0.01)

        base_A = out["candidates"]["C0_equal_tta"]["A"]
        ranked = []
        for name, cand in out["candidates"].items():
            if name.endswith("_ref") or "A" not in cand:
                continue
            if not rock_crater_ok(cand["A"], base_A):
                continue
            ranked.append((cand["A"]["mIoU"], name))
        ranked.sort(reverse=True)
        winner = ranked[0][1] if ranked else "C0_equal_tta"
        out["frozen_recipe"] = {
            "name": winner,
            "A_mIoU": out["candidates"][winner]["A"]["mIoU"],
            "B_mIoU": out["candidates"][winner].get("B", {}).get("mIoU"),
            "B_per_class": out["candidates"][winner].get("B", {}).get("per_class"),
            "note": "Selected on A only; B numbers are one-shot.",
        }
        # slim save
        full_path = Path(args.out_json).with_name("protocols_full.json")
        full_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        slim = json.loads(json.dumps(out))
        for cand in slim["candidates"].values():
            for split, val in list(cand.items()):
                if isinstance(val, dict) and "per_image_miou" in val:
                    val["per_image_miou_mean"] = float(np.nanmean(val["per_image_miou"]))
                    val.pop("per_image_miou", None)
        Path(args.out_json).write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
        print("PROTOCOLS_DONE", args.out_json, "winner", winner, flush=True)


if __name__ == "__main__":
    main()
