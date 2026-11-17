#!/usr/bin/env python3
"""[M2F COPY 2026-11-16: only change vs lusnar_small_crater_score_sc_b5.py = HF Mask2Former
Swin-B backbone via build_m2f() (m2f_wrapper, ADE20k-semantic init, 5-class head).
Scoring protocol / CLI / metrics byte-identical. Run with env lusnar_bb (needs transformers).
Original docstring: [SC copy: adds detail_residual (stage1+2) loading; scoring identical to lusnar_small_crater_e0_wsl.py]
WSL copy of lusnar_small_crater_e0.py (Scene5 I1a score, E0 row).

Changes vs original (core prediction/fusion identical):
  * --data-root / --out-dir / --max-images CLI args (paths only).
  * Fix boundary-band intersection: original called np.logical_and(band, pred==2, crater),
    whose 3rd positional arg is `out=`, so the intersection ignored the GT crater mask.
  * Extra annotation metrics (not gate): crater instance recall by area / short-side bucket.
Does not read Moon_2/4/9.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
sys.path.insert(0, str(SCRIPT.parent.parent / "portable_m2f"))  # m2f_wrapper lives there
from m2f_wrapper import build_m2f  # noqa: E402
from _lusnar_common import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    RowHeightAttn,
    build_segformer,
    build_segformer_b2,
    find_pretrain,
    resolve_project_root,
    scan_scene_pairs,
    sliding_window_logits,
)
from lusnar_oneshot_opensource import imread_bgr, load_gt_official  # noqa: E402

SCALES = (0.75, 1.0, 1.25)
I1A_W = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    np.float32,
)
AREA_BUCKETS = [(0, 16), (16, 64), (64, 256), (256, 1024), (1024, 4096), (4096, 1 << 62)]
SIDE_BUCKETS = [(0, 8), (8, 16), (16, 32), (32, 1 << 30)]


def fuse_i1a(logits_list):
    acc = torch.zeros_like(logits_list[0])
    for c in range(NUM_CLASSES):
        for s, logits in enumerate(logits_list):
            acc[c] += I1A_W[c, s] * logits[c]
    return acc


def boundary_band(crater: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    er = cv2.erode(crater, kernel)
    edge = (crater > 0) & (er == 0)
    return cv2.dilate(edge.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0


@torch.no_grad()
def predict(net, img_bgr, device, row_attn, residual=None):
    h0, w0 = img_bgr.shape[:2]
    logits_list = []
    for scale in SCALES:
        nh = max(1, int(round(h0 * scale)))
        nw = max(1, int(round(w0 * scale)))
        img_s = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        logits = sliding_window_logits(net, img_s, device, crop=512, stride=256, row_attn=row_attn, residual=residual)
        logits = torch.nn.functional.interpolate(logits.unsqueeze(0), size=(h0, w0), mode="bilinear", align_corners=False)[0]
        logits_list.append(logits)
    return fuse_i1a(logits_list).argmax(0).cpu().numpy().astype(np.uint8)


def new_bucket_stats(buckets):
    return [dict(lo=lo, hi=hi, n=0, hit50=0, hit_any=0, px=0, px_tp=0, px_as_reg=0) for lo, hi in buckets]


def bucket_of(v, buckets):
    for i, (lo, hi) in enumerate(buckets):
        if lo <= v < hi:
            return i
    return len(buckets) - 1


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--project-root", default=r"D:\LuSNAR_Seg")
    ap.add_argument("--data-root", default=None, help="dir containing Moon_5/color,label (default <project>/data/train)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()
    project = resolve_project_root(SCRIPT, args.project_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = build_m2f(find_pretrain(project, "pretrained/hf/mask2former-swin-base-IN21k-ade-semantic"), 5).to(device)
    row = RowHeightAttn().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model"])
    row.load_state_dict(ckpt["row_attn"])
    residual = None
    if ckpt.get("plain_residual"):
        from lusnar_row_blv_ddp import PlainStage1Residual

        residual = PlainStage1Residual().to(device)
        residual.load_state_dict(ckpt["plain_residual"])
        residual.eval()
    elif ckpt.get("detail_residual"):
        from lusnar_row_blv_ddp_sc import DetailResidual12

        sg = bool(ckpt.get("shading_gate"))
        residual = DetailResidual12(
            stages=ckpt.get("detail_residual_stages") or "12",
            shading_gate=sg,
            sg_orients=int(ckpt.get("sg_orients") or 8),
            sg_dists=tuple(ckpt.get("sg_dists") or (2, 4, 8)),
            sg_source=ckpt.get("sg_source") or "feat",
        ).to(device)
        residual.load_state_dict(ckpt["detail_residual"])
        residual.attach(net)
        residual.eval()
        print("[score] detail_residual alpha=", float(residual.alpha), "shading_gate=", sg, "sg_source=", getattr(residual, "sg_source", None), flush=True)
    net.eval()
    row.eval()
    data_root = Path(args.data_root) if args.data_root else project / "data" / "train"
    pairs = scan_scene_pairs(data_root, 5)
    if args.max_images:
        pairs = pairs[: args.max_images]
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    crater_as_reg = crater_gt = band_inter = band_union = 0.0
    area_st = new_bucket_stats(AREA_BUCKETS)
    side_st = new_bucket_stats(SIDE_BUCKETS)
    t0 = time.time()
    for i, (cp, lp) in enumerate(pairs, 1):
        pred = predict(net, imread_bgr(Path(cp)), device, row, residual)
        gt = load_gt_official(Path(lp))
        valid = gt < NUM_CLASSES
        for c in range(NUM_CLASSES):
            p = (pred == c) & valid
            g = gt == c
            inter[c] += np.logical_and(p, g).sum()
            union[c] += np.logical_or(p, g).sum()
        crater = gt == 2
        crater_gt += crater.sum()
        crater_as_reg += np.logical_and(crater, pred == 0).sum()
        band = boundary_band(crater.astype(np.uint8)) & valid
        pc = pred == 2
        band_inter += (band & pc & crater).sum()
        band_union += ((band & pc) | (band & crater)).sum()
        # instance buckets
        n, lab, stats, _ = cv2.connectedComponentsWithStats(crater.astype(np.uint8), connectivity=8)
        if n > 1:
            tp_cnt = np.bincount(lab[pc & crater].ravel(), minlength=n)
            reg_cnt = np.bincount(lab[crater & (pred == 0)].ravel(), minlength=n)
            for k in range(1, n):
                area = int(stats[k, cv2.CC_STAT_AREA])
                side = int(min(stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]))
                tp = int(tp_cnt[k])
                for st, b in ((area_st, bucket_of(area, AREA_BUCKETS)), (side_st, bucket_of(side, SIDE_BUCKETS))):
                    s = st[b]
                    s["n"] += 1
                    s["hit50"] += int(tp * 2 >= area)
                    s["hit_any"] += int(tp > 0)
                    s["px"] += area
                    s["px_tp"] += tp
                    s["px_as_reg"] += int(reg_cnt[k])
        if i % 50 == 0 or i == len(pairs):
            el = time.time() - t0
            print(f"[E0] {args.tag} {i}/{len(pairs)} {el:.0f}s eta {el / i * (len(pairs) - i):.0f}s", flush=True)

    def fin(st):
        out = []
        for s in st:
            d = dict(s)
            d["inst_recall50"] = s["hit50"] / s["n"] if s["n"] else None
            d["inst_recall_any"] = s["hit_any"] / s["n"] if s["n"] else None
            d["pixel_recall"] = s["px_tp"] / s["px"] if s["px"] else None
            d["pixel_as_regolith"] = s["px_as_reg"] / s["px"] if s["px"] else None
            out.append(d)
        return out

    iou = [float(inter[c] / union[c]) if union[c] else float("nan") for c in range(NUM_CLASSES)]
    out = {
        "tag": args.tag,
        "checkpoint": args.checkpoint,
        "protocol": "I1a",
        "n_images": len(pairs),
        "mIoU": float(np.nanmean(iou)),
        "crater_IoU": iou[2],
        "per_class_IoU": {n: iou[i] for i, n in enumerate(CLASS_NAMES)},
        "crater_to_regolith": float(crater_as_reg / crater_gt) if crater_gt else None,
        "crater_boundary_IoU": float(band_inter / band_union) if band_union else None,
        "crater_instances_by_area": fin(area_st),
        "crater_instances_by_short_side": fin(side_st),
        "runtime_s": time.time() - t0,
    }
    out_dir = Path(args.out_dir) if args.out_dir else project / "results" / "small_crater_E0"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{args.tag}.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ROW_DONE", args.tag, {k: out[k] for k in ("mIoU", "crater_IoU", "crater_to_regolith", "crater_boundary_IoU")}, flush=True)


if __name__ == "__main__":
    main()
