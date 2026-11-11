#!/usr/bin/env python3
"""Scene5 I1a score for the four-row small-crater baseline.

Uses the scale choice frozen on Scene5 half A. Does not read Moon_2/4/9.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
from _lusnar_common import (  # noqa: E402
    CLASS_NAMES,
    COLOR_TO_ID,
    NUM_CLASSES,
    RowHeightAttn,
    build_segformer_b2,
    find_pretrain,
    resolve_project_root,
    scan_scene_pairs,
    sliding_window_logits,
)
from lusnar_oneshot_opensource import imread_bgr, load_gt_official  # noqa: E402

SCALES = (0.75, 1.0, 1.25)
# Frozen on Scene5 A: regolith 1.0, rock 1.25, crater 0.75, mountain 0.75, sky 1.0.
I1A_W = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    np.float32,
)


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
        logits = sliding_window_logits(
            net, img_s, device, crop=512, stride=256, row_attn=row_attn, residual=residual
        )
        logits = torch.nn.functional.interpolate(
            logits.unsqueeze(0), size=(h0, w0), mode="bilinear", align_corners=False
        )[0]
        logits_list.append(logits)
    return fuse_i1a(logits_list).argmax(0).cpu().numpy().astype(np.uint8)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--project-root", default=r"D:\LuSNAR_Seg")
    args = ap.parse_args()
    project = resolve_project_root(SCRIPT, args.project_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = build_segformer_b2(find_pretrain(project, None)).to(device)
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
    net.eval()
    row.eval()
    pairs = scan_scene_pairs(project / "data" / "train", 5)
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    crater_as_reg = 0.0
    crater_gt = 0.0
    band_inter = 0.0
    band_union = 0.0
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
        band_inter += np.logical_and(band, pred == 2, crater).sum()
        band_union += np.logical_or(np.logical_and(band, pred == 2), np.logical_and(band, crater)).sum()
        if i % 50 == 0 or i == len(pairs):
            print(f"[E0] {args.tag} {i}/{len(pairs)}", flush=True)
    iou = [float(inter[c] / union[c]) if union[c] else float("nan") for c in range(NUM_CLASSES)]
    miou = float(np.nanmean(iou))
    out = {
        "tag": args.tag,
        "checkpoint": args.checkpoint,
        "protocol": "I1a",
        "n_images": len(pairs),
        "mIoU": miou,
        "crater_IoU": iou[2],
        "per_class_IoU": {n: iou[i] for i, n in enumerate(CLASS_NAMES)},
        "crater_to_regolith": float(crater_as_reg / crater_gt) if crater_gt else None,
        "crater_boundary_IoU": float(band_inter / band_union) if band_union else None,
    }
    dest = project / "results" / "small_crater_rows" / f"{args.tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ROW_DONE", args.tag, {k: out[k] for k in ("mIoU", "crater_IoU", "crater_to_regolith", "crater_boundary_IoU")}, flush=True)


if __name__ == "__main__":
    main()
