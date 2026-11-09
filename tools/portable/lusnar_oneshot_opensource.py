#!/usr/bin/env python3
"""One-shot score of Moon_2/4/9 against open-source GT.

Inference reads color only. Scoring remaps open-source RGB labels onto the
official order (rock/crater are swapped in the open palette). The score is
not fed back into training, checkpoint choice, or hyperparameters.
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
    NUM_CLASSES,
    RowHeightAttn,
    build_segformer_b2,
    ensure_mmcv_stub,
    find_pretrain,
    resolve_project_root,
    sliding_window_logits,
)

ensure_mmcv_stub(SCRIPT)

SCENES = (2, 4, 9)
EN_NAMES = ["regolith", "rock", "crater", "mountain", "sky"]
SCALES = (0.75, 1.0, 1.25)


def imread_bgr(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    return arr[:, :, ::-1].copy()


def load_gt_official(path: Path) -> np.ndarray:
    im = Image.open(path)
    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for color, cid in COLOR_TO_ID.items():
        mask = np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)
        out[mask] = cid
    return out


@torch.no_grad()
def predict_tta(net, img_bgr: np.ndarray, device, row_attn) -> np.ndarray:
    h0, w0 = img_bgr.shape[:2]
    acc = torch.zeros(NUM_CLASSES, h0, w0, device=device, dtype=torch.float32)
    for scale in SCALES:
        nh = max(1, int(round(h0 * scale)))
        nw = max(1, int(round(w0 * scale)))
        img_s = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        logits = sliding_window_logits(
            net, img_s, device, crop=512, stride=256, row_attn=row_attn
        )
        logits = F.interpolate(
            logits.unsqueeze(0), size=(h0, w0), mode="bilinear", align_corners=False
        )[0]
        acc += logits
    return acc.argmax(0).cpu().numpy().astype(np.uint8)


def load_model(checkpoint: Path, project: Path, device):
    pretrain = find_pretrain(project)
    core = build_segformer_b2(pretrain).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    core.load_state_dict(ckpt["model"])
    core.eval()
    row = None
    if ckpt.get("row_attn"):
        row = RowHeightAttn().to(device)
        row.load_state_dict(ckpt["row_attn"])
        row.eval()
    return core, row, ckpt


def iou_from_hist(hist: np.ndarray):
    ious = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
    for c in range(NUM_CLASSES):
        tp = hist[c, c]
        fp = hist[:, c].sum() - tp
        fn = hist[c, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            ious[c] = tp / denom
    return ious, float(np.nanmean(ious))


def write_report(path: Path, summary: dict):
    def fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{x:.4f}"

    lines = [
        f"# {summary['tag']} 开源测试集一次性评分（非官方）",
        "",
        "> 非官方事后评分。Moon_2/4/9 只在预测写完之后用来算分，不参与训练、调参或选模。",
        "",
        "## 设置",
        "",
        f"- 权重：`{summary['checkpoint']}`（iter {summary.get('ckpt_iter')}）",
        f"- 推理：ROW {'开' if summary['row_attn'] else '关'}，TTA 尺度 {summary['scales']}，滑窗 512/256",
        "- 预测：官方类别索引 0 月壤 / 1 岩石 / 2 撞击坑 / 3 山体 / 4 天空",
        "- GT：`data/Moon_{2,4,9}/image0/label`，按 RGB→官方 COLOR_TO_ID 对齐（开源调色板岩石/撞击坑对调）",
        f"- 图数：{summary['n_images']}；未映射 GT 像素：{summary['unmapped_gt_pixels']}；缺失：{summary['n_missing']}",
        "",
        "## 分场景",
        "",
        "| 场景 | 图数 | mIoU | 月壤 | 岩石 | 撞击坑 | 山体 | 天空 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scene, stats in summary["per_scene"].items():
        pc = stats["per_class_IoU"]
        lines.append(
            f"| {scene} | {stats['n_images']} | {fmt(stats['mIoU'])} | "
            + " | ".join(fmt(pc[name]) for name in EN_NAMES)
            + " |"
        )
    overall = summary["overall"]
    lines += [
        "",
        "## 总体",
        "",
        f"- **Overall mIoU（全像素微平均）**: **{fmt(overall['mIoU'])}**",
        f"- 场景 mIoU 宏平均: {fmt(summary['macro_scene_mIoU'])}",
        f"- 按图数加权: {fmt(summary['weighted_scene_mIoU'])}",
        "",
        "| 类别 | IoU | GT 像素数 |",
        "|---|---:|---:|",
    ]
    for i, name in enumerate(EN_NAMES):
        lines.append(
            f"| {i} {CLASS_NAMES[i]} ({name}) | {fmt(overall['per_class_IoU'][name])} | {overall['gt_pixels'][name]} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    project = resolve_project_root(SCRIPT, args.project_root)
    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    out_dir = project / "results" / f"oneshot_{args.tag}"
    pred_root = out_dir / "preds"
    pred_root.mkdir(parents=True, exist_ok=True)
    data = project / "data"

    print({"phase": "infer", "checkpoint": str(ckpt_path), "tag": args.tag}, flush=True)
    net, row, ckpt = load_model(ckpt_path, project, device)
    t0 = time.time()
    n_pred = 0
    for sid in SCENES:
        color_dir = data / "test" / f"Moon_{sid}" / "color"
        images = sorted(color_dir.glob("*.png"))
        if args.limit:
            images = images[: args.limit]
        scene_out = pred_root / f"Moon_{sid}"
        scene_out.mkdir(parents=True, exist_ok=True)
        for i, ip in enumerate(images, 1):
            pred = predict_tta(net, imread_bgr(ip), device, row)
            Image.fromarray(pred, mode="L").save(scene_out / ip.name)
            n_pred += 1
            if i % 20 == 0 or i == len(images):
                print(f"[infer] Moon_{sid} {i}/{len(images)}", flush=True)
    infer_sec = time.time() - t0

    print({"phase": "score", "n_pred": n_pred}, flush=True)
    overall = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_scene = {}
    missing = []
    unmapped = 0
    for sid in SCENES:
        scene = f"Moon_{sid}"
        hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        preds = sorted((pred_root / scene).glob("*.png"))
        n = 0
        for pp in preds:
            gp = data / scene / "image0" / "label" / pp.name
            if not gp.exists():
                missing.append(f"{scene}/{pp.name}")
                continue
            pred = np.asarray(Image.open(pp))
            gt = load_gt_official(gp)
            if pred.shape != gt.shape:
                raise RuntimeError(f"shape mismatch {pp}")
            valid = gt < NUM_CLASSES
            unmapped += int((~valid).sum())
            g = gt[valid].astype(np.int64)
            p = np.clip(pred[valid].astype(np.int64), 0, NUM_CLASSES - 1)
            np.add.at(hist, (g, p), 1)
            n += 1
        overall += hist
        ious, miou = iou_from_hist(hist)
        per_scene[scene] = {
            "n_images": n,
            "mIoU": miou,
            "per_class_IoU": {
                EN_NAMES[i]: (None if np.isnan(ious[i]) else float(ious[i]))
                for i in range(NUM_CLASSES)
            },
        }
        print(f"[score] {scene} n={n} mIoU={miou:.4f}", flush=True)

    ious_o, miou_o = iou_from_hist(overall)
    scene_mious = [per_scene[f"Moon_{sid}"]["mIoU"] for sid in SCENES]
    weights = [per_scene[f"Moon_{sid}"]["n_images"] for sid in SCENES]
    summary = {
        "disclaimer": "非官方一次性评分。Moon_2/4/9 未用于训练、调参或选模。",
        "tag": args.tag,
        "checkpoint": str(ckpt_path),
        "ckpt_iter": ckpt.get("iter"),
        "row_attn": row is not None,
        "scales": list(SCALES),
        "crop": 512,
        "stride": 256,
        "n_images": int(sum(weights)),
        "n_missing": len(missing),
        "unmapped_gt_pixels": unmapped,
        "infer_sec": round(infer_sec, 1),
        "per_scene": per_scene,
        "overall": {
            "mIoU": miou_o,
            "per_class_IoU": {
                EN_NAMES[i]: (None if np.isnan(ious_o[i]) else float(ious_o[i]))
                for i in range(NUM_CLASSES)
            },
            "gt_pixels": {EN_NAMES[i]: int(overall[i].sum()) for i in range(NUM_CLASSES)},
        },
        "macro_scene_mIoU": float(np.mean(scene_mious)),
        "weighted_scene_mIoU": float(np.average(scene_mious, weights=weights)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = project / "results" / f"oneshot_{args.tag}.md"
    write_report(report, summary)
    print("ONESHOT_DONE", report, "mIoU", round(miou_o, 4), flush=True)


if __name__ == "__main__":
    main()
