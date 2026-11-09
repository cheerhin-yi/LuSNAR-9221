#!/usr/bin/env python3
"""Attribute saved predictions to semantic-segmentation error types.

Reads predictions that already exist. Does not train, score a checkpoint
for selection, or change hyperparameters. Gates are fixed before the
open-set scores of the current runs are known.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
from _lusnar_common import CLASS_NAMES, COLOR_TO_ID, NUM_CLASSES  # noqa: E402

EN = ["regolith", "rock", "crater", "mountain", "sky"]
SCENES = (2, 4, 9)
SIZE_BINS = ("small", "mid", "large")
ROW_BINS = 4
GATES = {
    "majority_swallow": 0.50,
    "texture_confusion": 0.25,
    "boundary": 0.12,
    "scale": 0.15,
    "position_context": 0.15,
    "illumination": 15.0,
    "domain_shift": 0.12,
}
ZH = {
    "majority_swallow": "多数类吞并",
    "texture_confusion": "类间外观混淆",
    "boundary": "边界",
    "scale": "小目标与尺度",
    "position_context": "位置与上下文",
    "illumination": "光照",
    "domain_shift": "场景域偏移",
}
TEXTURE_PAIRS = {(1, 3), (3, 1), (4, 3), (3, 4), (2, 1), (1, 2)}
MIN_BIN_PIXELS = 50_000
MIN_ROW_FRACTION = 0.01
BAND_K = 17  # diameter; radius 8 px


def load_gt(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for color, cid in COLOR_TO_ID.items():
        out[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = cid
    return out


def load_pred(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def luminance(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def size_bin(area: int) -> int:
    if area < 32 * 32:
        return 0
    if area < 96 * 96:
        return 1
    return 2


def new_acc():
    return {
        "hist": np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64),
        "scene": {s: np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64) for s in SCENES},
        "band": np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64),
        "interior": np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64),
        "row": np.zeros((ROW_BINS, NUM_CLASSES, NUM_CLASSES), np.int64),
        "size_gt": np.zeros((NUM_CLASSES, 3), np.int64),
        "size_hit": np.zeros((NUM_CLASSES, 3), np.int64),
        "y_ok": np.zeros(NUM_CLASSES, np.float64),
        "y_bad": np.zeros(NUM_CLASSES, np.float64),
        "n_ok": np.zeros(NUM_CLASSES, np.int64),
        "n_bad": np.zeros(NUM_CLASSES, np.int64),
        "n_images": 0,
        "n_missing_pred": 0,
    }


def add_hist(hist, gt, pred, mask):
    g = gt[mask].astype(np.int64)
    p = pred[mask].astype(np.int64)
    keep = (g < NUM_CLASSES) & (p < NUM_CLASSES)
    if keep.any():
        np.add.at(hist, (g[keep], p[keep]), 1)


def gt_geometry(gt: np.ndarray):
    valid = gt < NUM_CLASSES
    edge = np.zeros(gt.shape, np.uint8)
    edge[1:] |= gt[1:] != gt[:-1]
    edge[:, 1:] |= gt[:, 1:] != gt[:, :-1]
    edge &= valid.astype(np.uint8)
    band = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BAND_K, BAND_K))) > 0
    band &= valid
    interior = valid & ~band
    h = gt.shape[0]
    rows = (np.arange(h) * ROW_BINS // max(h, 1))[:, None]
    comps = []
    for c in (1, 2, 3):
        nlab, labels, stats, _ = cv2.connectedComponentsWithStats((gt == c).astype(np.uint8), 8)
        for i in range(1, nlab):
            area = int(stats[i, cv2.CC_STAT_AREA])
            x, y, w, hh = (int(stats[i, k]) for k in range(4))
            crop = labels[y : y + hh, x : x + w] == i
            comps.append((c, size_bin(area), y, x, crop, area))
    return valid, band, interior, rows, comps


def accumulate(acc, gt, pred, yimg, scene: int, geo):
    valid, band, interior, rows, comps = geo
    pred = np.where(pred < NUM_CLASSES, pred, 0)
    add_hist(acc["hist"], gt, pred, valid)
    add_hist(acc["scene"][scene], gt, pred, valid)
    add_hist(acc["band"], gt, pred, band)
    add_hist(acc["interior"], gt, pred, interior)
    for b in range(ROW_BINS):
        add_hist(acc["row"][b], gt, pred, valid & (rows == b))
    for c, binv, y0, x0, crop, area in comps:
        acc["size_gt"][c, binv] += area
        acc["size_hit"][c, binv] += int(
            (pred[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]][crop] == c).sum()
        )
    for c in range(NUM_CLASSES):
        m = gt == c
        ok = m & (pred == c)
        bad = m & (pred != c)
        acc["y_ok"][c] += float(yimg[ok].sum())
        acc["y_bad"][c] += float(yimg[bad].sum())
        acc["n_ok"][c] += int(ok.sum())
        acc["n_bad"][c] += int(bad.sum())
    acc["n_images"] += 1


def iou_of(hist):
    ious = []
    for c in range(NUM_CLASSES):
        tp = float(hist[c, c])
        denom = float(hist[c, :].sum() + hist[:, c].sum() - tp)
        ious.append(tp / denom if denom > 0 else float("nan"))
    return ious


def recall_of(hist):
    out = []
    for c in range(NUM_CLASSES):
        support = float(hist[c, :].sum())
        out.append(float(hist[c, c]) / support if support > 0 else float("nan"))
    return out


def jnum(x):
    if isinstance(x, (np.floating, float)):
        x = float(x)
        if x != x:
            return None
        return round(x, 6)
    if isinstance(x, np.integer):
        return int(x)
    return x


def class_report(acc) -> list[dict]:
    hist = acc["hist"]
    band_iou = iou_of(acc["band"])
    in_iou = iou_of(acc["interior"])
    scene_iou = {s: iou_of(acc["scene"][s]) for s in SCENES}
    row_rec = [recall_of(acc["row"][b]) for b in range(ROW_BINS)]
    ious = iou_of(hist)
    rows = []
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
        small_gt = int(acc["size_gt"][c, 0])
        large_gt = int(acc["size_gt"][c, 2])
        small_r = float(acc["size_hit"][c, 0] / small_gt) if small_gt else float("nan")
        large_r = float(acc["size_hit"][c, 2] / large_gt) if large_gt else float("nan")
        row_support = [int(acc["row"][b, c, :].sum()) for b in range(ROW_BINS)]
        usable = [
            b
            for b in range(ROW_BINS)
            if support > 0 and row_support[b] >= MIN_ROW_FRACTION * support
        ]
        recs = [row_rec[b][c] for b in usable]
        row_span = (max(recs) - min(recs)) if len(recs) >= 2 else float("nan")
        y_ok = acc["y_ok"][c] / acc["n_ok"][c] if acc["n_ok"][c] else float("nan")
        y_bad = acc["y_bad"][c] / acc["n_bad"][c] if acc["n_bad"][c] else float("nan")
        dy = (y_bad - y_ok) if y_ok == y_ok and y_bad == y_bad else float("nan")
        scene_vals = [scene_iou[s][c] for s in SCENES]
        scene_span = (
            max(scene_vals) - min(scene_vals) if all(v == v for v in scene_vals) else float("nan")
        )
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
        if dy == dy and abs(dy) >= GATES["illumination"] and acc["n_bad"][c] >= MIN_BIN_PIXELS:
            side = "更暗" if dy < 0 else "更亮"
            flags.append(
                ("illumination", abs(dy) / 100.0, f"判错像素比判对像素{side} {abs(dy):.1f} 灰度")
            )
        if scene_span == scene_span and scene_span >= GATES["domain_shift"]:
            flags.append(("domain_shift", scene_span, f"三场景 IoU 极差 {scene_span:.3f}"))
        flags.sort(key=lambda t: -t[1])
        rows.append(
            {
                "class": EN[c],
                "name": CLASS_NAMES[c],
                "iou": jnum(ious[c]),
                "support": support,
                "fn": fn,
                "fn_to": [
                    {"pred": EN[p], "name": CLASS_NAMES[p], "pixels": n, "share_of_fn": jnum(s)}
                    for p, n, s in dest[:3]
                ],
                "boundary_iou": jnum(band_iou[c]),
                "interior_iou": jnum(in_iou[c]),
                "size_recall": {
                    SIZE_BINS[b]: {
                        "gt_pixels": int(acc["size_gt"][c, b]),
                        "recall": jnum(
                            float(acc["size_hit"][c, b] / acc["size_gt"][c, b])
                            if acc["size_gt"][c, b]
                            else float("nan")
                        ),
                    }
                    for b in range(3)
                },
                "row_recall": [jnum(row_rec[b][c]) for b in range(ROW_BINS)],
                "delta_y_error_minus_correct": jnum(dy),
                "scene_iou": {f"Moon_{s}": jnum(scene_iou[s][c]) for s in SCENES},
                "flags": [
                    {"type": name, "zh": ZH[name], "score": jnum(score), "evidence": text}
                    for name, score, text in flags
                ],
                "primary": flags[0][0] if flags else None,
            }
        )
    return rows


def val_curve(jsonl: Path):
    if not jsonl.exists():
        return []
    rows = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if "val_mIoU_scene5" not in line:
            continue
        obj = json.loads(line)
        rows.append({"iter": obj["iter"], "mIoU": obj["val_mIoU_scene5"], "per_class": obj["per_class_IoU"]})
    return rows


def plateau_note(curve):
    if len(curve) < 5:
        return "验证点不足 5 个，不能判断是否已经到平台。"
    tail = curve[-5:]
    bits = []
    for name in CLASS_NAMES:
        vals = [r["per_class"][name] for r in tail]
        span = max(vals) - min(vals)
        last = vals[-1]
        if span < 0.02 and last < 0.95:
            bits.append(f"{name} 最后 5 次验证停在 {last:.3f}（极差 {span:.3f}）")
        elif span >= 0.02:
            bits.append(f"{name} 最后 5 次仍在动（{vals[0]:.3f} → {last:.3f}）")
    return "；".join(bits) + "。"


def fmt(x):
    if x is None:
        return "—"
    return f"{float(x):.3f}"


def agreement(runs) -> str:
    bits = []
    for name in CLASS_NAMES:
        items = []
        for run in runs:
            row = next(r for r in run["classes"] if r["name"] == name)
            items.append((run["title"], row["primary"]))
        primaries = {p for _, p in items if p}
        if not primaries:
            bits.append(f"{name}：三组都没有单项过门槛。")
        elif len(primaries) == 1:
            bits.append(f"{name}：三组主因都是{ZH[next(iter(primaries))]}。")
        else:
            detail = "，".join(f"{title}={ZH.get(p, '无') if p else '无'}" for title, p in items)
            bits.append(f"{name}：主因不一致（{detail}）。")
    return "".join(bits)


def write_md(path: Path, payload: dict):
    lines = [
        "# 误差归因",
        "",
        "训练和一次性测试结束后，用已经写好的预测做统计。下面的门槛在看这批开源分数之前就定好。",
        "这里只说明不足落在语义分割的哪一类问题上，不写做法，也不据此改权重。",
        "",
        "## 口径",
        "",
        "- 混淆：每个类的漏检像素去了哪一类。",
        "- 边界：距类别交界 8 像素以内为边界带，其余有效像素为内部。",
        "- 尺度：连通域面积 &lt;32² 为小块，≥96² 为大块。只对岩石、撞击坑、山体计块。",
        "- 位置：图像从上到下切成 4 条，看同一类召回差多少。",
        "- 光照：同一类里，判错像素和判对像素的平均亮度差。",
        "- 域：Moon_2 / Moon_4 / Moon_9 上该类 IoU 的极差。",
        "",
        "过门槛才记一笔。多项同时过门槛时，分值最高的一项记为主因，其余照写。",
        "",
        "## 材料",
        "",
    ]
    for run in payload["runs"]:
        lines.append(
            f"- {run['title']}：{run['n_images']} 张，缺预测 {run['n_missing_pred']}，mIoU {fmt(run['mIoU'])}"
        )
    lines += ["", "## 开源测试集上的归因", ""]
    for run in payload["runs"]:
        lines += [f"### {run['title']}", "", "| 类 | IoU | 主因 | 同时成立的证据 |", "|---|---:|---|---|"]
        for row in run["classes"]:
            primary = ZH.get(row["primary"], "—")
            ev = "；".join(f["evidence"] for f in row["flags"]) or "没有单项过门槛"
            lines.append(f"| {row['name']} | {fmt(row['iou'])} | {primary} | {ev} |")
        lines.append("")
    lines += ["## 三组是不是同一类问题", "", payload["agreement"], "", "## 域内 Scene5 训练曲线", ""]
    lines.append("Scene5 只用来看本地验证有没有走到平台。加入 Moon_5 的那次没有验证集。")
    lines.append("")
    for name, text in payload["curves"].items():
        lines.append(f"- {name}：{text}")
    lines.append("")
    if payload.get("scene5_tta"):
        tta = payload["scene5_tta"]
        lines.append(
            f"不加 Moon_5 的 best.pth，Scene5 TTA：mIoU {fmt(tta.get('mIoU'))}，"
            f"分项 {tta.get('per_class')}，混淆率 {tta.get('rates')}。"
        )
        lines.append("")
    lines.append("Moon_2/4/9 的标签只参与这份统计，不回写训练。")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_specs(project: Path):
    return [
        ("B1 best，单尺度提交", project / "results" / "submit_B1_best"),
        ("ROW+BLV best + TTA", project / "results" / "oneshot_ROW_BLV" / "preds"),
        ("ROW+BLV+Moon_5 last + TTA", project / "results" / "oneshot_ROW_BLV_plus5" / "preds"),
    ]


def analyze(project: Path):
    data = project / "data"
    runs = [(title, new_acc(), folder) for title, folder in run_specs(project)]
    for sid in SCENES:
        gt_dir = data / f"Moon_{sid}" / "image0" / "label"
        color_dir = data / "test" / f"Moon_{sid}" / "color"
        names = sorted(p.name for p in gt_dir.glob("*.png"))
        print(f"[attrib] Moon_{sid} {len(names)}", flush=True)
        for i, name in enumerate(names, 1):
            gt = load_gt(gt_dir / name)
            yimg = luminance(color_dir / name)
            geo = gt_geometry(gt)
            for _, acc, folder in runs:
                pp = folder / f"Moon_{sid}" / name
                if not pp.exists():
                    acc["n_missing_pred"] += 1
                    continue
                pred = load_pred(pp)
                if pred.shape != gt.shape:
                    raise RuntimeError(f"shape {pp} {pred.shape} vs {gt.shape}")
                accumulate(acc, gt, pred, yimg, sid, geo)
            if i % 200 == 0 or i == len(names):
                print(f"[attrib] Moon_{sid} {i}/{len(names)}", flush=True)
    out_runs = []
    for title, acc, _ in runs:
        ious = iou_of(acc["hist"])
        valid = [v for v in ious if v == v]
        out_runs.append(
            {
                "title": title,
                "n_images": acc["n_images"],
                "n_missing_pred": acc["n_missing_pred"],
                "mIoU": jnum(float(np.mean(valid)) if valid else float("nan")),
                "classes": class_report(acc),
            }
        )
    curves = {
        "B1": plateau_note(val_curve(project / "work_dirs" / "train_B1" / "train_log.jsonl")),
        "ROW+BLV": plateau_note(val_curve(project / "work_dirs" / "train_ROW_BLV_20k" / "train_log.jsonl")),
    }
    tta_path = project / "work_dirs" / "train_ROW_BLV_20k" / "tta_final.json"
    tta = None
    if tta_path.exists():
        raw = json.loads(tta_path.read_text(encoding="utf-8"))
        if "error" not in raw:
            tta = {
                "mIoU": jnum(raw.get("mIoU")),
                "per_class": [jnum(v) for v in raw.get("per_class", [])],
                "rates": {k: jnum(v) for k, v in (raw.get("rates") or {}).items()},
            }
    payload = {
        "gates": GATES,
        "runs": out_runs,
        "agreement": agreement(out_runs),
        "curves": curves,
        "scene5_tta": tta,
        "note": "统计只用已有预测。不选模，不改超参。",
    }
    dest = project / "results" / "error_attrib.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(project / "results" / "error_attrib.md", payload)
    print("ATTRIB_DONE", dest, flush=True)
    return payload


def self_check():
    gt = np.zeros((64, 64), np.uint8)
    gt[:32, :] = 4
    gt[32:, :] = 0
    gt[40:48, 4:12] = 2
    pred = gt.copy()
    pred[:16, :] = 3
    pred[40:48, 4:12] = 0
    acc = new_acc()
    yimg = np.full(gt.shape, 100, np.float32)
    geo = gt_geometry(gt)
    accumulate(acc, gt, pred, yimg, 2, geo)
    for s in (4, 9):
        acc["scene"][s] = acc["hist"].copy()
    rows = {r["name"]: r for r in class_report(acc)}
    sky_types = [f["type"] for f in rows["天空"]["flags"]]
    crater_types = [f["type"] for f in rows["撞击坑"]["flags"]]
    if "texture_confusion" not in sky_types or "position_context" not in sky_types:
        raise SystemExit(f"sky flags {sky_types}")
    if "majority_swallow" not in crater_types:
        raise SystemExit(f"crater flags {crater_types}")
    if rows["月壤"]["primary"] is not None:
        raise SystemExit(f"regolith should be clean, got {rows['月壤']}")
    print("SELF_CHECK_OK", flush=True)


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
    else:
        root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\LuSNAR_Seg")
        analyze(root)
