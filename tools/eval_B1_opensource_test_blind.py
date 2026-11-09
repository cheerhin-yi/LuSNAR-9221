# ONE-SHOT POST-HOC blind eval: B1 preds vs open-source Moon_2/4/9 GT
# 非官方事后盲评；不参与调参/选模/配方。Do not reuse as model selection signal.
import json, time
from pathlib import Path
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "portable"))
from _lusnar_common import COLOR_TO_ID, CLASS_NAMES, NUM_CLASSES

ROOT = Path(r"D:\LuSNAR_Seg")
PRED_ROOT = ROOT / "results" / "submit_B1_best"
GT_ROOT = ROOT / "data"
OUT_MD = ROOT / "results" / "eval_B1_opensource_test_blind.md"
OUT_JSON = ROOT / "results" / "eval_B1_opensource_test_blind" / "summary.json"
SCENES = [("Moon_2", 1597), ("Moon_4", 1320), ("Moon_9", 2087)]
EN_NAMES = ["regolith", "rock", "crater", "mountain", "sky"]


def load_gt_official(path: Path) -> np.ndarray:
    """Open-source palette has rock/crater swapped; remap via RGB -> official COLOR_TO_ID."""
    im = Image.open(path)
    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for color, cid in COLOR_TO_ID.items():
        m = np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)
        out[m] = cid
    return out


def load_pred(path: Path) -> np.ndarray:
    im = Image.open(path)
    a = np.asarray(im)
    if a.ndim == 3:
        # unexpected RGB pred — remap
        out = np.full(a.shape[:2], 255, dtype=np.uint8)
        rgb = a[..., :3]
        for color, cid in COLOR_TO_ID.items():
            m = np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)
            out[m] = cid
        return out
    return a.astype(np.uint8)


def iou_from_hist(hist: np.ndarray):
    """hist[gt, pred] confusion, size CxC. Returns per-class IoU and mIoU (nanmean over classes with support)."""
    C = hist.shape[0]
    ious = np.full(C, np.nan, dtype=np.float64)
    for c in range(C):
        tp = hist[c, c]
        fp = hist[:, c].sum() - tp
        fn = hist[c, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            ious[c] = tp / denom
    valid = ~np.isnan(ious)
    miou = float(np.nanmean(ious)) if valid.any() else float("nan")
    return ious, miou


def main():
    t0 = time.time()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    overall = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    scene_stats = {}
    missing = []
    unmapped_gt = 0
    total_pairs = 0

    for scene, expect_n in SCENES:
        pred_dir = PRED_ROOT / scene
        gt_dir = GT_ROOT / scene / "image0" / "label"
        hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        n = 0
        preds = sorted(pred_dir.glob("*.png"))
        if len(preds) != expect_n:
            print(f"WARN {scene}: pred count {len(preds)} != expect {expect_n}")
        for pp in preds:
            gp = gt_dir / pp.name
            if not gp.exists():
                missing.append(f"{scene}/{pp.name}")
                continue
            pred = load_pred(pp)
            gt = load_gt_official(gp)
            if pred.shape != gt.shape:
                raise RuntimeError(f"shape mismatch {pp}: pred {pred.shape} gt {gt.shape}")
            valid = gt < NUM_CLASSES
            unmapped_gt += int((~valid).sum())
            p = pred[valid].astype(np.int64)
            g = gt[valid].astype(np.int64)
            # clamp unexpected pred ids
            p = np.clip(p, 0, NUM_CLASSES - 1)
            np.add.at(hist, (g, p), 1)
            n += 1
        overall += hist
        ious, miou = iou_from_hist(hist)
        scene_stats[scene] = {
            "n_images": n,
            "expect_n": expect_n,
            "mIoU": miou,
            "per_class_IoU": {EN_NAMES[i]: (None if np.isnan(ious[i]) else float(ious[i])) for i in range(NUM_CLASSES)},
            "per_class_IoU_zh": {CLASS_NAMES[i]: (None if np.isnan(ious[i]) else float(ious[i])) for i in range(NUM_CLASSES)},
            "gt_pixel_support": {EN_NAMES[i]: int(hist[i, :].sum()) for i in range(NUM_CLASSES)},
        }
        total_pairs += n
        print(f"{scene}: n={n} mIoU={miou:.4f} " + " ".join(f"{EN_NAMES[i]}={ious[i]:.4f}" for i in range(NUM_CLASSES)))

    ious_o, miou_o = iou_from_hist(overall)
    # macro over scenes (unweighted mean of scene mIoUs)
    scene_mious = [scene_stats[s]["mIoU"] for s, _ in SCENES]
    macro_scene = float(np.mean(scene_mious))
    # weighted by image count
    weighted = float(np.average(scene_mious, weights=[scene_stats[s]["n_images"] for s, _ in SCENES]))
    elapsed = time.time() - t0

    summary = {
        "disclaimer": "非官方事后盲评；开源 Moon_2/4/9 GT；不参与调参/选模/配方。ONE-SHOT POST-HOC ONLY.",
        "weight": str(ROOT / "work_dirs" / "train_B1" / "best.pth"),
        "weight_note": "iter 17000, Scene5 local mIoU 0.8630 (not this test score)",
        "pred_source": str(PRED_ROOT),
        "pred_reused": True,
        "re_inferred": False,
        "gt_root": str(GT_ROOT / "Moon_{2,4,9}/image0/label"),
        "gt_remap": "RGB via COLOR_TO_ID (open-source palette rock/crater swapped vs official)",
        "official_class_order": {i: EN_NAMES[i] for i in range(NUM_CLASSES)},
        "gpu_used": "none (CPU score of existing preds); GPU1 reserved if re-infer were needed",
        "n_images_total": total_pairs,
        "missing_gt": missing[:20],
        "n_missing": len(missing),
        "unmapped_gt_pixels": unmapped_gt,
        "wall_time_sec": round(elapsed, 2),
        "overall_micro_from_confusion": {
            "mIoU": float(miou_o),
            "per_class_IoU": {EN_NAMES[i]: (None if np.isnan(ious_o[i]) else float(ious_o[i])) for i in range(NUM_CLASSES)},
            "per_class_IoU_zh": {CLASS_NAMES[i]: (None if np.isnan(ious_o[i]) else float(ious_o[i])) for i in range(NUM_CLASSES)},
            "gt_pixel_support": {EN_NAMES[i]: int(overall[i, :].sum()) for i in range(NUM_CLASSES)},
        },
        "macro_mean_of_scene_mIoU": macro_scene,
        "image_count_weighted_mean_of_scene_mIoU": weighted,
        "official_scene_weights": None,
        "per_scene": scene_stats,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(x):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"

    lines = []
    lines.append("# B1 开源测试集事后盲评（非官方）")
    lines.append("")
    lines.append("> **非官方事后盲评**。使用开源 LuSNAR 的 Moon_2/4/9 标签，对已提交的 `train_B1/best.pth` 预测打分。")
    lines.append("> **不参与调参 / 选模 / 配方**；不回流任何训练决策。与比赛官方盲评分数不可等同。")
    lines.append("")
    lines.append("## 设置")
    lines.append("")
    lines.append(f"- 权重：`work_dirs/train_B1/best.pth`（iter 17000；本地 Scene5 mIoU 0.8630，仅作参照）")
    lines.append(f"- 预测：复用已有 `results/submit_B1_best/Moon_{{2,4,9}}`（共 {total_pairs} 张索引 PNG，未重新推理）")
    lines.append(f"- GT：`data/Moon_{{2,4,9}}/image0/label`（开源）；经 RGB→官方 `COLOR_TO_ID` 重映射（开源调色板岩石/撞击坑对调）")
    lines.append("- 类别序（官方提交序）：0 月壤 / 1 岩石 / 2 撞击坑 / 3 山体 / 4 天空")
    lines.append("- 计算：CPU；未占用 GPU（GPU0 冒烟保留；本任务无需 GPU1）")
    lines.append(f"- 墙钟：{elapsed:.1f}s；缺失 GT 文件：{len(missing)}；未映射 GT 像素：{unmapped_gt}")
    lines.append("- 官方场景权重：未知 → 同时给出 **混淆矩阵微平均 mIoU**、场景 mIoU 宏平均、按图数加权平均")
    lines.append("")
    lines.append("## 分场景 mIoU")
    lines.append("")
    lines.append("| 场景 | 图数 | mIoU | 月壤 | 岩石 | 撞击坑 | 山体 | 天空 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for scene, _ in SCENES:
        s = scene_stats[scene]
        pc = s["per_class_IoU"]
        lines.append(
            f"| {scene} | {s['n_images']} | {fmt(s['mIoU'])} | "
            + " | ".join(fmt(pc[k]) for k in EN_NAMES)
            + " |"
        )
    lines.append("")
    lines.append("## 总体")
    lines.append("")
    lines.append(f"- **Overall mIoU（全像素混淆矩阵微平均）**: **{fmt(miou_o)}**")
    lines.append(f"- 场景 mIoU 宏平均（3 景等权）: {fmt(macro_scene)}")
    lines.append(f"- 场景 mIoU 按图数加权: {fmt(weighted)}")
    lines.append("")
    lines.append("| 类别 | IoU | GT 像素数 |")
    lines.append("|---|---:|---:|")
    for i, name in enumerate(EN_NAMES):
        v = summary["overall_micro_from_confusion"]["per_class_IoU"][name]
        sup = summary["overall_micro_from_confusion"]["gt_pixel_support"][name]
        lines.append(f"| {i} {CLASS_NAMES[i]} ({name}) | {fmt(v)} | {sup} |")
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- 预测为 L 模式 0–4 索引图，与官方提交序一致；未改权重、未重选 checkpoint。")
    lines.append("- 开源 GT 调色板索引序为 0 月壤 / 1 撞击坑 / 2 岩石 / 3 山体 / 4 天空，打分前已按 RGB 映射到官方序。")
    lines.append("- 机器：DESKTOP-ORQLB8F（4090）；本评测仅 CPU。")
    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", OUT_MD)
    print("Wrote", OUT_JSON)
    print("OVERALL mIoU", miou_o)
    print("DONE", elapsed)


if __name__ == "__main__":
    main()
