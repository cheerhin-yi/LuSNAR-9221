#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Open-source Moon_2/4/9 eval of first-round best.pth under Scene5-frozen protocols.

Reuses lusnar_oneshot_opensource GT mapping (COLOR_TO_ID; rock/crater swapped
in the open-source palette) and the same TTA path (ROW + scales 0.75/1.0/1.25,
sliding 512/256, bf16 autocast via _lusnar_common.sliding_window_logits).

C0 / I1a / P1 / P1P2 share one forward stack per image (no extra GPU cost).
I1a weights and P1 bias are frozen from Scene5 A (protocols.json); they are
not re-searched on Moon_2/4/9.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
from _lusnar_common import (  # noqa: E402
    COLOR_TO_ID,
    NUM_CLASSES,
    ensure_mmcv_stub,
    find_pretrain,
    resolve_project_root,
    sliding_window_logits,
)
from lusnar_oneshot_opensource import (  # noqa: E402
    EN_NAMES,
    SCALES,
    SCENES,
    imread_bgr,
    iou_from_hist,
    load_gt_official,
    load_model,
)
from lusnar_scene5_protocols import (  # noqa: E402
    apply_logit_bias,
    geometric_cc_rules,
)

ensure_mmcv_stub(SCRIPT)

TZ = ZoneInfo("Asia/Shanghai")
PROTOCOLS = (
    "single_scale",
    "C0_equal_tta",
    "I1a_best_scale_per_class",
    "P1_logit_bias",
    "P1P2_bias_plus_geo",
)
OFFICIAL_BASELINE_S = 0.8046
PREV_C0_OVERALL = 0.8298977303524323


def softmax_logits(fused: torch.Tensor) -> np.ndarray:
    return torch.softmax(fused, dim=0).cpu().numpy().astype(np.float32)


def fuse_weighted_torch(logits_list, weights_CxS: np.ndarray) -> torch.Tensor:
    W = torch.as_tensor(weights_CxS, device=logits_list[0].device, dtype=logits_list[0].dtype)
    C, H, Wd = logits_list[0].shape
    S = len(logits_list)
    out = torch.zeros((C, H, Wd), device=logits_list[0].device, dtype=logits_list[0].dtype)
    for c in range(C):
        acc = torch.zeros((H, Wd), device=logits_list[0].device, dtype=logits_list[0].dtype)
        for si, z in enumerate(logits_list):
            acc = acc + W[c, si] * z[c]
        out[c] = acc
    return out


def predict_all(net, img_bgr, device, row_attn, W_i1a, bias) -> dict:
    h0, w0 = img_bgr.shape[:2]
    scale_logits = []
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
        scale_logits.append(logits)
        acc += logits
    pred_single = scale_logits[SCALES.index(1.0)].argmax(0).cpu().numpy().astype(np.uint8)
    pred_c0 = acc.argmax(0).cpu().numpy().astype(np.uint8)
    pred_i1a = fuse_weighted_torch(scale_logits, W_i1a).argmax(0).cpu().numpy().astype(np.uint8)
    # softmax(sum) == softmax(mean); matches cache_diag C0 probs
    probs = softmax_logits(acc)
    pred_p1 = apply_logit_bias(probs, bias).argmax(0).astype(np.uint8)
    pred_p1p2 = geometric_cc_rules(pred_p1)
    return {
        "single_scale": pred_single,
        "C0_equal_tta": pred_c0,
        "I1a_best_scale_per_class": pred_i1a,
        "P1_logit_bias": pred_p1,
        "P1P2_bias_plus_geo": pred_p1p2,
    }


def empty_state():
    state = {
        "per_scene": {
            f"Moon_{sid}": {
                name: {
                    "hist": np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64),
                    "n_images": 0,
                    "n_missing": 0,
                    "unmapped_gt_pixels": 0,
                }
                for name in PROTOCOLS
            }
            for sid in SCENES
        },
        "n_pred": 0,
        "n_scored": 0,
        "c0_ref_compared": 0,
        "c0_ref_equal": 0,
        "infer_sec": 0.0,
    }
    return state


def accumulate(slot, pred, gt):
    if pred.shape != gt.shape:
        raise RuntimeError(f"shape mismatch pred={pred.shape} gt={gt.shape}")
    valid = gt < NUM_CLASSES
    unmapped = int((~valid).sum())
    g = gt[valid].astype(np.int64)
    p = np.clip(pred[valid].astype(np.int64), 0, NUM_CLASSES - 1)
    np.add.at(slot["hist"], (g, p), 1)
    slot["n_images"] += 1
    slot["unmapped_gt_pixels"] += unmapped


def state_to_jsonable(state):
    out = {
        "n_pred": state["n_pred"],
        "n_scored": state["n_scored"],
        "c0_ref_compared": state["c0_ref_compared"],
        "c0_ref_equal": state["c0_ref_equal"],
        "infer_sec": state["infer_sec"],
        "per_scene": {},
    }
    for scene, protocols in state["per_scene"].items():
        out["per_scene"][scene] = {}
        for name, slot in protocols.items():
            out["per_scene"][scene][name] = {
                "hist": slot["hist"].tolist(),
                "n_images": int(slot["n_images"]),
                "n_missing": int(slot["n_missing"]),
                "unmapped_gt_pixels": int(slot["unmapped_gt_pixels"]),
            }
    return out


def state_from_jsonable(payload):
    state = empty_state()
    state["n_pred"] = int(payload.get("n_pred") or 0)
    state["n_scored"] = int(payload.get("n_scored") or 0)
    state["c0_ref_compared"] = int(payload.get("c0_ref_compared") or 0)
    state["c0_ref_equal"] = int(payload.get("c0_ref_equal") or 0)
    state["infer_sec"] = float(payload.get("infer_sec") or 0.0)
    for scene, protocols in (payload.get("per_scene") or {}).items():
        for name, slot in protocols.items():
            dst = state["per_scene"][scene][name]
            dst["hist"] = np.asarray(slot["hist"], dtype=np.int64)
            dst["n_images"] = int(slot["n_images"])
            dst["n_missing"] = int(slot["n_missing"])
            dst["unmapped_gt_pixels"] = int(slot["unmapped_gt_pixels"])
    return state


def list_items(data: Path, limit: int = 0):
    items = []
    for sid in SCENES:
        scene = f"Moon_{sid}"
        color_dir = data / "test" / scene / "color"
        images = sorted(color_dir.glob("*.png"))
        if limit:
            images = images[:limit]
        for ip in images:
            items.append((scene, ip))
    return items


def worker(rank, items, args_dict, out_path):
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    project = resolve_project_root(SCRIPT, args_dict["project_root"])
    data = Path(args_dict["data_root"])
    ckpt_path = Path(args_dict["checkpoint"])
    ref_pred_root = Path(args_dict["ref_pred_root"]) if args_dict.get("ref_pred_root") else None
    W_i1a = np.asarray(args_dict["W_i1a"], dtype=np.float32)
    bias = np.asarray(args_dict["bias"], dtype=np.float32)
    print(
        {"phase": "worker_start", "rank": rank, "n": len(items), "device": str(device)},
        flush=True,
    )
    net, row, ckpt = load_model(ckpt_path, project, device)
    state = empty_state()
    t0 = time.time()
    for i, (scene, ip) in enumerate(items, 1):
        img = imread_bgr(ip)
        preds = predict_all(net, img, device, row, W_i1a, bias)
        state["n_pred"] += 1
        gp = data / scene / "image0" / "label" / ip.name
        if not gp.exists():
            for name in PROTOCOLS:
                state["per_scene"][scene][name]["n_missing"] += 1
            print(f"[gpu{rank}] MISSING_GT {scene}/{ip.name}", flush=True)
        else:
            gt = load_gt_official(gp)
            for name, pred in preds.items():
                accumulate(state["per_scene"][scene][name], pred, gt)
            state["n_scored"] += 1
        if ref_pred_root is not None:
            rp = ref_pred_root / scene / ip.name
            if rp.exists():
                ref = np.asarray(Image.open(rp))
                state["c0_ref_compared"] += 1
                if ref.shape == preds["C0_equal_tta"].shape and np.array_equal(ref, preds["C0_equal_tta"]):
                    state["c0_ref_equal"] += 1
        if i % 20 == 0 or i == len(items):
            state["infer_sec"] = time.time() - t0
            Path(out_path).write_text(
                json.dumps(state_to_jsonable(state), ensure_ascii=False),
                encoding="utf-8",
            )
            elapsed = state["infer_sec"]
            eta = (elapsed / i) * (len(items) - i) if i else 0
            print(
                f"[gpu{rank}] {i}/{len(items)} scored={state['n_scored']} "
                f"c0_match={state['c0_ref_equal']}/{state['c0_ref_compared']} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    state["infer_sec"] = time.time() - t0
    Path(out_path).write_text(
        json.dumps(state_to_jsonable(state), ensure_ascii=False),
        encoding="utf-8",
    )
    print({"phase": "worker_done", "rank": rank, "n_pred": state["n_pred"]}, flush=True)


def merge_states(states):
    merged = empty_state()
    for st in states:
        merged["n_pred"] += st["n_pred"]
        merged["n_scored"] += st["n_scored"]
        merged["c0_ref_compared"] += st["c0_ref_compared"]
        merged["c0_ref_equal"] += st["c0_ref_equal"]
        merged["infer_sec"] = max(merged["infer_sec"], st["infer_sec"])
        for scene in merged["per_scene"]:
            for name in PROTOCOLS:
                a = merged["per_scene"][scene][name]
                b = st["per_scene"][scene][name]
                a["hist"] += b["hist"]
                a["n_images"] += b["n_images"]
                a["n_missing"] += b["n_missing"]
                a["unmapped_gt_pixels"] += b["unmapped_gt_pixels"]
    return merged


def metrics_from_hist(hist: np.ndarray, n_images: int, unmapped: int, n_missing: int):
    ious, miou = iou_from_hist(hist)
    return {
        "n_images": int(n_images),
        "n_missing": int(n_missing),
        "unmapped_gt_pixels": int(unmapped),
        "mIoU": (None if np.isnan(miou) else float(miou)),
        "per_class_IoU": {
            EN_NAMES[i]: (None if np.isnan(ious[i]) else float(ious[i]))
            for i in range(NUM_CLASSES)
        },
        "gt_pixels": {EN_NAMES[i]: int(hist[i].sum()) for i in range(NUM_CLASSES)},
    }


def converted_S(per_scene_miou: dict) -> float:
    m2 = per_scene_miou["Moon_2"]
    m4 = per_scene_miou["Moon_4"]
    m9 = per_scene_miou["Moon_9"]
    return (m2 + m4 + 2.0 * m9) / 4.0


def summarize(merged, meta):
    protocols = {}
    for name in PROTOCOLS:
        per_scene = {}
        overall_hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        unmapped = 0
        missing = 0
        n_images = 0
        for sid in SCENES:
            scene = f"Moon_{sid}"
            slot = merged["per_scene"][scene][name]
            per_scene[scene] = metrics_from_hist(
                slot["hist"], slot["n_images"], slot["unmapped_gt_pixels"], slot["n_missing"]
            )
            overall_hist += slot["hist"]
            unmapped += slot["unmapped_gt_pixels"]
            missing += slot["n_missing"]
            n_images += slot["n_images"]
        overall = metrics_from_hist(overall_hist, n_images, unmapped, missing)
        scene_mious = {scene: per_scene[scene]["mIoU"] for scene in per_scene}
        S = converted_S(scene_mious)
        protocols[name] = {
            "per_scene": per_scene,
            "overall": overall,
            "S": float(S),
            "S_minus_official_baseline": float(S - OFFICIAL_BASELINE_S),
            "m2": scene_mious["Moon_2"],
            "m4": scene_mious["Moon_4"],
            "m9": scene_mious["Moon_9"],
        }
    c0 = protocols["C0_equal_tta"]["overall"]["mIoU"]
    return {
        "created_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "Asia/Shanghai",
        "role": "competition-line open-source Moon_2/4/9 used as a dev set after Scene5 protocol freeze; not a paper test set; not used for training",
        "checkpoint": meta["checkpoint"],
        "ckpt_iter": meta.get("ckpt_iter"),
        "row_attn": True,
        "scales": list(SCALES),
        "crop": 512,
        "stride": 256,
        "amp": "bf16",
        "gpus": meta["gpus"],
        "n_images": int(merged["n_scored"]),
        "n_pred": int(merged["n_pred"]),
        "wall_s": round(float(merged["infer_sec"]), 1),
        "label_alignment": "open-source RGB remapped by official COLOR_TO_ID; rock/crater swapped vs official index order",
        "gt_root": "data/Moon_{2,4,9}/image0/label",
        "color_root": "data/test/Moon_{2,4,9}/color",
        "frozen_from_scene5": {
            "winner": "P1P2_bias_plus_geo",
            "I1a_weights": meta["W_i1a"],
            "I1a_scales": list(SCALES),
            "P1_bias": meta["bias"],
        },
        "official_competition_baseline_S": OFFICIAL_BASELINE_S,
        "previous_oneshot_C0_overall_mIoU": PREV_C0_OVERALL,
        "c0_vs_previous_oneshot_preds": {
            "compared": int(merged["c0_ref_compared"]),
            "equal": int(merged["c0_ref_equal"]),
            "match_rate": (
                None
                if merged["c0_ref_compared"] == 0
                else float(merged["c0_ref_equal"]) / float(merged["c0_ref_compared"])
            ),
        },
        "c0_overall_mIoU_delta_vs_previous": (
            None if c0 is None else float(c0) - PREV_C0_OVERALL
        ),
        "protocols": protocols,
    }


def fmt(x, nd=4):
    if x is None:
        return "\u2014"
    try:
        xf = float(x)
        if xf != xf:
            return "\u2014"
        return f"{xf:.{nd}f}"
    except Exception:
        return str(x)


def write_report(path: Path, summary: dict):
    lines = [
        "# Open-source Moon_2/4/9 eval (competition-line dev set)",
        "",
        f"- Time (Asia/Shanghai): {summary['created_at']}",
        f"- Checkpoint: `{summary['checkpoint']}` (iter {summary.get('ckpt_iter')})",
        "- Model: SegFormer-B2 + V_B1C + ROW + BLV, crop 512. One forward stack: single-scale 1.0, equal TTA 0.75/1.0/1.25, sliding 512/256, bf16.",
        "- GT: `data/Moon_{2,4,9}/image0/label`, RGB remapped by official COLOR_TO_ID (rock/crater swapped in the open-source palette). Same loader as `results/oneshot_ROW_BLV/`.",
        "- I1a weights and P1 bias are frozen from Scene5 A (`P1P2_bias_plus_geo` winner). They were not re-tuned on Moon_2/4/9.",
        "- These scores may be used to pick a competition submission. They are not a paper test result (paper test is Moon_3/5/7 under the official split).",
        f"- Images scored: {summary['n_images']}; wall {fmt(summary['wall_s'], 1)} s on {summary['gpus']} GPU(s).",
        "",
        "## Protocol comparison",
        "",
        "| protocol | m2 | m4 | m9 | S=(m2+m4+2*m9)/4 | vs official 0.8046 | overall micro mIoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in PROTOCOLS:
        p = summary["protocols"][name]
        lines.append(
            f"| {name} | {fmt(p['m2'])} | {fmt(p['m4'])} | {fmt(p['m9'])} | "
            f"{fmt(p['S'])} | {p['S_minus_official_baseline']:+.4f} | {fmt(p['overall']['mIoU'])} |"
        )
    lines += [
        "",
        "Previous oneshot C0 overall micro mIoU was "
        f"{PREV_C0_OVERALL:.4f}; this run C0 delta = {fmt(summary.get('c0_overall_mIoU_delta_vs_previous'))}.",
        "C0 pred match vs `results/oneshot_ROW_BLV/preds`: "
        f"{summary['c0_vs_previous_oneshot_preds']['equal']}/"
        f"{summary['c0_vs_previous_oneshot_preds']['compared']}"
        f" (rate {fmt(summary['c0_vs_previous_oneshot_preds']['match_rate'])}).",
        "",
        "## Per-scene per-class IoU",
        "",
    ]
    for name in PROTOCOLS:
        p = summary["protocols"][name]
        lines += [
            f"### {name}",
            "",
            "| scene | n | mIoU | regolith | rock | crater | mountain | sky |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for scene in ("Moon_2", "Moon_4", "Moon_9"):
            s = p["per_scene"][scene]
            pc = s["per_class_IoU"]
            lines.append(
                f"| {scene} | {s['n_images']} | {fmt(s['mIoU'])} | "
                + " | ".join(fmt(pc[k]) for k in EN_NAMES)
                + " |"
            )
        opc = p["overall"]["per_class_IoU"]
        lines.append(
            f"| overall | {p['overall']['n_images']} | {fmt(p['overall']['mIoU'])} | "
            + " | ".join(fmt(opc[k]) for k in EN_NAMES)
            + " |"
        )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_frozen(project: Path):
    proto = json.loads(
        (project / "results" / "inference_phase" / "protocols.json").read_text(encoding="utf-8")
    )
    cands = proto["candidates"]
    W = np.asarray(cands["I1a_best_scale_per_class"]["weights"], dtype=np.float32)
    bias = np.asarray(cands["P1P2_bias_plus_geo"]["bias"], dtype=np.float32)
    return W, bias, proto.get("frozen_recipe", {}).get("name")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=None)
    ap.add_argument(
        "--checkpoint",
        default="/mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/best.pth",
    )
    ap.add_argument(
        "--out-dir",
        default="/mnt/d/LuSNAR_Seg/results/opensource_final_ROW_BLV",
    )
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="Per-scene image cap (smoke).")
    ap.add_argument(
        "--ref-pred-root",
        default="/mnt/d/LuSNAR_Seg/results/oneshot_ROW_BLV/preds",
    )
    args = ap.parse_args()

    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    project = resolve_project_root(SCRIPT, args.project_root)
    data = Path(args.data_root) if args.data_root else (project / "data")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    W_i1a, bias, frozen_name = load_frozen(project)
    items = list_items(data, limit=args.limit)
    n_gpu_avail = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_gpu = max(1, min(int(args.gpus), n_gpu_avail if n_gpu_avail > 0 else 1, 2))
    if n_gpu_avail <= 0:
        n_gpu = 1
    print(
        {
            "phase": "start",
            "n_images": len(items),
            "gpus": n_gpu,
            "cuda_visible": n_gpu_avail,
            "frozen_recipe": frozen_name,
            "W_i1a": W_i1a.tolist(),
            "bias": bias.tolist(),
            "out_dir": str(out_dir),
        },
        flush=True,
    )

    args_dict = {
        "project_root": str(project),
        "data_root": str(data),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "ref_pred_root": args.ref_pred_root,
        "W_i1a": W_i1a.tolist(),
        "bias": bias.tolist(),
    }
    part_paths = [str(out_dir / f"_part_rank{r}.json") for r in range(n_gpu)]
    t0 = time.time()
    if n_gpu == 1:
        worker(0, items, args_dict, part_paths[0])
    else:
        import torch.multiprocessing as mp

        chunks = [items[i::n_gpu] for i in range(n_gpu)]
        ctx = mp.get_context("spawn")
        procs = []
        for r in range(n_gpu):
            pr = ctx.Process(target=worker, args=(r, chunks[r], args_dict, part_paths[r]))
            pr.start()
            procs.append(pr)
        for pr in procs:
            pr.join()
            if pr.exitcode != 0:
                raise RuntimeError(f"worker failed exit={pr.exitcode}")
    wall = time.time() - t0
    states = [state_from_jsonable(json.loads(Path(p).read_text(encoding="utf-8"))) for p in part_paths]
    merged = merge_states(states)
    merged["infer_sec"] = wall
    # ckpt iter
    ckpt = torch.load(args_dict["checkpoint"], map_location="cpu", weights_only=False)
    summary = summarize(
        merged,
        {
            "checkpoint": args_dict["checkpoint"],
            "ckpt_iter": ckpt.get("iter"),
            "gpus": n_gpu,
            "W_i1a": W_i1a.tolist(),
            "bias": bias.tolist(),
        },
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(out_dir / "report.md", summary)
    print(
        {
            "phase": "done",
            "summary": str(out_dir / "summary.json"),
            "report": str(out_dir / "report.md"),
            "single_mIoU": summary["protocols"]["single_scale"]["overall"]["mIoU"],
            "single_S": summary["protocols"]["single_scale"]["S"],
            "C0_mIoU": summary["protocols"]["C0_equal_tta"]["overall"]["mIoU"],
            "C0_S": summary["protocols"]["C0_equal_tta"]["S"],
            "I1a_mIoU": summary["protocols"]["I1a_best_scale_per_class"]["overall"]["mIoU"],
            "I1a_S": summary["protocols"]["I1a_best_scale_per_class"]["S"],
            "P1P2_mIoU": summary["protocols"]["P1P2_bias_plus_geo"]["overall"]["mIoU"],
            "P1P2_S": summary["protocols"]["P1P2_bias_plus_geo"]["S"],
            "c0_match": summary["c0_vs_previous_oneshot_preds"],
            "wall_s": summary["wall_s"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
