#!/usr/bin/env python3
"""I4 hybrid full-image + tiles. Pre-registered before any B scores are used.

Default merge: mountain/sky/regolith from a single native 1024 full-image
forward; rock/crater from cached tiled TTA (C0, 512 window, scales 0.75/1.0/1.25).

The first-round model was trained at crop 512, so 1024 full-image has a
train/test resolution mismatch; we still run it and record full-only quality.

Selection is Scene5 A only. Moon_2/4/9 labels are never used.
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

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
from _lusnar_common import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    resolve_project_root,
    rgb_label_to_index,
    set_seed,
)
from lusnar_scene5_cache_diag import (  # noqa: E402
    EN,
    CORE_SCALES,
    _forward_logits,
    _preprocess_bgr,
    imread_bgr,
    load_model,
    update_inter_union,
    iou_from_iu,
)
from lusnar_scene5_protocols import (  # noqa: E402
    load_stems,
    load_scale_logits,
    fuse_equal,
    per_image_iou,
    paired_tests,
    holm_correct,
    _merge_partial,
)

LARGE_CLASSES = (0, 3, 4)  # regolith, mountain, sky
SMALL_CLASSES = (1, 2)  # rock, crater
FULL_SIZE = (1024, 1024)  # native Scene5 / official MoonSeg whole-image size
MERGE_VARIANTS = ("M1_tile_small_overwrite", "M2_full_large_overwrite", "M3_logit_channels", "M4_per_class_source")


@torch.no_grad()
def full_image_logits(net, img_bgr, device, row_attn=None, infer_hw=None):
    """One forward at infer_hw (default: native). Logits returned at original HxW."""
    h0, w0 = img_bgr.shape[:2]
    img = img_bgr
    used_hw = (h0, w0)
    if infer_hw is not None and tuple(infer_hw) != (h0, w0):
        img = cv2.resize(img, (int(infer_hw[1]), int(infer_hw[0])), interpolation=cv2.INTER_LINEAR)
        used_hw = (int(infer_hw[0]), int(infer_hw[1]))
    t = _preprocess_bgr(img).unsqueeze(0).to(device)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        logits = _forward_logits(net, t, row_attn=row_attn)
        if logits.shape[-2:] != (h0, w0):
            logits = F.interpolate(
                logits, size=(h0, w0), mode="bilinear", align_corners=False
            )
    return logits.float()[0], used_hw


def _i4_worker(rank, items, args_dict, out_path):
    set_seed(42)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    project = resolve_project_root(SCRIPT, args_dict["project_root"])
    net, row, ckpt = load_model(Path(args_dict["checkpoint"]), project, device)
    infer_hw = tuple(args_dict["infer_hw"]) if args_dict.get("infer_hw") else None
    out_dir = Path(args_dict["full_logits_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    per_img = []
    n_fwd = 0
    used = None
    oom_fallback = None
    t0 = time.time()
    for j, (stem, ip, lp) in enumerate(items):
        img = imread_bgr(ip)
        mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
        try:
            logits, used = full_image_logits(net, img, device, row_attn=row, infer_hw=infer_hw)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            # fallback: 768x768, documented
            infer_hw = (768, 768)
            oom_fallback = "native 1024 OOM; fallback infer_hw=768x768"
            logits, used = full_image_logits(net, img, device, row_attn=row, infer_hw=infer_hw)
        n_fwd += 1
        arr = logits.cpu().numpy().astype(np.float16)
        np.save(out_dir / f"{stem}.npy", arr)
        pred = arr.argmax(0).astype(np.uint8)
        update_inter_union(inter, union, pred, mask)
        _, mi = per_image_iou(pred, mask)
        per_img.append(mi)
        if (j + 1) % 50 == 0 or (j + 1) == len(items):
            print(f"[I4 full gpu{rank}] {j+1}/{len(items)} used_hw={used}", flush=True)
    payload = {
        "inter": inter.tolist(),
        "union": union.tolist(),
        "per_image_miou": per_img,
        "forward_windows": int(n_fwd),
        "n_forwards": int(n_fwd),
        "wall_s": round(time.time() - t0, 2),
        "used_hw": list(used) if used else None,
        "oom_fallback": oom_fallback,
    }
    Path(out_path).write_text(json.dumps(payload), encoding="utf-8")


def run_full_image_cache(items, args):
    import torch.multiprocessing as mp

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_gpu = min(n_gpu, 2) if n_gpu else 1
    args_dict = {
        "project_root": args.project_root,
        "checkpoint": args.checkpoint,
        "full_logits_dir": args.full_logits_dir,
        "infer_hw": list(args.infer_hw) if args.infer_hw else None,
    }
    Path(args.full_logits_dir).mkdir(parents=True, exist_ok=True)
    if n_gpu <= 1 or len(items) < 8:
        tmp = Path(args.out_json).with_name("_part_I4full_0.json")
        _i4_worker(0, items, args_dict, str(tmp))
        part = json.loads(tmp.read_text(encoding="utf-8"))
        return _merge_partial([part]), part.get("used_hw"), part.get("oom_fallback")
    chunks = [items[i::n_gpu] for i in range(n_gpu)]
    tmp_paths = [str(Path(args.out_json).with_name(f"_part_I4full_{r}.json")) for r in range(n_gpu)]
    ctx = mp.get_context("spawn")
    procs = []
    for r in range(n_gpu):
        pr = ctx.Process(target=_i4_worker, args=(r, chunks[r], args_dict, tmp_paths[r]))
        pr.start()
        procs.append(pr)
    for pr in procs:
        pr.join()
        if pr.exitcode != 0:
            raise RuntimeError(f"I4 worker failed exit={pr.exitcode}")
    parts = [json.loads(Path(tp).read_text(encoding="utf-8")) for tp in tmp_paths]
    used = parts[0].get("used_hw")
    oom = next((p.get("oom_fallback") for p in parts if p.get("oom_fallback")), None)
    return _merge_partial(parts), used, oom


def load_full_logits(full_dir: Path, stem: str):
    p = full_dir / f"{stem}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


def merge_pred(full_z, tile_z, variant, source_is_tile=None):
    """Return uint8 pred. full_z/tile_z are C,H,W logits."""
    full_p = full_z.argmax(0).astype(np.uint8)
    tile_p = tile_z.argmax(0).astype(np.uint8)
    if variant == "M1_tile_small_overwrite":
        pred = full_p.copy()
        for c in SMALL_CLASSES:
            pred[tile_p == c] = c
        return pred
    if variant == "M2_full_large_overwrite":
        pred = tile_p.copy()
        for c in LARGE_CLASSES:
            pred[full_p == c] = c
        return pred
    if variant == "M3_logit_channels":
        z = tile_z.copy()
        for c in LARGE_CLASSES:
            z[c] = full_z[c]
        return z.argmax(0).astype(np.uint8)
    if variant == "M4_per_class_source":
        z = np.zeros_like(full_z)
        for c in range(NUM_CLASSES):
            src = tile_z if (source_is_tile is not None and source_is_tile[c]) else full_z
            z[c] = src[c]
        return z.argmax(0).astype(np.uint8)
    raise ValueError(variant)


def eval_variant(items, full_dir, logits_dir, variant, source_is_tile=None, scales=CORE_SCALES):
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    per_img = []
    t0 = time.time()
    n = 0
    for stem, ip, lp in items:
        full_z = load_full_logits(full_dir, stem)
        zs = load_scale_logits(logits_dir, stem, scales)
        if full_z is None or zs is None:
            continue
        tile_z = fuse_equal(zs)
        pred = merge_pred(full_z, tile_z, variant, source_is_tile=source_is_tile)
        mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
        update_inter_union(inter, union, pred, mask)
        _, mi = per_image_iou(pred, mask)
        per_img.append(mi)
        n += 1
    miou, ious = iou_from_iu(inter, union)
    return {
        "mIoU": miou,
        "per_class": {CLASS_NAMES[c]: ious[c] for c in range(NUM_CLASSES)},
        "per_class_en": {EN[c]: ious[c] for c in range(NUM_CLASSES)},
        "per_image_miou": per_img,
        "n": n,
        "wall_s": round(time.time() - t0, 2),
        "ms_per_img": round(1000 * (time.time() - t0) / max(n, 1), 2),
        "forward_passes_note": "merge is CPU; +1 full-image 1024 forward/img already cached",
    }


def pick_m4_sources(items_A, full_dir, logits_dir, scales=CORE_SCALES):
    """Per-class: compare full-only vs tile-only IoU on A; True if tiles win."""
    inter_f = np.zeros(NUM_CLASSES, np.float64)
    union_f = np.zeros(NUM_CLASSES, np.float64)
    inter_t = np.zeros(NUM_CLASSES, np.float64)
    union_t = np.zeros(NUM_CLASSES, np.float64)
    for stem, ip, lp in items_A:
        full_z = load_full_logits(full_dir, stem)
        zs = load_scale_logits(logits_dir, stem, scales)
        if full_z is None or zs is None:
            continue
        mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
        update_inter_union(inter_f, union_f, full_z.argmax(0).astype(np.uint8), mask)
        update_inter_union(inter_t, union_t, fuse_equal(zs).argmax(0).astype(np.uint8), mask)
    _, iou_f = iou_from_iu(inter_f, union_f)
    _, iou_t = iou_from_iu(inter_t, union_t)
    source_is_tile = [bool(iou_t[c] > iou_f[c]) for c in range(NUM_CLASSES)]
    return source_is_tile, iou_f, iou_t


def slim_split(d):
    out = dict(d)
    if "per_image_miou" in out:
        arr = np.asarray(out["per_image_miou"], np.float64)
        out["per_image_miou_mean"] = float(np.nanmean(arr)) if arr.size else None
        out.pop("per_image_miou", None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=r"/mnt/d/LuSNAR_Seg")
    ap.add_argument("--checkpoint", default=r"/mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/best.pth")
    ap.add_argument("--list-A", default=r"/mnt/d/LuSNAR_Seg/data/lists_portable/scene5_ab/scene5_A.txt")
    ap.add_argument("--list-B", default=r"/mnt/d/LuSNAR_Seg/data/lists_portable/scene5_ab/scene5_B.txt")
    ap.add_argument("--list-all", default=r"/root/lusnar_data/lists/val_scene5.txt")
    ap.add_argument("--logits-dir", default=r"/root/lusnar_data/cache/scene5_scale_logits")
    ap.add_argument("--full-logits-dir", default=r"/root/lusnar_data/cache/scene5_full1024_logits")
    ap.add_argument("--out-json", default=r"/mnt/d/LuSNAR_Seg/results/inference_phase/protocols.json")
    ap.add_argument("--infer-h", type=int, default=1024)
    ap.add_argument("--infer-w", type=int, default=1024)
    ap.add_argument("--skip-full-if-cached", action="store_true")
    args = ap.parse_args()
    args.infer_hw = (args.infer_h, args.infer_w)

    items_A = load_stems(Path(args.list_A))
    items_B = load_stems(Path(args.list_B))
    items_all = load_stems(Path(args.list_all))
    print("I4_PREREGISTERED merge variants", MERGE_VARIANTS, "default=M1 large=full small=tiles", flush=True)
    print(
        "I4 note: official MoonSeg baseline uses whole-image 1024; "
        "this ROW+BLV model was trained at crop 512 (resolution mismatch accepted).",
        flush=True,
    )

    full_dir = Path(args.full_logits_dir)
    n_cached = len(list(full_dir.glob("*.npy"))) if full_dir.exists() else 0
    need = len(items_all)
    if args.skip_full_if_cached and n_cached >= need:
        print(f"[I4] reuse cached full logits n={n_cached}", flush=True)
        full_only_all = None
        used_hw = [args.infer_h, args.infer_w]
        oom = None
        t_full = 0.0
        n_fwd = n_cached
    else:
        print(f"[I4] full-image forwards n={need} infer_hw={args.infer_hw}", flush=True)
        t0 = time.time()
        full_only_all, used_hw, oom = run_full_image_cache(items_all, args)
        t_full = time.time() - t0
        n_fwd = int(full_only_all.get("forward_windows") or full_only_all.get("n") or need)
        print(f"[I4] full-image done wall={t_full:.1f}s used_hw={used_hw} oom={oom}", flush=True)

    # Full-only on A (diagnostic, not a competing frozen candidate by itself)
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    per_img = []
    for stem, ip, lp in items_A:
        z = load_full_logits(full_dir, stem)
        if z is None:
            continue
        mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
        pred = z.argmax(0).astype(np.uint8)
        update_inter_union(inter, union, pred, mask)
        _, mi = per_image_iou(pred, mask)
        per_img.append(mi)
    miou_f, ious_f = iou_from_iu(inter, union)
    full_only_A = {
        "mIoU": miou_f,
        "per_class": {CLASS_NAMES[c]: ious_f[c] for c in range(NUM_CLASSES)},
        "per_class_en": {EN[c]: ious_f[c] for c in range(NUM_CLASSES)},
        "per_image_miou": per_img,
        "n": len(per_img),
        "note": "diagnostic only: native/1024 full-image, no tiles; records 512-train vs 1024-infer mismatch",
    }
    print(
        f"[I4] full-only A mIoU={miou_f:.4f} rock={ious_f[1]:.4f} crater={ious_f[2]:.4f} mountain={ious_f[3]:.4f}",
        flush=True,
    )

    source_is_tile, iou_full_A, iou_tile_A = pick_m4_sources(items_A, full_dir, Path(args.logits_dir))
    print(f"[I4] M4 source_is_tile={source_is_tile} (True=tiles win that class on A)", flush=True)

    variant_A = {}
    for v in MERGE_VARIANTS:
        src = source_is_tile if v.startswith("M4") else None
        variant_A[v] = eval_variant(items_A, full_dir, Path(args.logits_dir), v, src)
        row = variant_A[v]
        print(
            f"[I4] {v} A mIoU={row['mIoU']:.4f} rock={row['per_class_en']['rock']:.4f} "
            f"crater={row['per_class_en']['crater']:.4f} mountain={row['per_class_en']['mountain']:.4f}",
            flush=True,
        )

    # Load C0 A for rock/crater guard if available
    full_proto = Path(args.out_json).with_name("protocols_cpu_full.json")
    c0_A = None
    if full_proto.exists():
        proto = json.loads(full_proto.read_text(encoding="utf-8"))
        c0_A = proto["candidates"]["C0_equal_tta"]["A"]
    else:
        proto = {"candidates": {}, "paired_tests_vs_C0_on_A": {}}

    def rock_crater_ok(cand_A):
        if c0_A is None:
            return True
        r0 = c0_A["per_class_en"]["rock"]
        cr0 = c0_A["per_class_en"]["crater"]
        return (cand_A["per_class_en"]["rock"] >= r0 - 0.01) and (
            cand_A["per_class_en"]["crater"] >= cr0 - 0.01
        )

    # Default intended merge is M1; if another variant is better on A and passes the
    # rock/crater guard, freeze that variant as I4's internal rule (A only).
    ranked = sorted(variant_A.items(), key=lambda kv: kv[1]["mIoU"], reverse=True)
    chosen = "M1_tile_small_overwrite"
    if rock_crater_ok(variant_A[chosen]) is False:
        # still keep M1 as fallback unless a legal variant beats it
        pass
    for name, res in ranked:
        if rock_crater_ok(res):
            chosen = name
            break
    else:
        chosen = "M1_tile_small_overwrite"
    print(f"[I4] chosen merge on A: {chosen} mIoU={variant_A[chosen]['mIoU']:.4f}", flush=True)

    src = source_is_tile if chosen.startswith("M4") else None
    chosen_B = eval_variant(items_B, full_dir, Path(args.logits_dir), chosen, src)
    print(
        f"[I4] B once ({chosen}) mIoU={chosen_B['mIoU']:.4f} "
        f"rock={chosen_B['per_class_en']['rock']:.4f} crater={chosen_B['per_class_en']['crater']:.4f} "
        f"mountain={chosen_B['per_class_en']['mountain']:.4f}",
        flush=True,
    )

    ms_full = (1000.0 * t_full / max(n_fwd, 1)) if n_fwd else None
    i4 = {
        "A": variant_A[chosen],
        "B": chosen_B,
        "merge_rule": chosen,
        "merge_default_intended": "M1_tile_small_overwrite: large(full 1024) + small(tiles C0)",
        "merge_variants_A": {k: slim_split(v) for k, v in variant_A.items()},
        "m4_source_is_tile": source_is_tile,
        "m4_full_iou_A": list(map(float, iou_full_A)),
        "m4_tile_iou_A": list(map(float, iou_tile_A)),
        "full_only_A": slim_split(full_only_A),
        "full_image": {
            "infer_hw_requested": [args.infer_h, args.infer_w],
            "infer_hw_used": used_hw,
            "native_scene5": [1024, 1024],
            "train_crop": 512,
            "resolution_mismatch": True,
            "oom_fallback": oom,
            "n_forwards_all": n_fwd,
            "wall_s_all": round(t_full, 2) if t_full else None,
            "ms_per_img_full_forward": None if ms_full is None else round(ms_full, 2),
            "logits_dir": str(full_dir),
        },
        "tile_source": "C0_equal_tta cached logits (scales 0.75/1.0/1.25, window 512, stride 256)",
        "cost": {
            "extra_forwards_per_image": 1,
            "ms_per_img_full_forward": None if ms_full is None else round(ms_full, 2),
            "ms_per_img_merge_A": variant_A[chosen].get("ms_per_img"),
            "note": "Tiles reused from cache (0 new tiled forwards). I4 adds one 1024 full-image forward per image.",
        },
    }
    if c0_A is not None:
        i4_test = paired_tests(c0_A["per_image_miou"], variant_A[chosen]["per_image_miou"])
    else:
        i4_test = None

    proto.setdefault("candidates", {})
    proto["candidates"]["I4_hybrid_full_tiles"] = i4
    proto.setdefault("paired_tests_vs_C0_on_A", {})
    if i4_test is not None:
        proto["paired_tests_vs_C0_on_A"]["I4_hybrid_full_tiles"] = i4_test
        keys = list(proto["paired_tests_vs_C0_on_A"].keys())
        pvals = [proto["paired_tests_vs_C0_on_A"][k].get("wilcoxon_p") or 1.0 for k in keys]
        holm = holm_correct(pvals)
        for i, k in enumerate(keys):
            proto["paired_tests_vs_C0_on_A"][k]["holm_p"] = holm[i]
    prereg = proto.get("preregistered") or []
    if "I4_hybrid_full_tiles" not in prereg:
        prereg.append("I4_hybrid_full_tiles")
    proto["preregistered"] = prereg
    proto["i4_note"] = (
        "Added as a 9th pre-registered candidate before B scores were used for selection. "
        "Internal merge variant chosen on A only; B reported once for the chosen rule."
    )

    full_path = Path(args.out_json).with_name("protocols_cpu_full.json")
    # Keep existing per-image arrays; write full then slim
    full_path.write_text(json.dumps(proto, ensure_ascii=False, indent=2), encoding="utf-8")
    slim = json.loads(json.dumps(proto))
    for cand in slim.get("candidates", {}).values():
        for split, val in list(cand.items()):
            if isinstance(val, dict) and "per_image_miou" in val:
                arr = np.asarray(val["per_image_miou"], np.float64)
                val["per_image_miou_mean"] = float(np.nanmean(arr)) if arr.size else None
                val.pop("per_image_miou", None)
    Path(args.out_json).write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    print("I4_DONE", args.out_json, "merge", chosen, flush=True)


if __name__ == "__main__":
    main()
