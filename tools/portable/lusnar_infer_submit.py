"""LuSNAR official-style submission inference (sliding-window, no test GT).

Reads color PNGs only under data/test/Moon_{2,4,9}/color.
Writes same-basename 8-bit single-channel PNGs with pixel values 0-4.
Never opens label / GT paths.
"""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from _lusnar_common import (
    MEAN,
    STD,
    NUM_CLASSES,
    build_segformer_b2,
    ensure_mmcv_stub,
    find_pretrain,
    resolve_project_root,
)


def imread_bgr(path: str | Path) -> np.ndarray:
    """Match LuSNARDataset._imread_bgr: PIL RGB -> BGR, cv2 fallback."""
    path = str(path)
    try:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
        return arr[:, :, ::-1].copy()
    except Exception:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read image: {path}")
        return img


def preprocess_bgr(img_bgr: np.ndarray) -> torch.Tensor:
    """MEAN/STD on BGR uint8 float32 (same quirk as training). CHW float32."""
    img_f = (img_bgr.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(img_f.transpose(2, 0, 1)).float()


def sliding_windows(h: int, w: int, tile: int, stride: int):
    """Yield (y0, x0, y1, x1) covering the full image; last tiles clamp to edge."""
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if not ys or ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    if not xs or xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    # unique while preserving order
    ys = list(dict.fromkeys(ys))
    xs = list(dict.fromkeys(xs))
    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            # clamp start if tile would overhang (pad later)
            y0c = max(0, y1 - tile) if (y1 - y0) < tile and h >= tile else y0
            x0c = max(0, x1 - tile) if (x1 - x0) < tile and w >= tile else x0
            yield y0c, x0c, min(y0c + tile, h), min(x0c + tile, w)


@torch.no_grad()
def infer_image(
    model,
    img_bgr: np.ndarray,
    device: torch.device,
    tile: int,
    stride: int,
    amp_dtype,
    amp_enabled: bool,
    tile_batch: int = 16,
) -> np.ndarray:
    """Sliding-window logits average -> argmax. Returns HxW uint8 class map."""
    h, w = img_bgr.shape[:2]
    tensor = preprocess_bgr(img_bgr)  # 3,H,W
    logit_sum = torch.zeros((NUM_CLASSES, h, w), dtype=torch.float32, device=device)
    count = torch.zeros((1, h, w), dtype=torch.float32, device=device)

    windows = list(sliding_windows(h, w, tile, stride))
    # pad small images to tile if needed
    for i0 in range(0, len(windows), tile_batch):
        batch_wins = windows[i0 : i0 + tile_batch]
        tiles = []
        for y0, x0, y1, x1 in batch_wins:
            patch = tensor[:, y0:y1, x0:x1]
            ph, pw = patch.shape[1], patch.shape[2]
            if ph != tile or pw != tile:
                pad = torch.zeros((3, tile, tile), dtype=patch.dtype)
                pad[:, :ph, :pw] = patch
                patch = pad
            tiles.append(patch)
        batch = torch.stack(tiles, dim=0).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            feats = model.extract_feat(batch)
            logits = model.decode_head.forward(feats)
            if logits.shape[-2:] != (tile, tile):
                logits = F.interpolate(
                    logits, size=(tile, tile), mode="bilinear", align_corners=False
                )
        logits = logits.float()
        for j, (y0, x0, y1, x1) in enumerate(batch_wins):
            th, tw = y1 - y0, x1 - x0
            logit_sum[:, y0:y1, x0:x1] += logits[j, :, :th, :tw]
            count[:, y0:y1, x0:x1] += 1.0

    avg = logit_sum / count.clamp_min(1.0)
    pred = avg.argmax(0).cpu().numpy().astype(np.uint8)
    return pred


def load_model(checkpoint: Path, project: Path, device: torch.device):
    ensure_mmcv_stub(Path(__file__))
    pretrain = find_pretrain(project)
    model = build_segformer_b2(pretrain)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        meta = {k: ckpt.get(k) for k in ("iter", "miou", "recipe") if k in ckpt}
    else:
        state = ckpt
        meta = {}
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, meta, missing, unexpected


def list_color_images(color_dir: Path):
    return sorted(color_dir.glob("*.png"))


def write_png_uint8(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    # single-channel 8-bit; use PIL to avoid palette surprises
    Image.fromarray(arr, mode="L").save(str(path))


def zip_submit(out_dir: Path, zip_path: Path, scenes: list[int]):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for sid in scenes:
            scene = f"Moon_{sid}"
            sdir = out_dir / scene
            for p in sorted(sdir.glob("*.png")):
                zf.write(p, arcname=f"{scene}/{p.name}")


def parse_args():
    ap = argparse.ArgumentParser(description="LuSNAR sliding-window submit inference")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--test-subdir", default="test")
    ap.add_argument("--scenes", default="2,4,9")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16, help="tile batch size")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--limit", type=int, default=0, help="per-scene image limit (smoke)")
    ap.add_argument("--zip", action="store_true", help="also write submit zip")
    ap.add_argument("--manifest", default=None, help="manifest json path")
    return ap.parse_args()


def main():
    args = parse_args()
    script = Path(__file__).resolve()
    project = resolve_project_root(script, args.project_root)
    data_root = (
        Path(args.data_root).expanduser().resolve()
        if args.data_root
        else (project / "data").resolve()
    )
    test_root = (data_root / args.test_subdir).resolve()
    scenes = [int(x) for x in args.scenes.split(",") if x.strip()]
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if args.amp == "bf16":
        amp_dtype, amp_enabled = torch.bfloat16, device.type == "cuda"
    elif args.amp == "fp16":
        amp_dtype, amp_enabled = torch.float16, device.type == "cuda"
    else:
        amp_dtype, amp_enabled = torch.float32, False

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    print(
        {
            "checkpoint": str(ckpt_path),
            "test_root": str(test_root),
            "out_dir": str(out_dir),
            "scenes": scenes,
            "tile": args.tile,
            "stride": args.stride,
            "batch": args.batch,
            "device": str(device),
            "amp": args.amp,
            "limit": args.limit,
            "note": "color-only; no GT/label paths opened",
        },
        flush=True,
    )

    t0 = time.time()
    model, meta, missing, unexpected = load_model(ckpt_path, project, device)
    print(
        {
            "ckpt_meta": meta,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "load_s": round(time.time() - t0, 2),
        },
        flush=True,
    )

    counts = {}
    smoke_checks = []
    total = 0
    errors = []

    for sid in scenes:
        color_dir = test_root / f"Moon_{sid}" / "color"
        # NEVER touch label dir
        if not color_dir.is_dir():
            raise FileNotFoundError(color_dir)
        images = list_color_images(color_dir)
        if args.limit > 0:
            images = images[: args.limit]
        scene_out = out_dir / f"Moon_{sid}"
        scene_out.mkdir(parents=True, exist_ok=True)
        n_ok = 0
        for i, ip in enumerate(images):
            try:
                img = imread_bgr(ip)
                pred = infer_image(
                    model,
                    img,
                    device,
                    args.tile,
                    args.stride,
                    amp_dtype,
                    amp_enabled,
                    tile_batch=args.batch,
                )
                if pred.shape != img.shape[:2]:
                    raise RuntimeError(
                        f"shape mismatch pred={pred.shape} img={img.shape[:2]}"
                    )
                if pred.dtype != np.uint8:
                    pred = pred.astype(np.uint8)
                uniq = sorted(int(x) for x in np.unique(pred))
                if any(u < 0 or u > 4 for u in uniq):
                    raise RuntimeError(f"bad class ids {uniq} in {ip.name}")
                out_p = scene_out / ip.name
                write_png_uint8(out_p, pred)
                n_ok += 1
                total += 1
                if len(smoke_checks) < 6:
                    smoke_checks.append(
                        {
                            "scene": sid,
                            "name": ip.name,
                            "shape": list(pred.shape),
                            "dtype": str(pred.dtype),
                            "unique": uniq,
                        }
                    )
                if (i + 1) % 50 == 0 or (i + 1) == len(images):
                    elapsed = time.time() - t0
                    rate = total / max(elapsed, 1e-6)
                    print(
                        {
                            "scene": sid,
                            "done": i + 1,
                            "scene_total": len(images),
                            "global_done": total,
                            "elapsed_s": round(elapsed, 1),
                            "img_per_s": round(rate, 3),
                            "last_unique": uniq,
                        },
                        flush=True,
                    )
            except Exception as e:
                err = {"scene": sid, "file": ip.name, "error": str(e)}
                errors.append(err)
                print(err, flush=True)
                raise
        counts[f"Moon_{sid}"] = n_ok

    wall = time.time() - t0
    manifest = {
        "checkpoint": str(ckpt_path),
        "ckpt_meta": meta,
        "method": f"slide{args.tile}/stride{args.stride}_avg_logits",
        "amp": args.amp,
        "device": str(device),
        "tile_batch": args.batch,
        "counts": counts,
        "total_images": total,
        "wall_sec": round(wall, 1),
        "smoke_checks": smoke_checks,
        "errors": errors,
        "note": "No test GT/label files were opened. Official score from organizer only.",
        "official_baseline_reminder": {
            "test_weighted_mIoU": 0.8046,
            "Moon_9": 0.7186,
            "crater": 0.3096,
        },
        "out_dir": str(out_dir),
    }
    man_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else out_dir / "manifest.json"
    )
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({"manifest": str(man_path), "wall_sec": round(wall, 1), "counts": counts}, flush=True)

    if args.zip:
        zip_path = out_dir.parent / f"{out_dir.name}.zip"
        zip_submit(out_dir, zip_path, scenes)
        print({"zip": str(zip_path), "size_mb": round(zip_path.stat().st_size / 1e6, 2)}, flush=True)
        manifest["zip"] = str(zip_path)
        man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
