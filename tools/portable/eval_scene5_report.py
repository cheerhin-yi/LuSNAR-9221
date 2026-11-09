#!/usr/bin/env python3
"""Evaluate a checkpoint on Scene5 and write a markdown/json report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
from _lusnar_common import (  # noqa: E402
    CLASS_NAMES,
    build_segformer_b2,
    build_split_lists,
    ensure_mmcv_stub,
    evaluate,
    find_pretrain,
    jfloat,
    resolve_data_root,
    resolve_project_root,
    train_dir,
    write_markdown_table,
)
from lusnar_train_ddp import LuSNARDataset  # noqa: E402

ensure_mmcv_stub(SCRIPT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--train-subdir", default="train")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pretrain", default=None, help="only needed if building arch; weights come from ckpt")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out-name", default="eval_scene5")
    args = ap.parse_args()

    project = resolve_project_root(SCRIPT, args.project_root)
    data_root = resolve_data_root(project, args.data_root)
    scene_root = train_dir(data_root, args.train_subdir)
    lists_dir = project / "data" / "lists_portable"
    _, val_file, _, n_va = build_split_lists(scene_root, lists_dir)
    pretrain = find_pretrain(project, args.pretrain)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_segformer_b2(pretrain).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    val_ds = LuSNARDataset(val_file, train=False, crop=args.crop)
    loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)
    miou, ious = evaluate(model, loader, device)
    summary = {
        "run_name": args.out_name,
        "exp": args.out_name,
        "best_mIoU_scene5": jfloat(miou),
        "per_class_IoU_best": {n: jfloat(v) for n, v in zip(CLASS_NAMES, ious)},
        "iou_rock": jfloat(ious[1]),
        "iou_crater": jfloat(ious[2]),
        "wall_sec": None,
        "peak_mem_mb": None,
        "n_val": n_va,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "note": "本地 Scene5 验证，非官方测试成绩",
        "error": None,
    }
    out_dir = project / "work_dirs" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_table(project / "results" / f"{args.out_name}.md", [summary], f"Eval {args.out_name}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
