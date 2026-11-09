#!/usr/bin/env python3
"""Prepare smoke subsets / rarity / CC bank with relative-path aware roots.

Standalone entry so it can be run before lusnar_smoke_compare.py on a fresh box.
Never reads Moon_2/4/9 labels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))

from _lusnar_common import (  # noqa: E402
    resolve_data_root,
    resolve_project_root,
    resolve_scene_root,
)


def main():
    ap = argparse.ArgumentParser(description="Prepare LuSNAR smoke assets")
    ap.add_argument("--project-root", default=None)
    ap.add_argument(
        "--data-root",
        default=None,
        help="folder with train/Moon_* OR Moon_* directly (cheerhin LuSNAR_train)",
    )
    ap.add_argument("--train-subdir", default="train")
    ap.add_argument("--smoke-dir", default=None, help="default <project>/data/smoke")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    project = resolve_project_root(SCRIPT, args.project_root)
    data_root = resolve_data_root(project, args.data_root)
    scene_root = resolve_scene_root(data_root, args.train_subdir)
    smoke_dir = (
        Path(args.smoke_dir).expanduser().resolve()
        if args.smoke_dir
        else (project / "data" / "smoke")
    )
    if not scene_root.is_dir():
        raise SystemExit(f"scene root not found: {scene_root}")

    # reuse prepare implementation from smoke script
    from lusnar_smoke_compare import prepare_smoke_assets

    print(
        {
            "project": str(project),
            "data_root": str(data_root),
            "scene_root": str(scene_root),
            "smoke_dir": str(smoke_dir),
        },
        flush=True,
    )
    prepare_smoke_assets(scene_root, smoke_dir, force=args.force)


if __name__ == "__main__":
    main()
