#!/usr/bin/env python3
"""Scene5 A/B split (seed 42, ~50/50 by image). Selection only on A; B report-once."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-list", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lines = [ln.strip() for ln in args.val_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rng = random.Random(args.seed)
    order = list(range(len(lines)))
    rng.shuffle(order)
    mid = len(order) // 2
    a_idx = sorted(order[:mid])
    b_idx = sorted(order[mid:])
    # if odd, B gets the extra one so A is the smaller/equal select set
    if len(order) % 2 == 1 and len(b_idx) < len(a_idx):
        b_idx, a_idx = a_idx, b_idx

    args.out_dir.mkdir(parents=True, exist_ok=True)
    a_lines = [lines[i] for i in a_idx]
    b_lines = [lines[i] for i in b_idx]
    (args.out_dir / "scene5_A.txt").write_text("\n".join(a_lines) + "\n", encoding="utf-8")
    (args.out_dir / "scene5_B.txt").write_text("\n".join(b_lines) + "\n", encoding="utf-8")
    meta = {
        "seed": args.seed,
        "source": str(args.val_list),
        "n_total": len(lines),
        "n_A": len(a_lines),
        "n_B": len(b_lines),
        "rule": "shuffle indices with Random(seed); first half -> A (select), rest -> B (report-once)",
        "A_stems": [Path(l.split("\t")[0]).stem for l in a_lines],
        "B_stems": [Path(l.split("\t")[0]).stem for l in b_lines],
    }
    (args.out_dir / "split_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"SPLIT_OK A={len(a_lines)} B={len(b_lines)} -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
