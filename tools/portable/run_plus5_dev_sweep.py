#!/usr/bin/env python3
"""Score every plus5 iter_*.pth on Moon_2/4/9 and pick the highest S.

User confirmed on 2026-11-11 that the dev set may be used to choose the
plus5 checkpoint. Protocols stay frozen (single-scale, C0, I1a).
Selection rule, fixed before the scores exist:
  candidate S = max S among single_scale, C0_equal_tta, I1a_best_scale_per_class
  designated = highest candidate S; ties go to higher overall mIoU, then earlier iter.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\LuSNAR_Seg")
PY = Path(r"F:\Conda\envs\lusnar\python.exe")
CKPT_DIR = ROOT / "work_dirs" / "train_ROW_BLV_20k_plus5"
OUT = ROOT / "results" / "opensource_plus5_sweep"
EVAL = ROOT / "tools" / "portable" / "lusnar_opensource_final_eval.py"
PROTOCOLS = ("single_scale", "C0_equal_tta", "I1a_best_scale_per_class")


def checkpoints():
    found = []
    for path in CKPT_DIR.glob("iter_*.pth"):
        stem = path.stem.replace("iter_", "")
        if stem.isdigit():
            found.append((int(stem), path))
    return [p for _, p in sorted(found)]


def done(out_dir: Path) -> bool:
    summary = out_dir / "summary.json"
    if not summary.exists():
        return False
    obj = json.loads(summary.read_text(encoding="utf-8"))
    return int(obj.get("n_images") or 0) >= 5004 and "protocols" in obj


def row_from(summary_path: Path) -> dict:
    obj = json.loads(summary_path.read_text(encoding="utf-8"))
    protocols = obj["protocols"]
    best_name = None
    best_s = -1.0
    best_miou = -1.0
    packed = {}
    for name in PROTOCOLS:
        slot = protocols[name]
        s = float(slot["S"])
        miou = float(slot["overall"]["mIoU"])
        packed[name] = {"S": s, "mIoU": miou, "m2": slot["m2"], "m4": slot["m4"], "m9": slot["m9"]}
        if s > best_s or (s == best_s and miou > best_miou):
            best_s = s
            best_miou = miou
            best_name = name
    return {
        "iter": obj.get("ckpt_iter"),
        "checkpoint": obj["checkpoint"],
        "best_protocol": best_name,
        "best_S": best_s,
        "best_mIoU": best_miou,
        "protocols": packed,
    }


def write_compare(rows: list[dict]):
    ranked = sorted(rows, key=lambda r: (-r["best_S"], -r["best_mIoU"], r["iter"]))
    winner = ranked[0]
    payload = {
        "selection_rule": "max S over single_scale, C0_equal_tta, I1a_best_scale_per_class; tie -> higher mIoU, then earlier iter",
        "dev_set": "Moon_2/4/9",
        "designated": winner,
        "rows": ranked,
    }
    (OUT / "compare.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# plus5 checkpoint sweep on Moon_2/4/9",
        "",
        "开发集被用来选择 plus5 检查点。这是 2026-11-11 确认过的。",
        "S = (Moon_2 + Moon_4 + 2×Moon_9) / 4。每个检查点取单尺度、C0、I1a 里最高的 S。",
        "并列时先看该协议的全图像素 mIoU，再取更早的 iter。",
        "",
        f"选定：iter {winner['iter']}，协议 {winner['best_protocol']}，S {winner['best_S']:.4f}。",
        "",
        "| iter | 最高协议 | S | mIoU | 单尺度 S | C0 S | I1a S |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in ranked:
        p = r["protocols"]
        lines.append(
            f"| {r['iter']} | {r['best_protocol']} | {r['best_S']:.4f} | {r['best_mIoU']:.4f} | "
            f"{p['single_scale']['S']:.4f} | {p['C0_equal_tta']['S']:.4f} | {p['I1a_best_scale_per_class']['S']:.4f} |"
        )
    lines.append("")
    (OUT / "compare.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    ckpts = checkpoints()
    print(f"SWEEP_START n={len(ckpts)}", flush=True)
    for path in ckpts:
        out_dir = OUT / path.stem
        if done(out_dir):
            print(f"SKIP {path.name}", flush=True)
        else:
            print(f"RUN {path.name}", flush=True)
            subprocess.check_call(
                [
                    str(PY),
                    "-u",
                    str(EVAL),
                    "--project-root",
                    str(ROOT),
                    "--checkpoint",
                    str(path),
                    "--out-dir",
                    str(out_dir),
                    "--data-root",
                    str(ROOT / "data"),
                    "--gpus",
                    "2",
                    "--ref-pred-root",
                    str(OUT / "_no_ref"),
                ],
                cwd=str(ROOT),
            )
        rows.append(row_from(out_dir / "summary.json"))
        write_compare(rows)
    print("PLUS5_SWEEP_DONE", OUT / "compare.md", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        print("PLUS5_SWEEP_FAILED\n" + traceback.format_exc(), flush=True)
        sys.exit(1)
