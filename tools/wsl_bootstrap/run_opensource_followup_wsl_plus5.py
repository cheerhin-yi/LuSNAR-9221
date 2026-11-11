#!/usr/bin/env python3
"""Wait for the live plus5 train to finish, then score it. NEVER launches training."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/d/LuSNAR_Seg")
PY = Path("/opt/miniconda3/envs/lusnar/bin/python")
WORK_B = ROOT / "work_dirs" / "train_ROW_BLV_20k_plus5"
CHAIN_LOG = ROOT / "logs" / "opensource_followup_plus5.log"
ONESHOT_B = ROOT / "results" / "oneshot_ROW_BLV_plus5"
ONESHOT_A = ROOT / "results" / "oneshot_ROW_BLV"
FINAL_B = ROOT / "results" / "opensource_final_ROW_BLV_plus5"


def log(msg: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    print(line, flush=True)
    CHAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with CHAIN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def train_alive(run_name: str) -> bool:
    needle = f"--run-name {run_name}"
    try:
        out = subprocess.check_output(["ps", "-ef"], text=True, errors="replace")
    except Exception:
        return False
    return any(needle in line and "lusnar_row_blv_ddp" in line for line in out.splitlines())


def plus5_ckpt() -> Path | None:
    last = WORK_B / "last.pth"
    it = WORK_B / "iter_20000.pth"
    if last.is_file() and last.stat().st_size > 0:
        return last
    if it.is_file() and it.stat().st_size > 0:
        return it
    return None


def plus5_ready() -> bool:
    return (not train_alive("train_ROW_BLV_20k_plus5")) and plus5_ckpt() is not None


def wait_until(pred, what: str, poll_s: int = 60) -> None:
    while not pred():
        log(f"waiting: {what}")
        time.sleep(poll_s)


def run_oneshot(checkpoint: Path, tag: str) -> None:
    cmd = [
        str(PY),
        "-u",
        str(ROOT / "tools" / "portable" / "lusnar_oneshot_opensource.py"),
        "--checkpoint",
        str(checkpoint),
        "--tag",
        tag,
        "--project-root",
        str(ROOT),
        "--device",
        "cuda:0",
    ]
    log("start oneshot " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def run_protocol_eval(checkpoint: Path) -> None:
    cmd = [
        str(PY),
        "-u",
        str(ROOT / "tools" / "portable" / "lusnar_opensource_final_eval.py"),
        "--checkpoint",
        str(checkpoint),
        "--out-dir",
        str(FINAL_B),
        "--project-root",
        str(ROOT),
        "--gpus",
        "2",
    ]
    log("start protocol eval " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def write_compare() -> None:
    a_path = ONESHOT_A / "summary.json"
    b_path = ONESHOT_B / "summary.json"
    if not a_path.is_file() or not b_path.is_file():
        log("skip compare: missing oneshot summary")
        return
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))
    da = a["overall"]["mIoU"]
    db = b["overall"]["mIoU"]
    lines = [
        "# ROW+BLV opensource test compare (non-official, one-shot)",
        "",
        "Both runs use the locked oneshot protocol (equal TTA). Moon_2/4/9 were not used to train.",
        "",
        "| run | train scenes | ckpt | overall mIoU | macro | weighted |",
        "|---|---|---|---:|---:|---:|",
        f"| no Moon_5 | 1,3,6,7,8 | Scene5 best.pth | {da:.4f} | {a['macro_scene_mIoU']:.4f} | {a['weighted_scene_mIoU']:.4f} |",
        f"| plus Moon_5 | 1,3,5,6,7,8 | plus5 final | {db:.4f} | {b['macro_scene_mIoU']:.4f} | {b['weighted_scene_mIoU']:.4f} |",
        "",
        f"Moon_5 delta overall mIoU {db - da:+.4f}",
        "",
        "per-scene mIoU:",
        "",
        "| scene | no Moon_5 | plus Moon_5 | delta |",
        "|---|---:|---:|---:|",
    ]
    for scene in ("Moon_2", "Moon_4", "Moon_9"):
        xa = a["per_scene"][scene]["mIoU"]
        xb = b["per_scene"][scene]["mIoU"]
        lines.append(f"| {scene} | {xa:.4f} | {xb:.4f} | {xb - xa:+.4f} |")
    lines.append("")
    out = ROOT / "results" / "oneshot_ROW_BLV_compare.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote {out}")


def main() -> None:
    if not PY.is_file():
        raise SystemExit(f"missing python {PY}")
    log("plus5 eval waiter started; will not launch any training")
    wait_until(plus5_ready, "plus5 to finish with last.pth or iter_20000.pth")
    ckpt = plus5_ckpt()
    if ckpt is None:
        raise RuntimeError("plus5 finished but no last.pth or iter_20000.pth")
    log(f"plus5 finished; scoring {ckpt}")
    if not (ONESHOT_B / "summary.json").is_file():
        run_oneshot(ckpt, "ROW_BLV_plus5")
    else:
        log("oneshot_ROW_BLV_plus5 summary already present; skip oneshot")
    if not (FINAL_B / "report.md").is_file():
        run_protocol_eval(ckpt)
    else:
        log("opensource_final_ROW_BLV_plus5 report already present; skip protocol eval")
    write_compare()
    log("PLUS5_EVAL_DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        log("FOLLOWUP_FAILED\n" + traceback.format_exc())
        sys.exit(1)
