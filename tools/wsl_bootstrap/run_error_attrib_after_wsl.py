#!/usr/bin/env python3
"""WSL variant of run_error_attrib_after.py."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/d/LuSNAR_Seg")
PY = Path("/opt/miniconda3/envs/lusnar/bin/python")
NEED = [
    ROOT / "results" / "oneshot_ROW_BLV" / "summary.json",
    ROOT / "results" / "oneshot_ROW_BLV_plus5" / "summary.json",
    ROOT / "results" / "oneshot_ROW_BLV" / "preds" / "Moon_9",
    ROOT / "results" / "oneshot_ROW_BLV_plus5" / "preds" / "Moon_9",
]
FOLLOW = ROOT / "logs" / "opensource_followup.log"


def log(msg: str):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    print(line, flush=True)


def follow_failed() -> bool:
    if not FOLLOW.exists():
        return False
    text = FOLLOW.read_text(encoding="utf-8", errors="replace")
    return "FOLLOWUP_FAILED" in text and not all(p.exists() for p in NEED)


def main():
    if not PY.is_file():
        raise SystemExit(f"missing python {PY}")
    log("attrib waiter started (WSL); gates locked in lusnar_error_attrib.py")
    while not all(p.exists() for p in NEED):
        if follow_failed():
            log("ATTRIB_FAILED follow-up failed before both one-shots existed")
            return 1
        log("waiting: both one-shot summaries and Moon_9 preds")
        time.sleep(60)
    log("one-shots present; start attribution")
    subprocess.check_call(
        [str(PY), "-u", str(ROOT / "tools" / "portable" / "lusnar_error_attrib.py"), str(ROOT)],
        cwd=str(ROOT),
    )
    done = ROOT / "results" / "error_attrib.md"
    if not done.exists():
        log("ATTRIB_FAILED report missing")
        return 1
    log("ATTRIB_DONE " + str(done))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        log("ATTRIB_FAILED\n" + traceback.format_exc())
        sys.exit(1)