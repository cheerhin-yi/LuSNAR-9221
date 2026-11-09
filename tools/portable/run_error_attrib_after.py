#!/usr/bin/env python3
"""Wait until both open-set one-shots exist, then write the error attribution.

Does not start training and does not read the attribution back into a run.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\LuSNAR_Seg")
PY = Path(r"F:\Conda\envs\lusnar\python.exe")
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
    log("attrib waiter started; gates already locked in lusnar_error_attrib.py")
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
