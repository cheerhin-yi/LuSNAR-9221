#!/usr/bin/env python3
"""Local supervisor for both 4090s. Logs status and state transitions.

Does not start, stop, or restart training. Does not create STOP files.
Does not official-submit. Writes:
  /mnt/d/LuSNAR_Seg/logs/gpu_watch_status.json
  /mnt/d/LuSNAR_Seg/logs/gpu_watch_events.jsonl
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path("/mnt/d/LuSNAR_Seg")
LOGDIR = ROOT / "logs"
FI_LOGDIR = ROOT / "fullimage_compare" / "logs"
STATUS = LOGDIR / "gpu_watch_status.json"
EVENTS = LOGDIR / "gpu_watch_events.jsonl"
INTERVAL = 60
TZ = timezone(timedelta(hours=8))

GPU0_QUEUE = "06_backbone_smoke_wsl.sh|06b_rerun_s1_wsl.sh"
GPU0_PY = "lusnar_smoke_compare.py"
GPU1_QUEUE = "05_fullimage_queue_wsl.sh"
GPU1_PY = "fullimage_smoke.py"
GPU0_DONE = "ALL BACKBONE S0/S1 DONE"
# queue script also says this when S0/S1 finish and M2F is not started
GPU0_DONE_ALT = "Mask2Former not started"
GPU1_DONE_MARKERS = (
    "ALL FULLIMAGE DONE",
    "ALL DONE",
    "queue complete",
    "QUEUE DONE",
)

ITER_RE = re.compile(r"\{['\"]iter['\"]:\s*(\d+)")
ITER_RE2 = re.compile(r"\biter(?:eration)?[=\s:]+(\d+)", re.I)
FAIL_RE = re.compile(
    r"(Traceback|CUDA out of memory|OutOfMemoryError|Killed|FAIL |RuntimeError|Error:)",
    re.I,
)


def now_str() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %z")


def run(cmd: list[str]) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return f"ERR {e}"


def pgrep(pattern: str) -> list[dict]:
    out = run(["pgrep", "-af", pattern])
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line or "pgrep" in line or "08_gpu_watch" in line:
            continue
        if pattern not in line and not re.search(pattern, line):
            continue
        pid, _, rest = line.partition(" ")
        if pid.isdigit():
            rows.append({"pid": int(pid), "cmd": rest})
    return rows


def nvidia() -> list[dict]:
    out = run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "util": int(float(parts[1])),
                "mem_mb": int(float(parts[2])),
                "total_mb": int(float(parts[3])),
            }
        )
    return gpus


def tail_text(path: Path, n: int = 80) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 120000), os.SEEK_SET)
            data = f.read().decode("utf-8", "replace")
        return "\n".join(data.splitlines()[-n:])
    except Exception:
        return ""


def last_iter(text: str) -> int | None:
    found = ITER_RE.findall(text) or ITER_RE2.findall(text)
    return int(found[-1]) if found else None


def has_fail(text: str) -> str | None:
    m = FAIL_RE.search(text)
    return m.group(1) if m else None


def newest_log(dirpath: Path, patterns: tuple[str, ...]) -> Path | None:
    if not dirpath.is_dir():
        return None
    cands = []
    for p in dirpath.iterdir():
        if p.is_file() and any(k in p.name for k in patterns):
            cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]


def classify(alive_queue: bool, alive_py: bool, done: bool, fail: str | None, stalled: bool) -> str:
    if done and not alive_py:
        return "done"
    if fail and not alive_py:
        return "crashed"
    if not alive_queue and not alive_py and not done:
        return "unexpected_stop"
    if stalled and (alive_queue or alive_py):
        return "stalled"
    if alive_py or alive_queue:
        return "running"
    return "idle"


def load_prev() -> dict:
    if STATUS.exists():
        try:
            return json.loads(STATUS.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def emit(event: str, detail: dict) -> None:
    rec = {"ts": now_str(), "event": event, **detail}
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def gpu0_stage(text: str) -> str:
    # last START line wins
    starts = re.findall(r"START (\S+)", text)
    if starts:
        return starts[-1]
    if "S1_B5_ROW_BLV" in text:
        return "S1_B5_ROW_BLV"
    if "S0_B2_TTA" in text:
        return "S0_B2_TTA"
    if "S0_B2_ROW_BLV" in text:
        return "S0_B2_ROW_BLV"
    return "unknown"


def snapshot() -> dict:
    gpus = {g["index"]: g for g in nvidia()}
    q0 = pgrep(GPU0_QUEUE)
    p0 = [x for x in pgrep(GPU0_PY) if "fullimage" not in x["cmd"]]
    q1 = pgrep(GPU1_QUEUE)
    p1 = pgrep(GPU1_PY)

    bb_log = LOGDIR / "backbone_smoke_wsl.log"
    bb_txt = tail_text(bb_log, 120)
    stage = gpu0_stage(bb_txt)
    stage_log = LOGDIR / f"{stage}_wsl.log"
    if not stage_log.exists():
        alt = newest_log(LOGDIR, ("S0_", "S1_", "M2F", "backbone"))
        stage_log = alt or stage_log
    s0_txt = tail_text(stage_log, 80)
    fi_q = FI_LOGDIR / "queue_DESKTOP-ORQLB8F-WSL.log"
    if not fi_q.exists():
        fi_q = newest_log(FI_LOGDIR, ("queue_",)) or fi_q
    fi_txt = tail_text(fi_q, 120)
    probe = FI_LOGDIR / "vram_probe_wsl.log"
    fi_stage_log = newest_log(FI_LOGDIR, ("_wsl.log", "probe"))
    fi_stage_txt = tail_text(fi_stage_log, 80) if fi_stage_log else ""

    merged0 = bb_txt + "\n" + s0_txt
    merged1 = fi_txt + "\n" + fi_stage_txt + "\n" + tail_text(probe, 40)

    done0 = (GPU0_DONE in bb_txt) or (GPU0_DONE_ALT in bb_txt and "ALL BACKBONE" in bb_txt)
    # only treat GPU1 as done if the queue log says so clearly
    done1 = any(m.lower() in fi_txt.lower() for m in GPU1_DONE_MARKERS) and (
        "done" in fi_txt.lower()
    )
    # narrower: look for explicit queue-finished lines written by our launcher
    done1 = bool(
        re.search(r"ALL FULLIMAGE DONE|QUEUE ALL DONE|fullimage queue ALL DONE", fi_txt, re.I)
    )

    it0 = last_iter(s0_txt) or last_iter(bb_txt)
    it1 = last_iter(fi_stage_txt) or last_iter(fi_txt)
    fail0 = has_fail(s0_txt[-4000:]) or has_fail(bb_txt[-4000:])
    fail1 = has_fail(fi_stage_txt[-4000:]) or has_fail(fi_txt[-4000:])

    prev = load_prev()
    now = time.time()

    def progress_ts(key: str, it, prev_it, prev_ts):
        if it is not None and it != prev_it:
            return now
        return prev_ts or now

    p0_prev = (prev.get("gpu0") or {})
    p1_prev = (prev.get("gpu1") or {})
    ts0 = progress_ts("gpu0", it0, p0_prev.get("last_iter"), p0_prev.get("progress_unix"))
    ts1 = progress_ts("gpu1", it1, p1_prev.get("last_iter"), p1_prev.get("progress_unix"))

    # stall thresholds: TTA/probe val can be long
    stall0_lim = 90 * 60 if ("TTA" in stage or "DRY" in stage) else 40 * 60
    stall1_lim = 90 * 60  # probe + whole-image val
    stalled0 = bool((q0 or p0) and not done0 and (now - ts0) > stall0_lim)
    stalled1 = bool((q1 or p1) and not done1 and (now - ts1) > stall1_lim)

    def pack(idx, queue, py, stage_name, it, fail, done, stalled, log_path, prog_unix):
        g = gpus.get(idx, {})
        state = classify(bool(queue), bool(py), done, fail, stalled)
        return {
            "index": idx,
            "util": g.get("util"),
            "mem_mb": g.get("mem_mb"),
            "queue_alive": bool(queue),
            "python_alive": bool(py),
            "queue_pids": [x["pid"] for x in queue[:4]],
            "python_pids": [x["pid"] for x in py[:4]],
            "stage": stage_name,
            "last_iter": it,
            "fail_hint": fail,
            "done": done,
            "stalled": stalled,
            "state": state,
            "log": str(log_path) if log_path else None,
            "progress_unix": prog_unix,
            "quiet_sec": int(now - prog_unix) if prog_unix else None,
        }

    fi_stage = "vram_probe"
    if "FI_" in fi_txt:
        ms = re.findall(r"(FI_[A-Z0-9_]+)", fi_txt)
        if ms:
            fi_stage = ms[-1]
    if done1:
        fi_stage = "all_done"

    snap = {
        "ts": now_str(),
        "unix": now,
        "gpu0": pack(0, q0, p0, stage, it0, fail0, done0, stalled0, stage_log, ts0),
        "gpu1": pack(
            1,
            q1,
            p1,
            fi_stage,
            it1,
            fail1,
            done1,
            stalled1,
            fi_stage_log or fi_q,
            ts1,
        ),
    }
    return snap, prev


def alerts_of(snap: dict) -> list[str]:
    out = []
    for key, label in (("gpu0", "GPU0 backbone"), ("gpu1", "GPU1 full-image")):
        g = snap[key]
        if g["state"] == "unexpected_stop":
            out.append(f"{label} stopped unexpectedly (stage={g.get('stage')}, last_iter={g.get('last_iter')})")
        elif g["state"] == "crashed":
            out.append(f"{label} crashed ({g.get('fail_hint')}, stage={g.get('stage')}, last_iter={g.get('last_iter')})")
        elif g["state"] == "stalled":
            out.append(f"{label} stalled for {g.get('quiet_sec')}s (stage={g.get('stage')}, last_iter={g.get('last_iter')})")
        elif g["state"] == "done":
            out.append(f"{label} finished (stage={g.get('stage')})")
    return out


def main() -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    emit("watch_start", {"interval_sec": INTERVAL})
    while True:
        try:
            snap, prev = snapshot()
            alerts = alerts_of(snap)
            snap["alerts"] = alerts
            # transitions vs previous
            events = []
            if prev:
                for key in ("gpu0", "gpu1"):
                    a, b = (prev.get(key) or {}).get("state"), snap[key]["state"]
                    if a != b:
                        events.append(
                            {
                                "event": "state_change",
                                "gpu": key,
                                "from": a,
                                "to": b,
                                "stage": snap[key].get("stage"),
                                "last_iter": snap[key].get("last_iter"),
                                "fail_hint": snap[key].get("fail_hint"),
                                "alert": next((x for x in alerts if key[-1] in x or key in x.lower()), None),
                            }
                        )
                    # stage change while still running
                    if (prev.get(key) or {}).get("stage") != snap[key].get("stage"):
                        events.append(
                            {
                                "event": "stage_change",
                                "gpu": key,
                                "from": (prev.get(key) or {}).get("stage"),
                                "to": snap[key].get("stage"),
                                "state": snap[key]["state"],
                            }
                        )
            else:
                emit("initial", {"gpu0": snap["gpu0"]["state"], "gpu1": snap["gpu1"]["state"],
                                 "gpu0_stage": snap["gpu0"]["stage"], "gpu1_stage": snap["gpu1"]["stage"]})
            for ev in events:
                emit(ev.pop("event"), ev)
            snap["new_events"] = events
            tmp = STATUS.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(STATUS)
        except Exception as e:
            emit("watch_error", {"error": str(e)})
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
