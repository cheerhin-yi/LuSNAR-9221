#!/usr/bin/env python3
"""Wait for the current ROW+BLV 20k run, then do two predetermined jobs.

1. One-shot open-set score of that run's Scene5-selected best.pth.
2. Retrain with Moon_5 added, same hyperparameters, then one-shot score last.pth.

Test labels are never used to choose a checkpoint or a hyperparameter.
The second command is fixed before either score is known.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\LuSNAR_Seg")
PY = Path(r"F:\Conda\envs\lusnar\python.exe")
LOG = ROOT / "logs" / "train_ROW_BLV_20k.log"
WORK_A = ROOT / "work_dirs" / "train_ROW_BLV_20k"
WORK_B = ROOT / "work_dirs" / "train_ROW_BLV_20k_plus5"
CHAIN_LOG = ROOT / "logs" / "opensource_followup.log"
PROTOCOL = ROOT / "results" / "oneshot_protocol.json"

PROTOCOL_BODY = {
    "locked_before_any_open_score": True,
    "test_scenes": [2, 4, 9],
    "label_alignment": "open-source RGB remapped by COLOR_TO_ID; rock/crater swapped vs official index order",
    "inference": "ROW + TTA scales 0.75/1.0/1.25, sliding window 512 stride 256",
    "no_tuning_on_test": True,
    "run_a": {
        "checkpoint": str(WORK_A / "best.pth"),
        "why": "best.pth already selected on Scene5 during the current train, before this test",
    },
    "run_b": {
        "checkpoint": str(WORK_B / "last.pth"),
        "train_scenes": [1, 3, 5, 6, 7, 8],
        "why": "Moon_5 joins training. No val set remains, so the tested weights are the predetermined final iter, not a test-picked checkpoint.",
        "hyperparams_from_scene5": {
            "recipe": "M_ROW_BLV",
            "iters": 20000,
            "batch_per_gpu": 48,
            "lr": 6e-5,
            "crop": 512,
            "amp": "bf16",
            "workers": 4,
            "val": False,
        },
    },
}


def log(msg: str):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    print(line, flush=True)
    CHAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with CHAIN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def train_alive(run_name: str) -> bool:
    needle = f"--run-name {run_name} "
    out = subprocess.check_output(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine",
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return any(needle in line for line in out.splitlines())


def wait_until(pred, what: str, poll_s: int = 60):
    while not pred():
        log(f"waiting: {what}")
        time.sleep(poll_s)


def run_oneshot(checkpoint: Path, tag: str):
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


def run_plus5():
    env = os.environ.copy()
    env.update(
        {
            "USE_LIBUV": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29621",
            "OMP_NUM_THREADS": "2",
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    log_path = ROOT / "logs" / "train_ROW_BLV_20k_plus5.log"
    cmd = [
        str(PY),
        "-u",
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--master_addr",
        "127.0.0.1",
        "--master_port",
        "29621",
        str(ROOT / "tools" / "portable" / "lusnar_row_blv_ddp.py"),
        "--recipe",
        "M_ROW_BLV",
        "--run-name",
        "train_ROW_BLV_20k_plus5",
        "--iters",
        "20000",
        "--batch",
        "48",
        "--workers",
        "4",
        "--amp",
        "bf16",
        "--crop",
        "512",
        "--lr",
        "6e-5",
        "--val-every",
        "1000",
        "--train-scenes",
        "1,3,5,6,7,8",
        "--no-val",
        "--list-dir",
        str(ROOT / "data" / "lists_plus5"),
        "--project-root",
        str(ROOT),
        "--data-root",
        str(ROOT / "data"),
        "--train-subdir",
        "train",
    ]
    log("start plus5 train")
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT)
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"plus5 train exited {code}")
    if not (WORK_B / "last.pth").exists():
        raise RuntimeError("plus5 finished without last.pth")


def write_compare():
    a = json.loads((ROOT / "results" / "oneshot_ROW_BLV" / "summary.json").read_text(encoding="utf-8"))
    b = json.loads((ROOT / "results" / "oneshot_ROW_BLV_plus5" / "summary.json").read_text(encoding="utf-8"))
    da = a["overall"]["mIoU"]
    db = b["overall"]["mIoU"]
    lines = [
        "# ROW+BLV 开源测试集对照（非官方，一次性）",
        "",
        "两组都使用同一套 Scene5 上已定的超参，推理都是 ROW+TTA。Moon_2/4/9 没有参与训练或选模。",
        "",
        "| 运行 | 训练场景 | 权重 | overall mIoU | 宏平均 | 加权 |",
        "|---|---|---|---:|---:|---:|",
        f"| 不加 Moon_5 | 1,3,6,7,8 | Scene5 选出的 best.pth | {da:.4f} | {a['macro_scene_mIoU']:.4f} | {a['weighted_scene_mIoU']:.4f} |",
        f"| 加入 Moon_5 | 1,3,5,6,7,8 | 预定的 last.pth（iter 20000） | {db:.4f} | {b['macro_scene_mIoU']:.4f} | {b['weighted_scene_mIoU']:.4f} |",
        "",
        f"加入 Moon_5 后，全像素 mIoU 变化 {db - da:+.4f}。",
        "",
        "分场景 mIoU：",
        "",
        "| 场景 | 不加 Moon_5 | 加入 Moon_5 | 差值 |",
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


def main():
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(PROTOCOL_BODY, ensure_ascii=False, indent=2), encoding="utf-8")
    log("protocol locked")

    def first_ready():
        alive = train_alive("train_ROW_BLV_20k")
        if (not alive) and LOG.exists() and "ChildFailedError" in LOG.read_text(encoding="utf-8", errors="replace"):
            if not (WORK_A / "best.pth").exists():
                raise RuntimeError("current 20k failed before best.pth")
        return (not alive) and (WORK_A / "best.pth").exists() and (WORK_A / "tta_final.json").exists()

    wait_until(first_ready, "current 20k to finish best.pth and tta_final.json")
    if "TTA_ERROR" in (WORK_A / "tta_final.json").read_text(encoding="utf-8"):
        log("scene5 TTA wrote an error; open-set test still uses best.pth")
    run_oneshot(WORK_A / "best.pth", "ROW_BLV")
    run_plus5()
    run_oneshot(WORK_B / "last.pth", "ROW_BLV_plus5")
    write_compare()
    log("FOLLOWUP_DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("FOLLOWUP_FAILED\n" + traceback.format_exc())
        sys.exit(1)
