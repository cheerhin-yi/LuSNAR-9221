"""Probe DESKTOP-ORQLB8F for LuSNAR B1 DDP max throughput params."""
from __future__ import annotations
import json, os, platform, subprocess, sys
from pathlib import Path

os.environ["USE_LIBUV"] = "0"
SCRIPT = Path(r"D:\LuSNAR_Seg\tools\portable")
sys.path.insert(0, str(SCRIPT))
import mmcv_stub  # noqa: F401 — installs mmcv._ext stub

import torch
import psutil
from _lusnar_common import build_segformer_b2, find_pretrain, NUM_CLASSES, ensure_mmcv_stub
ensure_mmcv_stub(SCRIPT)

import torch.nn.functional as F

OUT = Path(r"D:\LuSNAR_Seg\results\hw_tune_probe.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

def gpu_rows():
    raw = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ], text=True, encoding="utf-8", errors="replace")
    rows = []
    for line in raw.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        rows.append({
            "index": int(p[0]), "name": p[1],
            "mem_total_mb": float(p[2]), "mem_free_mb": float(p[3]), "mem_used_mb": float(p[4]),
            "util": float(p[5]), "temp_c": float(p[6]),
            "power_w": float(p[7]) if p[7] not in ("", "[N/A]") else None,
            "power_limit_w": float(p[8]) if p[8] not in ("", "[N/A]") else None,
        })
    return rows

info = {
    "host": platform.node(),
    "os": platform.platform(),
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu_count": torch.cuda.device_count(),
    "cpu_physical": psutil.cpu_count(logical=False),
    "cpu_logical": psutil.cpu_count(logical=True),
    "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
    "gpus": gpu_rows(),
}

project = Path(r"D:\LuSNAR_Seg")
pretrain = find_pretrain(project)
device = torch.device("cuda:0")
torch.cuda.set_device(0)
torch.cuda.empty_cache()

model = build_segformer_b2(pretrain).to(device)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=6e-5)

crop = 512
probe = []
for bs in [32, 28, 24, 20, 18, 16, 14, 12, 10, 8]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    try:
        x = torch.randn(bs, 3, crop, crop, device=device, dtype=torch.float32)
        y = torch.randint(0, NUM_CLASSES, (bs, crop, crop), device=device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if hasattr(logits, "seg_logits"):
                logits = logits.seg_logits
            if torch.is_tensor(logits) and logits.dim() == 4 and logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits.float(), y, ignore_index=255)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(0) / (1024**2)
        probe.append({"batch": bs, "ok": True, "peak_alloc_mb": round(peak, 1)})
        print(f"OK batch={bs} peak_mb={peak:.1f}", flush=True)
        if sum(1 for p in probe if p["ok"]) >= 4:
            break
    except torch.cuda.OutOfMemoryError:
        print(f"OOM batch={bs}", flush=True)
        probe.append({"batch": bs, "ok": False, "peak_alloc_mb": None})
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"ERR batch={bs}: {type(e).__name__}: {e}", flush=True)
        probe.append({"batch": bs, "ok": False, "error": f"{type(e).__name__}: {e}"})
        torch.cuda.empty_cache()

ok_batches = [p["batch"] for p in probe if p.get("ok")]
max_bs = max(ok_batches) if ok_batches else 8
safe_bs = max(4, int(max_bs * 0.85))
for cand in [32, 28, 24, 20, 18, 16, 14, 12, 10, 8, 6, 4]:
    if cand <= safe_bs:
        safe_bs = cand
        break

logical = info["cpu_logical"] or 16
ram_gb = info["ram_total_gb"]
workers = max(4, min(16, logical // 4))
if ram_gb < 48:
    workers = min(workers, 6)
elif ram_gb < 96:
    workers = min(workers, 12)
omp = max(1, min(4, logical // max(1, 2 * workers)))

rec = {
    "recipe": "B1",
    "nproc_per_node": 2,
    "batch_per_gpu": safe_bs,
    "global_batch": safe_bs * 2,
    "crop": 512,
    "workers_per_proc": workers,
    "amp": "bf16",
    "iters": 20000,
    "val_every": 1000,
    "MASTER_ADDR": "127.0.0.1",
    "USE_LIBUV": "0",
    "OMP_NUM_THREADS": omp,
    "backend_note": "gloo on win32",
    "max_batch_probe_ok": max_bs,
}
info["probe"] = probe
info["recommend"] = rec
OUT.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({
    "recommend": rec,
    "probe": probe,
    "cpu_phys_logic": [info["cpu_physical"], info["cpu_logical"]],
    "ram_gb": [info["ram_total_gb"], info["ram_available_gb"]],
    "gpus": info["gpus"],
}, indent=2, ensure_ascii=False))
del model, opt
torch.cuda.empty_cache()
