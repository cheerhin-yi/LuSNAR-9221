import os, glob, subprocess, json
root = r"/mnt/d/LuSNAR_Seg"
wd = root + "/work_dirs/train_ROW_BLV_20k"
ckpts = sorted(glob.glob(wd + "/iter_*.pth"), key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
print("NEWEST_CKPT:", ckpts[-1] if ckpts else "NONE")
for p in ckpts[-5:]:
    print("CKPT", os.path.basename(p), os.path.getsize(p), int(os.path.getmtime(p)))
logp = wd + "/train_log.jsonl"
if os.path.exists(logp):
    lines = open(logp, encoding="utf-8", errors="replace").read().strip().splitlines()
    print("LOG_LINES", len(lines))
    for ln in lines[-5:]:
        print("LOG", ln[:300])
print("---PS---")
ps = subprocess.check_output(["ps","-ef"], text=True, errors="replace")
for ln in ps.splitlines():
    if any(k in ln for k in ["lusnar_row_blv","torch.distributed","run_opensource","run_error_attrib","03_resume","04_relaunch"]) and "grep" not in ln:
        print(ln)
print("---GPU---")
print(subprocess.check_output(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,memory.total","--format=csv,noheader"], text=True))
print("---NPROC---")
print(subprocess.check_output(["nproc"], text=True).strip())
print("---MEM---")
print(subprocess.check_output(["free","-h"], text=True))
print("---SHM---")
print(subprocess.check_output(["df","-h","/dev/shm"], text=True))
