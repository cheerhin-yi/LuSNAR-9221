
import os, subprocess, datetime, glob
print("=== TRAIN_LOG_TAIL ===")
logp = r"D:\LuSNAR_Seg\work_dirs\train_ROW_BLV_20k\train_log.jsonl"
if os.path.exists(logp):
    lines = open(logp, encoding="utf-8", errors="replace").readlines()
    for l in lines[-20:]:
        print(l.rstrip())
else:
    print("NO_LOG")
print("=== CKPTS ===")
path = r"D:\LuSNAR_Seg\work_dirs\train_ROW_BLV_20k"
for f in sorted(glob.glob(os.path.join(path, "*.pth")), key=os.path.getmtime):
    st = os.stat(f)
    mt = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(os.path.basename(f), st.st_size, mt)
print("=== GPU ===")
try:
    print(subprocess.check_output(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,memory.total","--format=csv,noheader"], text=True))
except Exception as e:
    print("gpu err", e)
print("=== PAGEFILE ===")
try:
    out = subprocess.check_output(["powershell","-NoProfile","-Command","Get-CimInstance Win32_PageFileUsage | ForEach-Object { $_.Name + \" AllocMB=\" + $_.AllocatedBaseSize + \" UsedMB=\" + $_.CurrentUsage }"], text=True)
    print(out)
except Exception as e:
    print("pagefile err", e)
print("=== PROCS ===")
ps = """Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -match \"lusnar_row_blv|opensource_followup|error_attrib|distributed\\.run|torch\\.distributed\") } | ForEach-Object { $cmd = $_.CommandLine; if ($cmd.Length -gt 220) { $cmd = $cmd.Substring(0,220) + \"...\" }; \"PID=\" + $_.ProcessId + \" PPID=\" + $_.ParentProcessId + \" Name=\" + $_.Name + \" Cmd=\" + $cmd }"""
print(subprocess.check_output(["powershell","-NoProfile","-Command", ps], text=True))
print("=== LOG_TAIL_FILE ===")
tlog = r"D:\LuSNAR_Seg\logs\train_ROW_BLV_20k.log"
if os.path.exists(tlog):
    lines = open(tlog, encoding="utf-8", errors="replace").readlines()
    for l in lines[-40:]:
        print(l.rstrip())
else:
    print("NO_TRAIN_LOG")
