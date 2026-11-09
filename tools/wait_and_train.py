"""Wait for GPU free of j6_uavdt_eval, train SegFormer-B2 phase1, then finalize report section."""
import os, sys, time, subprocess, json, shutil
from pathlib import Path

LOG = Path(r"D:\LuSNAR_Seg\logs\train_waiter.log")
TRAIN = Path(r"D:\LuSNAR_Seg\tools\train_segformer_b2_phase1.py")
PY = r"F:\Conda\envs\lusnar\python.exe"
WORK = Path(r"D:\LuSNAR_Seg\work_dirs\segformer_b2_phase1")
REPORT = Path(r"D:\LuSNAR_Seg\项目环境与数据检查报告.md")
STAGE = Path(r"C:\Users\ChHao\Downloads\lusnar_phase1_stage")

def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def uav_running():
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine"],
            text=True, stderr=subprocess.DEVNULL)
        return "j6_uavdt_eval" in out
    except Exception as e:
        log(f"proc check err {e}")
        return True

def gpu_used_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True)
        return float(out.strip().split("\n")[0])
    except Exception:
        return 99999

def finalize():
    summary = {}
    sp = WORK / "summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text(encoding="utf-8"))
    # append/replace section 7 in report
    text = REPORT.read_text(encoding="utf-8")
    block = f"""## 7. SegFormer-B2 最小试跑（已完成）
- 配置：512 crop，bs8，BF16 AMP，AdamW lr=6e-5，iters≈1000
- 训练场景：1/3/6/7/8（6403 对）；验证：Scene 5（1599 对）
- 预训练：ImageNet-1K MiT-B2（`pretrained/mit_b2_20220624-66e8bf70.pth`）
- **本地 Scene5 val mIoU = {summary.get('best_mIoU_scene5', 'N/A')}（非官方测试成绩）**
- 摘要 JSON：`work_dirs/segformer_b2_phase1/summary.json`
- checkpoint：`work_dirs/segformer_b2_phase1/best.pth` 与 `iter_*.pth`
- 日志：`work_dirs/segformer_b2_phase1/train_log.jsonl`
- 可视化：`work_dirs/segformer_b2_phase1/vis_val_*.png`
- 原始 summary：`{json.dumps(summary, ensure_ascii=False)}`
"""
    import re
    if "## 7. SegFormer-B2" in text:
        text = re.sub(r"## 7\. SegFormer-B2[\s\S]*?(?=\n## 8\.|\Z)", block.strip()+"\n\n", text, count=1)
    else:
        text += "\n" + block
    REPORT.write_text(text, encoding="utf-8")
    STAGE.mkdir(parents=True, exist_ok=True)
    for src in [REPORT, WORK/"summary.json", WORK/"train_log.jsonl"]:
        if src.exists():
            shutil.copy2(src, STAGE / src.name)
    for p in sorted(WORK.glob("vis_val_*.png"))[:3]:
        shutil.copy2(p, STAGE / p.name)
    # marker for parent
    (Path(r"D:\LuSNAR_Seg\results")/"TRAIN_DONE.flag").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log("finalize done")

def main():
    log("waiter start v2")
    while True:
        uav = uav_running()
        mem = gpu_used_mib()
        log(f"check uav={uav} mem_mib={mem}")
        if (not uav) and mem < 4000:
            log("GPU free enough; launching train")
            break
        time.sleep(60)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    rc = subprocess.call([PY, str(TRAIN)], env=env)
    log(f"train exit rc={rc}")
    if rc == 0:
        try:
            finalize()
        except Exception as e:
            log(f"finalize err {e}")
    return rc

if __name__ == "__main__":
    sys.exit(main() or 0)
