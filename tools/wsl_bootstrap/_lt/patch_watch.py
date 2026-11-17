import re, shutil, time
from pathlib import Path
f = Path("/mnt/d/LuSNAR_Seg/tools/wsl_bootstrap/08_gpu_watch_wsl.py")
bak = f.with_name(f.name + ".bak_pre_lt_" + time.strftime("%Y%m%d_%H%M"))
shutil.copy2(f, bak)
s = f.read_text(encoding="utf-8")
s = s.replace('GPU1_PY = "fullimage_smoke.py"\n',
 'GPU1_PY = "fullimage_smoke.py"\n'
 '# full-image longtrain (both GPUs, DDP): waiter + trainer\n'
 'LT_QUEUE = "09_fi_longtrain_wsl.sh"\n'
 'LT_PY = "lusnar_row_blv_ddp.py"\n'
 'LT_WAITER_LOG = LOGDIR / "LT_FI_ROW_BLV_waiter_wsl.log"\n', 1)
old = '''    snap = {
        "ts": now_str(),'''
new = '''    # ---- full-image longtrain (DDP on both GPUs) ----
    lq = pgrep(LT_QUEUE)
    lp = pgrep(LT_PY)
    lw_txt = tail_text(LT_WAITER_LOG, 60)
    lt_runs = re.findall(r"LAUNCH (\\S+)", lw_txt)
    lt_stage = lt_runs[-1] if lt_runs else ("waiting" if lq else "none")
    lt_log = LOGDIR / f"{lt_stage}_wsl.log"
    lt_txt = tail_text(lt_log, 80) if lt_runs else ""
    done_lt = "ALL LT_FI DONE" in lw_txt
    it_lt = last_iter(lt_txt)
    fail_lt = has_fail(lt_txt[-4000:]) or (has_fail(lw_txt[-2000:]) if "FAILED" in lw_txt else None)
    plt = (prev.get("lt") or {})
    ts_lt = progress_ts("lt", it_lt, plt.get("last_iter"), plt.get("progress_unix"))
    stalled_lt = bool(lp and not done_lt and (now - ts_lt) > 90 * 60)
    lt_pack = pack(1, lq, lp, lt_stage, it_lt, fail_lt, done_lt, stalled_lt, lt_log if lt_runs else LT_WAITER_LOG, ts_lt)
    lt_pack["index"] = "0+1"
    if lq and not lp and not done_lt and not fail_lt:
        lt_pack["state"] = "waiting" if not lt_runs else lt_pack["state"]
    if not lq and not lp and not lt_runs and not done_lt:
        lt_pack["state"] = "idle"

    snap = {
        "ts": now_str(),'''
assert old in s
s = s.replace(old, new, 1)
old2 = '''            ts1,
        ),
    }'''
assert old2 in s
s = s.replace(old2, '''            ts1,
        ),
        "lt": lt_pack,
    }''', 1)
old3 = 'for key, label in (("gpu0", "GPU0 backbone"), ("gpu1", "GPU1 full-image")):'
assert old3 in s
s = s.replace(old3, 'for key, label in (("gpu0", "GPU0 backbone"), ("gpu1", "GPU1 full-image"), ("lt", "FI longtrain DDP")):', 1)
old4 = 'for key in ("gpu0", "gpu1"):\n                    a, b'
assert old4 in s
s = s.replace(old4, 'for key in ("gpu0", "gpu1", "lt"):\n                    a, b', 1)
old5 = '''emit("initial", {"gpu0": snap["gpu0"]["state"], "gpu1": snap["gpu1"]["state"],'''
s = s.replace(old5, '''emit("initial", {"gpu0": snap["gpu0"]["state"], "gpu1": snap["gpu1"]["state"], "lt": snap["lt"]["state"],''', 1)
f.write_text(s, encoding="utf-8")
print("patched; backup", bak)