from pathlib import Path
from datetime import datetime
root = Path("/mnt/d/LuSNAR_Seg")
now = datetime.now().strftime("%Y-%m-%d %H:%M Asia/Shanghai")
status = root / "logs" / "wsl_migrate_status.md"
handoff = root / "HANDOFF.md"
block = """
## workers switch %s
- Tried WORKERS=16 from iter_5000 (~21:46): BAD. First 50 iters 381s (~7.6 s/iter). WSL Mem available~0, /dev/shm peaked 15G/16G (92%%), swap thrash. GPU util often one-card idle.
- Fell back WORKERS=12 from iter_5000 at 21:58:09. Launcher PID 2038, ranks 2088/2089, waiters 2466/2467.
- Steady ~2.17 s/iter (5100 to 5350 elapsed 234.3 to 777.3). First-50 after resume 133.6s (~2.67 incl startup). Better than prior workers=8 ~2.47 s/iter.
- Mem: available ~4.9GB (not zero), swap ~4.2GB, /dev/shm ~2%%. Host commit ~123/192 GB. nproc=48, memoryLimit=32GB, data /root/lusnar_data.
- Preferred workers now **12**; ladder **12->8** on DataLoader/shm OOM (16 rejected). Batch 48 unchanged. No reboot.
- ETA 20k @ ~2.17 s/iter from 5350: ~8.8h -> ~**07:00 Asia/Shanghai** 2026-11-10.
""" % now
status.write_text(status.read_text(encoding="utf-8", errors="replace") + block, encoding="utf-8")
note = "\n### WSL workers=12 (%s)\n- Dropped 16->12 after shm/mem thrash. Live: launcher 2038, workers=12, port 29601, resume iter_5000, ~2.17 s/iter, waiters 2466/2467. Ladder 12->8 on OOM. ETA ~07:00 Asia/Shanghai 2026-11-10.\n" % now
handoff.write_text(handoff.read_text(encoding="utf-8", errors="replace") + note, encoding="utf-8")
print("updated status+handoff")
p = root / "tools/wsl_bootstrap/03_resume_train_wsl.sh"
t = p.read_text(encoding="utf-8", errors="replace")
t2 = t.replace('WORKERS="${WORKERS:-16}"', 'WORKERS="${WORKERS:-12}"')
t2 = t2.replace('ladder: 16 -> 12 -> 8 (DataLoader/shm OOM only; preferred 16)', 'ladder: 12 -> 8 (DataLoader/shm OOM only; preferred 12; 16 rejected)')
t2 = t2.replace('(ladder 16->12->8)', '(ladder 12->8; 16 rejected)')
p.write_text(t2, encoding="utf-8")
print("patched resume default workers=12")