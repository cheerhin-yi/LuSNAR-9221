# 接手说明（2026-11-09 15:31）

给下一个 bot。先读本文件，再读 `近期实验记录.md` 和 `PROJECT_STATUS.md`。不要另开一轮训练，不要改已经写死的测试协议。

## 现在在跑

ROW+BLV × 20k，约 10:55 拉起，日志写到 **iter 3950 / 20000**，loss 约 0.063，显存 36510 MB。最近一次全量 Scene5 验证是 iter 3000，mIoU **0.818**（岩石 0.767，撞击坑 0.802，山体 0.653，天空 0.887）。同一步的 B1 是 0.823。最好权重目前是 iter 3000 的 `best.pth`。

| 进程 | 作用 | 状态 |
| --- | --- | --- |
| `lusnar_row_blv_ddp.py --run-name train_ROW_BLV_20k` | 当前长训，端口 29601 | 活着就不要动 |
| `tools/portable/run_opensource_followup.py` | 等 20k 结束，然后做两次一次性测试和 plus5 再训 | 仍在等 `best.pth` 和 `tta_final.json` |
| `tools/portable/run_error_attrib_after.py` | 等两次测试的预测落盘，再写误差归因 | 仍在等 |

日志：`logs/train_ROW_BLV_20k.log`，`logs/opensource_followup.log`，`logs/error_attrib_waiter.log`。

这三个进程是上一会话拉起来的。新会话先看进程还在不在。还在就继续等。不在，再按下面的方式重接，不要并行再开一份。

## 不要做

- 不要用 Moon_2/4/9 的标签训练、调参、选模。测试标签只在预测写完之后打分，以及最后那份归因。
- 不要看见开源分数之后改超参或换 checkpoint。协议在 `results/oneshot_protocol.json`，看分数之前就锁了。
- 不要覆盖 `work_dirs/train_B1`。
- 不要把 batch 改回 56/64，不要设 `PYTORCH_CUDA_ALLOC_CONF`，不要用 `Start-Process` 启动训练（父进程一退，子进程会被杀掉）。
- 不要加 SHAD，不要加 Albumentations，不要垂直翻转或旋转。
- 不要 force-push。
- 官方盲评没有批准，不要提交。

## 长训命令（只有进程死了才重跑）

工作目录 `D:\LuSNAR_Seg`。环境 `F:\Conda\envs\lusnar\python.exe`。Windows DDP 用 gloo：`USE_LIBUV=0`，`MASTER_ADDR=127.0.0.1`。启动时连 `DESKTOP-ORQLB8F` 出现 socket 10049 是预期警告。

```
F:\Conda\envs\lusnar\python.exe -u -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 29601 tools\portable\lusnar_row_blv_ddp.py --recipe M_ROW_BLV --run-name train_ROW_BLV_20k --iters 20000 --batch 48 --workers 4 --amp bf16 --crop 512 --lr 6e-5 --val-every 1000 --tta-final --project-root D:\LuSNAR_Seg --data-root D:\LuSNAR_Seg\data --train-subdir train
```

日志用 `cmd /c "... > logs\train_ROW_BLV_20k.log 2>&1"`。先确认 29601 没被占、两张 4090 空闲，再拉起。端口 29621 留给 plus5，不要占用。

死法参考：worker 6 时 copy-paste 的全图 `argwhere` 把 64 GB 内存打满。现在的代码已改成随机找粘贴点，worker 是 4。若再 OOM，先看是训练步还是验证步；验证 OOM 只降验证 batch，不要把训练 batch 降到 48 以下，除非训练步本身 OOM。

## 自动后续（脚本已写好）

`run_opensource_followup.py` 的顺序：

1. 等当前进程结束，且存在 `work_dirs/train_ROW_BLV_20k/best.pth` 和 `tta_final.json`。
2. 用 `best.pth` 对 Moon_2/4/9 做 ROW+TTA（尺度 0.75/1/1.25，滑窗 512/256）。GT 按 RGB→`COLOR_TO_ID` 对齐，岩石和撞击坑在开源调色板里是对调的。结果在 `results/oneshot_ROW_BLV/`。
3. 再训一次，场景 1/3/5/6/7/8，其余超参相同，`--no-val`，列表目录 `data/lists_plus5`，端口 29621，日志 `logs/train_ROW_BLV_20k_plus5.log`。没有验证集，所以不选模，测预定的 `last.pth`。
4. 第二次测试写入 `results/oneshot_ROW_BLV_plus5/`，对照表 `results/oneshot_ROW_BLV_compare.md`。

`run_error_attrib_after.py` 等两次预测都在，再跑 `lusnar_error_attrib.py`，写出 `results/error_attrib.md`。只归因，不写做法，不改权重。门槛已经写死在脚本里。

若排队脚本死了而训练还在，把这两个 python 按原命令重新拉起即可。它们只是轮询，不会重开训练。

粗算：从 iter 3950 起，这次 20k 加 Scene5 TTA 还要约 20 小时；两次开源测试各约 2 小时；plus5 再训约 21 小时；归因约半小时。报告大约在 11 月 11 日中午前后。

## Git

远程 `https://github.com/cheerhin-yi/LuSNAR-9221.git`，分支 `main`。

本地提交 `83f0a78` 还没有推上去，远程停在 `61149e9`。推送需要 GitHub 登录，这台机器没有存凭证。代理是 `127.0.0.1:7897`，要用 OpenSSL 后端：

```
git -c http.sslBackend=openssl push origin main
```

不要 force-push。`data/`、权重、`logs/`、正在写的 `work_dirs/train_ROW_BLV_20k/` 都不在这次提交里。

## 已经定下的结论

- 长训底座是 ROW+BLV，结尾 TTA。2k 冒烟里 ROW+BLV+TTA 的 Scene5 子集 mIoU 是 0.852。SHAD 叠上去会把撞击坑打下去，不进长训。
- B1×20k 最好 Scene5 mIoU 0.863（iter 17000）。开源事后全像素 mIoU 0.852；缺口主要在 Moon_9 的撞击坑 0.455。见 `results/eval_B1_opensource_test_blind.md`。
- 第三批在 cheerhin 已完成，没有超过噪声带的正向增益。分数在远程的 `PROJECT_STATUS.md` 第 6 节和 `results/smoke_method_eval_batch3.md`。不要在 4090 上重跑。
- 误差归因做完后，方法由用户自己定。bot 不要抢先改配方。

## 鐢ㄦ埛鎸囦护锛氱珛鍗抽噸鍚縼绉?WSL锛?Stamp Asia/Shanghai锛?
鐢ㄦ埛锛坱121u锛夋槑纭姹傦細**鐜板湪杩欒疆璁粌瀹岋紝鐩存帴閲嶅惎**銆備笉鍐嶇瓑 iter_5000.pth銆?
- 宸蹭紭闆呭仠姝?FIRST-ROUND ROW+BLV DDP锛堝師 launcher/workers 绾?PID 3208 / 42796 / 58232锛沺ort 29601锛夈€?- 鍋滆鏃舵渶鍚庡凡鐭?iter/loss锛?*4820 / 0.06612738966941833**锛堜互 train_log.jsonl 鏈涓哄噯锛夈€?- 鏈€鏂?checkpoint锛氳鑴氭湰杈撳嚭 NEWEST_CKPT锛堝彲鑳戒粛鏄?iter_4000.pth 鈥?鐢ㄦ埛鍏佽锛夈€?- Waiter PIDs锛堥噸鍚細鏉€鎺夛紝WSL 鐜灏辩华鍚庨渶閲嶆媺锛夛細48148, 4444, 21144, 20796
  - `tools/portable/run_opensource_followup.py`
  - `tools/portable/run_error_attrib_after.py`
- `.wslconfig`锛歮emory=64GB processors=24 swap=32GB锛堝凡纭/鍒锋柊锛夈€?- WSL MSI 宸叉槸 2.7.14.0锛涢噸鍚悗铏氭嫙鍖栫敓鏁堝啀瑁?Ubuntu锛屽啀璺?`setup_lusnar_inside_wsl.sh`銆佹寕杞?`D:\LuSNAR_Seg`銆?- **杩佺Щ闂ㄥ凡寮€锛堢敤鎴锋壒鍑嗭級**銆傞噸鍚悗涓嬩竴姝ワ細Ubuntu + 鐜 + remount锛涗紭鍏堜粠鏈€鏂?ckpt 鍦?WSL 缁绗竴杞紝鍚﹀垯 Windows 涓婄画瀹屽啀涓?WSL锛沺lus5 鍦?WSL锛泈aiters 鐜濂藉悗鍦?WSL 閲嶆媺銆?- 涓嶈鍔?`work_dirs\train_B1`锛涗笉瑕佺 Moon_2/4/9 璁粌锛沚atch 淇濇寔 >=48锛涗笉瑕佹敼閰嶆柟銆?

## WSL migrate live status (2026-11-09 18:41:59 Asia/Shanghai)
Machine rebooted 2026-11-09 18:27:25. First-round Windows DDP stopped at iter 4820 (loss 0.06613); newest checkpoint remains work_dirs/train_ROW_BLV_20k/iter_4000.pth. iter_5000.pth still absent. Ubuntu WSL is Running; /mnt/d/LuSNAR_Seg is mounted; /dev/shm is 32G. Original setup hit Anaconda ToS after miniconda install; setup_lusnar_continue.sh accepted ToS at 18:38:44 and is still running. The lusnar Python 3.10 env exists and is now installing torch==2.7.1+cu128 (log: logs/wsl_setup_lusnar.log). Do not start a second Windows long train while this setup/resume path is active. Do not reboot again. After setup verify passes, resume first-round on WSL from iter_4000.pth via tools/wsl_bootstrap/03_resume_train_wsl.sh with WORKERS=4, then relaunch waiters with 04_relaunch_waiters_wsl.sh. Details: logs/wsl_migrate_status.md.
## WSL migrate live (Asia/Shanghai 2026-11-09 19:20:50)

- Ubuntu 26.04.1 LTS installed (WSL2 2.7.14). .wslconfig 64GB/24CPU/32GB swap OK. HypervisorPresent=True.
- Env `lusnar` at `/opt/miniconda3/envs/lusnar`: py3.10.22, torch 2.7.1+cu128, mmseg 1.2.2, mmcv-lite 2.1.0 (same as Windows). CUDA 2×4090 visible.
- FIRST-ROUND ROW+BLV resumed on WSL (NOT Windows):
  - launcher PID **548**, workers **574/575**, port **29601**, WORKERS=**8** (ladder 8→6→4 on DataLoader/shm OOM only)
  - resume `work_dirs/train_ROW_BLV_20k/iter_4000.pth`; recipe M_ROW_BLV; batch 48; amp bf16; crop 512; lr 6e-5; val-every 1000; tta-final
  - launch: `bash /mnt/d/LuSNAR_Seg/tools/wsl_bootstrap/03_resume_train_wsl.sh` (WORKERS=8)
  - prior SIGHUP kill at 19:18 (workers=4 attempt); relaunched 19:19:46 with workers=8
- Waiters on WSL: followup PID **653** (`tools/wsl_bootstrap/run_opensource_followup_wsl.py`), attrib PID **654** (`run_error_attrib_after_wsl.py`). Logs: `logs/opensource_followup.log`, `logs/error_attrib_waiter.log`.
- Do NOT start a second Windows longtrain. Port 29621 reserved for plus5.
- RCS `full_crater_weights.json` Windows-path cache backed up; WSL regenerates with `/mnt/d` paths.

## WSL migrate live status (2026-11-09 19:31:04 Asia/Shanghai)
WSL env ready (torch 2.7.1+cu128, mmseg 1.2.2, mmcv-lite 2.1.0, 2 CUDA GPUs). First-round ROW+BLV 20k resumed on WSL at 19:25:18 from work_dirs/train_ROW_BLV_20k/iter_4000.pth (Windows had stopped at iter 4820 / loss 0.06613; iter_5000.pth still absent). Live DDP: launcher PID 1842, ranks 1868/1869, workers=8, batch 48, port 29601, recipe M_ROW_BLV. train_log.jsonl wrote resume from_iter 4000 best 0.8487. GPU mem ~39316 / 41207 MiB. Waiters 653 (run_opensource_followup_wsl.py) and 654 (run_error_attrib_after_wsl.py). Do not start a second train. Do not reboot. Do not touch work_dirs/train_B1 or Moon_2/4/9 labels.

## WSL migrate live (Asia/Shanghai 2026-11-09 19:34)

- First WSL resume 19:19:46 (workers=8, PIDs 548/574/575) died 19:23:54: ModuleNotFoundError: No module named ftfy
- ftfy 6.3.1 and regex 2026.9.29 are installed in /opt/miniconda3/envs/lusnar (visible to root)
- Relaunched 19:25:18 from tools/wsl_bootstrap/03_resume_train_wsl.sh; healthy as of 19:34:
  - launcher PID 1842, rank workers 1868/1869, port 29601, WORKERS=8, DataLoader children spawned
  - resume work_dirs/train_ROW_BLV_20k/iter_4000.pth (resume_miou 0.8487); recipe M_ROW_BLV; batch 48; amp bf16; crop 512; lr 6e-5; val-every 1000; tta-final
  - train_log.jsonl wrote resume at 19:27:59, then iter 4050 loss 0.06216 at 19:33:08; mem_mb 36796.4
  - GPU memory about 39316 MiB / 41221 MiB
- Waiters still alive: followup PID 653, attrib PID 654
- iter_5000.pth still absent; newest checkpoint remains iter_4000.pth
- Do NOT start a second Windows longtrain. Port 29621 reserved for plus5.
## 2026-11-09 19:58 (UTC+8) ext4 data migrate — BLOCKED by Windows commit exhaustion
- Data copied to WSL ext4: /root/lusnar_data/train/Moon_{1,3,5,6,7,8}/{color,label} (14G, 341 s, counts match; no Moon_2/4/9). Log: logs/wsl_copy_data_ext4.log, script tools/wsl_bootstrap/05_copy_data_ext4.sh
- 03_resume_train_wsl.sh now passes --data-root /root/lusnar_data --list-dir /root/lusnar_data/lists (env DATA_ROOT/LIST_DIR/CKPT override; CKPT = newest iter_*.pth). RCS weights json copied with path prefix rewritten (6403 pairs, same order; check passed). Backup: 03_resume_train_wsl.sh.bak_pre_ext4. D: lists_portable untouched. Recipe unchanged.
- 9p run (19:25, PIDs 1842/1868/1869) stopped 19:36 at iter ~4050 (6.2 s/iter, GPU 0-2%).
- Relaunches 19:45, 19:47 (mine), 19:51, 19:54 (watch routine) all failed: first-step CUDA OOM at ~34.8 GiB/GPU (PyTorch 32-34 GiB allocated; need ~36.8 GB) and/or the WSL VM was killed (19:52:40, 19:56:06).
- Root cause (Windows System log, Resource-Exhaustion-Detector 2004): vmmemWSL used 118-121 GB virtual memory; commit limit 95.8 GB (64 GB RAM + pagefile C: 32 GB initial / 64 GB max). WSL GPU-PV device memory (~2x38 GB) is charged to vmmem commit, plus guest RAM (fast ext4 reads fill page cache + pinned buffers) -> commit exhausted -> OOM / VM killed. The 9p run fit only because the guest stayed small.
- State now: NO train, NO waiters (VM restart killed 620/621 & 653/654). iter_4000.pth still newest. Do NOT keep auto-relaunching: each attempt re-crashes the VM.
- Fix options (need approval): (A) .wslconfig memory=64GB -> 32GB then `wsl --shutdown` (no host reboot); (B) bigger pagefile on D: (e.g. 96-128 GB), may need a reboot; (C) go back to /mnt/d data (works, ~6 s/iter, ~27 h).

## 2026-11-09 20:39 reboot #2 (user-approved)
- Windows train stopped at iter 4350 (newest ckpt iter_4000.pth).
- Added D:\pagefile.sys 98304-163840 MB (C kept 32768-65536). .wslconfig memory=32GB (backup .wslconfig.bak_64GB).
- HKCU Run: Clash Verge, Grok Bot. Next: WSL resume from iter_4000 with ext4 data (/root/lusnar_data) via 03_resume_train_wsl.sh workers=8, then 04_relaunch_waiters_wsl.sh.


## 2026-11-09 20:58 (UTC+8) WSL ext4 resume HEALTHY (after reboot #2)
- Commit limit now 191.8 GB (C: 32-64 GB + D: 96-160 GB pagefile). WSL guest 31 GB RAM, /dev/shm 16G, 24 vCPU.
- Pre-check: /root/lusnar_data/lists/{train_scenes.txt 6403, val_scene5.txt 1599, full_crater_weights.json 6403 pairs} all /root/lusnar_data paths, same order, files exist. Script's pairs-equality check passed (resume event written).
- Launched 20:45:01 `WORKERS=8 bash tools/wsl_bootstrap/03_resume_train_wsl.sh` (setsid): launcher PID 500, ranks 526/527, 8 DataLoader workers each. Port 29601. Resume iter_4000.pth (best 0.8487). Recipe unchanged (M_ROW_BLV, 20000, batch 48, bf16, crop 512, lr 6e-5, val-every 1000, tta-final). No PYTORCH_CUDA_ALLOC_CONF.
- Speed: iter 4050 @148.4s, 4100 @281.4s, 4150 @403.1s, 4200 @536.2s, 4250 @645.6s -> steady ~2.4-2.7 s/iter (Windows workers=4 was ~5.0 s/iter; WSL 9p was ~6.2 s/iter).
- GPU: mem 39.3/40.7 GB; util bursty (runs of 90-100% at ~2.7 GHz/230-290 W, then idle 210 MHz gaps). WSL CPU ~65-72% busy (CPU-bound augmentation, iowait ~0). Data-loader still the limiter; more CPU (wslconfig processors>24 or workers>8) would help but needs wsl --shutdown / user OK.
- Windows commit: 121.6-122.1 GB / 191.8 GB limit, vmmemWSL WS 31.6 GB. No Resource-Exhaustion 2004 events since boot. dmesg only shows benign dxgkio_escape -22 (nvidia-smi queries), no OOM.
- Waiters relaunched 20:55 via 04_relaunch_waiters_wsl.sh: followup PID 966, attrib PID 967 (both "protocol/gates locked", waiting).
- ETA 20k: ~2.4 s/iter x 15750 = ~10.6 h + val/TTA -> around 2026-11-10 08:00-09:00 Asia/Shanghai. iter_5000.pth expected ~21:25.
- Next watch: if train dies, resume on WSL with same script (newest iter_*.pth); workers ladder 8->6->4 only on DataLoader/shm OOM. Windows train stays paused (user).

## 2026-11-09 21:45 (UTC+8) WSL processors 24 -> 48 (user-approved), resumed from iter_5000
- iter 5000 val (Scene5, local): mIoU 0.8559 (soil 0.9827 / rock 0.8146 / crater 0.8525 / mountain 0.6904 / sky 0.9392) -> new best.pth (21:27:54). iter_5000.pth 21:27:52, size stable.
- 21:29 stopped train pgid 500 (SIGTERM) and waiters 966/967; .wslconfig backed up to C:\Users\Administrator\.wslconfig.bak_24cpu; set processors=48 (memory=32GB, swap=32GB unchanged); wsl --shutdown; restart OK: nproc 48, RAM 31 GB, /dev/shm 16G, ext4 /root/lusnar_data intact (Moon_1/3/5/6/7/8 only). Host NOT rebooted.
- 21:30:04 relaunched `WORKERS=8 bash tools/wsl_bootstrap/03_resume_train_wsl.sh` (setsid): launcher PID 543, ranks 593/594, port 29601, ckpt iter_5000.pth (resume event best 0.8559). Recipe unchanged (M_ROW_BLV, 20000, batch 48, bf16, crop 512, lr 6e-5, val-every 1000, tta-final). No PYTORCH_CUDA_ALLOC_CONF.
- 21:30:53 waiters via 04_relaunch_waiters_wsl.sh: followup PID 789, attrib PID 790 (both locked/waiting).
- Result: speed unchanged. 5050->5300 = 2.47 s/iter (before at 24 vCPU: ~2.48 s/iter). GPU util still bursty/alternating (0-100%). WSL CPU ~34-36% busy of 48 (load ~16): the 16 DataLoader workers (8 per rank, OMP=1) each sit at ~100% of one core -> limiter is worker COUNT, not vCPUs. Raising workers above 8 (e.g. 12-16) would likely help but needs user approval (not done).
- Commit 122.2 / 191.8 GB, free phys ~23.5 GB, no event 2004, no OOM in dmesg, shm 1%.
- ETA 20k: ~14700 iters x 2.47 s + val/TTA -> ~2026-11-10 08:15-09:00 Asia/Shanghai.
- If WSL crashes from commit: restore .wslconfig.bak_24cpu, wsl --shutdown, relaunch same script.
### WSL workers=12 (2026-11-09 22:12 Asia/Shanghai)
- Dropped 16->12 after shm/mem thrash. Live: launcher 2038, workers=12, port 29601, resume iter_5000, ~2.17 s/iter, waiters 2466/2467. Ladder 12->8 on OOM. ETA ~07:00 Asia/Shanghai 2026-11-10.

## 2026-11-09 22:20 Asia/Shanghai user decision: plus5 on Windows
- Finish current ROW+BLV 20k (+TTA) on WSL workers=12 (ladder 12->8), do not interrupt for this.
- After first-round done: plus5 on **Windows** (not WSL). Start workers=4, try 6 then 8; on 1455/shared-memory drop back. Same recipe/batch constraints; port 29621; USE_LIBUV=0; F:\Conda\envs\lusnar. Watch crater-weights/list paths stay D:\ (not /mnt/d). No Moon_2/4/9 for train/select. Don't touch train_B1.

## 2026-11-09 23:31 Asia/Shanghai user decision: pipeline order (inference before plus5)
1. After first-round 20k+TTA: diagnostics on first-round weights (no train): confusion/error attribution, T0 scale audit (train vs infer apparent size of rock/crater), per-scale x per-class IoU table; split Scene5 into halves A (select) / B (report once).
2. Inference protocols on FIRST-ROUND weights, Scene5 only, BEFORE plus5: I1 per-class scale fusion -> I2 overlap Gaussian/cosine weighting + drop window-edge preds -> I3 small-object re-zoom (DMNet/AutoFocus style). Candidates pre-registered (<=8), paired per-image test, report rock/crater/mountain IoU + cost (forwards, ms/img).
3. Then plus5 on Windows (recipe unchanged, scenes 1/3/5/6/7/8, --no-val, last.pth, port 29621, workers 4->6->8, drop on 1455).
4. Fine-tunes (control + T5 FixRes-style head FT, T1 zoom-biased multiscale, T2 Lovasz/boundary/dice; depth aux head smoke if mountain is the main error). NOTE: plus5 has seen Scene5 -> select fine-tune recipes on FIRST-ROUND weights with Scene5, freeze, then apply to plus5 weights; final test once on Moon_2/4/9. Never select on Scene5 with plus5 weights.
5. Later long trains per plan (T6 CroW-style fg sampling merged with crater sampler, depth head, then copy-paste/T7 lowest). ROW/BLV kept (lit review 23:22: no clear replacement).

## 2026-11-10 00:27 Asia/Shanghai user decision: official-split comparison (LAST step)
- After all of: inference I1-I3, plus5, fine-tunes, later long trains -> run ONE paper-only comparison on the LuSNAR paper's official split: train Moon_1/2/4/6/8/9, test Moon_3/5/7 (paper SegFormer baseline mIoU 89.96%).
- Same recipe as the final pipeline; separate work_dir (e.g. work_dirs/train_official_split_*) and results/official_split_*; NEVER used for method/recipe selection or competition submission.
- This is the only sanctioned exception to "never touch Moon_2/4/9 labels". Re-confirm with the user right before launching.
- Plus5 fine-tune: recipe selected on first-round weights with Scene5, then frozen and applied to plus5 (user confirmed 2026-11-09 23:57).

- Folder: ALL official-split files go ONLY in D:\LuSNAR_Seg\official_split_compare\ (lists/work_dirs/results/logs/scripts); never mixed with main outputs (user 2026-11-10 00:28).

## 2026-11-10 09:43 Asia/Shanghai user decisions: backbone smoke (results/backbone_postproc_plan.md)
- Backbone smokes (SegFormer-B5, Mask2Former Swin-B HF): ALLOWED to run 2 single-GPU smokes in parallel (one per GPU). This is a sanctioned exception to "no second concurrent train" for these smokes only.
- Winning backbone 20k: if VRAM insufficient, lower per-GPU batch 48 -> 24 (user-approved exception to batch 48 rule for that run).
- Allowed: clone env lusnar -> lusnar_bb and install transformers (4.49.0) there; never modify env lusnar.
- Order unchanged: inference phase -> plus5 (Windows) -> backbone smokes. Env clone/weight download/adapter code may be prepared on CPU during plus5.
- Still pending from user: contest rules (inference time limit, ensemble allowed, test-image pseudo-labels allowed).

## 2026-11-10 10:14 Asia/Shanghai user: add I4 hybrid full-image + tiles
- User asked why the official tiny baseline (DeepLabV3+ MobileNetV3-L 4.71M, 1024 whole-image) scores S=0.8046, and to ADD a hybrid inference: full-image for large classes (mountain/sky/regolith), tiled/sliced for small classes (rock/crater), then merge.
- Added as pre-registered I4 on first-round best.pth, Scene5 A-select / B-report-once only. Do not use I4 selection on Moon_2/4/9. Still no plus5 until inference phase is reported.

## 2026-11-10 10:29 Asia/Shanghai user: T8 MSA head; 512 is not a model constraint
- Crop 512 is a RECIPE choice (batch 48, 2x4090 memory), not a SegFormer/MiT requirement. The backbone accepts variable sizes (typically multiples of 32). Official MoonSeg baseline trains/infers at 1024x1024 whole image. User said IGNORE the train-512 vs infer-1024 mismatch risk.
- T8 (new fine-tune, after inference I1-I4 and plus5, recipe selected on FIRST-ROUND weights + Scene5 A only, then frozen onto plus5 weights): add a light hierarchical multi-scale attention head (Tao et al. 2020, arXiv:2005.10821) that predicts per-pixel fusion weights over full-image vs tiled/high-res branches; end-to-end short fine-tune (2k-4k iters) with an unchanged-continuation control at the same iter. Default large/layout vs small-object branches: full-image for mountain/sky/regolith path, tiles/rezoom for rock/crater path, but the head learns soft weights (not a hard class switch). I4 (rule-based merge, no train) still runs first in the current inference phase; T8 is the learned version. Do not use Moon_2/4/9 labels to select T8.

## 2026-11-10 10:34 Asia/Shanghai user: keep crop pipeline; whole-image is a SEPARATE later experiment
- Do NOT change the current crop-512 training recipe or plan (inference I1-I4 -> plus5 -> fine-tunes incl. T8 -> later long trains). Finish it to measure each module's effect.
- AFTER that, open a NEW separate experiment line: whole-image (1024) training, porting the modules/recipe that proved effective in the crop line, for a clean crop-vs-whole-image comparison. Use its own run names/work_dirs/results (e.g. *_wholeimg_*), never mixed with crop-line outputs. Not started now.

## 2026-11-10 10:40 master plan written: results\MASTER_PLAN.md (authoritative stage order 1-8)


## Stage 7 refinement (2026-11-10 10:43, user t205u)
Whole-image line re-smokes ALL crop-line modules, including ones rejected under crop (e.g. SHAD, CTRAST, SAPA, HARD, DYSAMPLE, V_B1E), because context/scale changes can flip results.
Protocol: 7a whole-image 1024 baseline (same B2, same 2k iters, batch reduced to fit, lr scaled; same seed) -> 7b single-module smoke matrix @2k vs that baseline, Scene5 val, same eval as crop line -> 7c pairwise/three-module stacks of whole-image winners -> 7d 20k long train of best stack -> 7e crop-vs-whole table (per module: crop delta vs whole delta). All under fullimage_compare\, never mixed.

## 2026-11-10 10:50 time estimate: contest path 7-9d (to 11-17/19); all-in 11-15d (after deadline). Details in results\MASTER_PLAN.md


## Speed record correction (2026-11-10 11:06, t209u)
Do not treat later Windows workers=4 (~5 s/iter after 1455) as the Windows best. Agreed measured segments (same recipe, batch 48, 2-card DDP) come from training logs. The old claim that early Windows ran at ~0.87 or 0.84 s/iter is NOT in any train log and must not be used as a target.
- First-round Windows start (iter 50-2050) and plus5 Windows pre-stall (~50-1300): ~3.4-3.7 s/iter. This is the measured Windows pace.
- Later Windows workers=4 fallback, and plus5 Windows auto-resume 1050-1100: ~4.9-5.2 s/iter.
- WSL /mnt/d: ~6 s/iter. WSL ext4 workers=8: ~2.5. WSL ext4 workers=12: ~2.18-2.2 (best verified stable).
Plan: plus5 was moved to WSL ext4 workers=12 on 2026-11-10 15:48 Asia/Shanghai, resumed from iter_1000.pth. Do not relaunch plus5 on Windows. Do not chase unverified 0.84.

## 2026-11-10 11:22 Asia/Shanghai user t213u: cheerhin whole-image smoke handoff (not started)
User confirmed: (1) data/code under the same D: project root on both machines, new folder only (planned D:\LuSNAR_Seg\fullimage_compare\); machines already git-synced, most content consistent. (2) VRAM smoke first on 5060 Ti (~16 GiB) to set batch; cannot copy 4090 batch 48 / crop 512 to 1024 whole-image. (3) one-smoke-at-a-time, pause immediately if P0 needs the GPU. cheerhin finishes already-scheduled P0 work first (WP2 seed 1 + queued DT val / WP2 evals), then smokes; no queue-jump, do not start now. Asked LuSNAR to confirm path / git coverage / missing data-weights-env / whole-image 2k scripts.

## 2026-11-10 11:55 Asia/Shanghai: whole-image (1024) smoke scripts written (t215u), NOT launched, NOT committed
- New isolated folder D:\LuSNAR_Seg\fullimage_compare\ (README.md, .gitignore, scripts\fullimage_smoke.py, scripts\run_fullimage_queue.ps1; lists\ work_dirs\ results\ logs\). Whole-image line only; never mixed with crop line / official_split_compare / P0.
- Reuses tools\portable + tools\portable_b3 module code by import (no edits to those files). Includes VRAM probe, OOM batch-halving fallback, 1455 worker fallback, STOP file, resume, module self-check, UTF-8 English-key summaries.
- CPU-only checks passed (results\cpu_check_DESKTOP-ORQLB8F.md). No GPU used; WSL inference not touched.
- Target machine: cheerhin, only after P0's queued work finishes. LuSNAR must review before running; git push needs user approval.
- 2026-11-10 12:31 user decision: competition line now selects each long-train model on open-source Moon_2/4/9 (dev set, not test); Stage 8 paper uses official split, competition results as secondary evidence. See results\MASTER_PLAN.md.

- 2026-11-10 12:32 correction (t226u): official PAPER split is train Moon_1/2/4/6/8/9, test Moon_3/5/7 (not Moon_2/4/9). Moon_2/4/9 is the COMPETITION test set only; in the paper split those three scenes are training data. 12:31 note updated in results\MASTER_PLAN.md.


## Inference phase report written (2026-11-10 12:47:20 Asia/Shanghai)
- Scene5 A/B split seed42: A=799 B=800; lists in data/lists_portable/scene5_ab/
- Diagnostics + I1-I4 + postproc on first-round best.pth done; report: results/inference_phase_report.md
- TTA probs: /root/lusnar_data/cache/scene5_tta_probs/
- Frozen: P1P2_bias_plus_geo; B mIoU=0.8905125750548605
- HARD STOP: no plus5 launched by this report script.

## Inference phase done, plus5 deferred (2026-11-10 12:59:46 Asia/Shanghai)
- Inference report already written 12:47:20: results/inference_phase_report.md. Frozen P1P2_bias_plus_geo; Scene5 B mIoU 0.8905. No plus5 from that script.
- This watch (12:54 routine): first-round 20k is finished (best=iter_17000, tta_final.json present, STAGE1_DONE 09:30:59). No train/plus5/waiter processes. last.pth for first-round is absent (iter_20000.pth exists); plus5 is a fresh run and will write its own last.pth.
- Do NOT start plus5 yet: both 4090s are busy with WSL eval lusnar_opensource_final_eval.py (PID 504, started 12:51, out results/opensource_final_ROW_BLV, 5004 images, 2 GPUs, frozen Scene5 protocols C0/I1a/P1/P1P2, not retuned on Moon_2/4/9). Progress ~220/2502 per GPU at 12:56, ETA about 13:40 Asia/Shanghai. Do not kill it. Do not launch a second long train.
- data/lists_plus5 does not exist yet; the plus5 trainer can build it via build_split_lists + auto RCS weights. When starting plus5, use Windows F:\Conda\envs\lusnar\python.exe, USE_LIBUV=0, port 29621, scenes 1/3/5/6/7/8, --no-val, data/lists_plus5, D: paths only, workers 4 then 6 then 8, pre-check + short smoke first. Never launch run_opensource_followup_wsl.py or 04_relaunch_waiters_wsl.sh. Official blind submit not approved.
- User should be notified of inference results (if not already sent in chat) and that plus5 waits for this eval.

## Open-source final eval still running, plus5 deferred (2026-11-10 13:12 Asia/Shanghai)
- Eval lusnar_opensource_final_eval.py still healthy (PID 504 / workers 654,796, since 12:51). Progress 1060/2502 per GPU at 13:12 (Moon_2+Moon_4 done; Moon_9 next). ETA about 13:40 Asia/Shanghai. Do not kill it. Do not start plus5 or a second long train yet.

## Open-source final eval still running, plus5 deferred (2026-11-10 13:22 Asia/Shanghai)
- First-round 20k finished earlier (best=iter_17000, tta_final.json, STAGE1_DONE 09:30:59). Inference report already written 12:47:20: results/inference_phase_report.md. Frozen P1P2_bias_plus_geo; Scene5 B mIoU 0.8905. No plus5 from that script.
- WSL eval lusnar_opensource_final_eval.py still healthy: parent PID 504, workers 654/796, started 12:51, out results/opensource_final_ROW_BLV, 5004 images, 2 GPUs, frozen Scene5 protocols C0/I1a/P1/P1P2, not retuned on Moon_2/4/9. At 13:22 both ranks 1580/2502 (Moon_2+Moon_4 done; Moon_9 in progress: rank0 121 / rank1 122 images), C0 match 1580/1580, ~1.15 s/img, remaining ~17-18 min, ETA about 13:40 Asia/Shanghai. GPU0 29% 1196 MiB, GPU1 44% 2878 MiB. Do not kill it. Do not launch a second long train or plus5 while it occupies both 4090s.
- No train / plus5 / forbidden waiters (run_opensource_followup_wsl.py or 04_relaunch_waiters_wsl.sh). Ports 29601/29621 free. No Windows python train processes. lists_plus5 still absent.
- Guest mem 31Gi, available 27Gi, swap 0 used. Host free physical ~46.6 GB of ~63.8 GB. dmesg only benign dxgkio_escape -22.
- When starting plus5 after this eval finishes and is reported: Windows F:\Conda\envs\lusnar\python.exe, USE_LIBUV=0, port 29621, scenes 1/3/5/6/7/8, --no-val, data/lists_plus5, D: paths only, workers 4 then 6 then 8, pre-check + short smoke first. Official blind submit not approved.

## Open-source final eval still running, plus5 deferred (2026-11-10 13:35 Asia/Shanghai)
- First-round 20k finished earlier (best=iter_17000, tta_final.json, STAGE1_DONE 09:30:59). Inference report already written 12:47:20: results/inference_phase_report.md. Frozen P1P2_bias_plus_geo; Scene5 B mIoU 0.8905. No plus5 from that script.
- WSL eval lusnar_opensource_final_eval.py still healthy: parent PID 504, workers 654/796, started 12:51, out results/opensource_final_ROW_BLV, 5004 images, 2 GPUs, frozen Scene5 protocols C0/I1a/P1/P1P2, not retuned on Moon_2/4/9. At 13:35 both ranks 2240/2502 (Moon_2+Moon_4 done; Moon_9 in progress: rank0 781 / rank1 782 images), C0 match 2240/2240, ~1.15 s/img, remaining ~5 min, ETA about 13:40-13:41 Asia/Shanghai. GPU0 34% 1196 MiB, GPU1 15% 2886 MiB at check (bursty). Do not kill it. Do not launch a second long train or plus5 while it occupies both 4090s.
- No train / plus5 / forbidden waiters (run_opensource_followup_wsl.py or 04_relaunch_waiters_wsl.sh). Ports 29601/29621 free. No Windows python train processes.
- Guest mem 31Gi, available 27Gi, swap 0 used. dmesg only benign dxgkio_escape -22.
- When starting plus5 after this eval finishes and is reported: Windows F:\Conda\envs\lusnar\python.exe, USE_LIBUV=0, port 29621, scenes 1/3/5/6/7/8, --no-val, data/lists_plus5, D: paths only, workers 4 then 6 then 8, pre-check + short smoke first. Official blind submit not approved.

## Open-source final eval finished (2026-11-10 13:40:16 Asia/Shanghai)

- WSL `lusnar_opensource_final_eval.py` (parent PID 504, workers 654/796, started 12:51) exited on its own and wrote `results/opensource_final_ROW_BLV/summary.json` and `report.md`. Wall 2900.0 s (~48.3 min) on 2 GPUs. Images scored 5004/5004. No missing images. No unmapped GT pixels. UTF-8 English keys in summary.json confirmed.
- Checkpoint: `work_dirs/train_ROW_BLV_20k/best.pth` (iter 17000). Protocols frozen from Scene5 A, not retuned on Moon_2/4/9. Competition-line dev set only; not a paper test (paper test is Moon_3/5/7).
- C0 reproduced the earlier oneshot exactly: overall micro mIoU 0.82989773 (delta 0.0000 vs 0.8299), S 0.83000833 (~0.8300). Per-scene C0: Moon_2 0.9215, Moon_4 0.9136, Moon_9 0.7425. Overall crater IoU 0.5436. C0 pred match vs `results/oneshot_ROW_BLV/preds`: 5004/5004 (rate 1.0000).

| protocol | m2 | m4 | m9 | S=(m2+m4+2*m9)/4 | vs official 0.8046 | overall micro mIoU | rock | crater | mountain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_equal_tta | 0.9215 | 0.9136 | 0.7425 | 0.8300 | +0.0254 | 0.8299 | 0.8174 | 0.5436 | 0.8126 |
| I1a_best_scale_per_class | 0.9248 | 0.9171 | 0.7529 | 0.8369 | +0.0323 | 0.8348 | 0.7972 | 0.5730 | 0.8270 |
| P1_logit_bias | 0.9218 | 0.9123 | 0.7435 | 0.8303 | +0.0257 | 0.8303 | 0.8182 | 0.5464 | 0.8114 |
| P1P2_bias_plus_geo | 0.9214 | 0.9138 | 0.7333 | 0.8254 | +0.0208 | 0.8236 | 0.7980 | 0.5463 | 0.7976 |

- Best converted S on this open-source set is I1a (0.8369, +0.0323 vs official 0.8046), mainly from crater (0.5730 vs C0 0.5436; Moon_9 crater 0.3103 vs 0.2589) and mountain, with a rock tradeoff (0.7972 vs C0 0.8174). Scene5 winner P1P2 did not transfer: S 0.8254 is below C0, mainly Moon_9 0.7333 vs 0.7425 (rock/mountain drop). P1 is essentially tied with C0.
- After eval: no remaining eval processes; GPU0 0 MiB, GPU1 ~1.6 GiB desktop with no compute apps; ports 29601/29621 free. plus5 is a fresh Windows run (no existing plus5 last.pth). Official blind submit not approved.

## plus5 Windows long train started (2026-11-10 13:47:37 Asia/Shanghai)

- Pre-check OK: `data/lists_plus5` built on Windows, 8002 train pairs (Moon_1/3/5/6/7/8 = 1094/1230/1599/1102/1626/1351), empty val list (`--no-val`), all `D:\LuSNAR_Seg\data\train\` paths, files exist, no Moon_2/4/9. RCS `full_crater_weights.json` 8002 D: pairs, cache match. Ports 29601/29621 were free. No existing plus5 `last.pth` (fresh run, not a resume). `work_dirs\train_B1` not touched (mtime still 2026-11-09 16:43:52).
- Short smoke `--dry-run` 5 iters, run-name `train_ROW_BLV_20k_plus5_smoke`, same recipe/list-dir, exit 0 at 13:46:11. ROW_BLV_EFFECT ok. Loss 2.612 → 2.483. mem 36509.6 MiB.
- Long train launched detached via WMI Win32_Process.Create (not Start-Process). Fresh 20k, workers=4, port 29621, USE_LIBUV=0, F:\Conda\envs\lusnar\python.exe, scenes 1,3,5,6,7,8, `--no-val`, D: paths only. PYTORCH_CUDA_ALLOC_CONF unset. No WSL followup/waiters, no official blind submit.
- Start 2026-11-10 13:47:37 Asia/Shanghai. Launcher cmd PID 19060 (parent WmiPrvSE); torch.distributed.run PID 4872 listening on 29621; workers 11544 / 34196. Log: `logs/train_ROW_BLV_20k_plus5.log`. Work dir: `work_dirs/train_ROW_BLV_20k_plus5`. Bat: `logs/launch_plus5.bat`.
- Cmd: `F:\Conda\envs\lusnar\python.exe -u -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 29621 tools\portable\lusnar_row_blv_ddp.py --recipe M_ROW_BLV --run-name train_ROW_BLV_20k_plus5 --iters 20000 --batch 48 --workers 4 --amp bf16 --crop 512 --lr 6e-5 --val-every 1000 --train-scenes 1,3,5,6,7,8 --no-val --list-dir D:\LuSNAR_Seg\data\lists_plus5 --project-root D:\LuSNAR_Seg --data-root D:\LuSNAR_Seg\data --train-subdir train`
- Expected DESKTOP-ORQLB8F socket 10049 warning appeared; workers still connected via 127.0.0.1:29621. Known DDP grad-stride warning, not an error. No 1455 so far.
- iter 1: loss 2.6124, elapsed 37.8 s (warmup), mem 36509.6 MiB. iter 50: loss 1.6054, elapsed 191.8 s (3.14 s/iter over 1→50). iter 100: loss 1.1921, elapsed 371.9 s (3.60 s/iter over 50→100; 3.37 s/iter over 1→100). Designated ckpt is last.pth (no val). At ~3.4 s/iter, 20k is roughly 19 h, around 08:50 2026-11-11 Asia/Shanghai if this rate holds. Workers left at 4 for now (raise 4→6→8 later; drop on 1455).

## plus5 stall auto-resume (2026-11-10 15:46 Asia/Shanghai)
- Stuck plus5 Windows DDP (cmd 19060, launcher 4872, ranks 11544/34196, port 29621, workers=4) had no new train_log.jsonl after iter 1300 / loss 0.1229 at 15:06:24. At 15:30-15:35: GPU0 ~0% 11W 210 MHz, GPU1 ~6-7% ~69W 2520 MHz, processes and 8 DataLoader workers still alive, no Event 1455 or Resource-Exhaustion 2004, WSL Ubuntu Stopped, commit 112/192 GB, free phys ~39 GB. Stall exceeded the 20-minute rule (last watch at 15:14 was only ~9 min quiet).
- 15:35:31 killed process tree of cmd 19060 (all ranks and workers). GPU memory released (GPU0 0 MiB, GPU1 ~1.9 GiB desktop). Ports 29601/29621 free.
- 15:36:56 resumed via WMI Win32_Process.Create, bat logs\launch_plus5_resume.bat, from work_dirs\train_ROW_BLV_20k_plus5\iter_1000.pth (newest iter_*.pth; no last.pth; iters 1001-1300 not saved). Same recipe M_ROW_BLV, batch 48, workers 4, port 29621, USE_LIBUV=0, F:\Conda\envs\lusnar\python.exe, scenes 1/3/5/6/7/8, --no-val, D: paths only. No PYTORCH_CUDA_ALLOC_CONF. Did not launch run_opensource_followup_wsl.py or 04_relaunch_waiters_wsl.sh. Did not touch work_dirs\train_B1 or train_ROW_BLV_20k.
- Live PIDs: cmd 4752, torch.distributed.run 12256, ranks 14736/27580, 4 DataLoader workers per rank. Resume event from_iter 1000, resume_miou None (no-val). iter 1050 at 15:41:28 loss 0.1499 elapsed 255.6 s mem 36797.2 MiB; iter 1100 at 15:45:46 loss 0.1461 elapsed 513.9 s (~5.2 s/iter over 1050-1100, slower than earlier ~3.6 s/iter). If ~5.2 s/iter holds, 20k ETA around 18:50 2026-11-11 Asia/Shanghai; if it returns to ~3.6 s/iter, around late morning 11-11. Do not start a second long train. Official blind submit not approved. Next watch: if GPU util ~0% and no new jsonl iters for >20 minutes, kill and resume from newest iter_*.pth the same way; try workers 6 then 8 only after a healthy stretch; drop on 1455/shared-memory.

## plus5 moved to WSL ext4 (2026-11-10 15:48 Asia/Shanghai)
- Windows plus5 stalled after iter 1300 at 15:06:24, then the watch auto-resumed on Windows from iter_1000.pth at 15:37. That resume ran at ~5.2 s/iter (1050-1100). It was stopped at 15:47:49 (cmd 4752 / launcher 12256 / ranks 14736,27580). GPUs cleared. work_dirs/train_B1 and train_ROW_BLV_20k were not touched.
- WSL resume launched 15:48:37 with tools/wsl_bootstrap/03b_resume_plus5_wsl.sh: env /opt/miniconda3/envs/lusnar, port 29621, workers=12, batch 48, --no-val, run-name train_ROW_BLV_20k_plus5, data /root/lusnar_data, lists /root/lusnar_data/lists_plus5 (8002 pairs, Moon_1/3/5/6/7/8), resume work_dirs/train_ROW_BLV_20k_plus5/iter_1000.pth, log logs/train_ROW_BLV_20k_plus5_wsl.log, USE_LIBUV=0, no PYTORCH_CUDA_ALLOC_CONF, setsid. Confirmed resume_iter 1000. Live log is the WSL file; the Windows log is stale.
- Next watch: treat WSL plus5 on 29621 as the one healthy train. Do not start Windows plus5. If this WSL run dies, relaunch only 03b_resume_plus5_wsl.sh from the newest iter_*.pth. Official blind submit not approved.

## plus5 WSL healthy (2026-11-10 15:57 Asia/Shanghai)
- Live WSL plus5 only: launcher PID 521, ranks 571/572, 12 DataLoader workers per rank, port 29621, started 15:48:37 from work_dirs/train_ROW_BLV_20k_plus5/iter_1000.pth. Recipe M_ROW_BLV, batch 48, --no-val, data /root/lusnar_data, lists /root/lusnar_data/lists_plus5, run-name train_ROW_BLV_20k_plus5, log logs/train_ROW_BLV_20k_plus5_wsl.log (Windows log is stale). No Windows python train. No forbidden waiters. Did not touch work_dirs/train_B1 or train_ROW_BLV_20k.
- Progress: resume event from_iter 1000; iter 1050 @151.8s loss 0.1567; 1100 @277.3s loss 0.1246; 1150 @394.4s loss 0.1338; 1200 @15:57:13 loss 0.1438 elapsed 506.4s mem 36796.4 MiB. Pace 1150-1200 = 2.24 s/iter; 1050-1200 = 2.36 s/iter. GPU mem ~39316 / 41198 MiB; util bursty then both ~99%. Guest avail 3.5Gi, swap 3.9Gi used, /dev/shm 1%. Host free phys ~21.9/63.8 GB. dmesg only 15:48 start-up dxg -22, no OOM.
- Newest ckpt still iter_1000.pth (no last.pth yet). If this dies, relaunch only tools/wsl_bootstrap/03b_resume_plus5_wsl.sh from newest iter_*.pth, workers=12 (ladder 12->8 on DataLoader/shm OOM). Do not start Windows plus5 or a second train. Official blind submit not approved. Rough ETA ~2.24 s/iter x 18800 remaining ~11.7 h, around 03:40 2026-11-11 Asia/Shanghai if this rate holds. Deadline 2026-11-12 12:00 not reached; not ALL DONE.

## USER PAUSE after iter_2000 (requested 2026-11-10 16:28 Asia/Shanghai)
The user asked to stop plus5 as soon as iter_2000.pth is saved, so they can use the computer. After this pause, resume only when the user says so, from iter_2000.pth, WSL ext4, workers=14 (ladder 14->12 on shm/mem thrash). Do NOT auto-relaunch after the stop. Do NOT start Windows plus5. Do not touch work_dirs/train_B1 or train_ROW_BLV_20k. Official blind submit not approved.


## plus5 PAUSED at iter_2000 (2026-11-10 16:31 Asia/Shanghai)
User-requested pause is in effect. Do NOT auto-relaunch plus5 on WSL or Windows until the user explicitly says to resume.
- iter_2000.pth saved 16:30:11 Asia/Shanghai, size 297113142 bytes, path work_dirs/train_ROW_BLV_20k_plus5/iter_2000.pth. No last.pth. train_log.jsonl ends at iter 2000, loss 0.0954187661409378, elapsed_s 2479.9, mem_mb 36796.4. Log also has checkpoint iter_2000 / no_val true.
- WSL plus5 (launcher PID 521, ranks 571/572, port 29621, workers=12, resumed 15:48:37 from iter_1000.pth) received SIGTERM at 16:30:52, after the checkpoint was written. By 16:31:41: no plus5 processes, port 29621 free, GPU0 0 MiB / 0% / 12 W, GPU1 ~1736 MiB desktop / 1% / no compute apps. No Windows plus5. Did not launch 03b_resume_plus5_wsl.sh, launch_plus5.bat, or launch_plus5_resume.bat.
- Did not touch work_dirs/train_B1 (mtime still 2026-11-09 16:43:52) or work_dirs/train_ROW_BLV_20k (mtime still 2026-11-10 07:57:44). Official blind submit not approved.
- When the user says resume: WSL ext4 only, tools/wsl_bootstrap/03b_resume_plus5_wsl.sh, resume iter_2000.pth, workers=14 (ladder 14->12 on shm/mem thrash), batch 48, port 29621, same lists/data /root/lusnar_data. Do not start Windows plus5.

## plus5 RESUMED after reboot (2026-11-10 17:36 Asia/Shanghai)

USER_PAUSE is cleared. The user asked to reboot this machine and then continue plus5, and to try to max out CPU and GPU before/during resume.

- Host last boot: 2026-11-10 17:24:20 Asia/Shanghai. Reconnected about 17:26. `.wslconfig` memory raised 32GB -> 40GB (backup `C:\Users\Administrator\.wslconfig.bak-2026-11-10`; processors=48, swap=32GB, localhostForwarding=true unchanged). Guest `free -h` shows 39Gi total (MemTotal 41061904 kB).
- plus5 resumed on WSL ext4 only, not Windows, at 17:27:55 Asia/Shanghai via `tools/wsl_bootstrap/03b_resume_plus5_wsl.sh` (Unix LF; CKPT=iter_2000.pth; WORKERS=14). Port 29621, batch 48, --no-val, USE_LIBUV=0, PYTORCH_CUDA_ALLOC_CONF unset, data `/root/lusnar_data`, lists `/root/lusnar_data/lists_plus5` (8002 pairs), run-name `train_ROW_BLV_20k_plus5`. Log: `logs/train_ROW_BLV_20k_plus5_wsl.log` (Windows plus5 log is stale). Recipe unchanged (SegFormer-B2 + V_B1C + ROW + BLV, crop 512, lr 6e-5, bf16, scenes 1/3/5/6/7/8).
- Live: launcher PID 569, ranks 619/620, 14 DataLoader workers per rank, port 29621. resume_iter 2000 from `work_dirs/train_ROW_BLV_20k_plus5/iter_2000.pth` (297113142 bytes, mtime 16:30:11). No last.pth. Did not resume from iter_1000.pth.
- Progress (Asia/Shanghai): iter 2050 around 17:31, loss 0.08467, elapsed_s 154.5 (warmup included); iter 2100, loss 0.10406, elapsed_s 271.1; iter 2150 around 17:35, loss 0.07338, elapsed_s 362.3, mem_mb 36796.4. Settled pace 2100-2150 = 1.82 s/iter; 2050-2150 = 2.08 s/iter (first 50 after resume still includes leftover warmup). Kept workers=14; did not step to 16 (GPUs already high-90s, host CPU 83-100%, WSL swap in use, no clear unused RAM headroom).
- GPU (healthy samples): 17:31:32 GPU0 99% 280 W 40819 MiB, GPU1 99% 284 W 39339 MiB; 17:36:04 GPU0 99% 264 W 40960 MiB, GPU1 97% 268 W 39339 MiB. A 17:35:49 reading showed GPU1 0% / 14 W with VRAM still held — treat as a spurious nvidia-smi dip, not a stall.
- Host: 48 threads, CPU 83% (17:31) / 100% (17:33) / 87% (17:36). RAM about 63.8 GB total, free physical about 13.2-13.5 GB.
- WSL at 17:35:50: Mem 39Gi total, about 32Gi used, shared 25Gi, available 6.2Gi, swap 3.7Gi used (climbed 1.6Gi at 17:31 -> 3.3Gi at 17:33 -> 3.7Gi at 17:35, then slowing; not treated as thrash). Load 28.31 / 21.93 / 10.93.
- Did not start a second train. Did not launch `run_opensource_followup.py`, `run_opensource_followup_wsl.py`, `04_relaunch_waiters_wsl.sh`, or any Windows plus5 bat. Did not touch `work_dirs/train_B1` or `train_ROW_BLV_20k`. Official blind submit not approved.
- If this run dies: relaunch only `03b_resume_plus5_wsl.sh` from the newest intact iter_*.pth, workers=14 (ladder 14->12 on shm/mem thrash). Do not start Windows plus5. Rough ETA about 1.82 s/iter x 17850 remaining about 9.0 h, around 02:40 2026-11-11 Asia/Shanghai if this rate holds.
