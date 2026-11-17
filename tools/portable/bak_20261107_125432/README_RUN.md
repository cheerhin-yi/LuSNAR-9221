# LuSNAR portable scripts — run book

Files in this folder:

| file | role |
|------|------|
| `lusnar_smoke_compare.py` | Ablation smoke A0 / A1 / B1 (C1 optional) |
| `lusnar_train_ddp.py` | Formal train; single GPU or `torchrun` 2×4090 DDP |
| `prepare_smoke_assets.py` | Build smoke subset + rarity + CC bank |
| `mmcv_stub.py` | Required with mmcv-lite |
| `_lusnar_common.py` | Shared helpers (do not delete) |

Class order (official): 0 月壤, 1 岩石, 2 撞击坑, 3 山体, 4 天空.

**Never** read Moon_2 / 4 / 9 labels. Scene5 is local val only.

---

## Data layouts

**A. Portable (recommended on 4090 server)**

```
<data-root>/
  train/Moon_{1,3,5,6,7,8}/{color,label}/*.png
  test/                         # reserved; scripts never read labels here
```

Lists look like: `train/Moon_1/color/xxx.png<TAB>train/Moon_1/label/xxx.png`

**B. Cheerhin bridge (current machine)**

`--data-root D:\LuSNAR_Seg\data\raw\LuSNAR_train`  
(Moon_* sits directly under that folder — no extra `train/` prefix)

Also OK: `--data-root D:\LuSNAR_Seg\data\raw --train-subdir LuSNAR_train`

---

## 0) Kill leftover smoke (cheerhin only, if GPU busy)

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'train_smoke_matrix' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

---

## 1) Cheerhin dry-run verify (must pass before long train)

```powershell
cd D:\LuSNAR_Seg

F:\Conda\envs\lusnar\python.exe D:\LuSNAR_Seg\tools\portable\lusnar_smoke_compare.py --help

F:\Conda\envs\lusnar\python.exe D:\LuSNAR_Seg\tools\portable\lusnar_smoke_compare.py `
  --exp A0 --iters 2 --batch 2 --dry-run `
  --project-root D:\LuSNAR_Seg `
  --data-root D:\LuSNAR_Seg\data\raw\LuSNAR_train
```

Expect: prints project/scene_root/pretrain, runs 1–2 train steps, writes
`work_dirs\smoke_A0\summary.json` with `"dry_run": true`, exit code 0.

Helpful: reuse existing smoke assets with `--skip-prepare` if `data\smoke\` already present.

---

## 2) Cheerhin short ablation (A0 vs A1 or A0 vs B1)

```powershell
# A0 baseline vs A1 (rare sampling + mosaic + CC paste)
F:\Conda\envs\lusnar\python.exe D:\LuSNAR_Seg\tools\portable\lusnar_smoke_compare.py `
  --exps A0,A1 --iters 200 --batch 8 `
  --project-root D:\LuSNAR_Seg `
  --data-root D:\LuSNAR_Seg\data\raw\LuSNAR_train `
  --mem-fraction 0.92

# or baseline vs long-tail ACW (B1 includes A1 features + ACW)
F:\Conda\envs\lusnar\python.exe D:\LuSNAR_Seg\tools\portable\lusnar_smoke_compare.py `
  --exps A0,B1 --iters 200 --batch 8 `
  --project-root D:\LuSNAR_Seg `
  --data-root D:\LuSNAR_Seg\data\raw\LuSNAR_train `
  --mem-fraction 0.92
```

Bring back for analysis:

```
work_dirs/smoke_A0/summary.json
work_dirs/smoke_A1/summary.json   # or smoke_B1
work_dirs/smoke_*/train_log.jsonl
work_dirs/smoke_*/best.pth
results/smoke_matrix_report.md
```

`summary.json` fields: `best_mIoU_scene5`, `per_class_IoU_best`, `iou_rock`,
`iou_crater`, `peak_mem_mb`, `wall_sec`, `flags`, `error`.

---

## 3) Dual 4090 (48G) formal train

Assume portable layout under project `data/train/...` and same conda stack
(torch 2.7+cu128, mmseg 1.2.2, mmcv-lite + stub, timm).

```bash
cd /path/to/LuSNAR_Seg

# single-GPU sanity
python tools/portable/lusnar_train_ddp.py \
  --recipe A0 --iters 5 --batch 2 --dry-run \
  --project-root . --data-root ./data

# 2×4090 DDP — recipe A1 (sampling+paste) or B1 (+ACW); C1 optional later
torchrun --nproc_per_node=2 tools/portable/lusnar_train_ddp.py \
  --recipe A1 \
  --iters 20000 \
  --batch 16 \
  --crop 512 \
  --workers 8 \
  --amp bf16 \
  --val-every 1000 \
  --project-root . \
  --data-root ./data
```

Outputs: `work_dirs/train_<recipe>/summary.json`, `best.pth`, `train_log.jsonl`,
`results/train_report_train_<recipe>.md`.

---

## Recipe flags

| recipe | RCS + mosaic + CC paste | ACW long-tail | EPS + multi-scale + photo + EMA |
|--------|-------------------------|---------------|----------------------------------|
| A0     |                         |               |                                  |
| A1     | yes                     |               |                                  |
| B1     | yes                     | yes           |                                  |
| C1     | yes                     | yes           | yes (optional; default off)      |

First ablation recommended by user: **baseline + one or two of A1 / B1**.
