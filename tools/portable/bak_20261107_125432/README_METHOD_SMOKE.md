# Method smoke recipes (V_B1C / M_ROW / M_BLV / M_TTA)

Scene5 val_sub only. **Never** read Moon_2 / 4 / 9 labels.
Control for new methods = **V_B1C** (B1 + crater RCS/ACW boost).
Official DeepLab numbers stay the formal baseline; these smokes are relative only.

| recipe | vs V_B1C | train? | infer cost | targets |
|--------|---------|--------|------------|---------|
| **V_B1C** | control | yes | same as B1 | crater sampling/weight |
| **M_ROW** | + RowHeightAttn (HANet-style vertical prior on logits) | yes | +tiny MLP | sky↔mountain |
| **M_BLV** | + BLV logit variation (train-only, σ=0.5) | yes | **zero** | crater/rock tails |
| **M_TTA** | sliding-window + multi-scale eval on **V_B1C** weights | **no** (trains V_B1C if `best.pth` missing) | ×3–5 | small-object diagnosis |

Default: iters **2000**, batch **8**, lr **6e-5**, crop **512**, val_sub **350**.

## Cheerhin PowerShell

```powershell
cd D:\LuSNAR_Seg

# optional: kill leftover smoke
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'lusnar_smoke_compare|train_smoke_matrix' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

# dry-run syntax/GPU check (few iters)
F:\Conda\envs\lusnar\python.exe D:\LuSNAR_Seg\tools\portable\lusnar_smoke_compare.py `
  --exps V_B1C --iters 2 --batch 2 --dry-run `
  --project-root D:\LuSNAR_Seg `
  --data-root D:\LuSNAR_Seg\data\raw\LuSNAR_train `
  --mem-fraction 0.92

# full method smoke matrix (2k)
F:\Conda\envs\lusnar\python.exe D:\LuSNAR_Seg\tools\portable\lusnar_smoke_compare.py `
  --exps V_B1C,M_ROW,M_BLV,M_TTA `
  --iters 2000 --batch 8 --lr 6e-5 `
  --project-root D:\LuSNAR_Seg `
  --data-root D:\LuSNAR_Seg\data\raw\LuSNAR_train `
  --mem-fraction 0.92
```

If Moon_* sits under `data\raw\LuSNAR_train` (cheerhin layout), the `--data-root` above is correct.
If you use a `data\train\Moon_*` junction instead:

```powershell
  --data-root D:\LuSNAR_Seg\data
```

Reuse existing `data\smoke\` assets with `--skip-prepare` when already built.

## Sync portable folder

Copy `/workspace/lusnar_portable/*` → `D:\LuSNAR_Seg\tools\portable\` (overwrite).
Required files: `_lusnar_common.py`, `lusnar_smoke_compare.py`, `mmcv_stub.py`, this README.

## Outputs

```
work_dirs/smoke_V_B1C/{best.pth,summary.json,train_log.jsonl}
work_dirs/smoke_M_ROW/...
work_dirs/smoke_M_BLV/...
work_dirs/smoke_M_TTA/summary.json   # includes tta_vs_single
results/smoke_method_eval.md
```

Each summary has `per_class_IoU_best` (all 5), `confusion_rates`
(`sky_to_mountain`, `crater_to_regolith`, …), and flags.

## Pass criteria (vs V_B1C @ same iters; Δ&lt;0.02 = noise)

1. **M_ROW**: sky→mountain &lt; 12% **and** mountain IoU +≥0.03; other classes drop ≤0.01
2. **M_BLV**: crater→regolith &lt; 7% **or** crater IoU +≥0.02
3. **M_TTA**: if small-rock bucket +≥0.05 → pursue FADE/HRDA; portable stack has **no** size-bucket helper yet (noted in summary)

## Blockers

- cheerhin / 5060 Ti must be online with `lusnar` env + mit_b2 pretrain.
- M_TTA needs `work_dirs\smoke_V_B1C\best.pth`; script trains V_B1C first if missing.
