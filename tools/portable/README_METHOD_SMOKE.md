# Method smoke recipes (extended)

Scene5 `val_sub` only. **Never** read Moon_2 / 4 / 9 labels.
Control for new methods = **V_B1C** (B1 + crater RCS/ACW boost).
Official DeepLab numbers stay the formal baseline; these smokes are relative only.

| recipe | vs V_B1C | train? | infer cost | targets |
|--------|---------|--------|------------|---------|
| **V_B1C** | control | yes | same as B1 | crater sampling/weight |
| **M_ROW** | + RowHeightAttn (HANet-style vertical prior on logits) | yes | +tiny MLP | sky↔mountain |
| **M_BLV** | + BLV logit variation (train-only, σ=0.5) | yes | **zero** | crater/rock tails |
| **M_SHAD** | + `shadow_augment` (gamma / lower-half / ellipses) | yes | **zero** | crater shadow / regolith |
| **M_ABL** | + ABL-lite BCE/Dice on soft boundary vs GT edge (ramp after 20% iters) | yes | **zero** | boundaries |
| **M_INVFORM** | stub → **falls back to ABL-lite** (no InverseForm net) | yes | zero | same as M_ABL |
| **M_FADE** | + FADE-lite (logits ⊕ backbone stage-0 → 3×3 residual) | yes | +tiny conv | small objects (gate after M_TTA) |
| **M_TTA** | sliding-window + multi-scale on **V_B1C** weights | **no** (trains V_B1C if `best.pth` missing) | ×3–5 | small-object diagnosis |
| **M_SEGFIX** | SegFix-lite boundary majority postprocess | **no** | cheap | boundary cleanup |
| **T_B1_TTA** | same TTA protocol on `work_dirs/train_B1/best.pth` | **no** | ×3–5 | long-train TTA; **fails if ckpt missing** (no auto 20k) |
| **M_HRDA** | **NOT implemented** | — | — | gate: M_TTA pass **and** M_FADE insufficient |

Default method smoke: iters **2000**, batch **16** (4090) or **8** (cheerhin 16GB), lr **6e-5**, crop **512**.

## 4090 PowerShell (primary)

```powershell
cd D:\LuSNAR_Seg

# Sync first: copy lusnar_portable/* → D:\LuSNAR_Seg\tools\portable\

# Wave A — already synced baseline methods
D:\path\to\lusnar\python.exe D:\LuSNAR_Seg\tools\portable\lusnar_smoke_compare.py `
  --exps V_B1C,M_ROW,M_BLV,M_TTA `
  --iters 2000 --batch 16 --lr 6e-5 `
  --project-root D:\LuSNAR_Seg `
  --data-root D:\LuSNAR_Seg\data `
  --skip-prepare

# Wave B — new train smokes (shadow / ABL / FADE-lite)
D:\path\to\lusnar\python.exe D:\LuSNAR_Seg\tools\portable\lusnar_smoke_compare.py `
  --exps M_SHAD,M_ABL,M_FADE `
  --iters 2000 --batch 16 --lr 6e-5 `
  --project-root D:\LuSNAR_Seg `
  --data-root D:\LuSNAR_Seg\data `
  --skip-prepare

# Wave C — eval-only (needs V_B1C best + optional train_B1 best)
D:\path\to\lusnar\python.exe D:\LuSNAR_Seg\tools\portable\lusnar_smoke_compare.py `
  --exps M_SEGFIX,T_B1_TTA `
  --iters 2000 --batch 16 `
  --project-root D:\LuSNAR_Seg `
  --data-root D:\LuSNAR_Seg\data `
  --skip-prepare

# One-shot example from the task brief
D:\path\to\lusnar\python.exe D:\LuSNAR_Seg\tools\portable\lusnar_smoke_compare.py `
  --exps M_SHAD,M_ABL,M_SEGFIX,T_B1_TTA `
  --iters 2000 --batch 16 --lr 6e-5 `
  --project-root D:\LuSNAR_Seg `
  --data-root D:\LuSNAR_Seg\data
```

Optional overrides:

- `--ckpt work_dirs\smoke_V_B1C\best.pth` for M_SEGFIX / M_TTA
- `--ckpt work_dirs\train_B1\best.pth` for T_B1_TTA (default already)
- `--abl-weight 0.5` for M_ABL / M_INVFORM
- `--blv-sigma 0.5` for M_BLV
- `--tta-scales 0.75,1.0,1.25 --tta-crop 512 --tta-stride 256`

If Moon_* sits under `data\raw\LuSNAR_train` (cheerhin):

```powershell
  --data-root D:\LuSNAR_Seg\data\raw\LuSNAR_train
```

## Cheerhin (16GB) note

Use `--batch 8 --mem-fraction 0.92`. Same `--exps` strings.

## Sync portable folder

Copy `/workspace/lusnar_portable/*` → `D:\LuSNAR_Seg\tools\portable\` (overwrite).
Required: `_lusnar_common.py`, `lusnar_smoke_compare.py`, `mmcv_stub.py`, this README.

## Outputs

```
work_dirs/smoke_V_B1C/{best.pth,summary.json,train_log.jsonl}
work_dirs/smoke_M_SHAD/...
work_dirs/smoke_M_ABL/...
work_dirs/smoke_M_SEGFIX/summary.json   # segfix_before_after
work_dirs/smoke_T_B1_TTA/summary.json   # tta_vs_single
work_dirs/smoke_M_FADE/...
work_dirs/smoke_M_HRDA/summary.json     # error: not implemented
results/smoke_method_eval.md
```

## Pass criteria (vs V_B1C @ same iters; |Δ|<0.02 = noise at 2k)

1. **M_ROW**: sky→mountain < 12% **and** mountain IoU +≥0.03
2. **M_BLV**: crater→regolith < 7% **or** crater IoU +≥0.02
3. **M_SHAD**: crater IoU +≥0.02 **or** crater→regolith drop ≥1pp
4. **M_ABL**: no crater crash; prefer rock/boundary gain without rock↑ crater↓ war
5. **M_SEGFIX**: ΔmIoU vs same ckpt +≥0.005 without crater collapse
6. **M_TTA / T_B1_TTA**: if small-rock bucket +≥0.05 → pursue FADE/HRDA (buckets not in portable yet)
7. **M_FADE**: only interpret after M_TTA Δ above noise
8. **M_HRDA**: not runnable — do not invent scores

## Gates / not implemented

- **Full InverseForm** (learned boundary offset net): not shipped → M_INVFORM = ABL-lite fallback
- **Full FADE / SAPA / HRDA**: M_FADE is FADE-lite only; **M_HRDA deliberately unimplemented** until M_TTA pass ∧ M_FADE insufficient
- Small-object IoU buckets: still absent in portable stack
