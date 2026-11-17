# fullimage_compare CPU checks on DESKTOP-ORQLB8F (2026-11-10 11:55 Asia/Shanghai)

CPU only (--cpu, CUDA hidden, 4 threads, BelowNormal priority). No GPU used; running WSL Scene5 inference untouched.
Dry runs downscale to 256x256 (CPU speed only); real runs are native 1024x1024 (all list images checked 1024x1024).

| Check | Result |
|---|---|
| py_compile fullimage_smoke.py / PowerShell parse of run_fullimage_queue.ps1 | OK / 0 errors |
| --prepare-lists (copy data\smoke -> lists\DESKTOP-ORQLB8F, 1600/350, seed 42, scene-leak check, all files exist) | OK |
| --check-only, all 16 training exps + 3 eval-only | all OK |
| --dry-run FI_V_B1C, FI_M_ROW_BLV (workers=2, Windows spawn) | OK; ROW grad 4.66, ROW weight moved, BLV CE delta 0.128 |
| --dry-run V_B1E, ROW, BLV, SHAD, ABL, INVFORM, FADE, HARD, DYSAMPLE, CTRAST, SAPA | all OK, every module self-check passed |
| STOP mid-run -> exit 3 + last.pth; start with STOP present -> exit 3; resume -> finished 40/40 | OK |
| eval-only FI_M_TTA / FI_M_SEGFIX / FI_M_MSINFER on a dry checkpoint | OK (whole_1.0 equals training val) |
| DataLoader 1455 simulation (worker raises WinError 1455) | workers 2 -> 1 -> 0, logged |
| queue launcher with STOP present | exit 3 before touching anything |
| summary JSON read by ConvertFrom-Json (UTF-8, English class keys) | OK |

Not testable without a GPU: VRAM probe, train-step OOM halving, val OOM -> slide fallback (code paths reviewed only).
Dry-run checkpoints were deleted after the test; summaries are kept in work_dirs\*_dryrun.