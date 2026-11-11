# Method smoke eval (Scene5)

Protocol: Scene5 val_sub only; never Moon_2/4/9. Control = V_B1C.
M_ROW / M_BLV / M_TTA are relative to V_B1C; 2k-iter smoke, Δ<0.02 = noise.

| exp | mIoU | rock | crater | mountain | sky | sky→mtn | crater→reg | wall_s | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V_B1C | 0.8014 | 0.7228 | 0.7407 | 0.6530 | 0.9180 | 0.0587 | 0.0491 | 12070.2 | B1 + crater RCS/ACW boost (control) |
| M_ROW | 0.8073 | 0.7572 | 0.7661 | 0.6390 | 0.8992 | 0.0900 | 0.0378 | 12509.8 | V_B1C + RowHeightAttn (HANet-style) |
| M_BLV | 0.8041 | 0.7540 | 0.7235 | 0.6801 | 0.8901 | 0.0987 | 0.0380 | 12055.9 | V_B1C + BLV logit variation (train-only) |
| M_SHAD | 0.8131 | 0.7397 | 0.7429 | 0.6891 | 0.9192 | 0.0542 | 0.0695 | 12724.3 | V_B1C + shadow_augment (train-only) |
| M_ROW_SHAD | 0.8083 | 0.7420 | 0.7343 | 0.6751 | 0.9145 | 0.0488 | 0.0575 | 10256.7 | V_B1C + RowHeightAttn + shadow_augment (combo) |
| M_BLV_SHAD | 0.7866 | 0.7113 | 0.7230 | 0.6101 | 0.9145 | 0.0656 | 0.0400 | 10240.4 | V_B1C + BLV + shadow_augment (combo) |
| M_ROW_BLV | 0.8219 | 0.7479 | 0.7828 | 0.6958 | 0.9055 | 0.0815 | 0.0443 | 8848.4 | V_B1C + RowHeightAttn + BLV (combo) |
| M_ROW_BLV_SHAD | 0.7818 | 0.7420 | 0.6802 | 0.6416 | 0.8715 | 0.1019 | 0.0491 | 10499.7 | V_B1C + RowHeightAttn + BLV + shadow_augment (3-module combo) |
| M_ABL | 0.7879 | 0.7631 | 0.6967 | 0.6390 | 0.8677 | 0.1221 | 0.0403 | 8935.5 | V_B1C + ABL-lite boundary BCE/Dice (train-only) |
| M_INVFORM | 0.8048 | 0.7648 | 0.7298 | 0.6554 | 0.8990 | 0.0883 | 0.0482 | 11260.3 | stub → ABL-lite fallback (no InverseForm net) |
| M_TTA | 0.8274 | 0.7219 | 0.7511 | 0.6889 | 0.9958 | 0.0026 | 0.0853 | 424.4 | TTA diagnostic on V_B1C weights (no retrain of M_TTA itself) | TTA vs single on  |
| M_SEGFIX | 0.8021 | 0.7237 | 0.7415 | 0.6564 | 0.9161 | 0.0583 | 0.0494 | 86.4 | SegFix-lite postprocess on V_B1C (or --ckpt); no retrain |
| T_B1_TTA | 0.8795 | 0.7835 | 0.8730 | 0.7599 | 0.9932 | 0.0058 | 0.0757 | 537.7 | TTA on train_B1 best (no retrain) | TTA vs single on same ckpt |

## Pass criteria (vs V_B1C @ same iters)
- M_ROW: sky→mountain < 12% AND mountain IoU +≥0.03; other classes drop ≤0.01
- M_BLV: crater→regolith < 7% OR crater IoU +≥0.02
- M_SHAD: crater IoU +≥0.02 OR crater→regolith drop ≥1pp (shadow-sensitive)
- M_ABL: boundary-sensitive — rock/crater drop ≤0.01 tradeoff; prefer Δ rock≥0 without crater crash
- M_SEGFIX: postprocess ΔmIoU vs same ckpt single-scale; keep if +≥0.005 without crater collapse
- T_B1_TTA / M_TTA: if small-rock bucket +≥0.05 (when available) → pursue FADE/HRDA; else note resolution not sole issue
- M_FADE: only interpret after M_TTA Δ above noise; else treat as stretch
- M_HRDA: not implemented in portable smoke

注：本地 Scene5 子集，非官方测试；未使用 Moon_2/4/9 标签。

## M_TTA single-scale vs TTA

- single mIoU=0.8014 → TTA mIoU=0.8274 (scales=[0.75, 1.0, 1.25], crop=512, stride=256)
  - small-object buckets: portable stack has no size-bucket helper; skipped
- single mIoU=0.8675 → TTA mIoU=0.8795 (scales=[0.75, 1.0, 1.25], crop=512, stride=256)
  - small-object buckets: portable stack has no size-bucket helper; skipped

