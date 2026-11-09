# Method smoke eval (Scene5)

Protocol: Scene5 val_sub only; never Moon_2/4/9. Control = V_B1C.
Methods are relative to same-machine V_B1C; 2k-iter smoke, |Δ|<0.02 = noise. M_MSINFER compares full-image modes only.

| exp | mIoU | rock | crater | mountain | sky | sky→mtn | crater→reg | BF1 | BIoU | wall_s | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V_B1C | 0.8126 | 0.7467 | 0.7464 | 0.6844 | 0.9096 | 0.0745 | 0.0638 | 0.7285 | 0.4302 | 5683.7 | B1 + crater RCS/ACW boost (control) |
| M_DYSAMPLE | 0.8079 | 0.7502 | 0.7504 | 0.6568 | 0.9058 | 0.0796 | 0.0586 | 0.7245 | 0.4272 | 5333.1 | DySample lp x4 groups4 scope0.25; final 5 logits padded to8 then sliced |
| M_MSINFER | 0.7707 | 0.6035 | 0.6656 | 0.6127 | 0.9918 | - | - | 0.6630 | 0.3863 | 1009.2 | No training. Four independent full-image modes; not comparable to center-crop va |
| M_HARD | 0.7860 | 0.7243 | 0.7341 | 0.6190 | 0.8785 | 0.1075 | 0.0657 | 0.7025 | 0.4081 | 5336.5 | Instantaneous detached hardness weighting alpha1 gamma2; no EMA map |
| M_CTRAST | 0.7941 | 0.7461 | 0.7400 | 0.6502 | 0.8580 | 0.1261 | 0.0816 | 0.7215 | 0.4157 | 6491.1 | Contextrast-lite: fused-feature local boundary-pair InfoNCE temp0.1 weight0.1 ca |
| M_SAPA | 0.7967 | 0.7266 | 0.7160 | 0.6496 | 0.9175 | 0.0661 | 0.0761 | 0.6973 | 0.4205 | 9131.9 | SAPA-B-lite 5x5 embedding32; interpolated stage1 guidance; final logits only; ch |

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

Boundary F1: inner 1-px class contours, Euclidean tolerance 3 px; Boundary IoU: inner erosion band d=3 px (fixed, not diagonal ratio). Dataset-pooled class counts; ignore-adjacent pixels excluded. Absent classes are null.

- M_MSINFER / whole_1.0: mIoU=0.7707; per-class={'月壤': 0.9797185219372575, '岩石': 0.603529463298181, '撞击坑': 0.6655573067535541, '山体': 0.612730251106046, '天空': 0.9918328035978901}; BF1=0.6630; BIoU=0.3863; s/image=0.0928
- M_MSINFER / slide_1.0: mIoU=0.7820; per-class={'月壤': 0.9814817137153539, '岩石': 0.6254802420636307, '撞击坑': 0.6976672538180636, '山体': 0.6120315510308139, '天空': 0.9933188287836593}; BF1=0.6784; BIoU=0.3908; s/image=0.3098
- M_MSINFER / slide_1.25: mIoU=0.7564; per-class={'月壤': 0.9748921681747451, '岩石': 0.6148660136772336, '撞击坑': 0.5728816314714491, '山体': 0.627473204176221, '天空': 0.9921091913404073}; BF1=0.6714; BIoU=0.3824; s/image=0.5548
- M_MSINFER / slide_1.5: mIoU=0.7008; per-class={'月壤': 0.968524734719429, '岩石': 0.5370969823128581, '撞击坑': 0.4312121568083039, '山体': 0.575977757796032, '天空': 0.9912183970319712}; BF1=0.6242; BIoU=0.3555; s/image=0.8336

## Per-class boundary metrics

| exp / mode | BF1 regolith | BF1 rock | BF1 crater | BF1 mountain | BF1 sky | BIoU regolith | BIoU rock | BIoU crater | BIoU mountain | BIoU sky |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V_B1C | 0.8384 | 0.8020 | 0.5529 | 0.5592 | 0.8901 | 0.5620 | 0.4166 | 0.1829 | 0.3059 | 0.6838 |
| M_DYSAMPLE | 0.8387 | 0.8014 | 0.5520 | 0.5400 | 0.8905 | 0.5622 | 0.4132 | 0.1844 | 0.2906 | 0.6857 |
| M_MSINFER | 0.7982 | 0.6801 | 0.3998 | 0.4771 | 0.9596 | 0.5017 | 0.3251 | 0.1238 | 0.2143 | 0.7665 |
| M_MSINFER/whole_1.0 | 0.7982 | 0.6801 | 0.3998 | 0.4771 | 0.9596 | 0.5017 | 0.3251 | 0.1238 | 0.2143 | 0.7665 |
| M_MSINFER/slide_1.0 | 0.8083 | 0.7072 | 0.4379 | 0.4802 | 0.9583 | 0.5141 | 0.3462 | 0.1365 | 0.2080 | 0.7493 |
| M_MSINFER/slide_1.25 | 0.7795 | 0.6952 | 0.4593 | 0.4802 | 0.9430 | 0.4954 | 0.3396 | 0.1421 | 0.2040 | 0.7310 |
| M_MSINFER/slide_1.5 | 0.7372 | 0.6139 | 0.3997 | 0.4329 | 0.9372 | 0.4617 | 0.2815 | 0.1249 | 0.1830 | 0.7262 |
| M_HARD | 0.8280 | 0.7876 | 0.5174 | 0.5114 | 0.8680 | 0.5511 | 0.4064 | 0.1745 | 0.2656 | 0.6429 |
| M_CTRAST | 0.8421 | 0.7990 | 0.5682 | 0.5429 | 0.8553 | 0.5622 | 0.4121 | 0.1916 | 0.2880 | 0.6246 |
| M_SAPA | 0.8135 | 0.7641 | 0.5150 | 0.4995 | 0.8942 | 0.5357 | 0.3833 | 0.1873 | 0.2730 | 0.7234 |

