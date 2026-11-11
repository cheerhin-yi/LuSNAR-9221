# Open-source Moon_2/4/9 eval (competition-line dev set)

- Time (Asia/Shanghai): 2026-11-10 12:50:52
- Checkpoint: `/mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/best.pth` (iter 17000)
- Model: SegFormer-B2 + V_B1C + ROW + BLV, crop 512. Inference: ROW + equal TTA 0.75/1.0/1.25, sliding 512/256, bf16.
- GT: `data/Moon_{2,4,9}/image0/label`, RGB remapped by official COLOR_TO_ID (rock/crater swapped in the open-source palette). Same loader as `results/oneshot_ROW_BLV/`.
- I1a weights and P1 bias are frozen from Scene5 A (`P1P2_bias_plus_geo` winner). They were not re-tuned on Moon_2/4/9.
- These scores may be used to pick a competition submission. They are not a paper test result (paper test is Moon_3/5/7 under the official split).
- Images scored: 12; wall 19.5 s on 1 GPU(s).

## Protocol comparison

| protocol | m2 | m4 | m9 | S=(m2+m4+2*m9)/4 | vs official 0.8046 | overall micro mIoU |
|---|---:|---:|---:|---:|---:|---:|
| C0_equal_tta | 0.7643 | 0.8445 | 0.8560 | 0.8302 | +0.0256 | 0.8745 |
| I1a_best_scale_per_class | 0.7791 | 0.8257 | 0.8614 | 0.8319 | +0.0273 | 0.8734 |
| P1_logit_bias | 0.7647 | 0.8433 | 0.8567 | 0.8303 | +0.0257 | 0.8743 |
| P1P2_bias_plus_geo | 0.7647 | 0.8469 | 0.8860 | 0.8459 | +0.0413 | 0.8910 |

Previous oneshot C0 overall micro mIoU was 0.8299; this run C0 delta = 0.0446.
C0 pred match vs `results/oneshot_ROW_BLV/preds`: 12/12 (rate 1.0000).

## Per-scene per-class IoU

### C0_equal_tta

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 4 | 0.7643 | 0.9948 | 0.9191 | 0.0000 | 0.9081 | 0.9994 |
| Moon_4 | 4 | 0.8445 | 0.9958 | 0.9292 | 0.3583 | 0.9399 | 0.9994 |
| Moon_9 | 4 | 0.8560 | 0.9813 | 0.7790 | 0.6760 | 0.8464 | 0.9972 |
| overall | 12 | 0.8745 | 0.9905 | 0.8546 | 0.6536 | 0.8749 | 0.9988 |

### I1a_best_scale_per_class

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 4 | 0.7791 | 0.9949 | 0.9173 | 0.0733 | 0.9103 | 0.9996 |
| Moon_4 | 4 | 0.8257 | 0.9953 | 0.9256 | 0.2638 | 0.9446 | 0.9995 |
| Moon_9 | 4 | 0.8614 | 0.9776 | 0.8074 | 0.6229 | 0.9016 | 0.9976 |
| overall | 12 | 0.8734 | 0.9891 | 0.8688 | 0.6017 | 0.9086 | 0.9990 |

### P1_logit_bias

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 4 | 0.7647 | 0.9949 | 0.9218 | 0.0000 | 0.9072 | 0.9994 |
| Moon_4 | 4 | 0.8433 | 0.9957 | 0.9288 | 0.3624 | 0.9301 | 0.9994 |
| Moon_9 | 4 | 0.8567 | 0.9810 | 0.7838 | 0.6706 | 0.8508 | 0.9972 |
| overall | 12 | 0.8743 | 0.9904 | 0.8576 | 0.6486 | 0.8763 | 0.9989 |

### P1P2_bias_plus_geo

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 4 | 0.7647 | 0.9949 | 0.9218 | 0.0000 | 0.9072 | 0.9994 |
| Moon_4 | 4 | 0.8469 | 0.9959 | 0.9288 | 0.3660 | 0.9444 | 0.9994 |
| Moon_9 | 4 | 0.8860 | 0.9811 | 0.8489 | 0.6706 | 0.9323 | 0.9971 |
| overall | 12 | 0.8910 | 0.9905 | 0.8911 | 0.6487 | 0.9258 | 0.9988 |

