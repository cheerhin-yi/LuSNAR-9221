# Open-source Moon_2/4/9 eval (competition-line dev set)

- Time (Asia/Shanghai): 2026-11-17 05:08:25
- Checkpoint: `/run/media/cui/数据/LuSNAR_Seg/work_dirs/LT_M2F_crop_20k_plus5/last.pth` (iter 20000)
- Model: SegFormer-B2 + V_B1C + ROW + BLV, crop 512. One forward stack: single-scale 1.0, equal TTA 0.75/1.0/1.25, sliding 512/256, bf16.
- GT: `data/Moon_{2,4,9}/image0/label`, RGB remapped by official COLOR_TO_ID (rock/crater swapped in the open-source palette). Same loader as `results/oneshot_ROW_BLV/`.
- I1a weights and P1 bias are frozen from Scene5 A (`P1P2_bias_plus_geo` winner). They were not re-tuned on Moon_2/4/9.
- These scores may be used to pick a competition submission. They are not a paper test result (paper test is Moon_3/5/7 under the official split).
- Images scored: 5004; wall 4048.3 s on 2 GPU(s).

## Protocol comparison

| protocol | m2 | m4 | m9 | S=(m2+m4+2*m9)/4 | vs official 0.8046 | overall micro mIoU |
|---|---:|---:|---:|---:|---:|---:|
| single_scale | 0.9402 | 0.9332 | 0.8046 | 0.8707 | +0.0661 | 0.8694 |
| C0_equal_tta | 0.9418 | 0.9346 | 0.8011 | 0.8696 | +0.0650 | 0.8687 |
| I1a_best_scale_per_class | 0.9372 | 0.9306 | 0.7997 | 0.8668 | +0.0622 | 0.8652 |
| P1_logit_bias | 0.9421 | 0.9344 | 0.8013 | 0.8698 | +0.0652 | 0.8688 |
| P1P2_bias_plus_geo | 0.9416 | 0.9349 | 0.7974 | 0.8678 | +0.0632 | 0.8663 |

Previous oneshot C0 overall micro mIoU was 0.8299; this run C0 delta = 0.0388.
C0 pred match vs `results/oneshot_ROW_BLV/preds`: 0/0 (rate —).

## Per-scene per-class IoU

### single_scale

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9402 | 0.9935 | 0.8758 | 0.8745 | 0.9592 | 0.9982 |
| Moon_4 | 1320 | 0.9332 | 0.9919 | 0.8852 | 0.8221 | 0.9688 | 0.9980 |
| Moon_9 | 2087 | 0.8046 | 0.9760 | 0.8085 | 0.3999 | 0.8516 | 0.9871 |
| overall | 5004 | 0.8694 | 0.9858 | 0.8354 | 0.6323 | 0.8995 | 0.9938 |

### C0_equal_tta

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9418 | 0.9938 | 0.8868 | 0.8692 | 0.9605 | 0.9987 |
| Moon_4 | 1320 | 0.9346 | 0.9923 | 0.8919 | 0.8150 | 0.9749 | 0.9987 |
| Moon_9 | 2087 | 0.8011 | 0.9762 | 0.8040 | 0.3756 | 0.8604 | 0.9890 |
| overall | 5004 | 0.8687 | 0.9861 | 0.8359 | 0.6214 | 0.9050 | 0.9950 |

### I1a_best_scale_per_class

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9372 | 0.9932 | 0.8757 | 0.8682 | 0.9508 | 0.9979 |
| Moon_4 | 1320 | 0.9306 | 0.9917 | 0.8814 | 0.8149 | 0.9668 | 0.9980 |
| Moon_9 | 2087 | 0.7997 | 0.9755 | 0.7879 | 0.3849 | 0.8613 | 0.9891 |
| overall | 5004 | 0.8652 | 0.9854 | 0.8218 | 0.6227 | 0.9015 | 0.9945 |

### P1_logit_bias

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9421 | 0.9939 | 0.8886 | 0.8710 | 0.9584 | 0.9987 |
| Moon_4 | 1320 | 0.9344 | 0.9923 | 0.8936 | 0.8157 | 0.9720 | 0.9986 |
| Moon_9 | 2087 | 0.8013 | 0.9763 | 0.8037 | 0.3788 | 0.8589 | 0.9888 |
| overall | 5004 | 0.8688 | 0.9861 | 0.8364 | 0.6232 | 0.9032 | 0.9948 |

### P1P2_bias_plus_geo

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9416 | 0.9939 | 0.8870 | 0.8710 | 0.9577 | 0.9987 |
| Moon_4 | 1320 | 0.9349 | 0.9923 | 0.8935 | 0.8162 | 0.9740 | 0.9986 |
| Moon_9 | 2087 | 0.7974 | 0.9764 | 0.7935 | 0.3786 | 0.8494 | 0.9891 |
| overall | 5004 | 0.8663 | 0.9862 | 0.8297 | 0.6233 | 0.8972 | 0.9950 |

