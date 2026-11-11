# Open-source Moon_2/4/9 eval (competition-line dev set)

- Time (Asia/Shanghai): 2026-11-11 11:30:11
- Checkpoint: `D:\LuSNAR_Seg\work_dirs\train_ROW_BLV_20k_plus5\last.pth` (iter 20000)
- Model: SegFormer-B2 + V_B1C + ROW + BLV, crop 512. One forward stack: single-scale 1.0, equal TTA 0.75/1.0/1.25, sliding 512/256, bf16.
- GT: `data/Moon_{2,4,9}/image0/label`, RGB remapped by official COLOR_TO_ID (rock/crater swapped in the open-source palette). Same loader as `results/oneshot_ROW_BLV/`.
- I1a weights and P1 bias are frozen from Scene5 A (`P1P2_bias_plus_geo` winner). They were not re-tuned on Moon_2/4/9.
- These scores may be used to pick a competition submission. They are not a paper test result (paper test is Moon_3/5/7 under the official split).
- Images scored: 5004; wall 3528.2 s on 2 GPU(s).

## Protocol comparison

| protocol | m2 | m4 | m9 | S=(m2+m4+2*m9)/4 | vs official 0.8046 | overall micro mIoU |
|---|---:|---:|---:|---:|---:|---:|
| single_scale | 0.9438 | 0.9386 | 0.8090 | 0.8751 | +0.0705 | 0.8733 |
| C0_equal_tta | 0.9369 | 0.9355 | 0.8033 | 0.8698 | +0.0652 | 0.8684 |
| I1a_best_scale_per_class | 0.9394 | 0.9357 | 0.8097 | 0.8736 | +0.0690 | 0.8716 |
| P1_logit_bias | 0.9374 | 0.9350 | 0.8042 | 0.8702 | +0.0656 | 0.8689 |
| P1P2_bias_plus_geo | 0.9370 | 0.9360 | 0.7979 | 0.8672 | +0.0626 | 0.8649 |

Previous oneshot C0 overall micro mIoU was 0.8299; this run C0 delta = 0.0385.
C0 pred match vs `results/oneshot_ROW_BLV/preds`: 0/0 (rate —).

## Per-scene per-class IoU

### single_scale

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9438 | 0.9944 | 0.8958 | 0.8619 | 0.9685 | 0.9983 |
| Moon_4 | 1320 | 0.9386 | 0.9932 | 0.8974 | 0.8393 | 0.9645 | 0.9986 |
| Moon_9 | 2087 | 0.8090 | 0.9791 | 0.8186 | 0.4356 | 0.8243 | 0.9873 |
| overall | 5004 | 0.8733 | 0.9877 | 0.8473 | 0.6511 | 0.8862 | 0.9941 |

### C0_equal_tta

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9369 | 0.9938 | 0.8979 | 0.8280 | 0.9665 | 0.9985 |
| Moon_4 | 1320 | 0.9355 | 0.9928 | 0.8953 | 0.8241 | 0.9663 | 0.9989 |
| Moon_9 | 2087 | 0.8033 | 0.9786 | 0.8091 | 0.4164 | 0.8225 | 0.9901 |
| overall | 5004 | 0.8684 | 0.9872 | 0.8413 | 0.6339 | 0.8840 | 0.9954 |

### I1a_best_scale_per_class

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9394 | 0.9940 | 0.8843 | 0.8570 | 0.9635 | 0.9982 |
| Moon_4 | 1320 | 0.9357 | 0.9928 | 0.8871 | 0.8330 | 0.9669 | 0.9988 |
| Moon_9 | 2087 | 0.8097 | 0.9777 | 0.7839 | 0.4566 | 0.8405 | 0.9898 |
| overall | 5004 | 0.8716 | 0.9869 | 0.8215 | 0.6608 | 0.8935 | 0.9951 |

### P1_logit_bias

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9374 | 0.9939 | 0.8980 | 0.8311 | 0.9656 | 0.9985 |
| Moon_4 | 1320 | 0.9350 | 0.9928 | 0.8945 | 0.8250 | 0.9636 | 0.9989 |
| Moon_9 | 2087 | 0.8042 | 0.9786 | 0.8103 | 0.4204 | 0.8219 | 0.9900 |
| overall | 5004 | 0.8689 | 0.9872 | 0.8420 | 0.6366 | 0.8831 | 0.9953 |

### P1P2_bias_plus_geo

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9370 | 0.9939 | 0.8964 | 0.8310 | 0.9654 | 0.9985 |
| Moon_4 | 1320 | 0.9360 | 0.9928 | 0.8943 | 0.8260 | 0.9677 | 0.9989 |
| Moon_9 | 2087 | 0.7979 | 0.9786 | 0.7936 | 0.4199 | 0.8073 | 0.9900 |
| overall | 5004 | 0.8649 | 0.9872 | 0.8313 | 0.6366 | 0.8740 | 0.9954 |

