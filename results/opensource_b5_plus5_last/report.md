# Open-source Moon_2/4/9 eval (competition-line dev set)

- Time (Asia/Shanghai): 2026-11-16 13:51:27
- Checkpoint: `/run/media/cui/数据/LuSNAR_Seg/work_dirs/LT_B5_crop_20k_plus5/last.pth` (iter 20000)
- Model: SegFormer-B2 + V_B1C + ROW + BLV, crop 512. One forward stack: single-scale 1.0, equal TTA 0.75/1.0/1.25, sliding 512/256, bf16.
- GT: `data/Moon_{2,4,9}/image0/label`, RGB remapped by official COLOR_TO_ID (rock/crater swapped in the open-source palette). Same loader as `results/oneshot_ROW_BLV/`.
- I1a weights and P1 bias are frozen from Scene5 A (`P1P2_bias_plus_geo` winner). They were not re-tuned on Moon_2/4/9.
- These scores may be used to pick a competition submission. They are not a paper test result (paper test is Moon_3/5/7 under the official split).
- Images scored: 5004; wall 3881.5 s on 2 GPU(s).

## Protocol comparison

| protocol | m2 | m4 | m9 | S=(m2+m4+2*m9)/4 | vs official 0.8046 | overall micro mIoU |
|---|---:|---:|---:|---:|---:|---:|
| single_scale | 0.9399 | 0.9393 | 0.8292 | 0.8844 | +0.0798 | 0.8831 |
| C0_equal_tta | 0.9380 | 0.9389 | 0.8281 | 0.8832 | +0.0786 | 0.8823 |
| I1a_best_scale_per_class | 0.9368 | 0.9381 | 0.8324 | 0.8849 | +0.0803 | 0.8835 |
| P1_logit_bias | 0.9385 | 0.9387 | 0.8286 | 0.8836 | +0.0790 | 0.8826 |
| P1P2_bias_plus_geo | 0.9382 | 0.9393 | 0.8230 | 0.8809 | +0.0763 | 0.8792 |

Previous oneshot C0 overall micro mIoU was 0.8299; this run C0 delta = 0.0524.
C0 pred match vs `results/oneshot_ROW_BLV/preds`: 0/0 (rate —).

## Per-scene per-class IoU

### single_scale

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9399 | 0.9939 | 0.8838 | 0.8554 | 0.9676 | 0.9986 |
| Moon_4 | 1320 | 0.9393 | 0.9930 | 0.8998 | 0.8314 | 0.9736 | 0.9990 |
| Moon_9 | 2087 | 0.8292 | 0.9802 | 0.8207 | 0.5074 | 0.8481 | 0.9897 |
| overall | 5004 | 0.8831 | 0.9880 | 0.8471 | 0.6847 | 0.9003 | 0.9953 |

### C0_equal_tta

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9380 | 0.9938 | 0.8938 | 0.8388 | 0.9650 | 0.9984 |
| Moon_4 | 1320 | 0.9389 | 0.9930 | 0.9048 | 0.8203 | 0.9771 | 0.9990 |
| Moon_9 | 2087 | 0.8281 | 0.9803 | 0.8205 | 0.4831 | 0.8641 | 0.9924 |
| overall | 5004 | 0.8823 | 0.9880 | 0.8498 | 0.6691 | 0.9085 | 0.9963 |

### I1a_best_scale_per_class

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9368 | 0.9938 | 0.8895 | 0.8393 | 0.9631 | 0.9983 |
| Moon_4 | 1320 | 0.9381 | 0.9929 | 0.9000 | 0.8268 | 0.9717 | 0.9990 |
| Moon_9 | 2087 | 0.8324 | 0.9803 | 0.8004 | 0.5169 | 0.8712 | 0.9931 |
| overall | 5004 | 0.8835 | 0.9879 | 0.8356 | 0.6859 | 0.9115 | 0.9965 |

### P1_logit_bias

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9385 | 0.9939 | 0.8949 | 0.8407 | 0.9644 | 0.9985 |
| Moon_4 | 1320 | 0.9387 | 0.9930 | 0.9049 | 0.8209 | 0.9756 | 0.9991 |
| Moon_9 | 2087 | 0.8286 | 0.9804 | 0.8208 | 0.4869 | 0.8625 | 0.9924 |
| overall | 5004 | 0.8826 | 0.9881 | 0.8503 | 0.6714 | 0.9072 | 0.9963 |

### P1P2_bias_plus_geo

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9382 | 0.9939 | 0.8937 | 0.8407 | 0.9642 | 0.9985 |
| Moon_4 | 1320 | 0.9393 | 0.9930 | 0.9047 | 0.8222 | 0.9774 | 0.9991 |
| Moon_9 | 2087 | 0.8230 | 0.9804 | 0.8068 | 0.4868 | 0.8484 | 0.9924 |
| overall | 5004 | 0.8792 | 0.9881 | 0.8413 | 0.6717 | 0.8986 | 0.9963 |

