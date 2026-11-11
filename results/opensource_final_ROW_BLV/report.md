# Open-source Moon_2/4/9 eval (competition-line dev set)

- Time (Asia/Shanghai): 2026-11-10 13:40:16
- Checkpoint: `/mnt/d/LuSNAR_Seg/work_dirs/train_ROW_BLV_20k/best.pth` (iter 17000)
- Model: SegFormer-B2 + V_B1C + ROW + BLV, crop 512. Inference: ROW + equal TTA 0.75/1.0/1.25, sliding 512/256, bf16.
- GT: `data/Moon_{2,4,9}/image0/label`, RGB remapped by official COLOR_TO_ID (rock/crater swapped in the open-source palette). Same loader as `results/oneshot_ROW_BLV/`.
- I1a weights and P1 bias are frozen from Scene5 A (`P1P2_bias_plus_geo` winner). They were not re-tuned on Moon_2/4/9.
- These scores may be used to pick a competition submission. They are not a paper test result (paper test is Moon_3/5/7 under the official split).
- Images scored: 5004; wall 2900.0 s on 2 GPU(s).

## Protocol comparison

| protocol | m2 | m4 | m9 | S=(m2+m4+2*m9)/4 | vs official 0.8046 | overall micro mIoU |
|---|---:|---:|---:|---:|---:|---:|
| C0_equal_tta | 0.9215 | 0.9136 | 0.7425 | 0.8300 | +0.0254 | 0.8299 |
| I1a_best_scale_per_class | 0.9248 | 0.9171 | 0.7529 | 0.8369 | +0.0323 | 0.8348 |
| P1_logit_bias | 0.9218 | 0.9123 | 0.7435 | 0.8303 | +0.0257 | 0.8303 |
| P1P2_bias_plus_geo | 0.9214 | 0.9138 | 0.7333 | 0.8254 | +0.0208 | 0.8236 |

Previous oneshot C0 overall micro mIoU was 0.8299; this run C0 delta = 0.0000.
C0 pred match vs `results/oneshot_ROW_BLV/preds`: 5004/5004 (rate 1.0000).

## Per-scene per-class IoU

### C0_equal_tta

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9215 | 0.9922 | 0.8900 | 0.7774 | 0.9495 | 0.9982 |
| Moon_4 | 1320 | 0.9136 | 0.9910 | 0.8906 | 0.8063 | 0.8807 | 0.9990 |
| Moon_9 | 2087 | 0.7425 | 0.9699 | 0.7756 | 0.2589 | 0.7231 | 0.9850 |
| overall | 5004 | 0.8299 | 0.9826 | 0.8174 | 0.5436 | 0.8126 | 0.9933 |

### I1a_best_scale_per_class

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9248 | 0.9923 | 0.8770 | 0.8058 | 0.9505 | 0.9982 |
| Moon_4 | 1320 | 0.9171 | 0.9911 | 0.8780 | 0.8152 | 0.9024 | 0.9991 |
| Moon_9 | 2087 | 0.7529 | 0.9711 | 0.7508 | 0.3103 | 0.7462 | 0.9862 |
| overall | 5004 | 0.8348 | 0.9831 | 0.7972 | 0.5730 | 0.8270 | 0.9938 |

### P1_logit_bias

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9218 | 0.9922 | 0.8897 | 0.7803 | 0.9483 | 0.9983 |
| Moon_4 | 1320 | 0.9123 | 0.9909 | 0.8894 | 0.8079 | 0.8739 | 0.9991 |
| Moon_9 | 2087 | 0.7435 | 0.9698 | 0.7771 | 0.2630 | 0.7231 | 0.9846 |
| overall | 5004 | 0.8303 | 0.9825 | 0.8182 | 0.5464 | 0.8114 | 0.9932 |

### P1P2_bias_plus_geo

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9214 | 0.9922 | 0.8883 | 0.7801 | 0.9483 | 0.9983 |
| Moon_4 | 1320 | 0.9138 | 0.9910 | 0.8891 | 0.8081 | 0.8815 | 0.9991 |
| Moon_9 | 2087 | 0.7333 | 0.9700 | 0.7451 | 0.2627 | 0.7033 | 0.9853 |
| overall | 5004 | 0.8236 | 0.9826 | 0.7980 | 0.5463 | 0.7976 | 0.9935 |

