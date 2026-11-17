# Open-source Moon_2/4/9 eval (competition-line dev set)

- Time (Asia/Shanghai): 2026-11-17 11:26:42
- Checkpoint: `/run/media/cui/数据/LuSNAR_Seg/work_dirs/LT_M2F_crop_20k/best.pth` (iter 4000)
- Model: SegFormer-B2 + V_B1C + ROW + BLV, crop 512. One forward stack: single-scale 1.0, equal TTA 0.75/1.0/1.25, sliding 512/256, bf16.
- GT: `data/Moon_{2,4,9}/image0/label`, RGB remapped by official COLOR_TO_ID (rock/crater swapped in the open-source palette). Same loader as `results/oneshot_ROW_BLV/`.
- I1a weights and P1 bias are frozen from Scene5 A (`P1P2_bias_plus_geo` winner). They were not re-tuned on Moon_2/4/9.
- These scores may be used to pick a competition submission. They are not a paper test result (paper test is Moon_3/5/7 under the official split).
- Images scored: 5004; wall 5011.9 s on 2 GPU(s).

## Protocol comparison

| protocol | m2 | m4 | m9 | S=(m2+m4+2*m9)/4 | vs official 0.8046 | overall micro mIoU |
|---|---:|---:|---:|---:|---:|---:|
| single_scale | 0.9400 | 0.9206 | 0.7922 | 0.8612 | +0.0566 | 0.8577 |
| C0_equal_tta | 0.9436 | 0.9235 | 0.7984 | 0.8660 | +0.0614 | 0.8633 |
| I1a_best_scale_per_class | 0.9366 | 0.9212 | 0.8037 | 0.8663 | +0.0617 | 0.8631 |
| P1_logit_bias | 0.9435 | 0.9229 | 0.7982 | 0.8657 | +0.0611 | 0.8629 |
| P1P2_bias_plus_geo | 0.9416 | 0.9236 | 0.7906 | 0.8616 | +0.0570 | 0.8578 |

Previous oneshot C0 overall micro mIoU was 0.8299; this run C0 delta = 0.0334.
C0 pred match vs `results/oneshot_ROW_BLV/preds`: 0/0 (rate —).

## Per-scene per-class IoU

### single_scale

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9400 | 0.9929 | 0.8682 | 0.9005 | 0.9407 | 0.9975 |
| Moon_4 | 1320 | 0.9206 | 0.9906 | 0.8731 | 0.8244 | 0.9177 | 0.9972 |
| Moon_9 | 2087 | 0.7922 | 0.9729 | 0.7832 | 0.4470 | 0.7725 | 0.9853 |
| overall | 5004 | 0.8577 | 0.9840 | 0.8154 | 0.6543 | 0.8424 | 0.9927 |

### C0_equal_tta

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9436 | 0.9935 | 0.8785 | 0.9050 | 0.9428 | 0.9981 |
| Moon_4 | 1320 | 0.9235 | 0.9912 | 0.8833 | 0.8193 | 0.9259 | 0.9980 |
| Moon_9 | 2087 | 0.7984 | 0.9734 | 0.7906 | 0.4495 | 0.7923 | 0.9859 |
| overall | 5004 | 0.8633 | 0.9845 | 0.8240 | 0.6598 | 0.8546 | 0.9934 |

### I1a_best_scale_per_class

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9366 | 0.9928 | 0.8674 | 0.8946 | 0.9311 | 0.9971 |
| Moon_4 | 1320 | 0.9212 | 0.9906 | 0.8759 | 0.8147 | 0.9278 | 0.9971 |
| Moon_9 | 2087 | 0.8037 | 0.9736 | 0.7819 | 0.4721 | 0.8037 | 0.9872 |
| overall | 5004 | 0.8631 | 0.9842 | 0.8153 | 0.6653 | 0.8576 | 0.9933 |

### P1_logit_bias

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9435 | 0.9935 | 0.8795 | 0.9063 | 0.9403 | 0.9980 |
| Moon_4 | 1320 | 0.9229 | 0.9911 | 0.8842 | 0.8202 | 0.9211 | 0.9978 |
| Moon_9 | 2087 | 0.7982 | 0.9732 | 0.7906 | 0.4503 | 0.7912 | 0.9856 |
| overall | 5004 | 0.8629 | 0.9844 | 0.8244 | 0.6601 | 0.8526 | 0.9931 |

### P1P2_bias_plus_geo

| scene | n | mIoU | regolith | rock | crater | mountain | sky |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moon_2 | 1597 | 0.9416 | 0.9935 | 0.8734 | 0.9062 | 0.9371 | 0.9980 |
| Moon_4 | 1320 | 0.9236 | 0.9912 | 0.8840 | 0.8208 | 0.9244 | 0.9977 |
| Moon_9 | 2087 | 0.7906 | 0.9733 | 0.7683 | 0.4504 | 0.7751 | 0.9858 |
| overall | 5004 | 0.8578 | 0.9845 | 0.8095 | 0.6603 | 0.8411 | 0.9932 |

