# Scene5 推理阶段报告（I1–I4 与后处理）

- 生成时间（Asia/Shanghai）：2026-11-10 12:47:20
- 权重：`work_dirs/train_ROW_BLV_20k/best.pth`（iter 17000，SegFormer-B2 + V_B1C + ROW + BLV，crop 512）
- 范围：仅 Scene5。Moon_2/4/9 标签未用于选型。A 半集选型，B 半集只报一次。
- A/B 划分：seed=42，A=799，B=800，清单 `data/lists_portable/scene5_ab/`

## 1. 诊断

### 1.1 等权 TTA（尺度 0.75/1.0/1.25，滑窗 512/256）

- 全 Scene5 mIoU：**0.8651**
- 分类 IoU：月壤 0.9865 / 岩石 0.7478 / 撞击坑 0.8686 / 山体 0.7263 / 天空 0.9962
- 混淆率：mountain→rock=0.1258，crater→regolith=0.0966，sky→mountain=0.0027
- 耗时 3248.1 s，窗口前向 87945 次

### 1.2 单尺度 mIoU 与分类 IoU

| scale | mIoU | 月壤 | 岩石 | 撞击坑 | 山体 | 天空 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.8519 | 0.9853 | 0.6858 | 0.8818 | 0.7134 | 0.9932 |
| 0.75 | 0.8694 | 0.9862 | 0.7396 | 0.8954 | 0.7304 | 0.9951 |
| 1.00 | 0.8449 | 0.9875 | 0.6884 | 0.8731 | 0.6798 | 0.9959 |
| 1.25 | 0.7876 | 0.9762 | 0.7354 | 0.5655 | 0.6708 | 0.9900 |
| 1.50 | 0.6692 | 0.9408 | 0.6254 | 0.3107 | 0.5543 | 0.9148 |

### 1.3 T0 尺度审计（GT 连通域尺寸）

- **rock**：连通域 33007 个，side p50=11.4 px，p90=50.3，<32px 占比 0.816
- **crater**：连通域 2946 个，side p50=5.6 px，p90=178.7，<32px 占比 0.715

- 建议：If many rock/crater CC have side_p50 << 64, prefer scales >=1.25 or re-zoom (I3). If large objects dominate, scales <=0.75 help context (mountain).

### 1.4 误差归因（Scene5 等权 TTA 预测）

- 图数 1599，mIoU 0.8651

| 类别 | IoU | 主因 | 证据 |
|---|---:|---|---|
| 月壤 (regolith) | 0.9865 | boundary | 内部 IoU 0.995，边界带 IoU 0.788 |
| 岩石 (rock) | 0.7478 | majority_swallow | 漏检里 80% 被判成月壤；判错像素比判对像素更暗 29.3 灰度 |
| 撞击坑 (crater) | 0.8686 | majority_swallow | 漏检里 99% 被判成月壤；小块召回 0.526，大块召回 0.910；判错像素比判对像素更暗 29.8 灰度；内部 IoU 0.915，边界带 IoU 0.758 |
| 山体 (mountain) | 0.7263 | texture_confusion | 漏检里 93% 被判成岩石；内部 IoU 0.807，边界带 IoU 0.566 |
| 天空 (sky) | 0.9962 | texture_confusion | 漏检里 85% 被判成山体；判错像素比判对像素更亮 17.6 灰度 |

## 2. 预注册方案（含 I2 / I3 / I4）

预注册：`C0_equal_tta`, `I1a_best_scale_per_class`, `I1b_soft_scale_weights`, `I2a_gaussian_overlap`, `I2b_cosine_edge_drop`, `I3_small_rezoom`, `P1_logit_bias`, `P1P2_bias_plus_geo`, `I4_hybrid_full_tiles`
选型规则：maximize A mIoU; require rock/crater drop <=0.01 vs C0 on A; B reported once

| 方案 | A mIoU | B mIoU | rock A/B | crater A/B | mountain A/B | ms/图 |
|---|---:|---:|---:|---:|---:|---:|
| C0_equal_tta | 0.8648 | 0.8655 | 0.7493/0.7464 | 0.8652/0.8722 | 0.7268/0.7258 | 146.8 |
| I1a_best_scale_per_class | 0.8853 | 0.8824 | 0.7770/0.7679 | 0.8973/0.8940 | 0.7690/0.7672 | 150.6 |
| I1b_soft_scale_weights | 0.8813 | 0.8790 | 0.7687/0.7589 | 0.8969/0.8972 | 0.7583/0.7568 | 150.7 |
| I2a_gaussian_overlap | 0.8852 | 0.8827 | 0.7755/0.7673 | 0.8979/0.8957 | 0.7693/0.7676 | 525.3 |
| I2b_cosine_edge_drop | 0.7980 | — | 0.7187/— | 0.8137/— | 0.7218/— | 514.9 |
| I3_small_rezoom | 0.8407 | — | 0.7356/— | 0.8450/— | 0.6485/— | 1131.8 |
| P1_logit_bias | 0.8723 | 0.8726 | 0.7671/0.7636 | 0.8728/0.8776 | 0.7393/0.7389 | 156.4 |
| P1P2_bias_plus_geo | 0.8912 | 0.8905 | 0.8118/0.8046 | 0.8730/0.8779 | 0.7887/0.7874 | 181.1 |
| I4_hybrid_full_tiles | 0.8651 | 0.8663 | 0.7490/0.7474 | 0.8585/0.8659 | 0.7354/0.7350 | 182.9 |
| P2_geo_only_ref | 0.8873 | — | 0.8016/— | 0.8654/— | 0.7870/— | 153.6 |

I4 说明：Added as a 9th pre-registered candidate before B scores were used for selection. Internal merge variant chosen on A only; B reported once for the chosen rule.
- 选中的合并规则（仅按 A）：`M4_per_class_source`；B 只报一次。
- 纯 1024 整图（诊断，不参与选型）A mIoU=0.8268，rock=0.6659，mountain=0.6420。

I3 基线：`I2a_gaussian_overlap`。I2b 与 I3 在 A 上明显低于 I2a / I1a，按选型规则未再报 B。

### 与 C0 的配对检验（A 集逐图 mIoU）

| 方案 | Δmean | Wilcoxon p | Holm p | bootstrap 95% CI |
|---|---:|---:|---:|---|
| I1a_best_scale_per_class | -0.0085 | 0.0233 | 0.0699 | [-0.0124, -0.0045] |
| I1b_soft_scale_weights | 0.0028 | 0.0000 | 0.0000 | [0.0003, 0.0052] |
| P1_logit_bias | 0.0019 | 0.0000 | 0.0000 | [0.0002, 0.0035] |
| P1P2_bias_plus_geo | 0.0248 | 0.0000 | 0.0000 | [0.0221, 0.0276] |
| I4_hybrid_full_tiles | -0.0062 | 0.7538 | 0.7538 | [-0.0084, -0.0041] |
| I2a_gaussian_overlap | -0.0100 | 0.0256 | 0.0699 | [-0.0189, -0.0011] |
| I2b_cosine_edge_drop | -0.1255 | 0.0000 | 0.0000 | [-0.1341, -0.1170] |
| I3_small_rezoom | -0.0372 | 0.0000 | 0.0000 | [-0.0450, -0.0293] |

## 3. 冻结方案

- **选中**：`P1P2_bias_plus_geo`
- A mIoU=0.8912，B 一次性 mIoU=0.8905
- B 分类 IoU：月壤=0.9865, 岩石=0.8046, 撞击坑=0.8779, 山体=0.7874, 天空=0.9962
- 说明：Selected on A only; B numbers are one-shot.

## 4. 缓存路径

- TTA 概率 float16：`/root/lusnar_data/cache/scene5_tta_probs/`（`stem.npy`，形状 5×1024×1024）
- 单尺度 logits float16：`/root/lusnar_data/cache/scene5_scale_logits/`（`stem_s{scale}.npy`）
- 预测图：`/root/lusnar_data/cache/scene5_preds/`
- 过程 JSON：`results/inference_phase/`

## 5. 下一步

- **尚未启动 plus5**。
- 推理阶段报告落地后，按计划再进入 Windows plus5。

