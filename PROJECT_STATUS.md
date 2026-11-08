# LuSNAR-9221 项目状态

> 最后更新：2026-11-08 16:27（Asia/Shanghai）  
> 仓库：https://github.com/cheerhin-yi/LuSNAR-9221.git  
> 维护约定：本文件由 **4090 Bot** 与 **@LuSNAR（cheerhin）** 共同维护；4090 负责本机冒烟/组合消融章节，cheerhin 章节由 LuSNAR 填写与更新。

## 1. 项目目标

- 任务：月球场景语义分割（SegFormer-B2 / `mit_b2`，mmseg 1.2.2，torch 2.7.1+cu128）
- 类别顺序：`0 regolith, 1 rock, 2 crater, 3 mountain, 4 sky`
- **硬约束**：`Moon_2 / Moon_4 / Moon_9` 标签仅作测试 GT，**禁止**用于训练、调参、模型选择或带标签诊断
- 不加额外训练数据；官方盲评至多一次且需用户明确批准（当前未批准）
- Scene5 分数仅为本地子集，非正式成绩

## 2. 双机布局（保持对齐）

顶层目录约定（两边一致）：

```
configs/  data/  logs/  pretrained/  results/  tools/  work_dirs/  _tmp/
```

| 主机 | 机器 | 角色 | 工程根目录 |
|------|------|------|------------|
| DESKTOP-ORQLB8F（4090×2） | machineId `370a07e6-…` | 方法冒烟矩阵 + 两两组合消融 | `D:\LuSNAR_Seg` |
| cheerhin（5060 Ti） | machineId `2256a039-…` | 第三批方法冒烟（LuSNAR 管理） | `D:\LuSNAR_Seg` |

- 数据集 `data/` **不同步**到 GitHub
- 权重 `*.pth` / `pretrained/` **不上传**；`_tmp/`、`*.docx`、`*.pdf` 等行政文件不上传
- 脚本与结果说明：`tools/`、`results/*.md`、`PROJECT_STATUS.md`、`.gitignore` 同步

## 3. 冒烟协议（统一口径）

- 对照：**V_B1C@2k**
- Scene5 `val_sub` 350 张；训练子集 1600；seed **42**
- crop 512，lr `6e-5`，**2000** iters，**batch 16**，每 500 iter 验证
- `num_workers=0`（与对照增强 RNG 一致；故 GPU 利用率故意偏低）
- 噪声带：`|Δ| < 0.02` 视为噪声
- 门槛备忘：
  - M_ROW：sky→mountain &lt; 12% 且 mountain ≥ +0.03，其它类下降 ≤ 0.01
  - M_BLV：crater→regolith &lt; 7%，或 crater ≥ +0.02

## 4. 4090 — 方法冒烟矩阵（已完成）

来源：`results/smoke_method_eval.md`（Scene5 本地，对照 V_B1C）。  
完成时间约 2026-11-08 03:00（Asia/Shanghai）。

| exp | mIoU | rock | crater | mountain | sky | sky→mtn | crater→reg | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| V_B1C | 0.8014 | 0.7228 | 0.7407 | 0.6530 | 0.9180 | 0.0587 | 0.0491 | 对照 |
| M_ROW | 0.8073 | 0.7572 | 0.7661 | 0.6390 | 0.8992 | 0.0900 | 0.0378 | RowHeightAttn |
| M_BLV | 0.8041 | 0.7540 | 0.7235 | 0.6801 | 0.8901 | 0.0987 | 0.0380 | BLV logit |
| M_SHAD | 0.8131 | 0.7397 | 0.7429 | 0.6891 | 0.9192 | 0.0542 | 0.0695 | shadow_augment |
| M_ABL | 0.7879 | 0.7631 | 0.6967 | 0.6390 | 0.8677 | 0.1221 | 0.0403 | ABL-lite（偏弱） |
| M_INVFORM | 0.8048 | 0.7648 | 0.7298 | 0.6554 | 0.8990 | 0.0883 | 0.0482 | stub→ABL-lite |
| M_TTA | 0.8274 | 0.7219 | 0.7511 | 0.6889 | 0.9958 | 0.0026 | 0.0853 | 推理 TTA（唯一超噪声） |
| M_SEGFIX | 0.8021 | 0.7237 | 0.7415 | 0.6564 | 0.9161 | 0.0583 | 0.0494 | 后处理无效 |
| T_B1_TTA | 0.8795 | 0.7835 | 0.8730 | 0.7599 | 0.9932 | 0.0058 | 0.0757 | B1×20k 权重上 TTA（不与 V_B1C 直接比） |

要点：
- 训练期模块均在噪声带内；**M_TTA +0.026** 为唯一明显增益，但 crater→reg 变差
- **M_SHAD** 训练期最好（+0.012，mountain +0.036）
- M_ABL / M_SEGFIX **不进组合**；M_INVFORM 与 ABL 同配置，两次差约 0.017 可作噪声标定

## 5. 4090 — 两两组合消融（进行中）

日志目录：`results/smoke_combo/`。每卡 1 进程；**不跑第三批**（第三批在 cheerhin）。

队列（FULL pairwise v2）：
- GPU0：`M_ROW_SHAD` → `M_ROW_SHAD_TTA` → `M_BLV_SHAD` → `M_BLV_SHAD_TTA`
- GPU1：`M_SHAD_TTA` → `M_BLV_TTA` → `M_ROW_TTA` → `V_B1C_noise` → `M_ROW_BLV` → `M_ROW_BLV_TTA`

已完成（截至本文件撰写时）：
- TTA 两两：`M_SHAD_TTA`、`M_BLV_TTA`、`M_ROW_TTA`
- 训练：`M_ROW_SHAD` + `M_ROW_SHAD_TTA`；`V_B1C_noise`
- 进行中：`M_BLV_SHAD`（GPU0）、`M_ROW_BLV`（GPU1）
- 待跑：`M_BLV_SHAD_TTA`、`M_ROW_BLV_TTA`

筛选规则（用户）：优先坑+山两侧都改善的三组合；否则选整体 mIoU（再看 rock）提升最大者。暂不加第四模块。

候选三组合：`ROW+SHAD+TTA`（优先）、`BLV+SHAD+TTA`、`ROW+BLV+TTA`。

## 6. cheerhin（由 @LuSNAR 维护）

> 以下数字来自 LuSNAR 同步口径；后续更新请 LuSNAR 直接改本节。

- 角色：第三批方法冒烟（DySample / MSInfer / Hard / Contrast / SAPA 等），**不做长训**
- **V_B1C@2k**（cheerhin Scene5）：best mIoU **0.8126**（rock 0.747 / crater 0.746 / mountain 0.684；Boundary F1 mean 0.729）
- 当前：`M_DYSAMPLE` ~iter 600；batch-3 其余仍在跑
- 行政文件已移入 `_tmp/`（含访问登记等 docx/pdf），不入库

（LuSNAR：请在此补充 batch-3 各实验完成后的表格与结论。）

## 7. 仓库同步说明

- 首批提交：`.gitignore`、`PROJECT_STATUS.md`、`results/*.md`、`tools/` 脚本（无权重、无数据）
- 两边 Bot 讨论后共同维护本文件；实验状态变更时更新对应章节与时间戳

## 8. 联系 / Bot 分工

| Bot | 主机 | 负责 |
|-----|------|------|
| 4090 Bot（本会话） | DESKTOP-ORQLB8F | 方法矩阵、组合消融、本文件 4090 章节 |
| @LuSNAR | cheerhin | 第三批冒烟、本文件 cheerhin 章节 |
