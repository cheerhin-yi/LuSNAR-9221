# LuSNAR-9221 项目状态

> 最后更新：2026-11-09 15:08（Asia/Shanghai）  
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
| DESKTOP-ORQLB8F（4090×2） | machineId `370a07e6-…` | 方法冒烟矩阵 + 两两组合消融 | `D:\\LuSNAR_Seg` |
| cheerhin（5060 Ti） | machineId `2256a039-…` | 第三批方法冒烟（LuSNAR 管理） | `D:\\LuSNAR_Seg` |

- 数据集 `data/` **不同步**到 GitHub
- 权重 `*.pth` / `pretrained/` **不上传**；`_tmp/`、`*.docx`、`*.pdf` 等行政文件不上传
- 脚本与结果说明：`tools/`、`results/*.md`、`PROJECT_STATUS.md`、`.gitignore` 同步

## 3. 冒烟协议（统一口径）

- 对照：**V_B1C@2k**（**必须本机重跑**；跨机数字不可直接比）
- Scene5 `val_sub` 350 张；训练子集 1600；seed **42**
- crop 512，lr `6e-5`，**2000** iters，**batch 16**，每 500 iter 验证
- `num_workers=0`（与对照增强 RNG 一致；故 GPU 利用率故意偏低）
- 噪声带：`|Δ| < 0.02` 视为噪声
- 门槛备忘：
  - M_ROW：sky→mountain < 12% 且 mountain ≥ +0.03，其它类下降 ≤ 0.01
  - M_BLV：crater→regolith < 7%，或 crater ≥ +0.02
- 评测附加：Boundary F1 / Boundary IoU（容差 3px）；M_MSINFER 为整图评测，**不可**与中心裁剪验证直接比

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

## 5. 4090 — 两两组合消融（ALL DONE）

- **完成时间**：2026-11-08 17:59（Asia/Shanghai）；launcher 日志写出 `ALL DONE`
- **日志**：`results/smoke_combo/`（launcher + 各实验 `.log`）
- **结果**：10 个 `summary.json` 齐全于 `work_dirs/smoke_*/summary.json`（`M_ROW_SHAD`、`M_ROW_SHAD_TTA`、`M_BLV_SHAD`、`M_BLV_SHAD_TTA`、`M_SHAD_TTA`、`M_BLV_TTA`、`M_ROW_TTA`、`V_B1C_noise`、`M_ROW_BLV`、`M_ROW_BLV_TTA`）
- **对照**：4090 本机 Scene5 `V_B1C` mIoU **0.8014**（**不可**与 cheerhin 本机 V_B1C 0.8126 直接比）
- **全文表 / 交互项**：见 `results/smoke_method_eval.md`（Pairwise combo ablation 节）
- **本机不跑第三批**（batch-3 在 cheerhin）；**从未触碰 Moon_2/4/9**

### 5.1 最优与训练侧结论

| 项 | 结果 |
|---|---|
| 全队列最优 | **M_ROW_BLV_TTA** mIoU **0.8522**，Δ=**+0.0509** |
| 训练侧只保留 | **ROW+BLV**（mIoU 0.8219，Δ=+0.0205，I=+0.0120 **协同**） |
| 负交互 | ROW×SHAD I=−0.0107；BLV×SHAD I=−0.0292（组合低于对照）→ **不要把 SHAD 与 ROW/BLV 同训** |
| TTA | 帮助 ROW+BLV（+0.0303）、BLV（+0.0319）；伤害 ROW、ROW+SHAD（≈0）、尤其 BLV+SHAD（−0.0333，岩石/山体崩） |
| 噪声标定 | `V_B1C_noise` Δ=+0.0052，落在 \|Δ\|<0.02 带内；单模块 ROW/BLV 的 +0.003～+0.006 **不可靠** |

### 5.2 三模块筛选（用户规则已应用）

优先 crater **与** mountain 同时提升；否则看整体 mIoU / rock。

- **入选**：仅 **ROW+BLV**（坑+山双升）；推理侧可挂 **TTA** → `M_ROW_BLV_TTA`
- **不加第三训练模块**；**不开第四模块**
- SHAD 与 ROW、BLV 均为负交互，不再并入

### 5.3 摘要表（Δ vs 4090 V_B1C 0.8014）

| exp | mIoU | Δ | rock | crater | mountain | note |
|---|---:|---:|---:|---:|---:|---|
| V_B1C | 0.8014 | — | 0.7228 | 0.7407 | 0.6530 | 对照 |
| M_ROW_BLV | 0.8219 | +0.0205 | 0.7479 | 0.7828 | 0.6958 | 训练侧最佳 |
| M_ROW_BLV_TTA | 0.8522 | +0.0509 | 0.7520 | 0.7897 | 0.7423 | **全队列最优** |
| M_BLV_TTA | 0.8359 | +0.0346 | 0.7556 | 0.7211 | 0.7294 | TTA 有助 BLV |
| M_BLV_SHAD | 0.7866 | −0.0148 | 0.7113 | 0.7230 | 0.6101 | 强负交互 |
| M_BLV_SHAD_TTA | 0.7533 | −0.0481 | 0.5716 | 0.7505 | 0.4814 | 严重负交互 |
| V_B1C_noise | 0.8066 | +0.0052 | 0.7721 | 0.7184 | 0.6805 | 噪声带内 |

完整交互项表与全部 10+ 对照行见 `results/smoke_method_eval.md`。

## 6. cheerhin（由 @LuSNAR 维护）

- **Host / GPU**：cheerhin · RTX 5060 Ti 16GB · `machineId 2256a039-ab6a-4308-8c10-5d88e0be1ab4`
- **工程**：`D:\\LuSNAR_Seg` · env `F:\\Conda\\envs\\lusnar` · torch 2.7.1+cu128 · MMSeg 1.2.2 · SegFormer-B2
- **数据路径**（不同步）：`--data-root D:\\LuSNAR_Seg\\data\\raw --train-subdir LuSNAR_train`
- **角色**：第三批方法冒烟（`M_DYSAMPLE` / `M_MSINFER` / `M_HARD` / `M_CTRAST` / `M_SAPA`），**本机重跑 V_B1C 对照**；不做长训
- **脚本**：`tools/portable/`（`_lusnar_common.py`、`lusnar_smoke_compare.py`、`README_METHOD_SMOKE.md`）
- **行政文件**：已移入 `_tmp/`（`数据访问登记表_已填写.docx/.pdf`、`挑战赛规则协议_已填写.docx/.pdf`），gitignore 排除 `_tmp/` 与 `*.docx/*.pdf`

### 6.1 Dry-run（2026-11-08，batch 16，无需 accum）

| exp | peak_mem_mb | note |
|---|---:|---|
| V_B1C | 12014 | OK |
| M_DYSAMPLE | 12065 | OK |
| M_CTRAST | 12015 | OK（contrastive loss 生效） |

### 6.2 Batch-3 full smoke（已完成，2026-11-09 03:46 Asia/Shanghai）

原始 launcher pid 21276；`M_CTRAST` / `M_SAPA` 由 resume pid 2132 补跑完成（日志 `D:\\LuSNAR_Seg\\logs\\batch3_resume.out.log`；`work_dirs/smoke_M_CTRAST/summary.json` 与 `work_dirs/smoke_M_SAPA/summary.json` 已写出）。六项 `summary.json` 均齐全。

命令：

```powershell
& 'F:\\Conda\\envs\\lusnar\\python.exe' 'D:\\LuSNAR_Seg\\tools\\portable\\lusnar_smoke_compare.py' `
  --project-root D:\\LuSNAR_Seg --data-root D:\\LuSNAR_Seg\\data\\raw --train-subdir LuSNAR_train `
  --exps V_B1C,M_DYSAMPLE,M_MSINFER,M_HARD,M_CTRAST,M_SAPA `
  --iters 2000 --batch 16 --val-every 500 --seed 42 --lr 6e-5
```

对照本机 V_B1C `best_mIoU_scene5` **0.812645**（表中四位小数 **0.8126**）。噪声带：`|Δ| < 0.02`。

| exp | status | best mIoU | rock | crater | mountain | BoundF1 | Δ vs 本机 V_B1C | note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| V_B1C | **done** | **0.8126** | 0.7467 | 0.7464 | 0.6844 | 0.7285 | — | 本机对照；wall 5684s；peak_mem 12014 MB |
| M_DYSAMPLE | **done** | 0.8079 | 0.7502 | 0.7504 | 0.6568 | 0.7245 | −0.0047 | 噪声带内略低；peak_mem 12162 MB |
| M_MSINFER | **done** | — | — | — | — | — | n/a | eval-only，整图/滑窗，不可与中心裁剪比；见下表 |
| M_HARD | **done** | 0.7860 | 0.7243 | 0.7341 | 0.6190 | 0.7025 | −0.0266 | 超出噪声带的下降；不进组合 |
| M_CTRAST | **done** | 0.7941 | 0.7461 | 0.7400 | 0.6502 | 0.7215 | −0.0185 | 噪声带内略低；lite boundary-pair InfoNCE；wall 6491s |
| M_SAPA | **done** | 0.7967 | 0.7266 | 0.7160 | 0.6496 | 0.6973 | −0.0159 | 噪声带内略低；SAPA-B-lite；wall 9132s |

M_MSINFER（权重 `smoke_V_B1C/best.pth`；四种独立整图模式；summary 的 `best_mIoU_scene5` 取 whole_1.0 = 0.7707）：

| mode | mIoU | rock | crater | mountain | BoundF1 | note |
|---|---:|---:|---:|---:|---:|---|
| whole_1.0 | 0.7707 | 0.6035 | 0.6656 | 0.6127 | 0.6630 | 整图 scale 1.0 |
| slide_1.0 | 0.7820 | 0.6255 | 0.6977 | 0.6120 | 0.6784 | 四种模式最高 |
| slide_1.25 | 0.7564 | 0.6149 | 0.5729 | 0.6275 | 0.6714 | crop 512 / stride 256 |
| slide_1.5 | 0.7008 | 0.5371 | 0.4312 | 0.5760 | 0.6242 | crop 512 / stride 256 |

> 跨机不可比：4090 的 V_B1C=0.8014，cheerhin 本机对照=0.812645；第三批 Δ 一律相对 **0.812645**。

结论：
- 训练期五项里，**没有任何方法相对本机 V_B1C 超出噪声带的正向增益**，因此 **无 stack candidate**
- M_DYSAMPLE（−0.0047）、M_CTRAST（−0.0185）、M_SAPA（−0.0159）均落在 `|Δ|<0.02` 噪声带内、略低于对照
- **M_HARD −0.0266** 是唯一超出噪声带的变化，方向为下降，不进组合
- M_MSINFER 不可与中心裁剪对照直接比较
- 从未触碰 Moon_2/4/9

归档（2026-11-09 15:08，cheerhin）：本机脚本刷新的完整第三批报告（Boundary F1/IoU 明细、M_MSINFER 四模式表、逐类边界指标）已存为 `results/smoke_method_eval_batch3.md`；推送前已抽查六项 `summary.json`（mIoU / wall_s / peak_mem_mb）与本表记录一致。4090 的 `results/smoke_method_eval.md`（方法矩阵 + 组合消融）保持不变，第三批报告不覆盖该文件。

## 7. 仓库同步说明

- 首批提交：`.gitignore`、`PROJECT_STATUS.md`、`results/*.md`、`tools/` 脚本（无权重、无数据）
- 两边 Bot 讨论后共同维护本文件；实验状态变更时更新对应章节与时间戳

## 8. 联系 / Bot 分工

| Bot | 主机 | 负责 |
|-----|------|------|
| 4090 Bot（本会话） | DESKTOP-ORQLB8F | 方法矩阵、组合消融、本文件 4090 章节 |
| @LuSNAR | cheerhin | 第三批冒烟、本文件 cheerhin 章节 |
