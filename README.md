# LuSNAR-9221 — LLM Agent README

> **给大模型 / Agent 的入口文档。** 数字与实验细节以 `PROJECT_STATUS.md` 与 `results/*.md` 为准；本文件只定方向与硬约束，不复制完整分数表。

仓库：https://github.com/cheerhin-yi/LuSNAR-9221.git  
工程根（双机）：`D:\LuSNAR_Seg`  
状态真源：`PROJECT_STATUS.md`

---

## 1. What this repo is

月球场景语义分割竞赛工程（LuSNAR / MoonSeg）。骨干 **SegFormer-B2**（`mit_b2`），训练与评测走 MMSegmentation。

- **官方类别顺序（不可改）**：`0 regolith, 1 rock, 2 crater, 3 mountain, 4 sky`
- **截止**：约 **2026-11-20（Asia/Shanghai / UTC+8）**（以官方通知为准）
- **本地验证场景**：Scene5（子集分数 ≠ 正式成绩）
- **开源测试集**：`Moon_2` / `Moon_4` / `Moon_9`（见下节硬约束）

---

## 2. Hard constraints（必读）

1. **`Moon_2` / `Moon_4` / `Moon_9` 的 GT 仅作测试**：禁止用于训练、调参、模型选择、带标签诊断或「偷看」后改配置。
2. **本地选模 / 冒烟只看 Scene5**（及协议写明的本地子集）；不要把开源测试当验证集循环。
3. **跨机分数不可直接比**：4090 与 cheerhin 的 `V_B1C` 数字不同；任何本机 Δ 必须相对 **本机重跑的 V_B1C**。
4. **噪声带**：相对本机对照 `|Δ| < 0.02` 视为噪声，不能当增益宣称「有效」。
5. **不擅自做官方盲评 / 提交**：需用户明确批准。
6. **不加未批准的额外训练数据**。

---

## 3. Dual-machine layout

| 主机 | GPU | machineId（短） | 角色 |
|------|-----|-----------------|------|
| DESKTOP-ORQLB8F | 4090×2 | `370a07e6-…` | 方法矩阵、两两组合、**ROW+BLV 长训** |
| cheerhin | 5060 Ti 16GB | `2256a039-…` | 第三批方法冒烟；**不做长训** |

- 两边工程根均为 `D:\LuSNAR_Seg`，顶层约定：`configs/ data/ logs/ pretrained/ results/ tools/ work_dirs/ _tmp/`
- **Git 同步**：`tools/`、`results/*.md`、`PROJECT_STATUS.md`、`.gitignore`、本 `README.md` 等说明与脚本
- **不同步**：`data/`、`*.pth` / `pretrained/` / `work_dirs` 权重、`_tmp/`、`*.docx` / `*.pdf`

分工写状态：4090 Bot 写 4090 章节；@LuSNAR（cheerhin）写 cheerhin 章节。

---

## 4. Stack & env

| 项 | 值 |
|----|----|
| MMSeg | 1.2.2 |
| PyTorch | 2.7.1+cu128 |
| Backbone / model | `mit_b2` / SegFormer-B2 |
| cheerhin conda | `F:\Conda\envs\lusnar` |
| 冒烟协议摘要 | crop 512，lr `6e-5`，2000 iters，batch 16，seed 42，`num_workers=0`，每 500 iter 验证 |

冒烟入口（cheerhin 示例）：`tools/portable/lusnar_smoke_compare.py`；长训（4090）：`tools/portable/lusnar_row_blv_ddp.py`。细节见 `tools/portable/`（及 `portable_b3/`，若存在）与 `tools/portable/README_METHOD_SMOKE.md`。

---

## 5. Current decisions（对齐 tip 时点）

以 `PROJECT_STATUS.md` 时间戳为准；摘要如下：

- **长训底座（4090）**：**ROW + BLV**；推理侧可挂 **TTA**（`M_ROW_BLV_TTA`）。
- **不要**把 **SHAD** 与 ROW/BLV 同训（组合消融负交互）。
- cheerhin **batch-3**（`M_DYSAMPLE` / `M_MSINFER` / `M_HARD` / `M_CTRAST` / `M_SAPA`）：相对本机 V_B1C **无超出噪声带的正向增益** → **无 stack candidate**；不并入长训。
- `M_ABL` / `M_SEGFIX` 等不进组合；完整矩阵与组合表见结果文档，勿在 README 抄表。

---

## 6. Where to look

| 路径 | 用途 |
|------|------|
| `PROJECT_STATUS.md` | **状态真源**（双机进度、结论、同步约定） |
| `results/smoke_method_eval.md` | 4090 方法矩阵 + 两两组合消融（含交互项） |
| `results/smoke_method_eval_batch3.md` | cheerhin 第三批完整报告 |
| `tools/portable/` | 可移植冒烟 / 训练脚本（含 `portable_b3` 若有） |
| `work_dirs/*/summary.json` | **本机**实验摘要（不进仓或勿依赖远端有权重） |
| `results/oneshot_protocol.json` | 开源测试 / 后续加 Scene5 的预定协议（4090） |

需要具体 mIoU / 逐类 / Boundary 指标时：**打开上表文档**，不要凭记忆编造。

---

## 7. Git discipline

1. 改前：`git pull --rebase origin main`
2. **谁改谁推**；两边共维 `PROJECT_STATUS.md` 时只改自己负责章节 + 时间戳
3. **禁止 force-push**
4. **永不提交**：`data/`、`*.pth`、权重目录、`_tmp/`、`*.docx`、`*.pdf`、大体量 `work_dirs` 产物（以 `.gitignore` 为准）
5. 推送前确认 `git status`：只暂存说明与脚本

---

## 8. Official palette `COLOR_TO_ID`（提交 / 评测脚本）

RGB → class id（官方调色板）：

| RGB | id | name |
|-----|----|------|
| `(187, 70, 156)` | 0 | regolith |
| `(232, 250, 80)` | 1 | rock |
| `(120, 0, 200)` | 2 | crater |
| `(173, 69, 31)` | 3 | mountain |
| `(34, 201, 248)` | 4 | sky |

---

## 9. For LLM agents

```
1. 先读 PROJECT_STATUS.md，再动实验或改结论。
2. 不要编造分数；引用时写清机器、对照 V_B1C 数值与文档路径。
3. 训练 / 调参 / 选模禁止触碰 Moon_2/4/9；Scene5 才是本地验证。
4. 宣称「显卡空闲」前检查本机相关 python/训练进程，勿杀用户作业。
5. 实验结束后更新 PROJECT_STATUS 中【本机章节】与「最后更新」时间（Asia/Shanghai）。
6. 跨机协作：对齐 tip 后再改共享文件；冲突用 rebase，不用 force。
7. 数字表在 results/*.md；本 README 过时以 PROJECT_STATUS 为准。
```

---

## Quick pointers

- 冒烟对照名：**V_B1C@2k**（必须本机重跑）
- 长训目录（4090）：`work_dirs/train_ROW_BLV_20k/`、日志 `logs/train_ROW_BLV_20k.log`
- cheerhin 第三批 launcher / resume 日志见 `PROJECT_STATUS` §6
