# LuSNAR portable training scripts

两份可带走脚本（相对路径），环境与 cheerhin `lusnar` 一致：Python 3.10、torch cu128、mmsegmentation 1.2.2、mmcv-lite + `mmcv_stub.py`。

## 目录约定

```
<project>/                          # 默认：本脚本所在目录的上两级（tools/portable → 项目根）
  data/
    train/                          # Moon_{1,3,5,6,7,8}/{color,label}
                                    # cheerhin 可 junction: data\train -> data\raw\LuSNAR_train
                                    # 或 --data-root <project>/data/raw --train-subdir LuSNAR_train
    test/                           # 仅推理预留；训练脚本绝不读 label
    smoke/                          # 冒烟子集 / rarity / CC bank（自动生成）
  pretrained/
    mit_b2_20220624-66e8bf70.pth
  tools/portable/
    lusnar_smoke_compare.py         # 脚本1：A0–C1 短训对比
    lusnar_train_ddp.py             # 脚本2：单卡或 torchrun 双卡正式训
    eval_scene5_report.py           # 可选：读 checkpoint 复现 Scene5 表
    _lusnar_common.py
    mmcv_stub.py
    README_PORTABLE.md
  work_dirs/<run_name>/summary.json
  results/smoke_matrix_report.md
  results/train_report_<run>.md
```

类别官方序：0 月壤，1 岩石，2 撞击坑，3 山体，4 天空。

## 消融默认

冒烟矩阵默认 **A0,A1,B1**（不含 C1）。正式训默认 `--recipe B1`。C1 仅显式传入时启用。

## cheerhin 验证

```powershell
cd D:\LuSNAR_Seg

# 可选：建立相对约定目录
# cmd /c mklink /J "D:\LuSNAR_Seg\data\train" "D:\LuSNAR_Seg\data\raw\LuSNAR_train"

F:\Conda\envs\lusnar\python.exe tools\portable\lusnar_smoke_compare.py --help
F:\Conda\envs\lusnar\python.exe tools\portable\lusnar_train_ddp.py --help

# 不建 junction 时用官方解压目录名：
F:\Conda\envs\lusnar\python.exe tools\portable\lusnar_smoke_compare.py `
  --project-root D:\LuSNAR_Seg `
  --data-root D:\LuSNAR_Seg\data\raw `
  --train-subdir LuSNAR_train `
  --exps A0 --iters 5 --batch 2 --dry-run --skip-prepare

# 正式冒烟矩阵（会跑较久；先 prepare）
F:\Conda\envs\lusnar\python.exe tools\portable\lusnar_smoke_compare.py `
  --data-root D:\LuSNAR_Seg\data\raw --train-subdir LuSNAR_train `
  --exps A0,A1,B1 --iters 1000 --batch 8
# C1 需显式: --exps A0,A1,B1,C1
```

## 双卡 4090（48G）示例

```bash
cd /path/to/LuSNAR_Seg
# 数据：data/train/Moon_*/{color,label} ，预训练放 pretrained/

torchrun --nproc_per_node=2 tools/portable/lusnar_train_ddp.py \
  --recipe B1 \
  --iters 20000 \
  --batch 12 \
  --crop 512 \
  --workers 6 \
  --amp bf16 \
  --val-every 1000
```

单卡：

```bash
python tools/portable/lusnar_train_ddp.py --recipe B1 --batch 8 --iters 20000
```

Recipe 含义：

| recipe | RCS+mosaic+CC | ACW | EPS+MS+Photo+EMA |
|--------|---------------|-----|------------------|
| A0     |                 |     |                    |
| A1     | ✓               |     |                    |
| B1     | ✓               | ✓   |                    |
| C1     | ✓               | ✓   | ✓                  |

## 结果打包

跑完后拷回分析：

```
work_dirs/smoke_A0/summary.json
work_dirs/smoke_A1/summary.json
...
work_dirs/train_C1/summary.json
work_dirs/train_C1/best.pth
work_dirs/*/train_log.jsonl
results/smoke_matrix_report.md
results/train_report_train_C1.md
```

`summary.json` 字段含：`best_mIoU_scene5`、`per_class_IoU_*`、`iou_rock`、`iou_crater`、`wall_sec`、`peak_mem_mb`、`flags`、`world_size`、`error`、`config`（正式训）。

复现评估：

```bash
python tools/portable/eval_scene5_report.py \
  --checkpoint work_dirs/train_C1/best.pth \
  --out-name eval_train_C1
```

## 硬规则提醒

- 禁止使用测试场景 2/4/9 真实标签训练/验证/调参
- 禁止额外数据；不得伪造日志/权重；本地 Scene5 ≠ 官方测试成绩
