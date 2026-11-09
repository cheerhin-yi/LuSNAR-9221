# SegNeXt-S / LuSNAR：Windows Conda 训练包

当前默认使用第二张 CUDA 可见显卡 `cuda:1`，读取 `image0` 的原图与标签，按下文固定划分进行训练、验证和测试。

在已配置好的 `lusnar` 环境中，从本文件夹运行 `python train_segnext_s.py train` 或 `run_train.bat` 即可。默认训练 **100 个 epoch**，使用 standard 数据增强。

若设置了 `CUDA_VISIBLE_DEVICES`，显卡编号以 CUDA 可见设备的顺序为准。
`model.py` 根据官方源码移植，使用 MSCAN-S + LightHamHead / NMF，约 1389.6 万参数（5 类）。
保留官方小模型通道、层数、MLP 比率与解码器结构；把 SyncBN 改成单 GPU BatchNorm，移除 MMCV 依赖，并让 NMF 在当前设备用 FP32 运算。
这是便于 Windows 运行的移植版，训练轮数和数据划分为本任务设定，不是论文基准复现。

## 1. 放置文件与安装

解压整个文件夹到 `D:\LuSNAR_Seg\segnext_s_lusnar`，所有文件保持在一起。
打开 **Anaconda Prompt**。截图中的 `>>>` 表示仍在 Python 内，先输入 `exit()` 返回命令行。

```bat
conda activate lusnar
cd /d D:\LuSNAR_Seg\segnext_s_lusnar
python -m pip install -r requirements.txt
python train_segnext_s.py inspect --data-root D:\LuSNAR_Seg\data
```

如果检查输出 `CUDA available: False`、torch 未安装，或 GPU 架构不受当前 torch 支持，安装以下 CUDA 版本后再检查：

```bat
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

这是 PyTorch 官方提供的 Windows CUDA 12.8 安装源；使用 Python 3.10。
若已安装的同版本 torch 是 CPU 构建，给该命令加 `--force-reinstall`。
无需安装 mmcv、mmsegmentation、torchvision 或单独编译 CUDA 扩展。
截图没有显示完整 GPU 型号，因此脚本会实际检测设备，并执行 CUDA 张量计算检查兼容性。

## 2. 数据路径（已按你的补充截图设置）

```text
D:\LuSNAR_Seg\data\
  Moon_1\
    image0\
      color\1693274724086398208.png
      label\1693274724086398208.png
      depth\...
    image1\
      color\...
      label\...
    LiDAR\...
  Moon_3\...
  Moon_5\...
  ...
```

文件扩展名以实际文件为准，支持 png/jpg/jpeg/bmp/tif/tiff；color 与 label 按不含扩展名的相对路径严格配对。
默认使用 `image0`，灰度图自动复制成三个通道；输入采用 ImageNet 均值方差归一化。
`--camera image1` 切换到另一相机，`--camera all` 使用两个相机；两相机会随所属 Moon 场景进入同一数据划分。
depth、LiDAR、gt、imu 不参与这次标准 2D 语义分割。

默认标签使用 LuSNAR 官方公布的 **RGB** 五类颜色，P 模式标签会先按自身调色板转为 RGB：

|训练 ID|类别|RGB|
|---|---|---|
|0|月壤 lunar_regolith|187,70,156|
|1|撞击坑 impact_crater|120,0,200|
|2|岩石 rock|232,250,80|
|3|山体 mountain|173,69,31|
|4|天空 sky|34,201,248|

不把天空当忽略类，不对颜色做容差匹配。正式训练前逐个检查所有图像/标签及颜色；未知颜色会报文件名和颜色值，避免训练错类。
若数据标签与官方颜色不同，请根据原始标签说明修改 `schema_rgb.json`，并添加 `--schema schema_rgb.json`。
`schema_ids_EXAMPLE.json` 仅演示单通道 ID 标签，里面的 0..4 **不是对你原始数据 ID 的断言**。
不要拿截图缩略图颜色或 LiDAR 类别 ID 代替原始 2D 标签定义。

## 3. 先检查，再训练

只检查设备、配对、尺寸、标签颜色、划分及类别像素数量：

```bat
python train_segnext_s.py train --check-only
```

开始训练（单 GPU、512×512 随机裁剪、batch=4、100 轮、AMP）：

```bat
python train_segnext_s.py train
```

或在已激活 lusnar 的 Anaconda Prompt 运行：

```bat
run_train.bat
```

如果有官方 ImageNet 预训练 MSCAN-S 权重，推荐使用：

```bat
python train_segnext_s.py train --pretrained D:\LuSNAR_Seg\pretrained\mscan_s.pth
```

权重由 SegNeXt 官方仓库 README 的 ImageNet Pre-trained models 链接获取。
本包不附带权重，不会自动下载；未传 `--pretrained` 时明确从随机初始化开始训练，准确率与收敛速度可能不同。
加载时严格核对 MSCAN-S 主干所需张量的名称和形状，拒绝错误的模型型号。

默认固定划分如下。

| 用途 | 场景 |
|---|---|
| 训练 | Moon_1、Moon_3、Moon_5、Moon_6、Moon_7、Moon_8 |
| 验证 | Moon_2 |
| 测试 | Moon_4、Moon_9 |

请把这九个场景放在同一数据根目录下。脚本默认读取 `D:\LuSNAR_Seg\data`，可用 `--data-root` 指定实际目录。缺少任一指定场景的配对样本时，脚本会列出缺失场景并停止。

如要进行其他实验，可用 `--train-all` 显式选择全量训练，或显式指定验证和测试场景：

```bat
python train_segnext_s.py train --scene-split --val-scenes Moon_7 --test-scenes Moon_8 --output runs\segnext_s_scene_split
```

验证与测试保留原始分辨率，采用 512 裁剪、384 步长滑窗并平均重叠 logits。训练增强的配置见下文。
损失为交叉熵，AdamW 主干学习率 6e-5、解码器 10 倍，线性预热及多项式衰减。
每轮根据 Moon_2 的验证 mIoU 选择最佳权重。完成训练后，用这份权重评估合并的 Moon_4 和 Moon_9 测试集。
类别 IoU 的分母为 0 时记为 null，不计入 mIoU；指标为 0..1，比率乘 100 才是百分比。

## 4. 输出、恢复及设备选择

默认输出目录为 `runs\segnext_s_gpu1_scene_split_aug`。

- `best.pth` 保存验证 mIoU 最高的模型。
- `last.pth`：最近完整一轮的模型、优化器、AMP 和随机状态，用于恢复。
- `history.csv` 记录每轮训练损失、验证 mIoU、像素准确率和耗时。
- `best_metrics.json` 记录最佳验证指标，`test_metrics.json` 记录训练完成后的独立测试指标。
- `schema.json` / `split_manifest.json` / `pixel_counts.json`：类别、实际数据划分及像素数。
- `args.json` / `run_identity.json`：参数及恢复时的配置校验信息。
- `augmentation.json`：本次增强模式、强度、概率、填充值和模块版本。

新的场景划分使用独立实验目录。恢复训练时使用同一划分产生的 `last.pth`。

恢复本次划分的训练，使用相同参数、数据路径和输出目录：

```bat
python train_segnext_s.py train --resume
```

例如首次设置了 `--epochs 20 --batch-size 2`，恢复时也要带同样参数。
恢复到最近已完成的一轮；中途未完成的一轮会重新训练。多进程数据加载情况下不承诺逐位一致。
新实验通过 `--output runs\experiment2` 使用新目录，防止误覆盖已保存实验。

默认使用 `cuda:1`，batch size 为 4，裁剪尺寸为 512×512。
若显存不足，先把 batch 降至 2，再考虑减小 size（至少 64 且为 32 的倍数）：

```bat
python train_segnext_s.py train --batch-size 2 --size 384 --output runs\small
```

Windows 默认 workers=0；成功运行后可尝试 `--workers 4` 提高读取速度。
`--no-amp` 可关闭混合精度；`--device cpu` 用于小规模流程测试。

## 训练数据增强

新增 `augmentations.py`，仅依赖现有 NumPy 和 Pillow，无需安装其他包。增强在读取每个训练样本时随机生成，不修改原始图片或标签，也不额外保存扩增数据。每轮读取同一图片时可能得到不同增强结果；样本数量不因在线增强而增加。

默认 `--augmentation standard`：

| 增强 | 默认设置 |
|---|---|
| 缩放与裁剪 | 原图尺寸随机缩放至 0.5–2.0 倍，再裁剪为 size×size |
| 水平翻转 | 概率 50% |
| 小角度旋转 | 概率 30%，角度 -5° 到 +5° |
| 亮度、对比度、Gamma | 概率 50%，亮度/对比度系数 0.8–1.2，Gamma 0.9–1.1 |
| 高斯噪声 | 概率 15%，像素标准差 2（RGB 取值 0–255） |
| 高斯模糊 | 概率 10%，半径 0–0.6 像素 |

几何参数在原图和标签间共享；标签使用最近邻插值，填充/旋转空白标为 255（忽略），不会生成新的类别编号。亮度等变化只作用于 RGB 原图。没有默认上下翻转或 90° 旋转。验证和测试始终不使用随机增强。

保留基础几何增强：

```cmd
python train_segnext_s.py train --augmentation basic --output runs\segnext_s_basic
```

关闭随机增强（仍以固定的等比例缩放和居中填充适配输入尺寸）：

```cmd
python train_segnext_s.py train --augmentation none --output runs\segnext_s_noaug
```

自定义概率或强度，可编辑随附 JSON，配置文件只需要写想覆盖的参数：

```cmd
python train_segnext_s.py train --augmentation standard --augmentation-config augmentation_config_example.json
```

增强参数纳入断点一致性校验。更改增强设置后请用新的输出目录开始实验，不可直接恢复之前无此模块的旧断点。Python、NumPy、CPU Torch 和所选显卡的随机状态随新断点保存；多进程 worker 情况下仍不承诺逐位一致。

模块检查：`python test_augmentations.py`。这会验证配对、类别值、随机复现和填充规则，不会启动 GPU 训练。当前执行环境未实跑含新增强的 GPU 训练；这些增强强度是起步配置，尚未在你的数据上调优。

## 首批梯度非有限错误的修正

如果旧版在 `clip_grad_norm_` 提示 `total norm ... non-finite`，说明反向传播后的
梯度或其范数不是有限值；不能通过忽略报错就认定训练正确。
旧版会在 AMP 动态 loss scale 调整之前直接报错退出。
更新版保留严格梯度裁剪：遇到确认的 AMP 非有限梯度，交给 GradScaler 跳过本次
优化器更新并降低 scale，打印受影响张量；连续 20 批溢出、整轮没有成功更新或
FP32 梯度仍异常时会明确停止，不保存无更新的这一轮为最佳模型。

排查时先关闭 AMP，以新目录从头运行：

```bat
python train_segnext_s.py train --no-amp --output runs\segnext_s_fp32
```

第一批就失败的旧运行没有完整一轮的 checkpoint，不应添加 `--resume`。
若 FP32 仍报错，需提供新 traceback，再定位具体层；不能仅凭旧截图断定都是 AMP 导致。

## 来源与验证范围

- SegNeXt 官方源码与模型配置：https://github.com/Visual-Attention-Network/SegNeXt
- LuSNAR 官方类别颜色与数据说明：https://github.com/zqyu9/LuSNAR-dataset
- PyTorch 官方安装命令：https://pytorch.org/get-started/previous-versions/
- 固定源码提交与移植说明：`UPSTREAM.txt`；原项目 Apache-2.0 许可证：`LICENSE`。

开发端可见的是 Python 3.12 / CPU PyTorch，无法读取你截图里的 D:\LuSNAR_Seg 或验证截图设备。
本包用合成彩色标签和图像完成模型与训练流程检查，详见 `VALIDATION.txt`；这些结果不是 LuSNAR 训练指标。
