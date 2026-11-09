# SegFormer-B3：LuSNAR / Windows / Conda 训练包

根据已确认的设备、环境和数据准备：

- 环境：`F:\Conda\envs\lusnar`，Python 3.10.22。
- 项目和数据：`D:\LuSNAR_Seg` / `D:\LuSNAR_Seg\data`。
- 默认使用第一张 CUDA 可见显卡 `cuda:0`，启动时打印实际设备。
- 数据：`Moon_*\image0\color` 图像与同级 `label` 标签，按时间戳文件名配对。
- 模型：直接调用 Transformers 的标准 SegFormer-B3，MiT-B3 主干 + 768 维 MLP 解码器，5 类时 47,226,309 个参数。
- 修复版默认初始化：**离线随机初始化**，不访问 Hugging Face；全模型参与训练。可通过 `--pretrained` 显式加载 NVIDIA MiT-B3 ImageNet 主干。
- 默认精度：**FP32**。不默认启用此前发生溢出的 FP16 AMP。
- 使用 `image0` 按第三节的固定场景划分训练、验证和测试，训练时启用 standard 在线增强。
- 默认训练 **100 个 epoch**，输出目录为 `runs\segformer_b3_gpu0_scene_split_aug`。

## 1. 解压与激活

把整个 `segformer_b3_lusnar` 文件夹解压到 `D:\LuSNAR_Seg`。文件应位于：

```text
D:\LuSNAR_Seg\segformer_b3_lusnar\train_segformer_b3.py
D:\LuSNAR_Seg\segformer_b3_lusnar\model_segformer.py
```

在 CMD / Anaconda Prompt 中逐条执行（如果仍在 Python 的 `>>>` 内，先输入 `exit()`）：

```bat
conda activate "F:\Conda\envs\lusnar"
cd /d D:\LuSNAR_Seg\segformer_b3_lusnar
python -m pip install -r requirements.txt
python train_segformer_b3.py inspect
```

复用已有 CUDA PyTorch，不需要重装 torch、MMCV 或 MMSegmentation。
你的环境已配置好时，跳过上面的 pip 安装步骤。此次增强只使用现有 NumPy 和 Pillow，不需要新依赖。
requirements 固定 Transformers 4.57.1，支持 Python 3.10；安装会按其依赖约束更新相关 Python 包。
检查输出中的 Python 路径、版本、GPU 名称和 `CUDA available: True`。
可选 NVIDIA 模型包含 `.bin` 权重，加载时需使用 PyTorch 2.6 或更高；之前安装的 2.10 CUDA 版本满足要求。
若检查显示 CUDA 不可用，应先解决现有 CUDA PyTorch 安装问题再训练。

## 2. 数据检查和正式训练

数据结构：

```text
D:\LuSNAR_Seg\data\
  Moon_1\
    image0\
      color\1693274724086398208.png
      label\1693274724086398208.png
    image1\...
  Moon_3\...
  Moon_6\...
  Moon_7\...
  Moon_8\...
```

默认使用 image0；不读取深度、LiDAR、位姿或 IMU。
`--camera image1` 切换另一相机，`--camera all` 使用两个相机，仍按 Moon 场景分组。
图片为灰度时转为三通道，输入按 ImageNet mean/std 归一化一次，不再叠加 ImageProcessor 预处理。
标签按最近邻缩放；五类 RGB 颜色与之前检查通过的数据保持一致：

|ID|类别|RGB|
|---|---|---|
|0|月壤 lunar_regolith|187, 70, 156|
|1|撞击坑 impact_crater|120, 0, 200|
|2|岩石 rock|232, 250, 80|
|3|山体 mountain|173, 69, 31|
|4|天空 sky|34, 201, 248|

不对 ID 做减一处理，月壤 0 是有效类；255 才是内部忽略值。天空也参与训练。
未知颜色、尺寸不一致或图像/标签缺失都会报出具体文件；自定义标签请使用 `--schema`。

先运行数据检查，不下载权重、不训练：

```bat
python train_segformer_b3.py train --check-only
```

出现 `Data check passed` 后运行：

```bat
python train_segformer_b3.py train
```

**已修复你贴出的 `huggingface.co` 连接超时：现在不带参数训练时，直接离线随机初始化，不请求 config.json 或外部权重。**
启动会打印 `Initialization: RANDOM / OFFLINE. No ImageNet pretrained weights will be loaded.`，并将随机初始化写入 initialization.json。
这表示从头训练，不等于已加载预训练。模型结构与 FP32 设置保持原样，数据范围和增强采用上述新默认设置。
若明确指定 `--pretrained`，则仍要求权重完整，加载失败会停止，不自动改变初始化方式。
本次数据范围与增强已改变，请开始新实验，不要直接恢复此前的划分训练断点。可直接使用默认新目录：

```bat
python train_segformer_b3.py train
```

本次输出恢复时也使用同一目录：

```bat
python train_segformer_b3.py train --resume
```

命令 `run_train.bat` 也可在激活环境后运行，额外参数会传给训练脚本。

默认：100 轮、batch=4、512×512 随机裁剪、单 GPU、workers=0。
训练默认使用 standard 增强，详细设置见后文；如启用场景划分，验证/测试不做随机增强，保留原分辨率，使用 512 窗口和 384 步长的滑窗预测。
损失为五类交叉熵；AdamW 主干 lr=6e-5，解码器 lr 为主干的 10 倍，weight decay=.01，线性预热和线性多项式衰减，梯度范数上限 1。

## 3. 场景划分

默认划分如下。

| 用途 | 场景 |
|---|---|
| 训练 | Moon_1、Moon_3、Moon_5、Moon_6、Moon_7、Moon_8 |
| 验证 | Moon_2 |
| 测试 | Moon_4、Moon_9 |

请把九个场景放在同一个数据根目录下。脚本默认读取 `D:\LuSNAR_Seg\data`，也可通过 `--data-root` 指定。缺少指定场景的配对数据时，脚本会列出缺失场景并停止。

其他实验可用 `--train-all` 选择全量训练，或显式指定验证和测试场景并使用独立输出目录：

```bat
python train_segformer_b3.py train --scene-split --val-scenes Moon_8 --test-scenes Moon_7 --output runs\segformer_b3_scene_split
```

为了严格复用此前某一次 SegNeXt 运行的文件列表，可使用那次运行的实际清单路径：

```bat
python train_segformer_b3.py train --scene-split --split-manifest "D:\LuSNAR_Seg\segnext_s_lusnar\runs\segnext_s\split_manifest.json" --output runs\segformer_b3_scene_split
```

如果之前运行保存在 `segnext_s_fp32` 等目录，请把路径改成实际目录。
清单复用会核验文件、配对、场景一致性及全集覆盖，拒绝重复图片或场景跨集合混用。
清单是自定义场景划分，不宣称是 LuSNAR 官方基准划分。

**比较模型时要记录初始化差异：本修复版默认随机初始化；主动加载 ImageNet 权重时，应记录该条件，避免把不同初始化的影响全部归于模型结构。**
`--from-scratch` 仍可用于显式表示从头训练：

```bat
python train_segformer_b3.py train --from-scratch --output runs\segformer_b3_scratch
```

## 4. 可选预训练权重

可以先在能联网的电脑安装依赖后执行：

```bat
python download_weights.py --output pretrained\mit-b3
```

把整个 `pretrained\mit-b3` 文件夹复制到目标电脑。本地文件夹至少包含 NVIDIA 模型的 `config.json` 与 `pytorch_model.bin`（或兼容 safetensors 权重）。
随后执行：

```bat
python train_segformer_b3.py train --pretrained "D:\LuSNAR_Seg\segformer_b3_lusnar\pretrained\mit-b3" --local-files-only
```

本地权重必须是 Hugging Face 格式的完整 MiT-B3；原版 NVIDIA / MMSegmentation 的 `mit_b3.pth` 不能直接当作这里的模型文件夹。
架构或主干权重不完整会明确拒绝。`--revision <commit>` 可固定远端版本，初始化来源和解析出的提交会写入结果目录。
只有明确写出 repo ID 才会在线下载，例如 `--pretrained nvidia/mit-b3`；默认训练不触发这个过程。
`--pretrained` 与 `--from-scratch` 互斥，不能同时指定。

## 5. 输出、恢复、预测对照

默认保存到项目目录下的 `runs\segformer_b3_gpu0_scene_split_aug`。

- `best.pth` 保存 Moon_2 验证 mIoU 最高的模型、类别和初始化信息。
- `last.pth`：最近完成一轮的权重、优化器、随机状态及训练进度，用于恢复。
- `history.csv` 记录每轮训练损失、验证指标和耗时。
- `best_metrics.json` 记录最佳验证指标。训练完成后，脚本用最佳权重评估合并的 Moon_4 和 Moon_9，并将指标写入 `test_metrics.json`。
- `previews\*.png` 保存最多 3 张验证样本对照图，左至右为原图、真实标签和预测。可用 `--preview-count 0` 关闭。
- `schema.json`、`split_manifest.json`、`pixel_counts.json`：标签、实际划分和像素数。
- `model_config\config.json`、`initialization.json`、`environment.json`：结构、初始化来源和实际环境版本。
- `augmentation.json`：增强模式、强度、概率和模块版本；增强参数也纳入恢复一致性校验。

mIoU 和 IoU 均为 0..1，比率乘 100 为百分数。某类别交并集分母为 0 时 IoU=null，不计入 mIoU；不会因为该类未出现就计为满分。
显式选择 `--train-all` 时，脚本按训练损失保存 `best_train_loss.pth`，对照图写入 `training_previews`。
预测对照图用于检查空间对齐和类别，不能仅凭训练 loss 下降认定分割有效。

恢复（与首次训练保持相同 epochs、batch-size、size、lr、seed、精度、数据及划分参数）：

```bat
python train_segformer_b3.py train --resume
```

恢复从本地 `last.pth` 读取，不会重新下载预训练权重。若此前指定了输出目录或数据清单，恢复时也带上它们。
只能恢复已完成的一轮；未完成的一轮会重跑。新实验换 `--output`，禁止无意覆盖已有 last.pth 的实验。

若显存不足，可减小 batch（FP32 比 FP16 占用更多显存）：

```bat
python train_segformer_b3.py train --batch-size 2 --output runs\segformer_b3_bs2
```

设置了 `CUDA_VISIBLE_DEVICES` 时，显卡编号以 CUDA 可见设备顺序为准。
默认不启用多 GPU；Windows 默认 workers=0，确认稳定后可尝试 `--workers 4`。
`--amp` 可主动启用 FP16，已包含异常梯度跳过和连续溢出停止保护；本任务推荐先保持 FP32。

## 6. 在线数据增强

本包的 `augmentations.py` 和示例配置与修改后的 SegNeXt-S 相同。

默认 `--augmentation standard`：随机缩放 0.5–2.0 倍和裁剪、50% 水平翻转、30% 概率小角度旋转（±5°）、50% 概率亮度/对比度/Gamma 调整、15% 轻微高斯噪声、10% 轻微模糊。原图和标签共享几何变换；标签使用最近邻插值、空白填 255，光度变化只用于原图。不默认做上下翻转或大角度旋转。

增强读取时在线生成，不修改原始文件，不增加清单中的样本数量，也不会把增强重复用于预测对照图。

保留基础缩放、裁剪、水平翻转：

```cmd
python train_segformer_b3.py train --augmentation basic --output runs\segformer_b3_basic
```

关闭随机增强（使用固定等比例缩放和居中填充）：

```cmd
python train_segformer_b3.py train --augmentation none --output runs\segformer_b3_noaug
```

修改随附示例 JSON 后，可指定自定义参数：

```cmd
python train_segformer_b3.py train --augmentation-config augmentation_config_example.json
```

恢复时保持相同数据范围、增强设置、精度和训练参数；改变配置后使用新输出目录，不要覆盖或直接恢复旧断点。Python、NumPy、CPU Torch 和所选 GPU 的随机状态随新断点保存，多进程 worker 下不承诺逐位一致。

## 验证范围

本包使用库内模型，不承诺百分之百正确或复现论文指标。
可运行 `python test_gpu1_all.py` 和 `python test_augmentations.py` 检查场景划分、训练流程和增强规则。真实 GPU 训练请在目标环境中验证。
来源和模型链接见 `SOURCES.md`。本包不附带预训练权重。
