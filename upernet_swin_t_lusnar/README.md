# UPerNet-Swin-T：LuSNAR / Windows Conda 训练包

直接使用 Transformers 4.57.1 的 `UperNetForSemanticSegmentation` 和 Swin-T 主干。
使用完整四级 Swin-T（96 维起始通道、depths=2/2/6/2、window=7），UPerNet 主解码器和 FCN 辅助头。
5 类含辅助头参数量 59,830,212。训练损失为主头交叉熵 + **0.4 × 辅助头交叉熵**，两头均参与反向传播。

**默认 FP32、随机初始化、离线训练，不下载预训练权重。**
这是针对你提供的 Hugging Face 连接超时情况设置的明确默认值，不是下载失败后的自动降级。
随机初始化不等于已有预训练，也不承诺预训练模型的效果；需要 ImageNet 权重时，使用第 5 节的本地加载方式。

## 1. 放置与激活

解压到以下位置，文件保持在同一目录：

```text
D:\LuSNAR_Seg\upernet_swin_t_lusnar\train_upernet_swin_t.py
D:\LuSNAR_Seg\upernet_swin_t_lusnar\model_upernet_swin.py
```

在 CMD / Anaconda Prompt 中输入（若在 Python 的 `>>>` 中，先 `exit()`）：

```bat
conda activate "F:\Conda\envs\lusnar"
cd /d D:\LuSNAR_Seg\upernet_swin_t_lusnar
python -m pip install -r requirements.txt
python train_upernet_swin_t.py inspect
```

沿用已经安装的 CUDA PyTorch，不重装 torch、MMCV 或 MMSegmentation。
requirements 与前一个 SegFormer 包相同，固定 Transformers 4.57.1，适用于 Python 3.10。
脚本默认使用 GPU 0；inspect 会列出实际 GPU、Python 路径和 CUDA 可用情况。

## 2. 数据与训练

默认读取：

```text
D:\LuSNAR_Seg\data\Moon_*\image0\color\时间戳.png
D:\LuSNAR_Seg\data\Moon_*\image0\label\时间戳.png
```

按不含扩展名的相对文件名配对，支持 png/jpg/jpeg/bmp/tif/tiff。
灰度图转为三个通道，输入按 ImageNet 均值方差归一化一次；mask 使用最近邻缩放。
不读取 depth、LiDAR、位姿或 IMU。
`--camera image1` 切换另一相机，`--camera all` 使用两个相机且仍按 Moon 场景统一划分。

使用之前已检查通过的 RGB 标签定义，不做 ID 减一：

|ID|类别|RGB|
|---|---|---|
|0|月壤 lunar_regolith|187,70,156|
|1|撞击坑 impact_crater|120,0,200|
|2|岩石 rock|232,250,80|
|3|山体 mountain|173,69,31|
|4|天空 sky|34,201,248|

0 和天空都是有效训练类；内部忽略值是 255。
未知颜色、图像损坏、配对缺失或尺寸不一致会报告文件并停止。
`schema_rgb.json` 可通过 `--schema` 自定义；`schema_ids_EXAMPLE.json` 只演示单通道标签，里面的 ID 不是对实际数据的断言。

先检查数据：

```bat
python train_upernet_swin_t.py train --check-only
```

出现 `Data check passed` 后正式训练：

```bat
python train_upernet_swin_t.py train
```

也可在已激活环境的窗口运行 `run_train.bat`。
默认训练 100 轮、batch=4、512×512 随机裁剪、FP32、单 GPU、workers=0。
训练使用 0.5..2.0 缩放、随机裁剪和水平翻转；验证/测试按原始分辨率以 512 窗口、384 步长滑窗，平均重叠区域 logits。
Swin 预训练名称中的 224 表示分类预训练输入尺寸，不限制本脚本的 512 训练裁剪。

优化器 AdamW：lr=6e-5，默认主干、主头和辅助头使用相同 lr；`--head-lr-mult` 可调整两头倍率。
weight decay=.01，norm/bias、相对位置偏置不衰减；线性预热和多项式衰减（power=1），梯度范数上限 1。
默认不启用 FP16；`--amp` 为主动选项，含溢出跳过与持续异常停止保护。
batch 至少 2，因为 UPerNet 的全局池化分支使用 BatchNorm，batch=1 训练会出问题。

## 3. 与之前模型保持同一数据划分

默认取数字最大的 Moon 场景作验证、第二大的作测试，其余训练。
按你最近日志中实际出现的场景：训练 Moon_1/3/6，验证 Moon_8，测试 Moon_7，预计 4520/1351/1626 张；数量以实际读取为准。
若目录增加新场景，自动划分也会改变；可明确指定：

```bat
python train_upernet_swin_t.py train --val-scenes Moon_8 --test-scenes Moon_7
```

严格复用某次 SegFormer 或 SegNeXt 的清单：

```bat
python train_upernet_swin_t.py train --split-manifest "D:\LuSNAR_Seg\segformer_b3_lusnar\runs\segformer_b3\split_manifest.json"
```

把路径换成实际存在的清单文件。脚本会验证每张图像、配对、完整覆盖以及场景互斥。
至少需要两个场景；不随机拆分同一场景的相邻帧。
这是自定义场景划分，不声称为 LuSNAR 官方基准划分。
比较不同模型还需记录预训练来源和超参数差异，不能把不同初始化造成的差异全部归于模型结构。

## 4. 输出与恢复

默认输出：`D:\LuSNAR_Seg\upernet_swin_t_lusnar\runs\upernet_swin_t\`。

- `best.pth`：验证 mIoU 最高的模型，含权重、schema、模型配置、初始化来源。
- `last.pth`：最近完整一轮，含优化器、AMP 和随机状态，用于恢复。
- `history.csv`：每轮 loss、验证 mIoU、像素准确率及耗时。
- `best_metrics.json` / `test_metrics.json`：最佳验证指标 / 最终独立测试指标。
- `previews\*.png`：最佳模型的验证样例，左到右为原图、标签、预测，默认最多 3 张。
- `schema.json` / `split_manifest.json` / `pixel_counts.json`：标签、划分、像素数量。
- `model_config\config.json` / `initialization.json` / `environment.json`：模型结构、初始化和环境记录。

按完整验证集 mIoU 选择 best；仅完整训练结束后，用 best 对测试集评估。
IoU、mIoU 为 0..1，比率乘 100 为百分数；交并集为 0 的类别 IoU=null，不计入均值。
不要只用训练 loss 判断效果，应检查预测图和每类 IoU。

恢复训练：

```bat
python train_upernet_swin_t.py train --resume
```

恢复时保持原 epochs、batch-size、size、lr、head-lr-mult、seed、精度和数据划分参数。
若指定了 output 或 split-manifest，恢复时也要带上。
恢复不下载外部权重，从最近完成的一轮继续；未完成的一轮会重跑。
新实验用新 `--output` 目录，防止误覆盖。
默认覆盖更新 best/last，不为每一轮累积整份权重；单次训练输出建议预留 5 GB，不含数据、依赖缓存及其他实验。

若显存不足：

```bat
python train_upernet_swin_t.py train --batch-size 2 --output runs\upernet_swin_t_bs2
```

`--device cuda:1` 可选择另一块实际存在的 GPU；默认没有多 GPU 并行。
若前一个模型仍在训练，选择空闲 GPU 或等它完成，避免多任务争用同一块显存。

## 5. 可选 ImageNet 预训练

你贴出的 SegFormer 日志中，目标机器连接 huggingface.co 超时，因此本包的默认训练命令完全不访问模型站点。
需要预训练时，可在能联网的电脑上运行：

```bat
python download_weights.py --output pretrained\swin-tiny
```

把整个 `pretrained\swin-tiny` 文件夹复制到目标电脑，再运行：

```bat
python train_upernet_swin_t.py train --pretrained "D:\LuSNAR_Seg\upernet_swin_t_lusnar\pretrained\swin-tiny" --local-files-only --output runs\upernet_swin_t_pretrained
```

本地 Hugging Face 格式至少包含 `config.json` 和 `model.safetensors`（约 113 MB）。
直接下载来源是 Microsoft 的 `microsoft/swin-tiny-patch4-window7-224`，ImageNet-1k 分类权重。
不直接接受 OpenMMLab/NVIDIA 原生 `.pth`，不把它当 Hugging Face 文件夹使用。
若当前机器能连接 Hugging Face，也可在 `--pretrained` 中填写上述 repo ID；`--revision` 可固定提交。

加载会检查 Swin-T 架构及所有共享主干权重；分类最后的 LayerNorm 复用至 stage4，stage1/2/3 输出归一化层和两种分割头新建训练。
不使用 ADE20K 预训练的 UPerNet 解码器；加载失败会停止，不自动退回随机初始化。
加载来源和新建层会写入 initialization.json。

## 6. SegFormer 连接超时的处理

原 SegFormer 脚本和数据检查已工作，失败发生在下载 `nvidia/mit-b3/config.json`，属于网络/缓存问题。
如果明确选择随机初始化，在原来的 SegFormer 文件夹中执行：

```bat
cd /d D:\LuSNAR_Seg\segformer_b3_lusnar
python train_segformer_b3.py train --from-scratch --output runs\segformer_b3_scratch
```

若需要 ImageNet 预训练，请用原包的 download_weights.py 在能联网的电脑下载，再带本地目录和 `--local-files-only` 运行。
不需要为了这个连接超时重装 CUDA 或修改图像标签。

## 验证范围与来源

使用标准库内模型，不承诺百分之百正确或复现论文指标。完整 CPU 合成数据验证记录见 VALIDATION.txt；来源见 SOURCES.md。
开发端无法访问你的实际数据或 48 GB GPU；真实 CUDA 和 512 裁剪训练还需在目标环境确认。
