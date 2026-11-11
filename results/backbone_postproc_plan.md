# LuSNAR 强主干冒烟 + 可插拔 Transformer 主干 + 后处理：调研与执行方案

- 编写时间：2026-11-10 凌晨（北京时间）。本文件只是方案，**不启动任何 GPU 任务**，也不改动现有文件。
- 口径（沿用硬规则）：只在 Scene5 上验证；Moon_2/4/9 的标签绝不用于训练、调参或选型；不碰 `work_dirs\train_B1`；第一轮 20k（WSL，端口 29601）跑完之前不开任何新 GPU 任务；端口 29621 留给 plus5；不设 PYTORCH_CUDA_ALLOC_CONF。
- LuSNAR 原文 SegFormer 的 89.96 mIoU 用的是它自己的划分（1/2/4/6/8/9 训练，3/5/7 测试），**和本地 Scene5 的数字不能直接比**。
- 标注约定：【实测】= 本机日志或文件里读到的数；【文献】= 论文或官方 model zoo 的数，附链接；【估计】= 我推算的，需要冒烟实测确认。

---

## 0. 结论速览

| 候选 | 能不能接进现有方案（ROW/BLV/坑采样） | 环境代价 | 结论 |
|---|---|---|---|
| **SegFormer-B5（MiT-B5 + 原 MLP 头）** | **完全兼容**：输出仍是 1/4 分辨率的逐像素 logits，ROW、BLV、采样器、ACW、TTA 一行不用改 | 只需下载 MiT-B5 预训练（326 MB），把 `build_segformer_b2` 改成按配置名构建 | **可行，优先级最高**。冒烟用 batch 16 单卡没问题；20k 正式训练 batch 48/卡【估计】会 OOM，需要降到 24/卡或开梯度检查点 |
| **Mask2Former（Swin-B，IN-22k）** | **部分兼容**：它输出的是 query 掩码和类别，不是逐像素 logits。要接 ROW/BLV，需要先合成语义图，再加一个辅助的逐像素 CE（见第 2 节） | mmseg 自带的版本要装 mmdet，还要 **mmcv 完整版的 CUDA 算子**（MSDeformAttn）。现在环境是 mmcv-lite 2.1.0，没有 torch 2.7/cu128 的预编译轮子，WSL 里也没有 nvcc → **建议改走 HuggingFace transformers 的实现**（纯 PyTorch） | **可行，但改造和风险都更大**。2k iter 对它可能偏短，需要用 ADE20K 全模型权重初始化，并预注册"曲线还在涨就把双方一起延到 4k"的规则 |
| Swin-B/L + UperNet | 完全兼容（逐像素 logits） | mmseg 原生支持，mmcv_stub 已证明 Swin 不需要自定义算子 | 备选，第二梯队 |
| ConvNeXt-B/L + UperNet | 完全兼容 | 需要装 mmpretrain（配置里写的是 `mmpretrain.ConvNeXt`） | 备选（CNN，不是 Transformer），第三梯队 |
| BEiT / ViT-Adapter / DINOv2 | 用 UperNet 或线性头时兼容；ViT-Adapter 需要 MSDeformAttn 算子 | 显存大，ViT-Adapter 要编译算子 | 本轮不做 |
| InternImage | 兼容 | **必须编译 DCNv3 CUDA 算子**，本机 WSL 没有 nvcc | 本轮不做 |

后处理推荐前三（全部在第一轮权重上、Scene5 A/B 两半上评估，零训练）：
1. **P1 按类 logit 偏置（post-hoc logit adjustment）**：4 个自由参数，专门针对 mountain↔rock 和 crater↔regolith 的此消彼长。
2. **P2 几何/拓扑规则**：天空在上、山体在天空和月壤之间的竖直顺序，加上按类别的连通域清理（**rock 不做小块删除**）。
3. **P3 集成时的温度校准 + 概率平均**：如果第二个主干胜出或者和 B2 互补，就用这个融合。

不推荐：DenseCRF（对强模型证据有好有坏、慢、月面纹理颜色单一）、CascadePSP（重，类别无关，要按类跑 5 次）、再做一遍 SegFix 类边界修正（本地 SegFix-lite 实测只有 +0.0006）。

---

## 1. 现有代码怎么依赖模型（读代码所得）

- 主训练脚本 `tools/portable/lusnar_row_blv_ddp.py`，模型由 `_lusnar_common.build_segformer_b2()` 构建：读 mmseg 配置 `segformer_mit-b2_8xb2-160k_ade20k-512x512.py`，类别数改成 5，SyncBN 改成 BN，backbone 的 `init_cfg` 指向 `pretrained/mit_b2_20220624-66e8bf70.pth`（ImageNet-1k 预训练，**不是** ADE20K 全模型）。
- 前向：`_SegLogits.forward = core.extract_feat → core.decode_head.forward → row_attn(logits)`。logits 形状是 **B×5×(H/4)×(W/4)**，加上 ROW 以后再双线性上采样到 512 算损失。
- **ROW**（`RowHeightAttn`）：输入只有归一化的行坐标 y∈[0,1]，经过一个小 MLP（1→16→5，最后一层零初始化）得到每行每类一个加性偏置。**只依赖"有 B×C×H×W 的逐像素 logits"**，跟主干无关。
- **BLV**（`blv_perturb_logits`）：训练时给逐像素 logits 加高斯噪声，σ=0.5，按 1/√freq 放大稀有类，只进 CE 不进 dice。**同样只依赖逐像素 logits**。
- **坑采样器 / ACW / mosaic / CC paste**：都在数据集和损失层面（`DistributedWeightedSampler` 读 `full_crater_weights.json`；ACW 是 CE 的 class weight，crater 再乘 1.75），**跟主干完全无关**。
- **TTA**：`sliding_window_logits` 调用 `_forward_logits(net, …)`，也就是同一个 `extract_feat + decode_head` 接口。只要新模型提供"图像 → B×5×h×w 分数"的函数就能直接用。
- 冒烟协议（`lusnar_smoke_compare.py`，从 summary 实测读到）：2000 iter、单卡 batch 16、lr 6e-5、crop 512、训练子集 1600 张、Scene5 val 子集 350 张、seed 42。单尺度验证用中心裁剪 512，TTA 用整图滑窗。**噪声带**：文档定的是 |Δ|<0.02；同配方重跑 V_B1C_noise 实测差 +0.0052。
- 正式训练（实测）：batch 48/卡 × 2 卡，bf16，峰值显存 36.8 GB/卡（卡上可见 49140 MiB），约 2.23 s/iter（iter 5000→11050 用了 13510 s）。

**移植结论**
- SegFormer-B5：把 `build_segformer_b2` 泛化成 `build_segformer(variant)`，配置换成 `segformer_mit-b5_8xb2-160k_ade20k-512x512.py`，预训练换成 `mit_b5_20220624-658746d9.pth`。ROW/BLV/采样器/TTA **零改动**。
- Mask2Former：mmseg 的 `Mask2FormerHead.predict` 输出的是 `einsum('bqc,bqhw->bchw', softmax(cls)[..., :-1], sigmoid(mask))`，即**每类概率之和（≥0），不是 logits**。所以：
  - **ROW**：在 `S = log(语义概率 + 1e-6)` 上加 ROW 偏置，得到"伪 logits" `Z = S + ROW(y)`；训练时对 Z 加一个辅助逐像素 CE（带 ACW 权重），推理和 TTA 时直接对 Z 取 argmax 或平均。这样 ROW 的作用（每行每类的先验偏置）和 B2 上一样。
  - **BLV**：预注册的做法是加在 Z 上，只进辅助 CE（和 B2 机制一致）。另一种更贴近 BLV 原文的做法是加在 query 的类别 logits（B×Q×(C+1)，只加前 C 类）上，只进分类损失、不进匈牙利匹配。这个做变体，**本轮冒烟不跑**。
  - **ACW**：Mask2Former 分类损失本来就有 `class_weight=[1]*C+[0.1]`，前 5 项换成我们的 ACW 权重；辅助 CE 也用 ACW。
  - **dice**：Mask2Former 自带 mask dice（点采样），不再加我们的逐像素 dice。
  - 风险：辅助 CE 和 Mask2Former 的集合损失之间权重要定（预注册 1.0，不调）。先例方面，据我所知 MaX-DeepLab / kMaX-DeepLab 这类 mask-transformer 都用过辅助语义损失，**本次没有逐条核实**。

---

## 2. 本机环境现状（只读检查）

| 项目 | WSL `/opt/miniconda3/envs/lusnar` | Windows `F:\Conda\envs\lusnar` |
|---|---|---|
| torch | 2.7.1+cu128 | 2.7.1+cu128 |
| mmseg / mmengine / mmcv | 1.2.2 / 0.10.7 / **mmcv-lite 2.1.0**（靠 `mmcv_stub.py` 假装有 `_ext`，调用任何 CUDA 算子都会报错） | 同 |
| mmdet | **未装** | 未装 |
| mmpretrain | 未装 | 未查到 |
| transformers / timm | **都未装** | transformers 4.49.0，timm 1.0.15 |
| pydensecrf | 未装 | 未装 |
| nvcc / CUDA toolkit | **没有**（`nvcc: command not found`，`/usr/local` 下没有 cuda） | 未查 |
| 预训练缓存 | `/root/.cache/torch`、`/root/.cache/huggingface` 都不存在 | `D:\LuSNAR_Seg\pretrained` 下只有 `mit_b2_20220624-66e8bf70.pth` |

mmseg 1.2.2 自带的配置都在（`.mim/configs/`）：`segformer_mit-b5_8xb2-160k_ade20k-512x512.py`、`mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py`、`mask2former_swin-l-…`、`swin-*-upernet`、`convnext-*-upernet`。

另有早期另一套流水线的产物（`segformer_b3_lusnar`、`upernet_swin_t_lusnar`、`segnext_s_lusnar`，HF 实现、随机初始化、按另一种场景划分）。它们的分数（0.85/0.81/0.91）**划分和初始化都不同，不能拿来做主干比较**。

### 要装和下载的东西（等 GPU 训练结束、经确认后再做）

**安全做法**：不要往正在训练的 env 里装包。先 `conda create -n lusnar_bb --clone lusnar`，所有新包都装在克隆的环境里。

1. SegFormer-B5（无新包）
   - ImageNet 预训练：`https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth`（已核实 HTTP 200，326,083,151 字节）
   - 可选的 ADE20K 全模型（只用于"初始化公平性"对照）：`https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_512x512_160k_ade20k/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth`（已核实 200，328 MB）
2. Mask2Former（**推荐路线 A：HuggingFace**）
   - `pip install transformers==4.49.0`（和 Windows env 版本一致），装在克隆 env 里，不动 torch。
   - 权重：`facebook/mask2former-swin-base-IN21k-ade-semantic`（https://huggingface.co/facebook/mask2former-swin-base-IN21k-ade-semantic ，页面已核实）；备选 `facebook/mask2former-swin-base-ade-semantic`。
   - HF 的 deformable attention 是纯 PyTorch（grid_sample）实现，不需要编译算子。代价是比 CUDA 算子慢、显存更高【估计】。
   - WSL 现在有"localhost 代理未镜像"的警告，下载可能要在 Windows 侧下好再拷进去。
3. Mask2Former（路线 B：mmseg 原生，**不推荐**）
   - 需要 `mmdet 3.x`，还需要 mmcv 完整版的 `MultiScaleDeformableAttention` CUDA 实现。OpenMMLab 的预编译索引最高到 torch2.4 + mmcv 2.2.0（已核实 `cu128/torch2.7` 索引返回 404），而 mmseg 1.2.2 要求 mmcv<2.2.0 → 只能源码编译 mmcv 2.1.0，又要先装 CUDA 12.8 toolkit（WSL 里没有）。
   - 退一步的办法：mmcv-lite + stub，把 `MultiScaleDeformableAttention.forward` 打补丁改成走 `multi_scale_deformable_attn_pytorch`（mmcv 源码 366 行判断 `IS_CUDA_AVAILABLE and value.is_cuda` 才走 CUDA 算子）。可以用，但侵入性大。
   - 权重：`https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235230-7ec0f569.pth`（已核实 200，519 MB）
4. 后处理：P1/P2/P3 只需要 numpy/cv2/scipy（已有）。不装 pydensecrf。

---

## 3. 显存与速度估计（2×RTX 4090 48 GB 版，crop 512，bf16）

基准（【实测】）：B2 冒烟单卡 batch 16，峰值 12.1 GB；正式训练 batch 48/卡，峰值 36.8 GB，2.23 s/iter。

| 模型 | 文献/官方数据 | 冒烟（单卡 b16）显存 | 正式 20k（b48/卡）| 每 iter 相对 B2 |
|---|---|---|---|---|
| SegFormer-B2 | mmseg metafile：ADE 512 训练显存 3.6 GB @2 张/卡；ADE 46.5/47.5 mIoU（SS/MS），27.5M 参数，62.4 GFLOPs @512（SegFormer 论文表 1） | 12.1 GB【实测】 | 36.8 GB【实测】 | 1.0× |
| **SegFormer-B5** | metafile：ADE 512 训练显存 **7.2 GB** @2 张/卡；ADE 51.0/51.8（640 crop），Cityscapes 82.4/84.0；84.7M 参数；183.3 GFLOPs @640 | **约 24 GB**【估计，按 7.2/3.6=2× 推】 | **约 70 GB+，会 OOM**【估计】→ 需要 b24/卡，或开激活检查点保持 b48 | 约 1.9×【估计，按 FLOPs 换算到 512】 |
| **Mask2Former Swin-B (22k)** | metafile：ADE 640 训练显存 5.66 GB @2 张/卡；ADE 53.9（SS）/55.1（MS），107M 参数，223 GFLOPs @640（Mask2Former 论文表 V）；Cityscapes 83.3/84.5 | **约 25–30 GB**【估计，按面积 0.64× 换算，HF 纯 PyTorch 实现可能更高】 | b48/卡基本肯定 OOM【估计】→ b16–24/卡 | 约 2.3–3×【估计】 |
| Mask2Former Swin-L (22k) | metafile 8.86 GB @2 张/卡 640；ADE 56.1/57.3；215M 参数 | 约 40 GB+【估计，太紧】 | 不适合 | — |

冒烟耗时【估计】：B2 冒烟历史墙钟 2.5–3.4 h（有数据加载瓶颈）→ B5 约 4–6 h，M2F Swin-B 约 5–8 h。

---

## 4. 可插拔 Transformer 主干调研（保留 ROW/BLV/坑采样）

只要"输出逐像素 logits"，ROW/BLV 就能原样保留。所以优先选**逐像素头**（SegFormer MLP 头、UperNet），Mask2Former 属于"要改造"的一类。

| 主干 + 头 | ADE20K mIoU（SS/MS，官方数） | 参数 / 代价 | 接入难度 | 证据链接 |
|---|---|---|---|---|
| MiT-B5 + MLP | 51.0 / 51.8（640） | 84.7M | 改一个配置名 | https://arxiv.org/abs/2105.15203 |
| MiT-B4 + MLP | 50.3 / 51.1（512） | 64.1M | 同上，B5 太慢时的退路 | 同上 |
| Swin-B(22k) + UperNet | 50.13 / 51.9（512，mmseg metafile） | 约 121M | mmseg 原生，不需要算子 | https://arxiv.org/abs/2103.14030 |
| ConvNeXt-B + UperNet | 48.71 / 49.54（512，mmseg）；52.13 / 52.66（640） | 显存 6.33 GB @2 张/卡 512 | 要装 mmpretrain | https://arxiv.org/abs/2201.03545 |
| Mask2Former Swin-B(22k) | 53.9 / 55.1 | 107M，223G | 要改造（第 1 节），并处理依赖 | https://arxiv.org/abs/2112.01527 |
| Mask2Former Swin-L(22k) | 56.1 / 57.3 | 215M | 显存紧 | 同上 |
| BEiT-L + UperNet | 56.33 / 56.84（640，mmseg） | 22.64 GB @1 张/卡 | 太大 | https://arxiv.org/abs/2106.08254 |
| ViT-Adapter-L（+Mask2Former） | 60.5（MS，论文摘要） | 571M（官方表） | 需要 MSDeformAttn 算子 | https://arxiv.org/abs/2205.08534 |
| DINOv2 ViT-g（冻结）+ ViT-Adapter + Mask2Former | 60.2（MS） | 1B+ | 太大；patch 14 对小目标不友好 | https://arxiv.org/abs/2304.07193 |
| InternImage-B + UperNet / -H + Mask2Former | 50.8 / 62.9 | 要编译 DCNv3 | WSL 里没有 nvcc | https://arxiv.org/abs/2211.05778 |

**行星/月面相关证据（都不是同一划分，只作定性参考）**
- LuSNAR 原文（https://arxiv.org/abs/2407.06512 ，TGRS 2026）：5 种方法里 SegFormer 最好（89.96），没测 Mask2Former。
- 火星 DepthMars（DepthFormer，https://ira.lib.polyu.edu.hk/bitstream/10397/115601/1/Ma_DepthFormer_Depth_Enhanced.pdf ）：Mask2Former 在 Rock（57.21 vs 42.96）和 Craters（34.75 vs 3.86）上明显强于 SegFormer，mIoU 75.84 vs 66.99。**这是支持试 Mask2Former 的最直接证据**（小类、稀有类）。
- AI4Mars / LabelMars（IEEE，https://ieeexplore.ieee.org/document/11436256 ）：Transformer 在大数据集上明显好于 DeepLabV3+，稀有的 Big Rock 提升最大；SegFormer 吞吐最高。
- 斯坦福 CS231n 报告在 LuSNAR 上 SegFormer 0.9686、MaskFormer 0.8955（**自划分，不可比**）：https://cs231n.stanford.edu/2025/papers/CS231N_Final_Project_Report%20-%20Brian%20Y%20Wu.pdf

**判断**：B5 是"稳妥加容量"，跟现有改进完全兼容，风险最低；Mask2Former 是"换范式"，在稀有类和小类上有行星数据的正面证据，但改造工作量大、2k 冒烟可能对它不公平。两个都做，正好对应用户的要求。

---

## 5. 后处理调研

错误画像（【实测】，第一轮 iter 11000 单尺度，Scene5 全集）：月壤 0.984、岩石 0.817、坑 0.868、**山体 0.693**、天空 0.932。冒烟 TTA 后：sky→mountain 只剩 0.5%，但 **mountain→rock 约 9%**（P(预测为 rock | 真值 mountain)），crater→regolith 7–12%。**山体是最大短板**，后处理应该优先针对 mountain↔rock 和 crater↔regolith。最终选什么，还要看诊断阶段（错误归因 + 混淆矩阵）的结果。

| 编号 | 方法 | 证据 | 预期收益 | 风险 | 代价 |
|---|---|---|---|---|---|
| **P1** | **按类 logit 偏置**：pred = argmax(z + b)，b∈R^5，固定 b_regolith=0，在 A 半上做坐标搜索（每维 −1.0…+1.0，步长 0.05），在 B 半上报一次 | Menon et al., Long-tail learning via logit adjustment（https://arxiv.org/abs/2007.07314 ）：事后按类先验平移 logits 能纠正频率偏差 | +0.2–1.0 点【估计】，主要来自 mountain/rock 和 crater/regolith 的分界 | 过拟合 A 半 → 只有 4 个参数，再用配对检验把关 | 几乎为零（每张图一次加法） |
| **P2** | **几何/拓扑规则**：(a) 逐列竖直顺序：天空连通域如果在山体或月壤下方，就按邻域多数重标；(b) "嵌在山体里"的 rock 连通域（≥T% 的边界和 mountain 相邻，且质心在本列地平线以上）改成 mountain；(c) 按类别的小连通域清理，**只对 mountain/sky/crater 碎块**，rock 一律不删 | HANet 证明行位置先验有效（https://arxiv.org/abs/2003.05128 ）；连通域清理是工程惯例，没有强论文证据 | (b) 直接对着 mountain→rock 的 9%，【估计】mountain +1–3 点，mIoU +0.2–0.6 | 误伤山上真实存在的岩石；规则阈值 T、面积要在 A 半上定 | CPU，每张约 10–50 ms【估计】 |
| **P3** | **集成：温度校准 + 概率平均**：每个模型在 A 半上拟合温度 T（最小化 NLL），softmax(z/T) 加权平均，权重网格 {0.3,…,0.7} | Guo et al., On Calibration of Modern Neural Networks（https://arxiv.org/abs/1706.04599 ） | 两个结构不同的模型，【估计】+0.5–1 点 | 推理时间翻倍 | 推理成本 ×模型数 |
| P4 | DenseCRF / ConvCRF | Krähenbühl & Koltun 2011（https://arxiv.org/abs/1210.5644 ）；ConvCRF（https://arxiv.org/abs/1805.04777 ）。在 DeepLab 时代的弱 CNN 上有效，在强 Transformer 上证据有好有坏 | ≈0 或为负【估计】 | 月面颜色单一，外观核区分力弱；还会抹掉小 rock | 每张约 1 s（CPU，1024²）【估计】 | 
| P5 | SegFix（学习式边界偏移） | https://arxiv.org/abs/2007.04269 ：Cityscapes 上约 +0.3–0.6 | 本地 **SegFix-lite 实测 +0.0006**（V_B1C） | 需要另训一个偏移网络 | 中 |
| P6 | CascadePSP | https://arxiv.org/abs/2005.02551 ：ADE20K 上 +0.7–2.1，类别无关 | 不确定 | 用自然图像预训练，要按类跑 5 次 | 高 |
| P7 | 形态学开闭运算 | 惯例 | 被 P2 覆盖 | 同 P2 | 低 |

**推荐顺序**：P1 → P2 → P1+P2 组合 →（有第二个模型时）P3。P4–P7 不进预注册名单。

---

## 6. 预注册的冒烟协议（强主干）

**固定不变**：训练子集 1600 张，Scene5 val 子集 350 张（与历史冒烟同一份 seed 42 抽样），2000 iter，crop 512，bf16，单卡 batch 16，V_B1C + ROW + BLV 全套（坑采样 / ACW / mosaic / CC paste），同一个 seed 42。单尺度中心裁剪验证和 TTA（0.75/1.0/1.25，滑窗 512/256）都报。

| 编号 | 内容 | 初始化 | 优化器（预注册，不调） |
|---|---|---|---|
| S0 | B2 对照**同一套新代码重跑**（检查新代码路径没有引入偏差；和历史 smoke_M_ROW_BLV 0.8219 / TTA 0.8522 的差应该 <0.0052 噪声） | MiT-B2 IN-1k | AdamW lr 6e-5, wd 0.01（同现在） |
| S1 | **SegFormer-B5** + ROW + BLV | MiT-B5 IN-1k（和 B2 的初始化同类，公平） | 同 S0 |
| S2 | **Mask2Former Swin-B** + 辅助语义 CE + ROW + BLV（第 1 节的移植） | HF ADE20K 全模型（类别头重新初始化成 5+1）。注：不用 ADE 全模型的话，2k iter 内匈牙利匹配基本收敛不了 | 官方：AdamW lr 1e-4, wd 0.05, 主干 lr×0.1, clip 0.01（用 B2 的 lr 会对它不公平，属于已声明的偏差） |
| S1′（可选） | B5 用 ADE20K 全模型初始化 | 用来区分"S2 赢在结构还是赢在 ADE 预训练" | 同 S1 |

**训练前必须先过的检查**（借鉴 plus5 前的预检）：CPU 上小尺寸前向/反向；GPU 上 5 iter dry-run，检查 `ROW_BLV_EFFECT` 自检（ROW 梯度 >0、BLV 的 CE 差 >1e-4）；DDP 不用（单卡）；记录显存峰值。

**判定指标**：主指标是 TTA mIoU（因为最终提交用 TTA）；副指标是 rock/crater/mountain 的 IoU、sky→mtn、crater→reg、mountain→rock，以及墙钟和推理 ms/图。

**统计检验**：350 张逐图 IoU 配对（每类交并按图累加），paired bootstrap 10,000 次给 ΔmIoU 的 95% CI，加 Wilcoxon 符号秩检验；多个候选做 Holm 校正。

**决策规则（预注册）**
1. **换主干的条件**：S1 或 S2 的 TTA ΔmIoU 相对 S0 **≥ +0.02**（超出文档噪声带），bootstrap CI 不含 0，并且 rock 和 crater 各自下降不超过 0.01。
2. **+0.005 ≤ Δ < +0.02**：不直接换，转入"集成候选"，用第 7 节的 P3 规则判断。
3. **Δ < +0.005**：放弃这个主干。
4. **S2 收敛修正**：如果 S2 在 1500→2000 iter 之间单尺度 mIoU 还涨了 ≥0.01（说明没收敛），**S0 和 S2 一起延长到 4000 iter** 再比（只允许延长这一次）。
5. 两个都达标时选 TTA mIoU 高的；差距 <0.005 时选推理更快的。

---

## 7. 集成的决策规则

- 用**第一轮**权重（没见过 Scene5）：B2 第一轮 + 胜出主干的第一轮 20k（如果做了）。冒烟阶段只能用 2k 权重做"互补性"预判：两个模型逐像素分歧率，以及 oracle 上限（任一模型对就算对）。
- 在 Scene5 A 半上拟合各模型的温度 T 和融合权重 w，在 B 半上报一次。
- **接受条件**：B 半上 ΔmIoU（集成 vs 最好的单模型，两边都带 TTA）≥ +0.005，bootstrap CI 不含 0，rock/crater 不降超过 0.01，推理耗时在比赛限制内（规则还没拿到，需要用户提供）。
- plus5 版本的集成照搬 A 半上定死的 T 和 w，不再调。

---

## 8. 后处理的评估协议（第一轮权重，Scene5 A/B）

1. 在推理阶段（I1–I3）**顺便**保存第一轮权重在 Scene5 全部 1599 张图上的 TTA 概率（float16 存到 WSL ext4；每张 5×H×W，总量约十几 GB【估计】）。之后所有后处理都在 CPU 上读这些概率做，**不占 GPU**。
2. A/B 划分和推理阶段用同一份（按图像 ID 固定种子对半分，预注册、只划一次）。
3. 候选**最多 5 个**：P1、P2(a)、P2(b)、P2(c)、P1+P2。参数只在 A 上搜，B 上每个候选只报一次。
4. 接受条件：B 上 ΔmIoU ≥ +0.003（后处理是零训练、几乎零成本，门槛比训练类低），bootstrap CI 不含 0（Holm 校正后），rock IoU 不降超过 0.005（防止误删小岩石）。
5. 选中的后处理定死参数，原样用到 plus5 和最终提交上。

---

## 9. 执行顺序和时间（放在现有流水线后面，不拖 plus5）

| 阶段 | 内容 | 用 GPU？ | 预计时间（北京时间）【估计】 |
|---|---|---|---|
| 0 | 第一轮 20k 收尾 + TTA + 第一次开源测试 | 是（在跑） | 今天约 07:00–08:00 结束 |
| 1 | 诊断 + 推理三案 I1–I3（第一轮权重，Scene5 A/B），**同时保存 TTA 概率** | 是 | 半天到 1 天 → 约 11-10 晚 到 11-11 上午 |
| 1′ | **后处理 P1/P2 评估**（读保存的概率，纯 CPU，和阶段 1 后半段并行） | 否 | +2–4 h，和阶段 1 一起结束 |
| 2 | Windows 预检 + 冒烟 → **plus5**（workers 4→6→8） | 是（两卡 DDP） | 12–16 h → 约 11-12 上午到中午 |
| 2′ | plus5 跑的同时做**主干准备**（不占 GPU）：克隆 env、装 transformers 4.49、下载权重、写 `build_segformer(variant)` 和 M2F 适配器、CPU 前向/反向单测 | 否（少量 CPU 和网络） | 和阶段 2 并行 |
| 3 | 主干冒烟 S0 / S1 / S2（+可选 S1′），按第 6 节判定 | 是 | 串行约 12–17 h；如果允许两张卡各跑一个单卡冒烟，约 6–8 h → 约 11-12 晚 到 11-13 上午 |
| 4 | 按原计划做微调（T5/T1 → T2 …），**在 B2 主线上** | 是 | 约 1 天 |
| 5 | 如果第 3 步有主干胜出：它的第一轮 20k（Scene5 验证；B5 batch 要降到 24/卡或开检查点，需要用户确认）→ 集成判定（第 7 节）→ plus5 版 | 是 | B5 20k 约 22–28 h【估计】 |
| 6 | 原计划的长训、多种子、官方划分对照 | 是 | 往后顺延 |

**为什么这样排**：plus5 是比赛用得上的成果，不往后推；主干冒烟要等 GPU 空出来，但准备工作可以在 plus5 训练时做完，GPU 一空就能开跑，冒烟只多占约半天。后处理完全不占 GPU，所以塞进推理阶段。

---

## 10. 需要用户拍板的事

1. 阶段 3 里"两张卡各跑一个单卡冒烟"算不算违反"不开第二个并发训练"？默认按串行算。
2. 主干胜出后的 20k，batch 是否允许从 48/卡降到 24/卡（或者开激活检查点保持 48，代价是约 +30% 时间【估计】）。
3. 克隆 env `lusnar_bb` 并装 transformers 4.49.0（不动现在的 `lusnar` env）。
4. 比赛规则：推理时间有没有限制、能不能集成、能不能用测试图做自训练。这决定 P3 和集成值不值得做。

## 11. 风险清单

- Mask2Former + ROW/BLV 的移植属于"自研改造"，冒烟结果不一定代表 Mask2Former 本身的上限 → 冒烟时同时记录辅助 CE 和集合损失两条曲线。
- HF 纯 PyTorch 的 deformable attention 速度和显存未知 → 先做 5 iter dry-run 量一下。
- 2k iter 对大模型偏短，可能低估 B5 和 M2F → 第 6 节第 4 条的延长规则只给一次机会，避免反复挑。
- WSL 的 localhost 代理警告可能影响下载 → 在 Windows 侧下载再拷。
- 所有选择都只在 Scene5 上做；Moon_2/4/9 只在最终一次性开源测试时用，不参与任何选择。
