# 环境统一管理（Ubuntu / 2×4090）

写于 2026-11-15。约定：**所有 Python 环境只放一套 conda root，不跨盘、不跨目录散放。**

## 1. 唯一布局

| 项 | 路径 | 磁盘 / 文件系统 |
|---|---|---|
| conda root | `/home/cui/miniconda3` | `/dev/sdb3` ext4 |
| 环境目录（唯一） | `/home/cui/miniconda3/envs` | 同上 |
| 包缓存（唯一） | `/home/cui/miniconda3/pkgs` | 同上 |
| 工程 + 数据 + 权重（**不放环境**） | `/run/media/cui/数据/LuSNAR_Seg` | `nvme1n1p3` **NTFS** |

用户级配置 `/home/cui/.condarc` 已把 `envs_dirs` / `pkgs_dirs` 钉死到上面两行。

> conda 仍会把 `~/.conda/envs` 列成 fallback（这是 conda 内建行为，改不掉），但它和主目录在**同一块盘**，所以不会出现"环境分盘"。该目录目前不存在，也不要有。

## 2. 环境清单

| env | 用途 | 关键版本 | 大小 |
|---|---|---|---|
| `base` | conda 自带，不作为项目环境 | — | — |
| `lusnar` | B2/B5 训练、Scene5 打分、开源评测、小坑实验 | `torch 2.7.1+cu128`、`mmseg 1.2.2`、`mmcv-lite 2.1.0`、Python 3.10 | 7.2 G |
| `lusnar_bb` | 仅 Mask2Former（HuggingFace 路线） | 上面全套 + `transformers 4.49.0`、`timm 1.0.15`、`ftfy 6.3.1` | ~7.2 G |

## 3. 规则

1. **新建环境**一律 `conda create -n <name> ...`（用名字，不用 `-p`），不要指定工程盘路径。
2. **装包**先 `conda activate <env>`，再在该环境内 `pip install`。
3. **禁止 `pip install --user`**。它写进 `~/.local/lib/python3.10/site-packages`，会被所有环境共享，等于制造隐形混装。当前该目录为空，保持为空。
4. **不跨环境装**：Mask2Former 的依赖只进 `lusnar_bb`；B5/B2 训练依赖只进 `lusnar`。长训期间不要往 `lusnar` 装任何包。
5. **不把环境放到工程盘**（NTFS）：NTFS 丢符号链接与权限位，conda 环境会坏，`import` 也慢。
6. **环境复制**用 `conda create -n NEW --clone OLD`（纯本地拷贝），不要 `cp -a` 环境目录。
7. **ToS 注意**：`defaults` 通道的 Anaconda 服务条款未接受时，`conda create` 会直接报 `CondaToSNonInteractiveError`。本地克隆可绕开：
   ```bash
   conda create -n NEW --clone OLD --offline --override-channels -c conda-forge -y
   ```
   不要为了绕过而去接受条款，除非用户明确同意。

## 4. 常用命令

```bash
conda env list
conda activate lusnar          # 或 lusnar_bb
conda config --show envs_dirs pkgs_dirs
du -sh /home/cui/miniconda3/envs/* /home/cui/miniconda3/pkgs
```

## 5. 排查

| 现象 | 先查 |
|---|---|
| 某个包版本不对/被覆盖 | `python -c "import X; print(X.__file__)"`，看是否落在 `~/.local` 或别的 env |
| 环境"消失" | `conda env list`；确认没在工程盘或 `/mnt` 下误建 |
| 磁盘吃紧 | `du -sh` 三项：envs、pkgs、工程盘；包缓存可 `conda clean -a`（会掉下载缓存，不影响已装环境） |
