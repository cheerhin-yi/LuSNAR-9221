#!/usr/bin/env python3
"""LuSNAR SegFormer-B2 formal training with optional torch DDP (2×4090).

Usage (dual GPU):
  torchrun --nproc_per_node=2 tools/portable/lusnar_train_ddp.py --recipe B1 ...

Usage (single GPU):
  python tools/portable/lusnar_train_ddp.py --recipe B1 ...

Never reads Moon_2/4/9 labels. Scene5 is local validation only.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
os.environ["USE_LIBUV"] = "0"
import random
import sys
import time
import traceback
from pathlib import Path

import cv2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, WeightedRandomSampler

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
from _lusnar_common import (  # noqa: E402
    RowHeightAttn, blv_perturb_logits, evaluate_tta,
    CCBank,
    CLASS_NAMES,
    EdgeAux,
    IGNORE,
    MEAN,
    NUM_CLASSES,
    SEED,
    STD,
    TRAIN_SCENES,
    VAL_SCENE,
    build_segformer_b2,
    build_split_lists,
    dice_loss,
    edge_map_from_mask,
    ensure_mmcv_stub,
    evaluate,
    find_pretrain,
    jfloat,
    load_pairs,
    photometric,
    recipe_flags,
    resolve_data_root,
    resolve_project_root,
    resolve_scene_root,
    rgb_label_to_index,
    set_seed,
    train_dir,
    update_ema,
    write_markdown_table,
)

ensure_mmcv_stub(SCRIPT)


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        # Windows CUDA builds have no NCCL; use gloo. Linux keeps nccl.
        backend = "gloo" if sys.platform == "win32" else "nccl"
        dist.init_process_group(backend=backend, timeout=__import__("datetime").timedelta(hours=3))
        torch.cuda.set_device(local_rank)
    return world_size, rank, local_rank


def cleanup_distributed(world_size: int):
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


class _SegLogits(torch.nn.Module):
    """Thin forward so DDP hooks run (must not call .module.extract_feat directly)."""

    def __init__(self, core: torch.nn.Module, row_attn=None):
        super().__init__()
        self.core = core
        self.row_attn = row_attn

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        feats = self.core.extract_feat(imgs)
        logits = self.core.decode_head.forward(feats)
        return self.row_attn(logits) if self.row_attn is not None else logits


class DistributedWeightedSampler(torch.utils.data.Sampler):
    def __init__(self, weights, rank, world_size):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.rank, self.world_size, self.epoch = rank, world_size, 0
        self.count = (len(weights) + world_size - 1) // world_size
    def set_epoch(self, epoch):
        self.epoch = epoch
    def __len__(self):
        return self.count
    def __iter__(self):
        g = torch.Generator().manual_seed(SEED + self.epoch)
        indices = torch.multinomial(self.weights, self.count * self.world_size, True, generator=g)
        return iter(indices[self.rank::self.world_size].tolist())


def seed_worker(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng.seed(torch.initial_seed())
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


class LuSNARDataset(Dataset):
    def __init__(
        self,
        list_file,
        train=True,
        crop=512,
        multi_scale=False,
        photometric_on=False,
        cc_bank=None,
        mosaic=False,
        seed=SEED,
    ):
        self.pairs = load_pairs(Path(list_file))
        self.train = train
        self.crop = crop
        self.multi_scale = multi_scale
        self.photometric_on = photometric_on
        self.cc_bank = cc_bank
        self.mosaic = mosaic
        self.mean = MEAN
        self.std = STD
        self.rng = random.Random(seed + (0 if train else 999))

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _imread_bgr(path: str):
        # Prefer PIL: cv2/libpng can hang or spam CRC errors on corrupt IDAT.
        try:
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGB"))
            return arr[:, :, ::-1].copy()
        except Exception:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            return img

    def _read(self, idx):
        n = len(self.pairs)
        cur = int(idx) % n
        for _ in range(min(32, n)):
            ip, lp = self.pairs[cur]
            img = self._imread_bgr(ip)
            lab = self._imread_bgr(lp)
            if img is None or lab is None:
                cur = (cur + 1) % n
                continue
            try:
                mask = rgb_label_to_index(lab)
            except Exception:
                cur = (cur + 1) % n
                continue
            return img, mask, ip
        raise RuntimeError(f"bad pair after retries around idx={idx}")

    def _mosaic4(self, idx):
        idxs = [idx] + [self.rng.randrange(len(self.pairs)) for _ in range(3)]
        tiles_i, tiles_m = [], []
        for j in idxs:
            img, mask, _ = self._read(j)
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            tiles_i.append(img)
            tiles_m.append(mask)
        top = np.concatenate([tiles_i[0], tiles_i[1]], axis=1)
        bot = np.concatenate([tiles_i[2], tiles_i[3]], axis=1)
        img = np.concatenate([top, bot], axis=0)
        topm = np.concatenate([tiles_m[0], tiles_m[1]], axis=1)
        botm = np.concatenate([tiles_m[2], tiles_m[3]], axis=1)
        mask = np.concatenate([topm, botm], axis=0)
        return img, mask

    def __getitem__(self, idx):
        if self.train and self.mosaic and self.rng.random() < 0.5:
            img, mask = self._mosaic4(idx)
            ip = self.pairs[idx][0]
        else:
            img, mask, ip = self._read(idx)

        if self.train and self.cc_bank is not None and self.rng.random() < 0.7:
            img, mask = self.cc_bank.sample_paste(
                img, mask, self.rng, n_paste=self.rng.randint(1, 3)
            )
        if self.train and self.photometric_on:
            img = photometric(img, self.rng)

        h, w = img.shape[:2]
        crop = self.crop
        if self.train and self.multi_scale:
            scale = self.rng.uniform(0.75, 1.5)
            nh, nw = int(h * scale), int(w * scale)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
            h, w = nh, nw
            crop = 768 if self.rng.random() < 0.5 else self.crop

        if self.train:
            y0 = self.rng.randint(0, max(0, h - crop))
            x0 = self.rng.randint(0, max(0, w - crop))
            if self.rng.random() < 0.5:
                img = img[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
                x0 = w - crop - x0 if w >= crop else 0
        else:
            crop = self.crop
            y0 = max(0, (h - crop) // 2)
            x0 = max(0, (w - crop) // 2)

        img = img[y0 : y0 + crop, x0 : x0 + crop]
        mask = mask[y0 : y0 + crop, x0 : x0 + crop]
        if img.shape[0] != crop or img.shape[1] != crop:
            pad = np.zeros((crop, crop, 3), np.uint8)
            mpad = np.full((crop, crop), IGNORE, np.uint8)
            pad[: img.shape[0], : img.shape[1]] = img
            mpad[: mask.shape[0], : mask.shape[1]] = mask
            img, mask = pad, mpad

        edge = edge_map_from_mask(mask)
        # unify spatial size for batching
        out_size = self.crop
        if crop != out_size:
            img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
            edge = cv2.resize(edge, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

        img_f = (img.astype(np.float32) - self.mean) / self.std
        return (
            torch.from_numpy(img_f.transpose(2, 0, 1)).float(),
            torch.from_numpy(mask.astype(np.int64)),
            torch.from_numpy(edge.astype(np.float32)),
            ip,
        )


def maybe_prepare_smoke_assets(project: Path, scene_root: Path, need_rcs_or_cc: bool):
    """Reuse / build smoke assets for RCS weights + CC bank when recipe needs them."""
    if not need_rcs_or_cc:
        return None
    smoke_dir = project / "data" / "smoke"
    if (smoke_dir / "rarity.json").exists() and (smoke_dir / "cc_bank_index.json").exists():
        return smoke_dir
    # light prepare via smoke script helpers
    from lusnar_smoke_compare import prepare_smoke_assets

    prepare_smoke_assets(scene_root, smoke_dir, force=False)
    return smoke_dir


def parse_args():
    ap = argparse.ArgumentParser(description="LuSNAR SegFormer-B2 DDP trainer")
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--train-subdir", default="train")
    ap.add_argument("--pretrain", default=None)
    ap.add_argument("--recipe", default="B1", choices=["A0", "A1", "B1", "C1", "M_ROW_BLV"], help="C1 opt-in; default B1")
    ap.add_argument("--run-name", default=None, help="default train_{recipe}")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16, help="per-GPU batch (4090 48G: 8-16)")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true", help="cap at 5 iters, no long train")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--mem-fraction", type=float, default=None, help="optional CUDA mem cap; omit on 4090")
    ap.add_argument("--list-dir", default=None, help="dir for train_scenes.txt / val_scene5.txt")
    ap.add_argument("--train-scenes", default="", help="comma scene ids; default 1,3,6,7,8")
    ap.add_argument("--no-val", action="store_true", help="no val selection; save last.pth at the final iter")
    ap.add_argument("--tta-final", action="store_true")
    ap.add_argument(
        "--resume",
        default=None,
        help="path to iter_*.pth / best.pth; restore weights+optimizer and continue from saved iter without wiping train_log.jsonl",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    world_size, rank, local_rank = setup_distributed()
    set_seed(SEED + rank)
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    project = resolve_project_root(SCRIPT, args.project_root)
    data_root = resolve_data_root(project, args.data_root)
    scene_root = resolve_scene_root(data_root, args.train_subdir)
    pretrain = find_pretrain(project, args.pretrain)
    run_name = args.run_name or f"train_{args.recipe}"
    work = project / "work_dirs" / run_name
    if is_main(rank):
        work.mkdir(parents=True, exist_ok=True)
        (project / "results").mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    flags = recipe_flags(args.recipe)
    train_scenes = (
        tuple(int(x) for x in args.train_scenes.split(",") if x.strip())
        if args.train_scenes.strip()
        else TRAIN_SCENES
    )
    lists_dir = Path(args.list_dir).expanduser().resolve() if args.list_dir else (project / "data" / "lists_portable")
    if is_main(rank):
        if not scene_root.is_dir():
            raise SystemExit(f"missing scene root {scene_root}")
        train_file, val_file, n_tr, n_va = build_split_lists(
            scene_root,
            lists_dir,
            train_scenes=train_scenes,
            val_scene=None if args.no_val else VAL_SCENE,
        )
        print(
            json.dumps(
                {
                    "project": str(project),
                    "scene_root": str(scene_root),
                    "train": n_tr,
                    "val_scene5": n_va,
                    "train_scenes": list(train_scenes),
                    "no_val": bool(args.no_val),
                    "recipe": args.recipe,
                    "world_size": world_size,
                    "batch_per_gpu": args.batch,
                    "pretrain": str(pretrain),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        # never touch test labels
        test_dir = data_root / "test"
        if test_dir.exists():
            print(f"[info] test dir present at {test_dir} — training will NOT read labels there", flush=True)
    else:
        train_file = lists_dir / "train_scenes.txt"
        val_file = lists_dir / "val_scene5.txt"

    if world_size > 1:
        dist.barrier()

    smoke_dir = None
    if flags["rcs"] or flags["cc_paste"]:
        if is_main(rank):
            smoke_dir = maybe_prepare_smoke_assets(project, scene_root, True)
        if world_size > 1:
            dist.barrier()
        smoke_dir = project / "data" / "smoke"

    cc_bank = CCBank(smoke_dir / "cc_bank_index.json") if (smoke_dir and flags["cc_paste"]) else None

    train_ds = LuSNARDataset(
        train_file,
        train=True,
        crop=args.crop,
        multi_scale=flags["multi_scale"],
        photometric_on=flags["photometric"],
        cc_bank=cc_bank,
        mosaic=flags["mosaic"],
        seed=SEED + rank,
    )
    val_ds = None if args.no_val else LuSNARDataset(val_file, train=False, crop=args.crop, seed=SEED)

    sampler = DistributedSampler(train_ds, shuffle=True, seed=SEED) if world_size > 1 else None
    if flags['rcs']:
        weight_file = lists_dir / 'full_crater_weights.json'
        if is_main(rank) and not weight_file.exists():
            from concurrent.futures import ThreadPoolExecutor
            def weight_pair(pair):
                lab = LuSNARDataset._imread_bgr(pair[1])
                mask = rgb_label_to_index(lab)
                valid = max(1, int((mask != IGNORE).sum()))
                return ((mask == 1).sum() / valid + 2.0 * (mask == 2).sum() / valid + 0.02) ** 0.5
            with ThreadPoolExecutor(max_workers=8) as pool:
                weights = list(pool.map(weight_pair, train_ds.pairs))
            print(f"[rcs] full-train weights ready: {len(weights)}", flush=True)
            weight_file.write_text(json.dumps({'pairs': train_ds.pairs, 'weights': weights}), encoding='utf-8')
        if world_size > 1:
            dist.barrier()
        cached = json.loads(weight_file.read_text(encoding='utf-8'))
        if cached['pairs'] != [list(p) for p in train_ds.pairs]:
            raise RuntimeError('Full-train sampling cache does not match training list')
        sampler = DistributedWeightedSampler(cached['weights'], rank, world_size)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, sampler=sampler, shuffle=sampler is None,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0, worker_init_fn=seed_worker,
        **({'prefetch_factor': 2} if args.workers else {}),
    )

    # Keep val well below the train batch so a full Scene5 pass cannot OOM a card that is already packed.
    val_loader = None
    if val_ds is not None:
        val_bs = min(12, max(2, args.batch // 2))
        val_loader = DataLoader(
            val_ds, batch_size=val_bs, shuffle=False, num_workers=max(1, args.workers // 2), pin_memory=True
        )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    core = build_segformer_b2(pretrain).to(device)
    row_attn = RowHeightAttn().to(device) if flags["row_attn"] else None
    model = _SegLogits(core, row_attn)
    edge_aux = EdgeAux().to(device) if flags["eps"] else None
    ema_model = copy.deepcopy(core) if flags["ema"] else None
    if ema_model is not None:
        for p in ema_model.parameters():
            p.requires_grad_(False)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if is_main(rank):
            print(f"[ddp] backend-ready world_size={world_size} local_rank={local_rank} device={device}", flush=True)
    else:
        if is_main(rank):
            print(f"[single] device={device}", flush=True)

    raw_model = model.module.core if isinstance(model, DDP) else model.core

    params = [p for p in model.parameters() if p.requires_grad]
    if edge_aux is not None:
        params += list(edge_aux.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    class_weights = None
    if flags["acw"]:
        # estimate from a quick pass over train list labels would be heavy; use smoke meta if present
        meta_path = project / "data" / "smoke" / "meta.json"
        if meta_path.exists():
            class_freq = np.array(json.loads(meta_path.read_text(encoding="utf-8"))["class_freq"], np.float64)
        else:
            class_freq = np.array([0.55, 0.02, 0.03, 0.15, 0.25], np.float64)
        beta = 0.5
        freq = np.clip(class_freq, 1e-6, None)
        cw = (1.0 / np.power(freq, beta)).astype(np.float32)
        cw = cw / cw.mean()
        if flags["crater_boost"]:
            cw[2] *= 1.75
            cw /= cw.mean()
        class_weights = torch.tensor(cw, device=device)
        if is_main(rank):
            print(f"ACW weights={cw.tolist()}", flush=True)

    if flags["blv"] and "class_freq" not in dir():
        class_freq = np.array([0.55, 0.02, 0.03, 0.15, 0.25], np.float64)
    if flags["row_attn"]:
        row_ids = {id(p) for p in row_attn.parameters()}
        opt_ids = {id(p) for group in opt.param_groups for p in group["params"]}
        if not row_ids <= opt_ids:
            raise RuntimeError("RowHeightAttn parameters are not in the optimizer")
        last = row_attn.mlp[-1]
        if float(last.weight.detach().abs().sum()) != 0.0 or float(last.bias.detach().abs().sum()) != 0.0:
            raise RuntimeError("RowHeightAttn last layer is not zero-initialized")
        if is_main(rank):
            n_row = sum(p.numel() for p in row_attn.parameters())
            print(f"[module] RowHeightAttn params={n_row} in_optimizer=yes zero_init_last=yes", flush=True)
    if flags["blv"]:
        freq_np = np.clip(np.asarray(class_freq, np.float64), 1e-6, None)
        blv_scale = (1.0 / np.sqrt(freq_np))
        blv_scale = blv_scale / blv_scale.mean()
        if not (blv_scale[1] > blv_scale[0] and blv_scale[2] > blv_scale[0]):
            raise RuntimeError(f"BLV rarity scale is not larger on rock/crater: {blv_scale.tolist()}")
        if is_main(rank):
            print(
                f"[module] BLV sigma=0.5 train_only scale={[round(float(x), 4) for x in blv_scale]}",
                flush=True,
            )

    amp_enabled = args.amp != "off" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    iters = 5 if args.dry_run else args.iters
    log_path = work / "train_log.jsonl"
    resume_ckpt = None
    if args.resume:
        resume_path = Path(args.resume).expanduser()
        if not resume_path.is_file():
            resume_path = work / args.resume
        if not resume_path.is_file():
            raise SystemExit(f"resume checkpoint not found: {args.resume}")
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(resume_ckpt["model"])
        if row_attn is not None:
            if not resume_ckpt.get("row_attn"):
                raise SystemExit("resume ckpt missing row_attn but recipe has RowHeightAttn")
            row_attn.load_state_dict(resume_ckpt["row_attn"])
        if resume_ckpt.get("optimizer"):
            opt.load_state_dict(resume_ckpt["optimizer"])
        if is_main(rank):
            print(
                {
                    "resume": str(resume_path),
                    "resume_iter": int(resume_ckpt.get("iter", 0)),
                    "resume_miou": resume_ckpt.get("miou"),
                    "resume_recipe": resume_ckpt.get("recipe"),
                },
                flush=True,
            )
    elif is_main(rank) and log_path.exists() and not args.dry_run:
        log_path.unlink()

    peak_mem = 0.0
    it = int(resume_ckpt.get("iter", 0)) if resume_ckpt is not None else 0
    t0 = time.time()
    best = -1.0
    best_ious = None
    last_ious = None
    error = None
    if resume_ckpt is not None and resume_ckpt.get("miou") is not None:
        best = float(resume_ckpt["miou"])
    best_path = work / "best.pth"
    if best_path.is_file():
        try:
            bck = torch.load(best_path, map_location="cpu", weights_only=False)
            if bck.get("miou") is not None and float(bck["miou"]) > best:
                best = float(bck["miou"])
                if bck.get("per_class_IoU"):
                    pass
        except Exception as exc:
            if is_main(rank):
                print(f"[resume] warn: could not read best.pth ({exc})", flush=True)
    if is_main(rank) and resume_ckpt is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "resume", "from_iter": it, "best": best}, ensure_ascii=False) + "\n")
    model.train()

    try:
        epoch = 0
        while it < iters:
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            for batch_data in train_loader:
                imgs, masks, edges = batch_data[0], batch_data[1], batch_data[2]
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                edges = edges.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    # Call DDP wrapper (not .module) so allreduce hooks fire on both ranks.
                    logits = model(imgs)
                    if logits.shape[-2:] != masks.shape[-2:]:
                        logits = F.interpolate(
                            logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                        )
                    net = raw_model
                    ce = F.cross_entropy(
                        (blv_perturb_logits(logits, class_freq, sigma=0.5) if flags["blv"] else logits).float(), masks, weight=class_weights, ignore_index=IGNORE
                    )
                    dsc = dice_loss(logits, masks)
                    loss = ce + dsc
                    if flags["eps"] and edge_aux is not None:
                        e_logits = edge_aux(logits)
                        valid = masks != IGNORE
                        eg = edges.unsqueeze(1)
                        e_loss_map = F.binary_cross_entropy_with_logits(
                            e_logits.float(), eg.float(), reduction="none"
                        ).squeeze(1)
                        e_loss = e_loss_map[valid].mean() if valid.any() else logits.new_zeros(())
                        loss = loss + 0.4 * e_loss
                row_grad_l1 = 0.0
                blv_ce_delta = 0.0
                if it == 0 and flags["blv"]:
                    with torch.no_grad():
                        ce_clean = F.cross_entropy(
                            logits.detach().float(), masks, weight=class_weights, ignore_index=IGNORE
                        )
                    blv_ce_delta = abs(float(ce.detach()) - float(ce_clean))
                loss.backward()
                if it == 0 and flags["row_attn"]:
                    w = row_attn.mlp[-1].weight
                    b = row_attn.mlp[-1].bias
                    if w.grad is None or b.grad is None:
                        row_grad_l1 = 0.0
                    else:
                        row_grad_l1 = float(w.grad.detach().abs().sum() + b.grad.detach().abs().sum())
                opt.step()
                if flags["ema"] and ema_model is not None and is_main(rank):
                    update_ema(ema_model, net, 0.999)

                it += 1
                if device.type == "cuda":
                    mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    peak_mem = max(peak_mem, mem)
                else:
                    mem = 0.0

                if it == 1 and world_size > 1:
                    # Verify that both core and ROW parameters were updated identically.
                    probe = torch.stack([p.detach().double().sum() for p in model.parameters()]).cpu()
                    gathered = [torch.empty_like(probe) for _ in range(world_size)]
                    dist.all_gather(gathered, probe)
                    if not all(torch.equal(gathered[0], x) for x in gathered[1:]):
                        raise RuntimeError('DDP parameter synchronization check failed')
                    if is_main(rank):
                        print('[ddp] core + ROW parameter synchronization verified', flush=True)
                if it == 1 and (flags["row_attn"] or flags["blv"]):
                    moved = 0.0
                    if flags["row_attn"]:
                        last = row_attn.mlp[-1]
                        moved = float(last.weight.detach().abs().sum() + last.bias.detach().abs().sum())
                    local_fx = torch.tensor(
                        [row_grad_l1, blv_ce_delta, moved], dtype=torch.float64
                    )
                    if world_size > 1:
                        gathered_fx = [torch.empty_like(local_fx) for _ in range(world_size)]
                        dist.all_gather(gathered_fx, local_fx)
                    else:
                        gathered_fx = [local_fx]
                    rows_fx = [[float(x) for x in g] for g in gathered_fx]
                    bad_fx = []
                    if flags["row_attn"] and any(r[0] <= 1e-6 or r[2] <= 1e-8 for r in rows_fx):
                        bad_fx.append("row")
                    if flags["blv"] and any(r[1] <= 1e-4 for r in rows_fx):
                        bad_fx.append("blv")
                    payload = {
                        "row_grad_l1": [r[0] for r in rows_fx],
                        "blv_ce_delta": [r[1] for r in rows_fx],
                        "row_weight_l1_after": [r[2] for r in rows_fx],
                        "ok": not bad_fx,
                    }
                    if is_main(rank):
                        print("[module] ROW_BLV_EFFECT " + json.dumps(payload), flush=True)
                        (work / "module_effect.json").write_text(
                            json.dumps(payload, indent=2), encoding="utf-8"
                        )
                    if bad_fx:
                        raise RuntimeError(f"ROW/BLV did not take effect: {bad_fx} {payload}")
                if it == 1:
                    print(
                        {
                            "iter": 1,
                            "rank": rank,
                            "local_rank": local_rank,
                            "device": str(device),
                            "cuda_current": int(torch.cuda.current_device()) if device.type == "cuda" else None,
                            "mem_mb": round(mem, 1),
                        },
                        flush=True,
                    )
                if is_main(rank) and (it % args.log_every == 0 or it == 1 or args.dry_run):
                    msg = {
                        "iter": it,
                        "loss": float(loss.item()),
                        "ce": float(ce.item()),
                        "dice": float(dsc.item()),
                        "elapsed_s": round(time.time() - t0, 1),
                        "mem_mb": round(mem, 1),
                        "rank": rank,
                    }
                    print(msg, flush=True)
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(msg) + "\n")

                save_now = (not args.dry_run) and (it % args.val_every == 0 or it == iters)
                if world_size > 1 and save_now:
                    dist.barrier()
                if is_main(rank) and save_now and args.no_val:
                    payload = {
                        "model": net.state_dict(),
                        "row_attn": row_attn.state_dict() if row_attn is not None else None,
                        "optimizer": opt.state_dict(),
                        "iter": it,
                        "miou": None,
                        "recipe": args.recipe,
                    }
                    torch.save(payload, work / f"iter_{it}.pth")
                    if it == iters:
                        torch.save(payload, work / "last.pth")
                    print({"iter": it, "checkpoint": "last" if it == iters else f"iter_{it}", "no_val": True}, flush=True)
                do_eval = (
                    is_main(rank)
                    and (not args.dry_run)
                    and (not args.no_val)
                    and (it % args.val_every == 0 or it == iters)
                )
                if do_eval:
                    miou, ious = evaluate(
                        net, val_loader, device, use_ema=ema_model if flags["ema"] else None, row_attn=row_attn
                    )
                    last_ious = ious
                    row = {
                        "iter": it,
                        "val_mIoU_scene5": jfloat(miou),
                        "per_class_IoU": {n: jfloat(v) for n, v in zip(CLASS_NAMES, ious)},
                        "note": "本地 Scene5 验证，非官方测试成绩",
                    }
                    print(row, flush=True)
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    torch.save(
                        {"model": net.state_dict(), "row_attn": row_attn.state_dict() if row_attn is not None else None, "optimizer": opt.state_dict(), "iter": it, "miou": miou, "recipe": args.recipe},
                        work / f"iter_{it}.pth",
                    )
                    if miou > best:
                        best = miou
                        best_ious = ious
                        torch.save(
                            {"model": net.state_dict(), "row_attn": row_attn.state_dict() if row_attn is not None else None, "optimizer": opt.state_dict(), "iter": it, "miou": miou, "recipe": args.recipe},
                            work / "best.pth",
                        )
                    model.train()
                if world_size > 1 and (it % args.val_every == 0 or it == iters) and not args.dry_run:
                    dist.barrier()

                if it >= iters:
                    break
            epoch += 1
    except Exception:
        error = traceback.format_exc()
        if is_main(rank):
            print("ERROR", error, flush=True)

    if error:
        raise RuntimeError(error)
    if world_size > 1:
        dist.barrier()

    if is_main(rank):
        wall = time.time() - t0

        def iou_dict(ious):
            return {n: jfloat(v) for n, v in zip(CLASS_NAMES, ious)} if ious else None

        config_snap = {
            "recipe": args.recipe,
            "iters": args.iters,
            "batch_per_gpu": args.batch,
            "crop": args.crop,
            "lr": args.lr,
            "workers": args.workers,
            "amp": args.amp,
            "tta_final": args.tta_final,
            "blv_sigma": 0.5,
            "sampling": "full_train_crater_weighted_distributed",
            "train_scenes": list(train_scenes),
            "val_scene": None if args.no_val else VAL_SCENE,
            "no_val": bool(args.no_val),
            "designated_checkpoint": "last.pth" if args.no_val else "best.pth",
            "flags": flags,
            "data_root": str(data_root),
            "train_subdir": args.train_subdir,
            "pretrain": str(pretrain),
        }
        summary = {
            "exp": args.recipe,
            "run_name": run_name,
            "iters": int(it),
            "target_iters": int(iters),
            "best_mIoU_scene5": jfloat(best) if best >= 0 else None,
            "last_mIoU_scene5": jfloat(np.nanmean(last_ious)) if last_ious else None,
            "per_class_IoU_best": iou_dict(best_ious),
            "per_class_IoU_last": iou_dict(last_ious),
            "iou_rock": jfloat(best_ious[1]) if best_ious else None,
            "iou_crater": jfloat(best_ious[2]) if best_ious else None,
            "peak_mem_mb": float(round(peak_mem, 1)),
            "wall_sec": float(round(wall, 1)),
            "world_size": world_size,
            "flags": flags,
            "config": config_snap,
            "dry_run": args.dry_run,
            "error": error,
            "note": "本地 Scene5 验证，非官方测试成绩；未使用 Moon_2/4/9 标签",
        }
        (work / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report = project / "results" / f"train_report_{run_name}.md"
        write_markdown_table(report, [summary], f"LuSNAR train {run_name}")
        print("DONE", summary, flush=True)

    if error:
        raise RuntimeError(error)
    if world_size > 1:
        dist.barrier()
    if is_main(rank) and args.tta_final and not args.dry_run and not args.no_val:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            ckpt = torch.load(work / "best.pth", map_location=device, weights_only=False)
            raw_model.load_state_dict(ckpt["model"])
            if row_attn is not None:
                row_attn.load_state_dict(ckpt["row_attn"])
            result = evaluate_tta(raw_model, val_ds.pairs, device, row_attn=row_attn)
            (work / "tta_final.json").write_text(
                json.dumps(
                    {
                        "best_iter": ckpt["iter"],
                        "mIoU": result[0],
                        "per_class": result[1],
                        "rates": result[3],
                        "meta": result[4],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[tta] final mIoU={result[0]}", flush=True)
        except Exception:
            err = traceback.format_exc()
            print("TTA_ERROR", err, flush=True)
            (work / "tta_final.json").write_text(
                json.dumps({"error": err}, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    if world_size > 1:
        dist.barrier()
    cleanup_distributed(world_size)


if __name__ == "__main__":
    main()
