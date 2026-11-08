#!/usr/bin/env python3
"""LuSNAR SegFormer-B2 smoke matrix + method smokes (portable).

Recipes: A0/A1/B1/C1, V_B1C/V_B1E, M_ROW/M_BLV/M_SHAD/M_ABL/M_INVFORM/M_FADE,
M_DYSAMPLE/M_HARD/M_CTRAST/M_SAPA (train); M_MSINFER/M_TTA/M_SEGFIX/T_B1_TTA (eval-only); M_HRDA stub.
Prepares smoke subsets + rarity + CC bank, then runs short trains / TTA.
Scene5 local val only; never reads Moon_2/4/9 labels.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
from _lusnar_common import (  # noqa: E402
    CCBank,
    BoundaryMeter,
    DySample,
    SAPALite,
    _forward_logits,
    assert_safe_label,
    hardness_ce,
    boundary_contrast_loss,
    evaluate_inference_modes,
    CLASS_NAMES,
    EdgeAux,
    FadeLite,
    IGNORE,
    MEAN,
    NUM_CLASSES,
    SEED,
    STD,
    RowHeightAttn,
    abl_lite_loss,
    blv_perturb_logits,
    build_segformer_b2,
    build_split_lists,
    crater_boost_sampler_weights,
    dice_loss,
    edge_map_from_mask,
    ensure_mmcv_stub,
    evaluate,
    evaluate_detailed,
    evaluate_tta,
    find_pretrain,
    jfloat,
    load_pairs,
    photometric,
    recipe_flags,
    resolve_data_root,
    resolve_project_root,
    resolve_scene_root,
    rgb_label_to_index,
    segfix_lite,
    set_seed,
    shadow_augment,
    train_dir,
    update_ema,
    write_markdown_table,
    write_method_eval_md,
)

ensure_mmcv_stub(SCRIPT)

# smoke prepare defaults
TRAIN_N = 1600
VAL_N = 350
MAX_CC_PER_CLASS = 400
MAX_CC_SIDE = 256
MIN_CC_AREA = 64


class SmokeDataset(Dataset):
    def __init__(
        self,
        list_file,
        train=True,
        crop=512,
        multi_scale=False,
        photometric_on=False,
        shadow_aug=False,
        cc_bank=None,
        mosaic=False,
        seed=SEED,
    ):
        self.pairs = load_pairs(Path(list_file))
        self.train = train
        self.crop = crop
        self.multi_scale = multi_scale
        self.photometric_on = photometric_on
        self.shadow_aug = shadow_aug
        self.cc_bank = cc_bank
        self.mosaic = mosaic
        self.mean = MEAN
        self.std = STD
        self.rng = random.Random(seed + (0 if train else 1))

    def __len__(self):
        return len(self.pairs)

    def _read(self, idx):
        ip, lp = self.pairs[idx]
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        lab = cv2.imread(lp, cv2.IMREAD_COLOR)
        if img is None or lab is None:
            raise RuntimeError(f"bad pair {ip}")
        mask = rgb_label_to_index(lab)
        return img, mask, ip

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
        if self.train and self.shadow_aug:
            img = shadow_augment(img, self.rng)

        h, w = img.shape[:2]
        crop = self.crop
        if self.train and self.multi_scale:
            scale = self.rng.uniform(0.75, 1.5)
            nh, nw = int(h * scale), int(w * scale)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
            h, w = nh, nw
            crop = 768 if self.rng.random() < 0.4 else 512

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
        if crop != 512:
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            edge = cv2.resize(edge, (512, 512), interpolation=cv2.INTER_NEAREST)

        img_f = (img.astype(np.float32) - self.mean) / self.std
        img_t = torch.from_numpy(img_f.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask.astype(np.int64))
        edge_t = torch.from_numpy(edge.astype(np.float32))
        return img_t, mask_t, edge_t, ip


def prepare_smoke_assets(scene_root: Path, smoke_dir: Path, force: bool = False, seed: int = SEED):
    smoke_dir.mkdir(parents=True, exist_ok=True)
    marker = smoke_dir / "meta.json"
    if marker.exists() and not force and (smoke_dir / "train_sub.txt").exists():
        print(f"[prepare] reuse existing assets under {smoke_dir}", flush=True)
        return

    lists_dir = smoke_dir / "_lists"
    train_file, val_file, n_tr, n_va = build_split_lists(scene_root, lists_dir)
    print(f"[prepare] full split train={n_tr} val={n_va} from {scene_root}", flush=True)

    rng = random.Random(seed)
    train_all = load_pairs(train_file)
    val_all = load_pairs(val_file)
    train_idx = list(range(len(train_all)))
    val_idx = list(range(len(val_all)))
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    train_n = min(TRAIN_N, len(train_all))
    val_n = min(VAL_N, len(val_all))
    train_sub = [train_all[i] for i in train_idx[:train_n]]
    val_sub = [val_all[i] for i in val_idx[:val_n]]

    def write_list(path, pairs):
        path.write_text(
            "\n".join(f"{a}\t{b}" for a, b in pairs) + "\n", encoding="utf-8"
        )

    write_list(smoke_dir / "train_sub.txt", train_sub)
    write_list(smoke_dir / "val_sub.txt", val_sub)

    rarity = []
    class_pix = np.zeros(5, np.float64)
    bank = {1: [], 2: []}
    bank_dir = smoke_dir / "cc_bank"
    bank_dir.mkdir(exist_ok=True)
    # clear old bank files if regenerating
    for old in bank_dir.glob("c*_*.npz"):
        old.unlink()
    counts = {1: 0, 2: 0}

    for i, (ip, lp) in enumerate(train_sub):
        lab = cv2.imread(lp, cv2.IMREAD_COLOR)
        if lab is None:
            raise RuntimeError(lp)
        mask = rgb_label_to_index(lab)
        valid = mask != IGNORE
        n_valid = int(valid.sum()) or 1
        rare_frac = float(((mask == 1) | (mask == 2)).sum()) / n_valid
        rarity.append({"img": ip, "lab": lp, "rare_frac": rare_frac, "idx": i})
        for c in range(5):
            class_pix[c] += int((mask == c).sum())

        img = None
        for cid in (1, 2):
            if counts[cid] >= MAX_CC_PER_CLASS:
                continue
            binm = (mask == cid).astype(np.uint8)
            if binm.sum() < MIN_CC_AREA:
                continue
            nlab, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
            for lab_id in range(1, nlab):
                if counts[cid] >= MAX_CC_PER_CLASS:
                    break
                x, y, w, h, area = stats[lab_id]
                if area < MIN_CC_AREA:
                    continue
                if w > MAX_CC_SIDE or h > MAX_CC_SIDE:
                    cx, cy = x + w // 2, y + h // 2
                    x0 = max(0, cx - MAX_CC_SIDE // 2)
                    y0 = max(0, cy - MAX_CC_SIDE // 2)
                    x1 = min(mask.shape[1], x0 + MAX_CC_SIDE)
                    y1 = min(mask.shape[0], y0 + MAX_CC_SIDE)
                    x, y = x0, y0
                    w, h = x1 - x0, y1 - y0
                if img is None:
                    img = cv2.imread(ip, cv2.IMREAD_COLOR)
                comp = labels[y : y + h, x : x + w] == lab_id
                if comp.sum() < MIN_CC_AREA:
                    continue
                patch_img = img[y : y + h, x : x + w].copy()
                patch_msk = np.full((h, w), IGNORE, np.uint8)
                patch_msk[comp] = cid
                fn = bank_dir / f"c{cid}_{counts[cid]:04d}.npz"
                np.savez_compressed(fn, img=patch_img, mask=patch_msk, cid=cid)
                bank[cid].append(str(fn))
                counts[cid] += 1

        if (i + 1) % 200 == 0:
            print(f"[prepare] scanned {i+1}/{len(train_sub)} bank={counts}", flush=True)

    fracs = np.array([r["rare_frac"] for r in rarity], np.float64)
    weights = (fracs + 0.02) ** 0.5
    weights = weights / weights.sum()

    meta = {
        "seed": seed,
        "train_n": train_n,
        "val_n": val_n,
        "class_pix": class_pix.tolist(),
        "class_freq": (class_pix / max(class_pix.sum(), 1)).tolist(),
        "bank_counts": counts,
        "scene_root": str(scene_root),
        "note": "Scene5 val_sub only; no Moon_2/4/9",
    }
    (smoke_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (smoke_dir / "rarity.json").write_text(
        json.dumps({"items": rarity, "weights": weights.tolist()}, ensure_ascii=False),
        encoding="utf-8",
    )
    (smoke_dir / "cc_bank_index.json").write_text(
        json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[prepare] DONE", meta, flush=True)


def _run_exp_impl(
    exp: str,
    project: Path,
    smoke_dir: Path,
    pretrain: Path,
    iters: int,
    batch: int,
    lr: float,
    dry_run: bool = False,
    mem_fraction: float | None = None,
    blv_sigma: float = 0.5,
    tta_scales=(0.75, 1.0, 1.25),
    tta_crop: int = 512,
    tta_stride: int = 256,
    ckpt: str | None = None,
    abl_weight: float = 0.5,
    accum_steps: int = 1,
    val_every: int = 500,
    seed: int = SEED,
):
    set_seed(seed)
    flags = recipe_flags(exp)
    micro_batch = batch // accum_steps

    if flags.get("hrda"):
        work = project / "work_dirs" / f"smoke_{exp}{'_dryrun' if dry_run else ''}"
        work.mkdir(parents=True, exist_ok=True)
        summary = {
            "exp": exp,
            "run_name": f"smoke_{exp}{'_dryrun' if dry_run else ''}",
            "error": (
                "M_HRDA not implemented for portable smoke. "
                "Gate: only after M_TTA passes (Δ above noise) AND M_FADE insufficient. "
                "Do not fake HRDA."
            ),
            "flags": flags,
            "method_note": "NOT IMPLEMENTED — see README_METHOD_SMOKE.md",
            "note": "本地 Scene5；未使用 Moon_2/4/9 标签",
        }
        (work / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("DONE", summary, flush=True)
        return summary

    if flags.get("invform_fallback"):
        print(
            "[M_INVFORM] InverseForm pretrained boundary net NOT bundled; "
            "falling back to ABL-lite (same as M_ABL).",
            flush=True,
        )

    # M_TTA: eval-only on V_B1C weights (train V_B1C first if missing)
    if flags["tta"]:
        return run_tta_exp(
            project=project,
            smoke_dir=smoke_dir,
            pretrain=pretrain,
            iters=iters,
            batch=batch,
            lr=lr,
            dry_run=dry_run,
            mem_fraction=mem_fraction,
            blv_sigma=blv_sigma,
            tta_scales=tta_scales,
            tta_crop=tta_crop,
            tta_stride=tta_stride,
            ckpt_override=ckpt,
            exp_name="M_TTA",
            require_ckpt=False,
            default_ckpt=project / "work_dirs" / "smoke_V_B1C" / "best.pth",
            auto_train_vb1c=True,
            accum_steps=accum_steps, val_every=val_every, seed=seed,
        )

    if flags.get("t_b1_tta"):
        return run_tta_exp(
            project=project,
            smoke_dir=smoke_dir,
            pretrain=pretrain,
            iters=iters,
            batch=batch,
            lr=lr,
            dry_run=dry_run,
            mem_fraction=mem_fraction,
            blv_sigma=blv_sigma,
            tta_scales=tta_scales,
            tta_crop=tta_crop,
            tta_stride=tta_stride,
            ckpt_override=ckpt,
            exp_name="T_B1_TTA",
            require_ckpt=True,
            default_ckpt=project / "work_dirs" / "train_B1" / "best.pth",
            auto_train_vb1c=False,
            accum_steps=accum_steps, val_every=val_every, seed=seed,
        )

    if flags.get("msinfer"):
        return run_msinfer_exp(project, smoke_dir, pretrain, dry_run=dry_run,
                               mem_fraction=mem_fraction, ckpt_override=ckpt)

    if flags.get("segfix"):
        return run_segfix_exp(
            project=project,
            smoke_dir=smoke_dir,
            pretrain=pretrain,
            dry_run=dry_run,
            mem_fraction=mem_fraction,
            ckpt_override=ckpt,
        )

    work = project / "work_dirs" / f"smoke_{exp}{'_dryrun' if dry_run else ''}"
    work.mkdir(parents=True, exist_ok=True)
    log_path = work / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    meta = json.loads((smoke_dir / "meta.json").read_text(encoding="utf-8"))
    class_freq = np.array(meta["class_freq"], np.float64)
    cc_bank = CCBank(smoke_dir / "cc_bank_index.json") if flags["cc_paste"] else None

    train_ds = SmokeDataset(
        smoke_dir / "train_sub.txt",
        train=True,
        crop=512,
        multi_scale=flags["multi_scale"],
        photometric_on=flags["photometric"],
        shadow_aug=flags.get("shadow_aug", False),
        cc_bank=cc_bank,
        mosaic=flags["mosaic"],
        seed=seed,
    )
    val_ds = SmokeDataset(smoke_dir / "val_sub.txt", train=False, crop=512, seed=seed)

    sample_generator = torch.Generator().manual_seed(seed)
    loader_generator = torch.Generator().manual_seed(seed + 1)
    if flags["rcs"]:
        rarity = json.loads((smoke_dir / "rarity.json").read_text(encoding="utf-8"))
        if flags["crater_boost"]:
            print("[rcs] crater_boost: reweighting sampler toward crater pixels", flush=True)
            w_np = crater_boost_sampler_weights(rarity["items"], boost=2.0)
            weights = torch.tensor(w_np, dtype=torch.double)
        else:
            weights = torch.tensor(rarity["weights"], dtype=torch.double)
        assert len(weights) == len(train_ds)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True, generator=sample_generator)
        train_loader = DataLoader(
            train_ds, batch_size=micro_batch, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True, generator=loader_generator
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=micro_batch, shuffle=True, num_workers=2, pin_memory=True, drop_last=True, generator=loader_generator
        )
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and mem_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(float(mem_fraction), 0)

    model = build_segformer_b2(pretrain).to(device)
    if flags.get("dysample"):
        model.method_upsampler = DySample().to(device)
    if flags.get("sapa"):
        model.method_upsampler = SAPALite().to(device)
    fused_features = {}
    feature_hook = None
    if flags.get("ctrast"):
        # SegformerHead.fusion_conv output is the fused feature before cls_seg.
        feature_hook = model.decode_head.fusion_conv.register_forward_hook(
            lambda module, inputs, output: fused_features.update(feature=output))
    edge_aux = EdgeAux().to(device) if flags["eps"] else None
    row_attn = RowHeightAttn().to(device) if flags["row_attn"] else None
    fade = FadeLite().to(device) if flags.get("fade") else None
    ema_model = copy.deepcopy(model) if flags["ema"] else None
    if ema_model is not None:
        for p in ema_model.parameters():
            p.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    if edge_aux is not None:
        params += list(edge_aux.parameters())
    if row_attn is not None:
        params += list(row_attn.parameters())
    if fade is not None:
        params += list(fade.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    if flags["acw"]:
        beta = 0.5
        freq = np.clip(class_freq, 1e-6, None)
        cw = (1.0 / np.power(freq, beta)).astype(np.float32)
        cw = cw / cw.mean()
        if flags["crater_boost"]:
            cw[2] *= 1.75  # crater class ACW boost
            cw = cw / cw.mean()
            print(f"ACW+crater_boost weights={cw.tolist()}", flush=True)
        else:
            print(f"ACW weights={cw.tolist()}", flush=True)
        class_weights = torch.tensor(cw, device=device)
    else:
        class_weights = None

    if dry_run:
        iters = min(max(iters, 1), 5)

    peak_mem = 0.0
    it = 0
    t0 = time.time()
    best = -1.0
    best_ious = None
    best_rates = None
    best_boundary = None
    last_boundary = None
    last_ious = None
    last_rates = None
    error = None
    model.train()
    if row_attn is not None:
        row_attn.train()
    if fade is not None:
        fade.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    micro_count = 0
    opt.zero_grad(set_to_none=True)
    try:
        while it < iters:
            for batch_data in train_loader:
                imgs, masks, edges = batch_data[0], batch_data[1], batch_data[2]
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                edges = edges.to(device, non_blocking=True)
                fused_features.clear()
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    logits = _forward_logits(model, imgs, row_attn=row_attn, fade=fade)
                    if logits.shape[-2:] != masks.shape[-2:]:
                        logits = F.interpolate(
                            logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                        )
                    ce_logits = logits
                    if flags["blv"]:
                        ce_logits = blv_perturb_logits(logits, class_freq, sigma=blv_sigma)
                    ce = F.cross_entropy(
                        ce_logits.float(), masks, weight=class_weights, ignore_index=IGNORE
                    )
                    if flags.get("hard"):
                        ce = hardness_ce(ce_logits, masks, class_weights, alpha=1., gamma=2.)
                    dsc = dice_loss(logits, masks)
                    loss = ce + dsc
                    contrastive = logits.new_zeros(())
                    if flags.get("ctrast"):
                        contrastive = boundary_contrast_loss(fused_features["feature"], masks, edges)
                        loss = loss + .1 * contrastive
                    if flags["eps"] and edge_aux is not None:
                        e_logits = edge_aux(logits)
                        valid = masks != IGNORE
                        eg = edges.unsqueeze(1)
                        e_loss_map = F.binary_cross_entropy_with_logits(
                            e_logits.float(), eg.float(), reduction="none"
                        ).squeeze(1)
                        e_loss = e_loss_map[valid].mean() if valid.any() else logits.new_zeros(())
                        loss = loss + 0.4 * e_loss
                    if flags.get("abl"):
                        # weight ramp after 20% iters
                        ramp = min(1.0, float(it + 1) / max(1.0, 0.2 * float(iters)))
                        a_loss = abl_lite_loss(logits, masks, edges)
                        loss = loss + float(abl_weight) * ramp * a_loss
                (loss / accum_steps).backward()
                fused_features.clear()
                micro_count += 1
                if micro_count % accum_steps:
                    continue
                opt.step()
                opt.zero_grad(set_to_none=True)
                if flags["ema"] and ema_model is not None:
                    update_ema(ema_model, model, 0.999)

                it += 1
                if device.type == "cuda":
                    mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    peak_mem = max(peak_mem, mem)
                else:
                    mem = 0.0

                if it % 50 == 0 or it == 1 or dry_run:
                    msg = {
                        "iter": it,
                        "loss": float(loss.item()),
                        "ce": float(ce.item()),
                        "dice": float(dsc.item()),
                        "contrastive": float(contrastive.item()) if flags.get("ctrast") else None,
                        "micro_batch": micro_batch, "accum_steps": accum_steps,
                        "elapsed_s": round(time.time() - t0, 1),
                        "mem_mb": round(mem, 1),
                    }
                    print(msg, flush=True)
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(msg) + "\n")

                do_eval = (not dry_run) and (it % val_every == 0 or it == iters)
                if do_eval:
                    boundary_eval = {}
                    miou, ious, _cm, rates = evaluate_detailed(
                        model,
                        val_loader,
                        device,
                        use_ema=ema_model if flags["ema"] else None,
                        row_attn=row_attn,
                        fade=fade,
                        metrics_out=boundary_eval,
                    )
                    last_boundary = boundary_eval
                    last_ious = ious
                    last_rates = rates
                    row = {
                        "iter": it,
                        "val_mIoU_scene5": jfloat(miou),
                        "per_class_IoU": {n: jfloat(v) for n, v in zip(CLASS_NAMES, ious)},
                        "confusion_rates": {k: jfloat(v) for k, v in rates.items()},
                        "boundary_metrics": boundary_eval,
                        "note": "本地 Scene5 子集验证，非官方测试成绩",
                    }
                    print(row, flush=True)
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    ckpt = {
                        "model": model.state_dict(),
                        "iter": it,
                        "miou": miou,
                        "flags": flags,
                    }
                    if row_attn is not None:
                        ckpt["row_attn"] = row_attn.state_dict()
                    if fade is not None:
                        ckpt["fade"] = fade.state_dict()
                    torch.save(ckpt, work / f"iter_{it}.pth")
                    if miou > best:
                        best = miou
                        best_ious = ious
                        best_rates = rates
                        best_boundary = boundary_eval
                        torch.save(ckpt, work / "best.pth")
                    model.train()
                    if row_attn is not None:
                        row_attn.train()
                    if fade is not None:
                        fade.train()

                if it >= iters:
                    break
    except Exception:
        error = traceback.format_exc()
        print("ERROR", error, flush=True)

    if feature_hook is not None:
        feature_hook.remove()
    fused_features.clear()
    wall = time.time() - t0

    def iou_dict(ious):
        return {n: jfloat(v) for n, v in zip(CLASS_NAMES, ious)} if ious else None

    method_note = {
        "V_B1C": "B1 + crater RCS/ACW boost (control)",
        "M_ROW": "V_B1C + RowHeightAttn (HANet-style)",
        "M_BLV": "V_B1C + BLV logit variation (train-only)",
        "M_SHAD": "V_B1C + shadow_augment (train-only)",
        "M_ABL": "V_B1C + ABL-lite boundary BCE/Dice (train-only)",
        "M_INVFORM": "stub → ABL-lite fallback (no InverseForm net)",
        "M_FADE": "V_B1C + FADE-lite (logits⊕backbone-s0 refine); gate after M_TTA",
        "V_B1E": "B1 + edge aux",
        "M_DYSAMPLE": "DySample lp x4 groups4 scope0.25; final 5 logits padded to8 then sliced",
        "M_HARD": "Instantaneous detached hardness weighting alpha1 gamma2; no EMA map",
        "M_CTRAST": "Contextrast-lite: fused-feature local boundary-pair InfoNCE temp0.1 weight0.1 cap512; no context bank",
        "M_SAPA": "SAPA-B-lite 5x5 embedding32; interpolated stage1 guidance; final logits only; checkpointed chunks",
    }.get(exp.upper(), "")

    summary = {
        "exp": exp,
        "run_name": f"smoke_{exp}{'_dryrun' if dry_run else ''}",
        "iters": int(it),
        "target_iters": int(iters),
        "best_mIoU_scene5": jfloat(best) if best >= 0 else None,
        "last_mIoU_scene5": jfloat(np.nanmean(last_ious)) if last_ious else None,
        "per_class_IoU_best": iou_dict(best_ious),
        "per_class_IoU_last": iou_dict(last_ious),
        "boundary_metrics_best": best_boundary,
        "boundary_metrics_last": last_boundary,
        "iou_rock": jfloat(best_ious[1]) if best_ious else None,
        "iou_crater": jfloat(best_ious[2]) if best_ious else None,
        "confusion_rates": {k: jfloat(v) for k, v in (best_rates or last_rates or {}).items()}
        if (best_rates or last_rates)
        else None,
        "peak_mem_mb": float(round(peak_mem, 1)),
        "wall_sec": float(round(wall, 1)),
        "batch": batch,
        "lr": lr,
        "world_size": 1,
        "flags": flags,
        "blv_sigma": blv_sigma if flags["blv"] else None,
        "subset": {"train": meta["train_n"], "val": meta["val_n"], "seed": meta.get("seed", SEED)},
        "dry_run": dry_run,
        "error": error,
        "method_note": method_note,
        "note": "本地 Scene5 子集验证，非官方测试成绩；未使用 Moon_2/4/9 标签",
    }
    (work / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DONE", summary, flush=True)
    return summary


def run_tta_exp(
    project: Path,
    smoke_dir: Path,
    pretrain: Path,
    iters: int,
    batch: int,
    lr: float,
    dry_run: bool = False,
    mem_fraction: float | None = None,
    blv_sigma: float = 0.5,
    tta_scales=(0.75, 1.0, 1.25),
    tta_crop: int = 512,
    tta_stride: int = 256,
    ckpt_override: str | None = None,
    exp_name: str = "M_TTA",
    require_ckpt: bool = False,
    default_ckpt: Path | None = None,
    auto_train_vb1c: bool = True,
    accum_steps: int = 1,
    val_every: int = 500,
    seed: int = SEED,
):
    """TTA diagnostic: single-scale vs multi-scale sliding window on a fixed ckpt.

    M_TTA: default smoke_V_B1C/best.pth; auto-trains V_B1C if missing (unless require).
    T_B1_TTA: default work_dirs/train_B1/best.pth; fail clearly if missing (no 20k train).
    """
    exp = exp_name
    work = project / "work_dirs" / f"smoke_{exp}{'_dryrun' if dry_run else ''}"
    work.mkdir(parents=True, exist_ok=True)

    if ckpt_override:
        ckpt_path = Path(ckpt_override).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = (project / ckpt_path).resolve()
        else:
            ckpt_path = ckpt_path.resolve()
    else:
        ckpt_path = Path(default_ckpt) if default_ckpt is not None else (
            project / "work_dirs" / "smoke_V_B1C" / "best.pth"
        )

    trained_here = False
    if not ckpt_path.exists():
        if require_ckpt or not auto_train_vb1c:
            summary = {
                "exp": exp,
                "run_name": f"smoke_{exp}{'_dryrun' if dry_run else ''}",
                "error": (
                    f"checkpoint missing: {ckpt_path}. "
                    "T_B1_TTA will not auto-train 20k; place best.pth or pass --ckpt."
                    if exp.upper() == "T_B1_TTA" or require_ckpt
                    else f"checkpoint missing: {ckpt_path}"
                ),
                "flags": recipe_flags(exp),
                "method_note": f"TTA blocked — missing {ckpt_path}",
                "note": "本地 Scene5；未使用 Moon_2/4/9 标签",
            }
            (work / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print("DONE", summary, flush=True)
            return summary
        print(
            f"[{exp}] missing {ckpt_path}; training V_B1C first (iters={iters})",
            flush=True,
        )
        run_exp(
            "V_B1C",
            project,
            smoke_dir,
            pretrain,
            iters=iters,
            batch=batch,
            lr=lr,
            dry_run=dry_run,
            mem_fraction=mem_fraction,
            blv_sigma=blv_sigma,
            accum_steps=accum_steps, val_every=val_every, seed=seed,
        )
        trained_here = True
        ckpt_path = project / "work_dirs" / "smoke_V_B1C" / "best.pth"
        if not ckpt_path.exists():
            cands = sorted((project / "work_dirs" / "smoke_V_B1C").glob("iter_*.pth"))
            if cands:
                ckpt_path = cands[-1]
            else:
                summary = {
                    "exp": exp,
                    "error": f"V_B1C checkpoint still missing after train: {ckpt_path}",
                    "note": f"{exp} blocked",
                }
                (work / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                return summary

    meta = json.loads((smoke_dir / "meta.json").read_text(encoding="utf-8"))
    val_ds = SmokeDataset(smoke_dir / "val_sub.txt", train=False, crop=512)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    pairs = load_pairs(smoke_dir / "val_sub.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and mem_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(float(mem_fraction), 0)

    model = build_segformer_b2(pretrain).to(device)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    # TTA recipes use plain SegFormer (no RowHeightAttn / Fade) unless present in ckpt
    fade = None
    if isinstance(ckpt, dict) and "fade" in ckpt:
        fade = FadeLite().to(device)
        fade.load_state_dict(ckpt["fade"], strict=False)
        print(f"[{exp}] loaded FadeLite from ckpt", flush=True)
    print(f"[{exp}] loaded {ckpt_path}", flush=True)

    t0 = time.time()
    error = None
    try:
        if dry_run:
            pairs_eval = pairs[:2]
            max_images = 2
        else:
            pairs_eval = pairs
            max_images = None
        single_boundary = {}
        miou_s, ious_s, _cm_s, rates_s = evaluate_detailed(
            model, val_loader, device, fade=fade, metrics_out=single_boundary
        )
        print(
            {"single_mIoU": jfloat(miou_s), "per_class": dict(zip(CLASS_NAMES, ious_s))},
            flush=True,
        )
        miou_t, ious_t, _cm_t, rates_t, tta_meta = evaluate_tta(
            model,
            pairs_eval,
            device,
            scales=tta_scales,
            crop=tta_crop,
            stride=tta_stride,
            max_images=max_images,
            fade=fade,
        )
        print(
            {"tta_mIoU": jfloat(miou_t), "per_class": dict(zip(CLASS_NAMES, ious_t)), **tta_meta},
            flush=True,
        )
    except Exception:
        error = traceback.format_exc()
        print("ERROR", error, flush=True)
        miou_s = ious_s = rates_s = miou_t = ious_t = rates_t = tta_meta = None

    wall = time.time() - t0
    peak_mem = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    def iou_dict(ious):
        return {n: jfloat(v) for n, v in zip(CLASS_NAMES, ious)} if ious else None

    summary = {
        "exp": exp,
        "run_name": f"smoke_{exp}{'_dryrun' if dry_run else ''}",
        "iters": 0,
        "target_iters": 0,
        "best_mIoU_scene5": jfloat(miou_t) if miou_t is not None else jfloat(miou_s),
        "per_class_IoU_best": iou_dict(ious_t) if ious_t is not None else iou_dict(ious_s),
        "iou_rock": jfloat(ious_t[1]) if ious_t else (jfloat(ious_s[1]) if ious_s else None),
        "iou_crater": jfloat(ious_t[2]) if ious_t else (jfloat(ious_s[2]) if ious_s else None),
        "confusion_rates": {k: jfloat(v) for k, v in (rates_t or rates_s or {}).items()}
        if (rates_t or rates_s)
        else None,
        "boundary_metrics": tta_meta if isinstance(tta_meta, dict) else None,
        "tta_vs_single": {
            "single_boundary_metrics": locals().get("single_boundary"),
            "tta_boundary_metrics": tta_meta,
            "single_mIoU": jfloat(miou_s),
            "single_per_class": iou_dict(ious_s),
            "single_confusion_rates": {k: jfloat(v) for k, v in (rates_s or {}).items()}
            if rates_s
            else None,
            "tta_mIoU": jfloat(miou_t),
            "tta_per_class": iou_dict(ious_t),
            "tta_confusion_rates": {k: jfloat(v) for k, v in (rates_t or {}).items()}
            if rates_t
            else None,
            "scales": list(tta_scales),
            "crop": tta_crop,
            "stride": tta_stride,
            "small_object_note": (tta_meta or {}).get("small_object_note")
            if isinstance(tta_meta, dict)
            else "portable stack has no size-bucket helper; skipped",
            "ckpt": str(ckpt_path),
            "trained_V_B1C_here": trained_here,
        },
        "peak_mem_mb": float(round(peak_mem, 1)),
        "wall_sec": float(round(wall, 1)),
        "batch": batch,
        "lr": lr,
        "world_size": 1,
        "flags": recipe_flags(exp),
        "subset": {"train": meta["train_n"], "val": meta["val_n"], "seed": meta.get("seed", SEED)},
        "dry_run": dry_run,
        "error": error,
        "method_note": (
            "TTA on train_B1 best (no retrain)"
            if exp.upper() == "T_B1_TTA"
            else "TTA diagnostic on V_B1C weights (no retrain of M_TTA itself)"
        ),
        "note": "本地 Scene5 子集验证，非官方测试成绩；未使用 Moon_2/4/9 标签",
    }
    (work / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DONE", summary, flush=True)
    return summary


@torch.no_grad()
def run_segfix_exp(
    project: Path,
    smoke_dir: Path,
    pretrain: Path,
    dry_run: bool = False,
    mem_fraction: float | None = None,
    ckpt_override: str | None = None,
):
    """M_SEGFIX: eval-only SegFix-lite postprocess; report before/after on Scene5 val_sub."""
    exp = "M_SEGFIX"
    work = project / "work_dirs" / f"smoke_{exp}{'_dryrun' if dry_run else ''}"
    work.mkdir(parents=True, exist_ok=True)

    if ckpt_override:
        ckpt_path = Path(ckpt_override).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = (project / ckpt_path).resolve()
        else:
            ckpt_path = ckpt_path.resolve()
    else:
        ckpt_path = project / "work_dirs" / "smoke_V_B1C" / "best.pth"
    if not ckpt_path.exists():
        summary = {
            "exp": exp,
            "run_name": f"smoke_{exp}{'_dryrun' if dry_run else ''}",
            "error": f"checkpoint missing: {ckpt_path}. Train V_B1C first or pass --ckpt.",
            "flags": recipe_flags(exp),
            "method_note": "SegFix-lite blocked — missing ckpt",
            "note": "本地 Scene5；未使用 Moon_2/4/9 标签",
        }
        (work / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("DONE", summary, flush=True)
        return summary

    meta = json.loads((smoke_dir / "meta.json").read_text(encoding="utf-8"))
    pairs = load_pairs(smoke_dir / "val_sub.txt")
    if dry_run:
        pairs = pairs[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and mem_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(float(mem_fraction), 0)

    model = build_segformer_b2(pretrain).to(device)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    fade = None
    row_attn = None
    if isinstance(ckpt, dict) and "fade" in ckpt:
        fade = FadeLite().to(device)
        fade.load_state_dict(ckpt["fade"], strict=False)
        fade.eval()
    if isinstance(ckpt, dict) and "row_attn" in ckpt:
        row_attn = RowHeightAttn().to(device)
        row_attn.load_state_dict(ckpt["row_attn"], strict=False)
        row_attn.eval()
    print(f"[M_SEGFIX] loaded {ckpt_path}", flush=True)

    def _metrics(inter, union, cm):
        iou = [
            float(inter[c] / union[c]) if union[c] > 0 else float("nan")
            for c in range(NUM_CLASSES)
        ]
        valid = [x for x in iou if x == x]
        miou = float(np.mean(valid)) if valid else float("nan")
        from _lusnar_common import confusion_rates

        return miou, iou, confusion_rates(cm)

    boundary_b, boundary_a = BoundaryMeter(), BoundaryMeter()
    inter_b = np.zeros(NUM_CLASSES, np.float64)
    union_b = np.zeros(NUM_CLASSES, np.float64)
    cm_b = np.zeros((NUM_CLASSES, NUM_CLASSES), np.float64)
    inter_a = np.zeros(NUM_CLASSES, np.float64)
    union_a = np.zeros(NUM_CLASSES, np.float64)
    cm_a = np.zeros((NUM_CLASSES, NUM_CLASSES), np.float64)

    t0 = time.time()
    error = None
    n_done = 0
    try:
        for ip, lp in pairs:
            img = cv2.imread(ip, cv2.IMREAD_COLOR)
            lab = cv2.imread(lp, cv2.IMREAD_COLOR)
            if img is None or lab is None:
                continue
            mask = rgb_label_to_index(lab)
            # center-crop 512 protocol matching SmokeDataset val
            h, w = img.shape[:2]
            crop = 512
            y0 = max(0, (h - crop) // 2)
            x0 = max(0, (w - crop) // 2)
            img_c = img[y0 : y0 + crop, x0 : x0 + crop]
            mask_c = mask[y0 : y0 + crop, x0 : x0 + crop]
            if img_c.shape[0] != crop or img_c.shape[1] != crop:
                pad = np.zeros((crop, crop, 3), np.uint8)
                mpad = np.full((crop, crop), IGNORE, np.uint8)
                pad[: img_c.shape[0], : img_c.shape[1]] = img_c
                mpad[: mask_c.shape[0], : mask_c.shape[1]] = mask_c
                img_c, mask_c = pad, mpad
            img_f = (img_c.astype(np.float32) - MEAN) / STD
            t = torch.from_numpy(img_f.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                from _lusnar_common import _forward_logits

                logits = _forward_logits(model, t, row_attn=row_attn, fade=fade)
                if logits.shape[-2:] != (crop, crop):
                    logits = F.interpolate(
                        logits, size=(crop, crop), mode="bilinear", align_corners=False
                    )
            probs = torch.softmax(logits.float(), dim=1)[0]
            conf = probs.max(0).values.cpu().numpy()
            pred = probs.argmax(0).cpu().numpy().astype(np.int64)
            pred_fix = segfix_lite(pred, conf=conf, k=5)
            boundary_b.update(pred, mask_c)
            boundary_a.update(pred_fix, mask_c)

            valid = mask_c != IGNORE
            for c in range(NUM_CLASSES):
                g = mask_c == c
                pb = (pred == c) & valid
                pa = (pred_fix == c) & valid
                inter_b[c] += np.logical_and(pb, g).sum()
                union_b[c] += np.logical_or(pb, g).sum()
                inter_a[c] += np.logical_and(pa, g).sum()
                union_a[c] += np.logical_or(pa, g).sum()
            pv = pred[valid].ravel()
            gv = mask_c[valid].ravel()
            ok = (gv >= 0) & (gv < NUM_CLASSES) & (pv >= 0) & (pv < NUM_CLASSES)
            cm_b += np.bincount(
                pv[ok] * NUM_CLASSES + gv[ok], minlength=NUM_CLASSES * NUM_CLASSES
            ).reshape(NUM_CLASSES, NUM_CLASSES)
            pv2 = pred_fix[valid].ravel()
            ok2 = (gv >= 0) & (gv < NUM_CLASSES) & (pv2 >= 0) & (pv2 < NUM_CLASSES)
            cm_a += np.bincount(
                pv2[ok2] * NUM_CLASSES + gv[ok2], minlength=NUM_CLASSES * NUM_CLASSES
            ).reshape(NUM_CLASSES, NUM_CLASSES)
            n_done += 1
    except Exception:
        error = traceback.format_exc()
        print("ERROR", error, flush=True)

    miou_b, ious_b, rates_b = _metrics(inter_b, union_b, cm_b)
    miou_a, ious_a, rates_a = _metrics(inter_a, union_a, cm_a)
    wall = time.time() - t0
    peak_mem = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    def iou_dict(ious):
        return {n: jfloat(v) for n, v in zip(CLASS_NAMES, ious)} if ious else None

    summary = {
        "exp": exp,
        "run_name": f"smoke_{exp}{'_dryrun' if dry_run else ''}",
        "iters": 0,
        "target_iters": 0,
        "best_mIoU_scene5": jfloat(miou_a),
        "per_class_IoU_best": iou_dict(ious_a),
        "iou_rock": jfloat(ious_a[1]) if ious_a else None,
        "iou_crater": jfloat(ious_a[2]) if ious_a else None,
        "confusion_rates": {k: jfloat(v) for k, v in rates_a.items()},
        "boundary_metrics": boundary_a.compute(),
        "segfix_before_after": {
            "before_boundary_metrics": boundary_b.compute(),
            "after_boundary_metrics": boundary_a.compute(),
            "before_mIoU": jfloat(miou_b),
            "before_per_class": iou_dict(ious_b),
            "before_confusion_rates": {k: jfloat(v) for k, v in rates_b.items()},
            "after_mIoU": jfloat(miou_a),
            "after_per_class": iou_dict(ious_a),
            "after_confusion_rates": {k: jfloat(v) for k, v in rates_a.items()},
            "delta_mIoU": jfloat(miou_a - miou_b) if (miou_a == miou_a and miou_b == miou_b) else None,
            "ckpt": str(ckpt_path),
            "n_images": n_done,
            "k": 5,
        },
        "peak_mem_mb": float(round(peak_mem, 1)),
        "wall_sec": float(round(wall, 1)),
        "flags": recipe_flags(exp),
        "subset": {"train": meta["train_n"], "val": meta["val_n"], "seed": meta.get("seed", SEED)},
        "dry_run": dry_run,
        "error": error,
        "method_note": "SegFix-lite postprocess on V_B1C (or --ckpt); no retrain",
        "note": "本地 Scene5 子集验证，非官方测试成绩；未使用 Moon_2/4/9 标签",
    }
    (work / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DONE", summary, flush=True)
    return summary


METHOD_EXPS = (
    "V_B1C",
    "M_ROW",
    "M_BLV",
    "M_SHAD",
    "M_ABL",
    "M_INVFORM",
    "M_FADE",
    "M_TTA",
    "M_SEGFIX",
    "T_B1_TTA",
    "M_HRDA",
    "M_DYSAMPLE", "M_MSINFER", "M_HARD", "M_CTRAST", "M_SAPA",
)
ALL_EXPS = (
    "A0",
    "A1",
    "B1",
    "C1",
    "V_B1C",
    "V_B1E",
    *METHOD_EXPS[1:],  # avoid dup V_B1C
)

def refresh_report(project: Path):
    rows = []
    for exp in ("A0", "A1", "B1", "C1"):
        p = project / "work_dirs" / f"smoke_{exp}" / "summary.json"
        if p.exists():
            rows.append(json.loads(p.read_text(encoding="utf-8")))
    out = project / "results" / "smoke_matrix_report.md"
    if rows:
        write_markdown_table(out, rows, "LuSNAR smoke matrix (Scene5 local)")
        print(f"[report] wrote {out}", flush=True)

    method_rows = []
    for exp in METHOD_EXPS:
        p = project / "work_dirs" / f"smoke_{exp}" / "summary.json"
        if p.exists():
            method_rows.append(json.loads(p.read_text(encoding="utf-8")))
    if method_rows:
        mout = project / "results" / "smoke_method_eval.md"
        write_method_eval_md(mout, method_rows)
        print(f"[report] wrote {mout}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "LuSNAR smoke ablation + method smokes. Scene5 val only. "
            "Recipes: A0/A1/B1/C1 + V_B1C/V_B1E + M_ROW/M_BLV/M_SHAD/M_ABL/"
            "M_INVFORM/M_FADE/M_DYSAMPLE/M_HARD/M_CTRAST/M_SAPA + "
            "M_MSINFER/M_TTA/M_SEGFIX/T_B1_TTA; M_HRDA stub."
        )
    )
    ap.add_argument("--project-root", default=None, help="project root (default: ../.. from script)")
    ap.add_argument(
        "--data-root",
        default=None,
        help="data root. May be folder with train/Moon_* OR folder with Moon_* directly "
        "(cheerhin: D:\\LuSNAR_Seg\\data\\raw\\LuSNAR_train)",
    )
    ap.add_argument(
        "--train-subdir",
        default="train",
        help="subdir under data-root when Moon_* not directly under data-root",
    )
    ap.add_argument("--smoke-dir", default=None, help="default <project>/data/smoke")
    ap.add_argument("--pretrain", default=None, help="relative to project or absolute")
    ap.add_argument(
        "--exp",
        default=None,
        choices=list(ALL_EXPS),
        help="single experiment (overrides --exps)",
    )
    ap.add_argument(
        "--exps",
        default="A0,A1",
        help=(
            "comma list. Method smoke e.g. "
            "V_B1C,M_ROW,M_BLV,M_SHAD,M_ABL,M_SEGFIX,T_B1_TTA,M_FADE. "
            "Default A0,A1."
        ),
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=2000,
        help="default 2000 for method smoke; use 200 for quick verify",
    )
    ap.add_argument("--batch", type=int, default=16, help="effective batch; method protocol requires16")
    ap.add_argument("--accum-steps", type=int, default=1, help="micro-batch=batch/accum; optimizer iterations unchanged")
    ap.add_argument("--val-every", type=int, default=500, help="validate every N optimizer steps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--blv-sigma", type=float, default=0.5, help="BLV noise sigma for M_BLV")
    ap.add_argument(
        "--tta-scales",
        default="0.75,1.0,1.25",
        help="comma scales for M_TTA multi-scale sliding window",
    )
    ap.add_argument("--tta-crop", type=int, default=512)
    ap.add_argument("--tta-stride", type=int, default=256)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--force-prepare", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build model + few iters then exit (syntax/GPU verify)",
    )
    ap.add_argument(
        "--skip-prepare",
        action="store_true",
        help="reuse existing data/smoke assets (cheerhin dry-run)",
    )
    ap.add_argument(
        "--mem-fraction",
        type=float,
        default=None,
        help="optional cuda set_per_process_memory_fraction (cheerhin 16GB); omit on 4090",
    )
    ap.add_argument(
        "--ckpt",
        default=None,
        help="optional checkpoint for M_MSINFER / M_SEGFIX / M_TTA / T_B1_TTA (project-relative or absolute)",
    )
    ap.add_argument(
        "--abl-weight",
        type=float,
        default=0.5,
        help="ABL-lite loss weight for M_ABL / M_INVFORM (ramped after 20%% iters)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if args.batch < 1 or args.accum_steps < 1 or args.batch % args.accum_steps:
        raise SystemExit("--batch must be positive and divisible by positive --accum-steps")
    if args.val_every < 1 or args.iters < 1:
        raise SystemExit("--val-every and --iters must be positive")
    project = resolve_project_root(SCRIPT, args.project_root)
    data_root = resolve_data_root(project, args.data_root)
    scene_root = resolve_scene_root(data_root, args.train_subdir)
    smoke_dir = Path(args.smoke_dir).expanduser().resolve() if args.smoke_dir else (project / "data" / "smoke")
    pretrain = find_pretrain(project, args.pretrain)

    print(
        json.dumps(
            {
                "project": str(project),
                "data_root": str(data_root),
                "scene_root": str(scene_root),
                "smoke_dir": str(smoke_dir),
                "pretrain": str(pretrain),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    if not scene_root.is_dir():
        raise SystemExit(
            f"train scenes not found: {scene_root}\n"
            "Create junction/symlink data/train -> raw/LuSNAR_train, "
            "or pass --data-root .../data/raw --train-subdir LuSNAR_train"
        )

    # sanity: one pair readable
    probe = next(scene_root.glob("Moon_*/color/*.png"), None)
    if probe is None:
        raise SystemExit(f"no Moon_*/color/*.png under {scene_root}")
    print(f"[ok] probe image {probe}", flush=True)

    assets_ok = (smoke_dir / "train_sub.txt").exists() and (smoke_dir / "meta.json").exists()
    if args.dry_run and assets_ok and not args.force_prepare:
        print(f"[dry-run] reuse existing smoke assets under {smoke_dir}", flush=True)
    elif not args.skip_prepare:
        prepare_smoke_assets(scene_root, smoke_dir, force=args.force_prepare, seed=args.seed)
    elif not assets_ok:
        raise SystemExit(f"--skip-prepare but missing assets under {smoke_dir}")
    if args.prepare_only:
        return

    if args.exp:
        exps = [args.exp.upper()]
    else:
        exps = [e.strip().upper() for e in args.exps.split(",") if e.strip()]
    unknown = [e for e in exps if e not in ALL_EXPS]
    if unknown:
        raise SystemExit(f"unknown exps {unknown}; known={list(ALL_EXPS)}")
    if any(e in METHOD_EXPS for e in exps) and args.batch != 16:
        raise SystemExit("Method protocol requires effective --batch 16; use --accum-steps 2/4/8/16 for memory")
    validate_smoke_assets(smoke_dir, strict=any(e in METHOD_EXPS for e in exps), seed=args.seed)
    tta_scales = tuple(float(x) for x in args.tta_scales.split(",") if x.strip())
    failed = []
    for exp in exps:
        result = run_exp(
            exp,
            project,
            smoke_dir,
            pretrain,
            iters=args.iters,
            batch=args.batch,
            lr=args.lr,
            dry_run=args.dry_run,
            mem_fraction=args.mem_fraction,
            blv_sigma=args.blv_sigma,
            tta_scales=tta_scales,
            tta_crop=args.tta_crop,
            tta_stride=args.tta_stride,
            ckpt=args.ckpt,
            abl_weight=args.abl_weight,
            accum_steps=args.accum_steps, val_every=args.val_every, seed=args.seed,
        )
        if result.get("error"):
            failed.append(exp)
        refresh_report(project)
    if failed:
        raise SystemExit(f"Failed experiments: {failed}; see summary.json (or *_dryrun/summary.json)")


def validate_smoke_assets(smoke_dir, strict=True, seed=SEED):
    """Validate cached provenance before any label image is opened."""
    import re
    def scene(path):
        match = re.search(r"(?:^|[/\\])Moon_(\d+)(?:[/\\]|$)", str(path), re.I)
        return int(match.group(1)) if match else None
    train = load_pairs(smoke_dir / "train_sub.txt")
    val = load_pairs(smoke_dir / "val_sub.txt")
    for pairs, allowed in ((train, {1, 3, 6, 7, 8}), (val, {5})):
        for ip, lp in pairs:
            if scene(lp) not in allowed or scene(ip) != scene(lp):
                raise ValueError(f"Invalid smoke split provenance: {ip}, {lp}; allowed scenes={allowed}")
    meta = json.loads((smoke_dir / "meta.json").read_text(encoding="utf-8"))
    if strict and (len(train) != TRAIN_N or len(val) != VAL_N or meta.get("seed") != seed):
        raise ValueError(f"Protocol assets must be train1600/val350/seed{seed}; use --force-prepare")
    if meta["train_n"] != len(train) or meta["val_n"] != len(val):
        raise ValueError("Cached meta/list sizes disagree; regenerate smoke assets")
    rarity = json.loads((smoke_dir / "rarity.json").read_text(encoding="utf-8"))
    if [(r["img"], r["lab"]) for r in rarity["items"]] != train:
        raise ValueError("Cached rarity items do not match train subset; use --force-prepare")
    return meta


def run_msinfer_exp(project, smoke_dir, pretrain, dry_run=False, mem_fraction=None, ckpt_override=None):
    """Evaluation only. Missing control never triggers implicit training."""
    ckpt_path = Path(ckpt_override) if ckpt_override else Path("work_dirs/smoke_V_B1C/best.pth")
    if not ckpt_path.is_absolute():
        ckpt_path = project / ckpt_path
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"M_MSINFER requires cheerhin V_B1C best: {ckpt_path}; train control first or --ckpt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and mem_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(float(mem_fraction), 0)
    model = build_segformer_b2(pretrain).to(device)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    pairs = load_pairs(smoke_dir / "val_sub.txt")
    if dry_run:
        pairs = pairs[:2]
    t0 = time.perf_counter()
    modes = evaluate_inference_modes(model, pairs, device, crop=512, stride=256)
    single = modes["whole_1.0"]
    return {"exp": "M_MSINFER", "run_name": "smoke_M_MSINFER", "iters": 0, "target_iters": 0,
            "best_mIoU_scene5": single["mIoU"], "per_class_IoU_best": single["per_class_IoU"],
            "boundary_metrics": single["boundary_metrics"], "inference_modes": modes,
            "ckpt": str(ckpt_path.resolve()), "wall_sec": time.perf_counter() - t0,
            "flags": recipe_flags("M_MSINFER"), "dry_run": dry_run, "error": None,
            "method_note": "No training. Four independent full-image modes; not comparable to center-crop val scores.",
            "note": "Scene5 val_sub350 only; no Moon_2/4/9 labels"}


def run_exp(exp, project, smoke_dir, pretrain, iters, batch, lr, dry_run=False,
            mem_fraction=None, blv_sigma=.5, tta_scales=(.75, 1., 1.25),
            tta_crop=512, tta_stride=256, ckpt=None, abl_weight=.5,
            accum_steps=1, val_every=500, seed=SEED):
    """Protocol/summary wrapper; legacy callable signature kept, extras optional.

    Auto-OOM retry is deliberately not supported: regenerate the entire local
    control+method comparison with the SAME manual accumulation factor instead.
    """
    import importlib.metadata
    if batch < 1 or accum_steps < 1 or batch % accum_steps:
        raise ValueError("positive batch must be divisible by accum_steps")
    if val_every < 1 or iters < 1:
        raise ValueError("positive iters and val_every required")
    work = project / "work_dirs" / f"smoke_{exp}{'_dryrun' if dry_run else ''}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        mmseg_version = importlib.metadata.version("mmsegmentation")
    except importlib.metadata.PackageNotFoundError:
        mmseg_version = "unavailable"
    config = {"model": "SegFormer-B2", "mmseg_version": mmseg_version,
              "control": "V_B1C@2k (same machine and accumulation)", "train_scenes": [1, 3, 6, 7, 8],
              "val_scene": 5, "train_subset": 1600, "val_subset": 350,
              "seed": seed, "crop": 512, "lr": lr, "iters": iters,
              "batch_effective": batch, "micro_batch": batch // accum_steps,
              "accum_steps": accum_steps, "val_every": val_every,
              "iteration_unit": "optimizer steps", "single_variable_ablation": True,
              "forbidden_label_scenes": [2, 4, 9], "dry_run_steps": min(iters, 5) if dry_run else None,
              "eval_protocol": "center crop512; M_MSINFER uses same full images across four modes",
              "boundary_f1_tolerance_px": 3, "boundary_iou_d_px": 3,
              "accumulation_note": ("Loss divided by accum_steps. Same effective16, but micro-batch BN and per-micro Dice/ACW normalization differ from true batch16; rerun local control with identical accumulation." if accum_steps > 1 else "No accumulation"),
              "auto_accum": False, "dry_run": dry_run}
    config["fixed_protocol_matches"] = (batch == 16 and seed == 42 and iters == 2000
                                         and lr == 6e-5 and val_every == 500 and mmseg_version == "1.2.2")
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        validate_smoke_assets(smoke_dir, strict=exp in METHOD_EXPS, seed=seed)
        summary = _run_exp_impl(exp, project, smoke_dir, pretrain, iters, batch, lr,
                               dry_run=dry_run, mem_fraction=mem_fraction, blv_sigma=blv_sigma,
                               tta_scales=tta_scales, tta_crop=tta_crop, tta_stride=tta_stride,
                               ckpt=ckpt, abl_weight=abl_weight, accum_steps=accum_steps,
                               val_every=val_every, seed=seed)
    except Exception:
        summary = {"exp": exp, "run_name": work.name, "error": traceback.format_exc(),
                   "wall_sec": time.perf_counter() - t0, "dry_run": dry_run}
        print("ERROR", summary["error"], flush=True)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.
    peak = max(peak, summary.get("peak_mem_mb", 0.))
    config["peak_mem_mb"] = round(peak, 1)
    summary.update(config=config, batch_effective=batch, micro_batch=batch // accum_steps,
                   accum_steps=accum_steps, peak_mem_mb=round(peak, 1),
                   accumulation_note=config["accumulation_note"])
    summary["dry_run"] = dry_run
    # Machine provenance prevents an accidental cross-machine control comparison.
    import platform
    summary["machine"] = {"hostname": platform.node(), "torch": torch.__version__,
                          "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    (work / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[memory] {exp}: effective={batch} micro={batch // accum_steps} accum={accum_steps} peak_mem_mb={peak:.1f}", flush=True)
    if summary.get("error") and ("out of memory" in summary["error"].lower()):
        print("[OOM] Retry the entire local control+method set with --batch 16 --accum-steps 2 (then4/8 if needed). Do not reduce effective batch.", flush=True)
    return summary


if __name__ == "__main__":
    main()
