"""LuSNAR SegFormer-B2 smoke matrix: A0 / A1 / B1 / C1.
Scene5 local val only; never touches Moon_2/4/9 labels.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time, copy, math
from pathlib import Path

sys.path.insert(0, r"D:\LuSNAR_Seg\tools")
import mmcv_stub  # noqa

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from mmseg.utils import register_all_modules
from mmseg.registry import MODELS
from mmengine.config import Config

register_all_modules()

ROOT = Path(r"D:\LuSNAR_Seg")
SMOKE = ROOT / "data" / "smoke"
PRETRAIN = ROOT / "pretrained" / "mit_b2_20220624-66e8bf70.pth"
COLOR_TO_ID = {
    (187, 70, 156): 0,
    (232, 250, 80): 1,
    (120, 0, 200): 2,
    (173, 69, 31): 3,
    (34, 201, 248): 4,
}
BGR_TO_ID = {(c[2], c[1], c[0]): i for c, i in COLOR_TO_ID.items()}
CLASS_NAMES = ["月壤", "岩石", "撞击坑", "山体", "天空"]
NUM_CLASSES = 5
IGNORE = 255
SEED = 42


def jfloat(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return float(x.item()) if hasattr(x, "item") else float(x)



def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def rgb_label_to_index(lab_bgr: np.ndarray) -> np.ndarray:
    out = np.full(lab_bgr.shape[:2], IGNORE, np.uint8)
    for bgr, cid in BGR_TO_ID.items():
        m = np.all(lab_bgr == np.asarray(bgr, np.uint8), axis=-1)
        out[m] = cid
    return out


def load_pairs(list_file: Path):
    pairs = []
    for line in Path(list_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        a, b = line.split("\t") if "\t" in line else line.split()
        pairs.append((a, b))
    return pairs


def photometric(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simple PhotoMetricDistortion on BGR uint8."""
    out = img.astype(np.float32)
    if rng.random() < 0.5:
        out += rng.uniform(-32, 32)  # brightness
    if rng.random() < 0.5:
        out *= rng.uniform(0.5, 1.5)  # contrast
    out = np.clip(out, 0, 255)
    # saturation via HSV
    if rng.random() < 0.5:
        hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= rng.uniform(0.5, 1.5)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    if rng.random() < 0.5:
        hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-18, 18)) % 180
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def dice_loss(logits, masks, ignore=IGNORE, eps=1e-6):
    # logits: B C H W, masks: B H W
    num_classes = logits.shape[1]
    valid = masks != ignore
    masks_cl = masks.clone()
    masks_cl[~valid] = 0
    probs = torch.softmax(logits.float(), dim=1)
    onehot = F.one_hot(masks_cl, num_classes).permute(0, 3, 1, 2).float()
    valid_f = valid.unsqueeze(1).float()
    probs = probs * valid_f
    onehot = onehot * valid_f
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    den = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


def edge_map_from_mask(mask: np.ndarray, ksize=5) -> np.ndarray:
    """Morphological gradient edge on valid labels; 1=edge, 0=non, ignore kept."""
    edge = np.zeros_like(mask, np.uint8)
    valid = mask != IGNORE
    if not valid.any():
        return edge
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    # per-class edges then OR; focus rare+all
    for c in range(NUM_CLASSES):
        binm = (mask == c).astype(np.uint8) * 255
        if binm.sum() == 0:
            continue
        dil = cv2.dilate(binm, kernel)
        ero = cv2.erode(binm, kernel)
        e = ((dil > 0) ^ (ero > 0)) & valid
        edge[e] = 1
    return edge


class CCBank:
    def __init__(self, index_json: Path):
        data = json.loads(index_json.read_text(encoding="utf-8"))
        self.items = []
        for cid in ("1", "2", 1, 2):
            key = str(cid)
            if key in data:
                for p in data[key]:
                    self.items.append(p)
        # dedupe
        self.items = list(dict.fromkeys(self.items))

    def sample_paste(self, img, mask, rng: random.Random, n_paste=2):
        if not self.items:
            return img, mask
        h, w = mask.shape
        out_i, out_m = img.copy(), mask.copy()
        for _ in range(n_paste):
            path = self.items[rng.randrange(len(self.items))]
            try:
                z = np.load(path)
            except Exception:
                continue
            pi, pm = z["img"], z["mask"]
            cid = int(z["cid"]) if "cid" in z else int(pm[pm != IGNORE][0]) if (pm != IGNORE).any() else 1
            if rng.random() < 0.5:
                pi = pi[:, ::-1].copy()
                pm = pm[:, ::-1].copy()
            ph, pw = pm.shape
            # find regolith location
            reg = np.argwhere(out_m == 0)
            if len(reg) < 100:
                continue
            # try a few positions
            placed = False
            for _try in range(8):
                yx = reg[rng.randrange(len(reg))]
                y0 = int(yx[0] - ph // 2)
                x0 = int(yx[1] - pw // 2)
                if y0 < 0 or x0 < 0 or y0 + ph > h or x0 + pw > w:
                    continue
                region = out_m[y0:y0+ph, x0:x0+pw]
                # only paste where currently regolith (or ignore)
                dest = (region == 0) | (region == IGNORE)
                src = pm != IGNORE
                use = dest & src
                if use.sum() < 32:
                    continue
                out_i[y0:y0+ph, x0:x0+pw][use] = pi[use]
                out_m[y0:y0+ph, x0:x0+pw][use] = cid
                placed = True
                break
            if not placed:
                continue
        return out_i, out_m


class SmokeDataset(Dataset):
    def __init__(self, list_file, train=True, exp="A0", crop=512, multi_scale=False,
                 photometric_on=False, cc_bank=None, mosaic=False, seed=SEED):
        self.pairs = load_pairs(Path(list_file))
        self.train = train
        self.exp = exp
        self.crop = crop
        self.multi_scale = multi_scale
        self.photometric_on = photometric_on
        self.cc_bank = cc_bank
        self.mosaic = mosaic
        self.mean = np.array([123.675, 116.28, 103.53], np.float32)
        self.std = np.array([58.395, 57.12, 57.375], np.float32)
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
        # 2x2 mosaic into 1024 then will crop
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
            n_paste = self.rng.randint(1, 3)
            img, mask = self.cc_bank.sample_paste(img, mask, self.rng, n_paste=n_paste)

        if self.train and self.photometric_on:
            img = photometric(img, self.rng)

        h, w = img.shape[:2]
        crop = self.crop
        if self.train and self.multi_scale:
            # scale jitter then crop to crop size
            scale = self.rng.uniform(0.75, 1.5)
            nh, nw = int(h * scale), int(w * scale)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
            h, w = nh, nw
            # sometimes use 768 crop then resize to 512 for batching
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

        img = img[y0:y0+crop, x0:x0+crop]
        mask = mask[y0:y0+crop, x0:x0+crop]
        if img.shape[0] != crop or img.shape[1] != crop:
            pad = np.zeros((crop, crop, 3), np.uint8)
            mpad = np.full((crop, crop), IGNORE, np.uint8)
            pad[:img.shape[0], :img.shape[1]] = img
            mpad[:mask.shape[0], :mask.shape[1]] = mask
            img, mask = pad, mpad

        # unify to 512 for batching when multi-scale used 768 crop
        edge = edge_map_from_mask(mask)
        if crop != 512:
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            edge = cv2.resize(edge, (512, 512), interpolation=cv2.INTER_NEAREST)

        img_f = img.astype(np.float32)
        img_f = (img_f - self.mean) / self.std
        img_t = torch.from_numpy(img_f.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask.astype(np.int64))
        edge_t = torch.from_numpy(edge.astype(np.float32))
        return img_t, mask_t, edge_t, ip


def build_model():
    import mmseg
    cfg_path = Path(mmseg.__file__).parent / ".mim" / "configs" / "segformer" / "segformer_mit-b2_8xb2-160k_ade20k-512x512.py"
    cfg = Config.fromfile(str(cfg_path))
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, "to_dict") else dict(cfg.model)
    if "data_preprocessor" in model_cfg:
        model_cfg.pop("data_preprocessor")

    def fix(obj):
        if isinstance(obj, dict):
            if obj.get("type") in ("SyncBN", "mmcv.SyncBN"):
                obj["type"] = "BN"
            if "num_classes" in obj:
                obj["num_classes"] = NUM_CLASSES
            if "pretrained" in obj and obj["pretrained"]:
                obj["pretrained"] = None
            for v in obj.values():
                fix(v)
        elif isinstance(obj, list):
            for v in obj:
                fix(v)

    fix(model_cfg)
    if "backbone" in model_cfg:
        model_cfg["backbone"]["init_cfg"] = dict(type="Pretrained", checkpoint=str(PRETRAIN))
    model = MODELS.build(model_cfg)
    model.init_weights()
    return model


class EdgeAux(nn.Module):
    """Simplified EPS: 1x1 on decode logits channels via detach feats mid — use logits side."""
    def __init__(self, in_ch=NUM_CLASSES):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, logits):
        return self.conv(logits.float())


@torch.no_grad()
def evaluate(model, loader, device, use_ema=None):
    was_training = model.training
    model.eval()
    net = use_ema if use_ema is not None else model
    net.eval()
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    for batch in loader:
        imgs, masks = batch[0], batch[1]
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = net.extract_feat(imgs)
            logits = net.decode_head.forward(x)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1)
        for c in range(NUM_CLASSES):
            p = (pred == c)
            g = (masks == c)
            inter[c] += (p & g).sum().item()
            union[c] += (p | g).sum().item()
    iou = [float(inter[c] / union[c]) if union[c] > 0 else float("nan") for c in range(NUM_CLASSES)]
    valid = [x for x in iou if x == x]
    miou = float(np.mean(valid)) if valid else float("nan")
    if was_training:
        model.train()
    return miou, iou


def update_ema(ema, model, decay=0.999):
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema.state_dict().items():
            if k in msd and v.dtype.is_floating_point:
                v.copy_(v * decay + msd[k] * (1.0 - decay))


def run_exp(exp: str, iters: int = 1000, batch: int = 8, lr: float = 6e-5):
    set_seed(SEED)
    work = ROOT / "work_dirs" / f"smoke_{exp}"
    work.mkdir(parents=True, exist_ok=True)
    log_path = work / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    assert PRETRAIN.exists(), PRETRAIN
    assert (SMOKE / "train_sub.txt").exists(), "run prepare_smoke_assets.py first"
    meta = json.loads((SMOKE / "meta.json").read_text(encoding="utf-8"))
    class_freq = np.array(meta["class_freq"], np.float64)

    use_rcs = exp in ("A1", "B1", "C1")
    use_mosaic = exp in ("A1", "B1", "C1")
    use_cc = exp in ("A1", "B1", "C1")
    use_acw = exp in ("B1", "C1")  # C1 builds on better A/B → include ACW
    use_eps = exp == "C1"
    use_ms = exp == "C1"
    use_photo = exp == "C1"
    use_ema = exp == "C1"

    cc_bank = CCBank(SMOKE / "cc_bank_index.json") if use_cc else None

    train_ds = SmokeDataset(
        SMOKE / "train_sub.txt", train=True, exp=exp, crop=512,
        multi_scale=use_ms, photometric_on=use_photo, cc_bank=cc_bank, mosaic=use_mosaic,
    )
    val_ds = SmokeDataset(SMOKE / "val_sub.txt", train=False, exp=exp, crop=512)

    if use_rcs:
        rarity = json.loads((SMOKE / "rarity.json").read_text(encoding="utf-8"))
        weights = torch.tensor(rarity["weights"], dtype=torch.double)
        assert len(weights) == len(train_ds)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.92, 0)
    model = build_model().to(device)
    edge_aux = EdgeAux().to(device) if use_eps else None
    ema_model = copy.deepcopy(model) if use_ema else None
    if ema_model is not None:
        for p in ema_model.parameters():
            p.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    if edge_aux is not None:
        params += list(edge_aux.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    if use_acw:
        # ACW: weight ∝ 1 / freq^beta
        beta = 0.5
        freq = np.clip(class_freq, 1e-6, None)
        cw = (1.0 / np.power(freq, beta)).astype(np.float32)
        cw = cw / cw.mean()
        class_weights = torch.tensor(cw, device=device)
        print(f"ACW weights={cw.tolist()}", flush=True)
    else:
        class_weights = None

    peak_mem = 0
    it = 0
    t0 = time.time()
    best = -1.0
    best_ious = None
    last_ious = None
    error = None
    model.train()

    try:
        while it < iters:
            for batch in train_loader:
                imgs, masks, edges = batch[0], batch[1], batch[2]
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                edges = edges.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    feats = model.extract_feat(imgs)
                    logits = model.decode_head.forward(feats)
                    if logits.shape[-2:] != masks.shape[-2:]:
                        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                    ce = F.cross_entropy(logits.float(), masks, weight=class_weights, ignore_index=IGNORE)
                    dsc = dice_loss(logits, masks)
                    loss = ce + dsc
                    if use_eps and edge_aux is not None:
                        e_logits = edge_aux(logits)
                        valid = masks != IGNORE
                        eg = edges.unsqueeze(1)
                        e_loss_map = F.binary_cross_entropy_with_logits(
                            e_logits.float(), eg.float(), reduction="none"
                        ).squeeze(1)
                        e_loss = e_loss_map[valid].mean() if valid.any() else logits.new_zeros(())
                        loss = loss + 0.4 * e_loss
                loss.backward()
                opt.step()
                if use_ema and ema_model is not None:
                    update_ema(ema_model, model, 0.999)

                it += 1
                mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
                if mem > peak_mem:
                    peak_mem = mem

                if it % 50 == 0 or it == 1:
                    msg = {"iter": it, "loss": float(loss.item()), "ce": float(ce.item()),
                           "dice": float(dsc.item()), "elapsed_s": round(time.time() - t0, 1),
                           "mem_mb": round(mem, 1)}
                    print(msg, flush=True)
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(msg) + "\n")

                if it % 500 == 0 or it == iters:
                    miou, ious = evaluate(model, val_loader, device, use_ema=ema_model if use_ema else None)
                    last_ious = ious
                    row = {"iter": it, "val_mIoU_scene5": jfloat(miou),
                           "per_class_IoU": {n: jfloat(v) for n, v in zip(CLASS_NAMES, ious)},
                           "note": "本地 Scene5 子集验证，非官方测试成绩"}
                    print(row, flush=True)
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    torch.save({"model": model.state_dict(), "iter": it, "miou": miou}, work / f"iter_{it}.pth")
                    if miou > best:
                        best = miou
                        best_ious = ious
                        torch.save({"model": model.state_dict(), "iter": it, "miou": miou}, work / "best.pth")
                    model.train()

                if it >= iters:
                    break
    except Exception as e:
        import traceback
        error = traceback.format_exc()
        print("ERROR", error, flush=True)

    wall = time.time() - t0
    def iou_dict(ious):
        return {n: jfloat(v) for n, v in zip(CLASS_NAMES, ious)} if ious else None

    summary = {
        "exp": exp,
        "iters": int(it),
        "target_iters": int(iters),
        "best_mIoU_scene5": jfloat(best),
        "last_mIoU_scene5": jfloat(np.nanmean(last_ious)) if last_ious else None,
        "per_class_IoU_best": iou_dict(best_ious),
        "per_class_IoU_last": iou_dict(last_ious),
        "iou_rock": jfloat(best_ious[1]) if best_ious else None,
        "iou_crater": jfloat(best_ious[2]) if best_ious else None,
        "peak_mem_mb": float(round(peak_mem, 1)),
        "wall_sec": float(round(wall, 1)),
        "batch": batch,
        "lr": lr,
        "flags": {
            "rcs": use_rcs, "mosaic": use_mosaic, "cc_paste": use_cc,
            "acw": use_acw, "eps": use_eps, "multi_scale": use_ms,
            "photometric": use_photo, "ema": use_ema,
        },
        "subset": {"train": meta["train_n"], "val": meta["val_n"], "seed": SEED},
        "error": error,
        "note": "本地 Scene5 子集验证，非官方测试成绩；未使用 Moon_2/4/9 标签",
    }
    (work / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", summary, flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["A0", "A1", "B1", "C1"])
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-5)
    args = ap.parse_args()
    run_exp(args.exp, iters=args.iters, batch=args.batch, lr=args.lr)


if __name__ == "__main__":
    main()
