"""Minimal SegFormer-B2 fine-tune on LuSNAR train scenes; val on Scene5.
Uses mmseg MiT-B2 + ImageNet weights; on-the-fly RGB label -> class id.
Data lists point at data/raw/LuSNAR_train/Moon_*/{color,label} (official layout, no image0/).
"""
from __future__ import annotations
import os, sys, time, json, random
from pathlib import Path

# mmcv stub before mmseg
sys.path.insert(0, r"D:\LuSNAR_Seg\tools")
import mmcv_stub  # noqa

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from mmseg.utils import register_all_modules
from mmseg.registry import MODELS
from mmengine.config import Config

register_all_modules()

ROOT = Path(r"D:\LuSNAR_Seg")
WORK = ROOT / "work_dirs" / "segformer_b2_phase1"
WORK.mkdir(parents=True, exist_ok=True)
PRETRAIN = ROOT / "pretrained" / "mit_b2_20220624-66e8bf70.pth"  # copied from torch hub cache
# fall back common locations
if not PRETRAIN.exists():
    for p in [
        ROOT / "weights" / "mit_b2_20220624-66e8bf70.pth",
        ROOT / "checkpoints" / "mit_b2_20220624-66e8bf70.pth",
        Path(r"D:\LuSNAR_Seg\pretrained\segformer\mit_b2_20220624-66e8bf70.pth"),
    ]:
        if p.exists():
            PRETRAIN = p
            break

# Official submission order (RGB)
COLOR_TO_ID = {
    (187, 70, 156): 0,   # #BB469C 月壤
    (232, 250, 80): 1,   # #E8FA50 岩石
    (120, 0, 200): 2,    # #7800C8 撞击坑
    (173, 69, 31): 3,    # #AD451F 山体
    (34, 201, 248): 4,   # #22C9F8 天空
}
BGR_TO_ID = {(c[2], c[1], c[0]): i for c, i in COLOR_TO_ID.items()}
CLASS_NAMES = ["月壤", "岩石", "撞击坑", "山体", "天空"]  # official submission order
NUM_CLASSES = 5
IGNORE = 255
CROP = 512
BATCH = 8
ITERS = 1000
LR = 6e-5
NUM_WORKERS = 2

def rgb_label_to_index(lab_bgr: np.ndarray) -> np.ndarray:
    out = np.full(lab_bgr.shape[:2], IGNORE, np.uint8)
    for bgr, cid in BGR_TO_ID.items():
        m = np.all(lab_bgr == np.asarray(bgr, np.uint8), axis=-1)
        out[m] = cid
    return out

class LuSNARList(Dataset):
    def __init__(self, list_file, train=True):
        self.pairs = []
        for line in Path(list_file).read_text(encoding="utf-8").splitlines():
            line=line.strip()
            if not line: continue
            a,b = line.split("\t") if "\t" in line else line.split()
            self.pairs.append((a,b))
        self.train = train
        self.mean = np.array([123.675, 116.28, 103.53], np.float32)
        self.std = np.array([58.395, 57.12, 57.375], np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, lp = self.pairs[idx]
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        lab = cv2.imread(lp, cv2.IMREAD_COLOR)
        if img is None or lab is None:
            raise RuntimeError(f"bad pair {ip}")
        mask = rgb_label_to_index(lab)
        # random/center crop to 512
        h, w = img.shape[:2]
        if self.train:
            y0 = random.randint(0, max(0, h - CROP))
            x0 = random.randint(0, max(0, w - CROP))
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
                x0 = w - CROP - x0 if w >= CROP else 0
        else:
            y0 = max(0, (h - CROP)//2)
            x0 = max(0, (w - CROP)//2)
        img = img[y0:y0+CROP, x0:x0+CROP]
        mask = mask[y0:y0+CROP, x0:x0+CROP]
        # pad if needed
        if img.shape[0] != CROP or img.shape[1] != CROP:
            pad = np.zeros((CROP, CROP, 3), np.uint8)
            mpad = np.full((CROP, CROP), IGNORE, np.uint8)
            pad[:img.shape[0], :img.shape[1]] = img
            mpad[:mask.shape[0], :mask.shape[1]] = mask
            img, mask = pad, mpad
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img.transpose(2,0,1)).float()
        mask = torch.from_numpy(mask.astype(np.int64))
        return img, mask, ip

def build_model():
    import mmseg
    cfg_path = Path(mmseg.__file__).parent / ".mim" / "configs" / "segformer" / "segformer_mit-b2_8xb2-160k_ade20k-512x512.py"
    cfg = Config.fromfile(str(cfg_path))
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, "to_dict") else dict(cfg.model)
    # strip preprocessor; SyncBN->BN; num_classes
    if "data_preprocessor" in model_cfg:
        model_cfg.pop("data_preprocessor")
    def fix(obj):
        if isinstance(obj, dict):
            if obj.get("type") in ("SyncBN", "mmcv.SyncBN"):
                obj["type"] = "BN"
            if "num_classes" in obj:
                obj["num_classes"] = NUM_CLASSES
            if "pretrained" in obj and obj["pretrained"]:
                # load manually
                obj["pretrained"] = None
            for v in obj.values():
                fix(v)
        elif isinstance(obj, list):
            for v in obj:
                fix(v)
    fix(model_cfg)
    # backbone init_cfg
    if "backbone" in model_cfg:
        model_cfg["backbone"]["init_cfg"] = dict(type="Pretrained", checkpoint=str(PRETRAIN))
    model = MODELS.build(model_cfg)
    model.init_weights()
    return model

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = model.extract_feat(imgs)
            logits = model.decode_head.forward(x)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1)
        for c in range(NUM_CLASSES):
            p = (pred == c)
            g = (masks == c)
            inter[c] += (p & g).sum().item()
            union[c] += (p | g).sum().item()
    iou = [(inter[c]/union[c] if union[c] > 0 else float("nan")) for c in range(NUM_CLASSES)]
    valid = [x for x in iou if x == x]
    miou = float(np.mean(valid)) if valid else float("nan")
    return miou, iou

def save_vis(model, loader, device, n=3):
    model.eval()
    palette = {
        0: (156, 70, 187),   # BGR #BB469C 月壤
        1: (80, 250, 232),   # BGR #E8FA50 岩石
        2: (200, 0, 120),    # BGR #7800C8 撞击坑
        3: (31, 69, 173),    # BGR #AD451F 山体
        4: (248, 201, 34),   # BGR #22C9F8 天空
    }
    saved = 0
    for imgs, masks, paths in loader:
        imgs_d = imgs.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = model.extract_feat(imgs_d)
            logits = model.decode_head.forward(x)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            if logits.shape[-2:] != (CROP, CROP):
                logits = F.interpolate(logits, size=(CROP,CROP), mode="bilinear", align_corners=False)
        pred = logits.argmax(1).cpu().numpy()
        for i in range(imgs.size(0)):
            # denorm
            im = imgs[i].numpy().transpose(1,2,0)
            im = im * np.array([58.395, 57.12, 57.375]) + np.array([123.675, 116.28, 103.53])
            im = np.clip(im, 0, 255).astype(np.uint8)
            gt = masks[i].numpy()
            pr = pred[i]
            def colorize(m):
                out = np.zeros((*m.shape, 3), np.uint8)
                for c, col in palette.items():
                    out[m==c] = col
                return out
            gt_c = colorize(gt)
            pr_c = colorize(pr)
            panel = np.concatenate([im, gt_c, pr_c], axis=1)
            outp = WORK / f"vis_val_{saved}.png"
            cv2.imwrite(str(outp), panel)
            saved += 1
            if saved >= n:
                return

def main():
    assert PRETRAIN.exists(), f"missing pretrain {PRETRAIN}"
    device = torch.device("cuda")
    # leave headroom for other process
    torch.cuda.set_per_process_memory_fraction(0.90, 0)

    train_ds = LuSNARList(ROOT/"data"/"train_scenes.txt", train=True)
    val_ds = LuSNARList(ROOT/"data"/"val_scene5.txt", train=False)
    print(f"train={len(train_ds)} val={len(val_ds)} pretrain={PRETRAIN}", flush=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model().to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    model.train()
    it = 0
    t0 = time.time()
    log_path = WORK / "train_log.jsonl"
    best = -1.0
    while it < ITERS:
        for imgs, masks, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                losses = model.loss(imgs, data_samples=None) if False else None
                # manual forward: extract + decode_head.loss / CE
                feats = model.extract_feat(imgs)
                # decode_head.forward
                logits = model.decode_head.forward(feats)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = F.cross_entropy(logits.float(), masks, ignore_index=IGNORE)
            loss.backward()
            opt.step()
            it += 1
            if it % 50 == 0 or it == 1:
                msg = {"iter": it, "loss": float(loss.item()), "lr": LR, "elapsed_s": round(time.time()-t0,1)}
                print(msg, flush=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(msg)+"\n")
            if it % 500 == 0 or it == ITERS:
                miou, ious = evaluate(model, val_loader, device)
                row = {"iter": it, "val_mIoU_scene5": miou, "per_class_IoU": dict(zip(CLASS_NAMES, ious))}
                print(row, flush=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row)+"\n")
                ckpt = WORK / f"iter_{it}.pth"
                torch.save({"model": model.state_dict(), "iter": it, "miou": miou}, ckpt)
                if miou > best:
                    best = miou
                    torch.save({"model": model.state_dict(), "iter": it, "miou": miou}, WORK/"best.pth")
                model.train()
            if it >= ITERS:
                break
    # final vis
    save_vis(model, val_loader, device, n=3)
    summary = {"iters": ITERS, "best_mIoU_scene5": best, "crop": CROP, "batch": BATCH, "lr": LR, "amp": "bf16",
               "train_scenes": [1,3,6,7,8], "val_scene": 5, "note": "本地 Scene5 验证，非官方测试成绩"}
    (WORK/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", summary, flush=True)

if __name__ == "__main__":
    main()




