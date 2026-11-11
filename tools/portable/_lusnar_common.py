"""Shared helpers for LuSNAR portable SegFormer-B2 scripts."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Official submission order (RGB)
COLOR_TO_ID = {
    (187, 70, 156): 0,  # 月壤
    (232, 250, 80): 1,  # 岩石
    (120, 0, 200): 2,  # 撞击坑
    (173, 69, 31): 3,  # 山体
    (34, 201, 248): 4,  # 天空
}
BGR_TO_ID = {(c[2], c[1], c[0]): i for c, i in COLOR_TO_ID.items()}
CLASS_NAMES = ["月壤", "岩石", "撞击坑", "山体", "天空"]
NUM_CLASSES = 5
IGNORE = 255
SEED = 42
TRAIN_SCENES = (1, 3, 6, 7, 8)
VAL_SCENE = 5
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


def jfloat(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return float(x.item()) if hasattr(x, "item") else float(x)


def set_seed(s: int = SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def ensure_mmcv_stub(script_file: Path):
    """Import mmcv_stub from portable/ or ../tools."""
    here = script_file.resolve().parent
    candidates = [here, here.parent]
    for c in candidates:
        if (c / "mmcv_stub.py").exists():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            import mmcv_stub  # noqa: F401

            return
    raise FileNotFoundError("mmcv_stub.py not found next to script or in tools/")


def resolve_project_root(script_file: Path, project_root: str | None) -> Path:
    if project_root:
        return Path(project_root).expanduser().resolve()
    # tools/portable/<script>.py -> project is parents[2]
    here = script_file.resolve().parent  # .../tools/portable
    if here.name == "portable" and here.parent.name == "tools":
        return here.parents[1]  # project root
    # fallback: two levels up from script file
    return script_file.resolve().parents[2]


def resolve_data_root(project: Path, data_root: str | None) -> Path:
    if data_root:
        p = Path(data_root).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()
        return p
    return (project / "data").resolve()


def train_dir(data_root: Path, train_subdir: str = "train") -> Path:
    return (data_root / train_subdir).resolve()


def has_moon_scenes(root: Path) -> bool:
    if not root.is_dir():
        return False
    return any(root.glob("Moon_*/color"))


def resolve_scene_root(data_root: Path, train_subdir: str = "train") -> Path:
    """Locate folder that directly contains Moon_*/{color,label}.

    Layouts:
      1) data-root itself is LuSNAR_train (cheerhin): Moon_* under data-root
      2) portable: data-root/train/Moon_*
      3) explicit: data-root/<train_subdir>/Moon_*
    """
    data_root = data_root.resolve()
    if has_moon_scenes(data_root):
        return data_root
    cand = (data_root / train_subdir).resolve()
    if has_moon_scenes(cand):
        return cand
    # also try common cheerhin name even if train_subdir differs
    for name in ("train", "LuSNAR_train"):
        c = (data_root / name).resolve()
        if has_moon_scenes(c):
            return c
    # fall back so callers can raise a clear error
    return cand


def find_pretrain(project: Path, explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = (project / p).resolve()
        else:
            p = p.resolve()
        if p.exists():
            return p
        raise FileNotFoundError(p)
    for rel in (
        "pretrained/mit_b2_20220624-66e8bf70.pth",
        "weights/mit_b2_20220624-66e8bf70.pth",
        "checkpoints/mit_b2_20220624-66e8bf70.pth",
        "pretrained/segformer/mit_b2_20220624-66e8bf70.pth",
    ):
        p = project / rel
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        f"mit_b2 pretrained not found under {project}/pretrained/"
    )


def rgb_label_to_index(lab_bgr: np.ndarray) -> np.ndarray:
    out = np.full(lab_bgr.shape[:2], IGNORE, np.uint8)
    for bgr, cid in BGR_TO_ID.items():
        m = np.all(lab_bgr == np.asarray(bgr, np.uint8), axis=-1)
        out[m] = cid
    return out


def load_pairs(list_file: Path, resolve_roots=None):
    """Load image/label pairs. Relative paths resolve against resolve_roots then CWD.

    resolve_roots: Path or list of Path candidates (project / data-root / scene-root).
    Absolute paths kept as-is (cheerhin existing lists).
    """
    roots = []
    if resolve_roots is not None:
        if isinstance(resolve_roots, (str, Path)):
            roots = [Path(resolve_roots)]
        else:
            roots = [Path(r) for r in resolve_roots]
    roots = [r for r in roots if r is not None]

    def _resolve(path_str: str) -> str:
        pp = Path(path_str)
        if pp.is_absolute():
            return str(pp)
        candidates = list(roots) + [Path(list_file).resolve().parent, Path.cwd()]
        for root in candidates:
            cand = (root / path_str).resolve()
            if cand.exists():
                return str(cand)
        return str((roots[0] / path_str).resolve()) if roots else path_str

    pairs = []
    for line in Path(list_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        a, b = line.split("\t") if "\t" in line else line.split()
        pairs.append((_resolve(a), _resolve(b)))
    return pairs



def scan_scene_pairs(scene_root: Path, scene_id: int):
    """Return list of (color_path, label_path) for Moon_{id}."""
    moon = scene_root / f"Moon_{scene_id}"
    color = moon / "color"
    label = moon / "label"
    if not color.is_dir() or not label.is_dir():
        raise FileNotFoundError(f"missing color/label under {moon}")
    pairs = []
    for ip in sorted(color.glob("*.png")):
        lp = label / ip.name
        if lp.exists():
            pairs.append((str(ip.resolve()), str(lp.resolve())))
    return pairs


def build_split_lists(scene_root: Path, out_dir: Path, train_scenes=None, val_scene=VAL_SCENE):
    """Write train and val pair lists under out_dir.

    val_scene=None writes an empty val list. Defaults stay 1/3/6/7/8 train, scene 5 val.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    use_train = TRAIN_SCENES if train_scenes is None else tuple(train_scenes)
    train_pairs = []
    for sid in use_train:
        train_pairs.extend(scan_scene_pairs(scene_root, sid))
    val_pairs = [] if val_scene is None else scan_scene_pairs(scene_root, val_scene)
    train_file = out_dir / "train_scenes.txt"
    val_file = out_dir / "val_scene5.txt"
    train_file.write_text(
        "\n".join(f"{a}\t{b}" for a, b in train_pairs) + "\n", encoding="utf-8"
    )
    val_file.write_text(
        "\n".join(f"{a}\t{b}" for a, b in val_pairs) + "\n", encoding="utf-8"
    )
    return train_file, val_file, len(train_pairs), len(val_pairs)


def photometric(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.astype(np.float32)
    if rng.random() < 0.5:
        out += rng.uniform(-32, 32)
    if rng.random() < 0.5:
        out *= rng.uniform(0.5, 1.5)
    out = np.clip(out, 0, 255)
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


def shadow_augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Crater / terrain shadow photometric (train-only).

    Random gamma darkening, lower-half wash, elliptical umbras, and patch darkening.
    Grayscale-friendly (operates on BGR intensities without relying on hue).
    """
    out = img.astype(np.float32)
    h, w = out.shape[:2]
    if rng.random() < 0.7:
        gamma = rng.uniform(1.2, 2.5)
        out = 255.0 * np.power(np.clip(out / 255.0, 0.0, 1.0), gamma)
    if rng.random() < 0.55:
        y0 = rng.randint(h // 3, max(h // 3 + 1, 2 * h // 3))
        out[y0:, :] *= rng.uniform(0.35, 0.75)
    n_ell = rng.randint(1, 4)
    for _ in range(n_ell):
        if rng.random() > 0.85:
            continue
        cx = rng.randint(0, max(0, w - 1))
        cy = rng.randint(0, max(0, h - 1))
        ax = rng.randint(max(8, w // 16), max(9, w // 3))
        ay = rng.randint(max(8, h // 16), max(9, h // 3))
        yy, xx = np.ogrid[:h, :w]
        ell = ((xx - cx) / float(ax)) ** 2 + ((yy - cy) / float(ay)) ** 2 <= 1.0
        out[ell] *= rng.uniform(0.25, 0.65)
    if rng.random() < 0.4:
        ph = rng.randint(max(8, h // 10), max(9, h // 2))
        pw = rng.randint(max(8, w // 10), max(9, w // 2))
        y0 = rng.randint(0, max(0, h - ph))
        x0 = rng.randint(0, max(0, w - pw))
        out[y0 : y0 + ph, x0 : x0 + pw] *= rng.uniform(0.3, 0.7)
    return np.clip(out, 0, 255).astype(np.uint8)


def dice_loss(logits, masks, ignore=IGNORE, eps=1e-6):
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
    edge = np.zeros_like(mask, np.uint8)
    valid = mask != IGNORE
    if not valid.any():
        return edge
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    for c in range(NUM_CLASSES):
        binm = (mask == c).astype(np.uint8) * 255
        if binm.sum() == 0:
            continue
        dil = cv2.dilate(binm, kernel)
        ero = cv2.erode(binm, kernel)
        e = ((dil > 0) ^ (ero > 0)) & valid
        edge[e] = 1
    return edge


def soft_boundary_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Differentiable soft boundary map B×1×H×W in [0,1] from class-prob disagreement."""
    probs = F.softmax(logits.float(), dim=1)
    pad = F.pad(probs, (1, 1, 1, 1), mode="replicate")
    neigh = (
        pad[:, :, 1:-1, 2:]
        + pad[:, :, 1:-1, :-2]
        + pad[:, :, 2:, 1:-1]
        + pad[:, :, :-2, 1:-1]
    ) / 4.0
    edge = (probs - neigh).abs().sum(dim=1, keepdim=True)
    # normalize per-sample to keep BCE scale stable
    flat_max = edge.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp(min=1e-6)
    return (edge / flat_max).clamp(0.0, 1.0)


def abl_lite_loss(logits, masks, edge_gt, ignore=IGNORE, eps=1e-6):
    """ABL-lite: BCE + Dice on soft pred boundary vs GT edge (no InverseForm net).

    edge_gt: B×H×W float {0,1} from edge_map_from_mask.
    """
    pred_e = soft_boundary_from_logits(logits)
    gt = edge_gt.unsqueeze(1).float()
    valid = (masks != ignore).unsqueeze(1).float()
    # BCE is autocast-unsafe: compute in fp32 with autocast disabled (math unchanged)
    with torch.autocast(device_type=pred_e.device.type, enabled=False):
        bce_map = F.binary_cross_entropy(pred_e.float().clamp(1e-4, 1 - 1e-4), gt.float(), reduction="none")
    bce = (bce_map * valid).sum() / valid.sum().clamp(min=1.0)
    inter = (pred_e * gt * valid).sum()
    den = (pred_e * valid).sum() + (gt * valid).sum()
    dice = 1.0 - (2 * inter + eps) / (den + eps)
    return bce + dice


@torch.no_grad()
def segfix_lite(pred: np.ndarray, conf: np.ndarray | None = None, k: int = 5) -> np.ndarray:
    """SegFix-lite: replace boundary pixels with majority label from nearby interior.

    pred: H×W int labels. conf: optional H×W confidence (max softmax); higher preferred.
    Classic idea: offset boundary toward higher-confidence interior / local majority.
    """
    pred = pred.astype(np.int64)
    h, w = pred.shape
    out = pred.copy()
    # morphological class boundary
    edge = np.zeros_like(pred, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for c in range(NUM_CLASSES):
        binm = (pred == c).astype(np.uint8) * 255
        if binm.sum() == 0:
            continue
        dil = cv2.dilate(binm, kernel)
        ero = cv2.erode(binm, kernel)
        edge[((dil > 0) ^ (ero > 0))] = 1
    ys, xs = np.where(edge > 0)
    if len(ys) == 0:
        return out
    r = max(1, k // 2)
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        patch = pred[y0:y1, x0:x1]
        epatch = edge[y0:y1, x0:x1]
        interior = patch[epatch == 0]
        if interior.size == 0:
            interior = patch.ravel()
        # majority; break ties with confidence if available
        vals, counts = np.unique(interior, return_counts=True)
        maj = vals[np.argmax(counts)]
        if conf is not None and len(vals) > 1:
            # prefer label with highest mean conf among tied max counts
            top = counts.max()
            cands = vals[counts == top]
            if len(cands) > 1:
                cpatch = conf[y0:y1, x0:x1]
                best, best_s = maj, -1.0
                for lab in cands:
                    m = (patch == lab) & (epatch == 0)
                    if not m.any():
                        m = patch == lab
                    s = float(cpatch[m].mean()) if m.any() else -1.0
                    if s > best_s:
                        best, best_s = int(lab), s
                maj = best
        out[y, x] = int(maj)
    return out


class FadeLite(nn.Module):
    """FADE-lite: one content-aware refine on low-res logits guided by backbone stage-0.

    Honest lite: not full FADE/SAPA upsample; concat(logits, resized feat0) → 3×3 → logits.
    Gate: only interpret after M_TTA shows meaningful Δ.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, feat_ch: int = 64, hidden: int = 32):
        super().__init__()
        self.proj = nn.Conv2d(feat_ch, hidden, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(num_classes + hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_classes, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, logits: torch.Tensor, feat0: torch.Tensor) -> torch.Tensor:
        f = F.interpolate(feat0.float(), size=logits.shape[-2:], mode="bilinear", align_corners=False)
        if f.shape[1] != self.proj.in_channels:
            # adaptive 1×1 if channel mismatch (rebuild not possible mid-forward)
            # fall back: average-pool channels to expected or pad/crop
            c_exp = self.proj.in_channels
            if f.shape[1] > c_exp:
                f = f[:, :c_exp]
            else:
                f = F.pad(f, (0, 0, 0, 0, 0, c_exp - f.shape[1]))
        g = self.proj(f)
        return logits.float() + self.fuse(torch.cat([logits.float(), g], dim=1))


class CCBank:
    def __init__(self, index_json: Path):
        data = json.loads(index_json.read_text(encoding="utf-8"))
        self.items = []
        for cid in ("1", "2", 1, 2):
            key = str(cid)
            if key in data:
                for p in data[key]:
                    self.items.append(p)
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
            if "cid" in z:
                cid = int(z["cid"])
            elif (pm != IGNORE).any():
                cid = int(pm[pm != IGNORE][0])
            else:
                cid = 1
            if rng.random() < 0.5:
                pi = pi[:, ::-1].copy()
                pm = pm[:, ::-1].copy()
            ph, pw = pm.shape
            # Random regolith pixels. argwhere() materializes every hit and
            # ran the machine out of RAM with two ranks of workers.
            placed = False
            for _try in range(8):
                yx = None
                for _pick in range(24):
                    y = rng.randrange(h)
                    x = rng.randrange(w)
                    if out_m[y, x] == 0:
                        yx = (y, x)
                        break
                if yx is None:
                    break
                y0 = int(yx[0] - ph // 2)
                x0 = int(yx[1] - pw // 2)
                if y0 < 0 or x0 < 0 or y0 + ph > h or x0 + pw > w:
                    continue
                region = out_m[y0 : y0 + ph, x0 : x0 + pw]
                dest = (region == 0) | (region == IGNORE)
                src = pm != IGNORE
                use = dest & src
                if int(use.sum()) < 32:
                    continue
                out_i[y0 : y0 + ph, x0 : x0 + pw][use] = pi[use]
                out_m[y0 : y0 + ph, x0 : x0 + pw][use] = cid
                placed = True
                break
            if not placed:
                continue
        return out_i, out_m


class EdgeAux(nn.Module):
    def __init__(self, in_ch=NUM_CLASSES):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, logits):
        return self.conv(logits.float())


def recipe_flags(recipe: str) -> dict:
    """Flags for smoke / train recipes.

    Families:
      A0: baseline
      A1: RCS + mosaic + CC paste
      B1: A1 + ACW
      C1: B1 + EPS + multi-scale + photo + EMA
      V_B1C: B1 + crater RCS/ACW boost (control for method smokes)
      V_B1E: B1 + edge aux (legacy weakness check)
      M_ROW: V_B1C + HANet-style row-height attention
      M_BLV: V_B1C + BLV logit variation (train only)
      M_SHAD: V_B1C + shadow_augment (train only)
      M_ROW_SHAD: V_B1C + RowHeightAttn + shadow_augment (combo train)
      M_BLV_SHAD: V_B1C + BLV + shadow_augment (combo train)
      M_ROW_BLV: V_B1C + RowHeightAttn + BLV (combo train)
      M_ROW_BLV_SHAD: V_B1C + RowHeightAttn + BLV + shadow_augment (3-module combo train)
      M_ABL: V_B1C + ABL-lite boundary loss (train only)
      M_INVFORM: stub → falls back to ABL-lite with warning (no InverseForm net)
      M_FADE: V_B1C + FADE-lite content-aware refine on low-res logits
      M_TTA: eval-only TTA on V_B1C weights (no retrain)
      M_SEGFIX: eval-only SegFix-lite postprocess on a checkpoint
      T_B1_TTA: eval-only TTA on work_dirs/train_B1/best.pth (fail if missing)
      M_HRDA: NOT implemented for smoke (gate documented)
    """
    r = recipe.upper()
    train_method = (
        "M_ROW", "M_BLV", "M_SHAD", "M_ROW_SHAD", "M_BLV_SHAD", "M_ROW_BLV",
        "M_ROW_BLV_SHAD",
        "M_ABL", "M_INVFORM", "M_FADE",
    )
    a1_fam = r in ("A1", "B1", "C1", "V_B1C", "V_B1E", *train_method, "M_TTA", "M_SEGFIX", "T_B1_TTA")
    b1_fam = r in ("B1", "C1", "V_B1C", "V_B1E", *train_method, "M_TTA", "M_SEGFIX", "T_B1_TTA")
    # crater boost for control + train methods; eval-only inherit V_B1C family for docs
    vb1c_fam = r in ("V_B1C", *train_method, "M_TTA", "M_SEGFIX")
    invform = r == "M_INVFORM"
    abl = r in ("M_ABL", "M_INVFORM")  # INVFORM falls back to ABL-lite
    return {
        "rcs": a1_fam and r not in ("M_SEGFIX", "T_B1_TTA", "M_TTA", "M_HRDA"),
        "mosaic": r in ("A1", "B1", "C1", "V_B1C", "V_B1E", *train_method),
        "cc_paste": r in ("A1", "B1", "C1", "V_B1C", "V_B1E", *train_method),
        "acw": r in ("B1", "C1", "V_B1C", "V_B1E", *train_method),
        "crater_boost": r in ("V_B1C", *train_method),
        "eps": r in ("C1", "V_B1E"),
        "multi_scale": r == "C1",
        "photometric": r == "C1",
        "ema": r == "C1",
        "row_attn": r in ("M_ROW", "M_ROW_SHAD", "M_ROW_BLV", "M_ROW_BLV_SHAD"),
        "blv": r in ("M_BLV", "M_BLV_SHAD", "M_ROW_BLV", "M_ROW_BLV_SHAD"),
        "shadow_aug": r in ("M_SHAD", "M_ROW_SHAD", "M_BLV_SHAD", "M_ROW_BLV_SHAD"),
        "abl": abl,
        "invform_fallback": invform,
        "fade": r == "M_FADE",
        "tta": r == "M_TTA",
        "segfix": r == "M_SEGFIX",
        "t_b1_tta": r == "T_B1_TTA",
        "hrda": r == "M_HRDA",
        # eval-only shortcuts
        "eval_only": r in ("M_TTA", "M_SEGFIX", "T_B1_TTA", "M_HRDA"),
    }


class RowHeightAttn(nn.Module):
    """HANet-style vertical prior: per-class additive bias from normalized row y.

    Applied on decoder logits (train + infer for M_ROW). Very light MLP on y∈[0,1].
    """

    def __init__(self, num_classes: int = NUM_CLASSES, hidden: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )
        # near-zero init so early training ≈ identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: B C H W
        b, c, h, w = logits.shape
        y = torch.linspace(0, 1, h, device=logits.device, dtype=logits.dtype).view(h, 1)
        bias = self.mlp(y)  # H x C
        bias = bias.t().unsqueeze(0).unsqueeze(-1)  # 1 C H 1
        return logits + bias


def blv_perturb_logits(
    logits: torch.Tensor,
    class_freq,
    sigma: float = 0.5,
) -> torch.Tensor:
    """Simplified BLV (CVPR 2023): train-only Gaussian noise scaled by class rarity.

    Tail classes get larger logit variation → expands feature volume.
    Orthogonal to ACW (weights on CE); zero cost at inference (do not call).
    """
    freq = torch.as_tensor(class_freq, device=logits.device, dtype=torch.float32)
    freq = freq.clamp(min=1e-6)
    # rarer → larger scale; normalize to mean 1
    scale = (1.0 / torch.sqrt(freq))
    scale = scale / scale.mean()
    noise = torch.randn_like(logits.float()) * float(sigma)
    return logits.float() + noise * scale.view(1, -1, 1, 1)


def crater_boost_sampler_weights(rarity_items, boost: float = 2.0) -> np.ndarray:
    """RCS weights with crater pixels emphasized (V_B1C / method family)."""
    weights = []
    for item in rarity_items:
        lab = cv2.imread(item["lab"], cv2.IMREAD_COLOR)
        if lab is None:
            weights.append(0.02)
            continue
        mask = rgb_label_to_index(lab)
        valid = mask != IGNORE
        n = int(valid.sum()) or 1
        rock = float((mask == 1).sum()) / n
        crater = float((mask == 2).sum()) / n
        weights.append((rock + boost * crater + 0.02) ** 0.5)
    w = np.asarray(weights, np.float64)
    s = w.sum()
    if s <= 0:
        w = np.ones_like(w) / max(len(w), 1)
    else:
        w = w / s
    return w


def confusion_rates(cm: np.ndarray) -> dict:
    """cm[pred, gt] counts. Rates are P(pred=a | gt=b)."""
    def rate(pred, gt):
        col = float(cm[:, gt].sum())
        if col <= 0:
            return float("nan")
        return float(cm[pred, gt] / col)

    return {
        "sky_to_mountain": rate(3, 4),  # pred mountain | gt sky
        "crater_to_regolith": rate(0, 2),  # pred regolith | gt crater
        "mountain_to_rock": rate(1, 3),
        "rock_to_mountain": rate(3, 1),
    }


_SEGFORMER_CFGS = {
    "b2": "segformer_mit-b2_8xb2-160k_ade20k-512x512.py",
    "b5": "segformer_mit-b5_8xb2-160k_ade20k-512x512.py",
}


def build_segformer(pretrain: Path, variant: str = "b2"):
    from mmseg.utils import register_all_modules
    from mmseg.registry import MODELS
    from mmengine.config import Config
    import mmseg

    key = (
        str(variant)
        .lower()
        .replace("mit-", "")
        .replace("mit_", "")
        .replace("segformer-", "")
        .replace("segformer_", "")
    )
    if key not in _SEGFORMER_CFGS:
        raise ValueError(
            f"unknown segformer variant {variant!r}; known={sorted(_SEGFORMER_CFGS)}"
        )

    register_all_modules()
    cfg_path = (
        Path(mmseg.__file__).parent
        / ".mim"
        / "configs"
        / "segformer"
        / _SEGFORMER_CFGS[key]
    )
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
        model_cfg["backbone"]["init_cfg"] = dict(
            type="Pretrained", checkpoint=str(pretrain)
        )
    model = MODELS.build(model_cfg)
    model.init_weights()
    return model


def build_segformer_b2(pretrain: Path):
    return build_segformer(pretrain, "b2")


def _decode_logits(net, feats, residual=None):
    """Decoder logits. Optional residual is added on the fused 1/4 map, before cls_seg."""
    head = net.decode_head
    if residual is None:
        return head.forward(feats)
    from mmseg.models.utils import resize

    inputs = head._transform_inputs(feats)
    outs = []
    for idx, x in enumerate(inputs):
        y = head.convs[idx](x)
        outs.append(
            resize(
                y,
                size=inputs[0].shape[2:],
                mode=head.interpolate_mode,
                align_corners=head.align_corners,
            )
        )
    h0 = head.fusion_conv(torch.cat(outs, dim=1))
    h0 = residual(inputs[0], h0)
    return head.cls_seg(h0)


def _forward_logits(net, imgs, row_attn=None, fade=None, residual=None):
    x = net.extract_feat(imgs)
    logits = _decode_logits(net, x, residual=residual)
    if fade is not None:
        logits = fade(logits, x[0])
    if row_attn is not None:
        logits = row_attn(logits)
    return logits


@torch.no_grad()
def evaluate(model, loader, device, use_ema=None, row_attn=None, fade=None, residual=None):
    was_training = model.training
    model.eval()
    net = use_ema if use_ema is not None else model
    net.eval()
    if row_attn is not None:
        row_attn.eval()
    if fade is not None:
        fade.eval()
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    for batch in loader:
        imgs, masks = batch[0], batch[1]
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = _forward_logits(net, imgs, row_attn=row_attn, fade=fade, residual=residual)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
        pred = logits.argmax(1)
        for c in range(NUM_CLASSES):
            p = pred == c
            g = masks == c
            inter[c] += (p & g).sum().item()
            union[c] += (p | g).sum().item()
    iou = [
        float(inter[c] / union[c]) if union[c] > 0 else float("nan")
        for c in range(NUM_CLASSES)
    ]
    valid = [x for x in iou if x == x]
    miou = float(np.mean(valid)) if valid else float("nan")
    if was_training:
        model.train()
    return miou, iou


@torch.no_grad()
def evaluate_detailed(model, loader, device, use_ema=None, row_attn=None, fade=None):
    """mIoU + per-class IoU + confusion (pred, gt) on Scene5 subset."""
    was_training = model.training
    model.eval()
    net = use_ema if use_ema is not None else model
    net.eval()
    if row_attn is not None:
        row_attn.eval()
    if fade is not None:
        fade.eval()
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.float64)  # [pred, gt]
    for batch in loader:
        imgs, masks = batch[0], batch[1]
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = _forward_logits(net, imgs, row_attn=row_attn, fade=fade)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
        pred = logits.argmax(1)
        valid = masks != IGNORE
        for c in range(NUM_CLASSES):
            p = pred == c
            g = masks == c
            inter[c] += (p & g).sum().item()
            union[c] += (p | g).sum().item()
        pv = pred[valid].reshape(-1).cpu().numpy().astype(np.int64)
        gv = masks[valid].reshape(-1).cpu().numpy().astype(np.int64)
        ok = (gv >= 0) & (gv < NUM_CLASSES) & (pv >= 0) & (pv < NUM_CLASSES)
        idx = pv[ok] * NUM_CLASSES + gv[ok]
        bc = np.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES)
        cm += bc.reshape(NUM_CLASSES, NUM_CLASSES)
    iou = [
        float(inter[c] / union[c]) if union[c] > 0 else float("nan")
        for c in range(NUM_CLASSES)
    ]
    valid_iou = [x for x in iou if x == x]
    miou = float(np.mean(valid_iou)) if valid_iou else float("nan")
    rates = confusion_rates(cm)
    if was_training:
        model.train()
    return miou, iou, cm, rates


def _preprocess_bgr(img_bgr: np.ndarray) -> torch.Tensor:
    img_f = (img_bgr.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(img_f.transpose(2, 0, 1)).float()


@torch.no_grad()
def sliding_window_logits(
    net,
    img_bgr: np.ndarray,
    device,
    crop: int = 512,
    stride: int = 256,
    row_attn=None,
    fade=None,
    residual=None,
):
    """Average logits over sliding windows on a single BGR image."""
    h, w = img_bgr.shape[:2]
    C = NUM_CLASSES
    acc = torch.zeros(C, h, w, device=device, dtype=torch.float32)
    cnt = torch.zeros(1, h, w, device=device, dtype=torch.float32)
    ys = list(range(0, max(h - crop, 0) + 1, stride))
    xs = list(range(0, max(w - crop, 0) + 1, stride))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]
    if ys[-1] != max(0, h - crop):
        ys.append(max(0, h - crop))
    if xs[-1] != max(0, w - crop):
        xs.append(max(0, w - crop))
    for y0 in ys:
        for x0 in xs:
            patch = img_bgr[y0 : y0 + crop, x0 : x0 + crop]
            ph, pw = patch.shape[:2]
            if ph != crop or pw != crop:
                pad = np.zeros((crop, crop, 3), np.uint8)
                pad[:ph, :pw] = patch
                patch = pad
            t = _preprocess_bgr(patch).unsqueeze(0).to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = _forward_logits(net, t, row_attn=row_attn, fade=fade, residual=residual)
                if logits.shape[-2:] != (crop, crop):
                    logits = F.interpolate(
                        logits, size=(crop, crop), mode="bilinear", align_corners=False
                    )
            logits = logits.float()[0, :, :ph, :pw]
            acc[:, y0 : y0 + ph, x0 : x0 + pw] += logits
            cnt[:, y0 : y0 + ph, x0 : x0 + pw] += 1.0
    cnt = cnt.clamp(min=1.0)
    return acc / cnt


@torch.no_grad()
def evaluate_tta(
    model,
    pairs,
    device,
    scales=(0.75, 1.0, 1.25),
    crop: int = 512,
    stride: int = 256,
    row_attn=None,
    use_ema=None,
    max_images: int | None = None,
    fade=None,
):
    """Multi-scale sliding-window TTA on full images (Scene5 val pairs).

    Small-object bucket helper is not present in portable stack → skipped (note in summary).
    """
    was_training = model.training
    model.eval()
    net = use_ema if use_ema is not None else model
    net.eval()
    if row_attn is not None:
        row_attn.eval()
    if fade is not None:
        fade.eval()
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.float64)
    n_done = 0
    for ip, lp in pairs:
        if max_images is not None and n_done >= max_images:
            break
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        lab = cv2.imread(lp, cv2.IMREAD_COLOR)
        if img is None or lab is None:
            continue
        mask = rgb_label_to_index(lab)
        h0, w0 = mask.shape
        acc = torch.zeros(NUM_CLASSES, h0, w0, device=device, dtype=torch.float32)
        for s in scales:
            nh, nw = max(1, int(round(h0 * s))), max(1, int(round(w0 * s)))
            img_s = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            logits_s = sliding_window_logits(
                net, img_s, device, crop=crop, stride=stride, row_attn=row_attn, fade=fade
            )
            logits_s = F.interpolate(
                logits_s.unsqueeze(0), size=(h0, w0), mode="bilinear", align_corners=False
            )[0]
            acc += logits_s
        pred = acc.argmax(0).cpu().numpy()
        valid = mask != IGNORE
        for c in range(NUM_CLASSES):
            p = pred == c
            g = mask == c
            inter[c] += np.logical_and(p, g).sum()
            union[c] += np.logical_or(p, g).sum()
        pv = pred[valid].ravel().astype(np.int64)
        gv = mask[valid].ravel().astype(np.int64)
        ok = (gv >= 0) & (gv < NUM_CLASSES) & (pv >= 0) & (pv < NUM_CLASSES)
        idx = pv[ok] * NUM_CLASSES + gv[ok]
        bc = np.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES)
        cm += bc.reshape(NUM_CLASSES, NUM_CLASSES)
        n_done += 1
        if n_done % 20 == 0:
            print(f"[tta] {n_done}/{len(pairs) if max_images is None else max_images}", flush=True)
    iou = [
        float(inter[c] / union[c]) if union[c] > 0 else float("nan")
        for c in range(NUM_CLASSES)
    ]
    valid_iou = [x for x in iou if x == x]
    miou = float(np.mean(valid_iou)) if valid_iou else float("nan")
    rates = confusion_rates(cm)
    if was_training:
        model.train()
    return miou, iou, cm, rates, {
        "scales": list(scales),
        "crop": crop,
        "stride": stride,
        "n_images": n_done,
        "small_object_buckets": None,
        "small_object_note": "portable stack has no size-bucket helper; skipped",
    }


def write_method_eval_md(path: Path, rows: list[dict], title: str = "Method smoke eval (Scene5)"):
    """Fill results/smoke_method_eval.md from per-exp summaries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "Protocol: Scene5 val_sub only; never Moon_2/4/9. Control = V_B1C.",
        "M_ROW / M_BLV / M_TTA are relative to V_B1C; 2k-iter smoke, Δ<0.02 = noise.",
        "",
        "| exp | mIoU | rock | crater | mountain | sky | sky→mtn | crater→reg | wall_s | note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        pc = r.get("per_class_IoU_best") or r.get("per_class_IoU") or {}
        rates = r.get("confusion_rates") or {}
        note = r.get("method_note") or r.get("note") or ""
        if r.get("tta_vs_single"):
            note = (note + " | " if note else "") + "TTA vs single on same ckpt"
        lines.append(
            "| {exp} | {miou} | {rock} | {crater} | {mtn} | {sky} | {s2m} | {c2r} | {wall} | {note} |".format(
                exp=r.get("exp") or r.get("run_name"),
                miou=_fmt(r.get("best_mIoU_scene5")),
                rock=_fmt(pc.get("岩石") if isinstance(pc, dict) else r.get("iou_rock")),
                crater=_fmt(pc.get("撞击坑") if isinstance(pc, dict) else r.get("iou_crater")),
                mtn=_fmt(pc.get("山体") if isinstance(pc, dict) else None),
                sky=_fmt(pc.get("天空") if isinstance(pc, dict) else None),
                s2m=_fmt(rates.get("sky_to_mountain")),
                c2r=_fmt(rates.get("crater_to_regolith")),
                wall=_fmt(r.get("wall_sec")),
                note=str(note)[:80],
            )
        )
    lines += [
        "",
        "## Pass criteria (vs V_B1C @ same iters)",
        "- M_ROW: sky→mountain < 12% AND mountain IoU +≥0.03; other classes drop ≤0.01",
        "- M_BLV: crater→regolith < 7% OR crater IoU +≥0.02",
        "- M_SHAD: crater IoU +≥0.02 OR crater→regolith drop ≥1pp (shadow-sensitive)",
        "- M_ABL: boundary-sensitive — rock/crater drop ≤0.01 tradeoff; prefer Δ rock≥0 without crater crash",
        "- M_SEGFIX: postprocess ΔmIoU vs same ckpt single-scale; keep if +≥0.005 without crater collapse",
        "- T_B1_TTA / M_TTA: if small-rock bucket +≥0.05 (when available) → pursue FADE/HRDA; else note resolution not sole issue",
        "- M_FADE: only interpret after M_TTA Δ above noise; else treat as stretch",
        "- M_HRDA: not implemented in portable smoke",
        "",
        "注：本地 Scene5 子集，非官方测试；未使用 Moon_2/4/9 标签。",
        "",
    ]
    # TTA detail section if present
    tta_rows = [r for r in rows if r.get("tta_vs_single")]
    if tta_rows:
        lines += ["## M_TTA single-scale vs TTA", ""]
        for r in tta_rows:
            ts = r["tta_vs_single"]
            lines.append(
                f"- single mIoU={_fmt(ts.get('single_mIoU'))} → TTA mIoU={_fmt(ts.get('tta_mIoU'))} "
                f"(scales={ts.get('scales')}, crop={ts.get('crop')}, stride={ts.get('stride')})"
            )
            if ts.get("small_object_note"):
                lines.append(f"  - small-object buckets: {ts['small_object_note']}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_ema(ema, model, decay=0.999):
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema.state_dict().items():
            if k in msd and v.dtype.is_floating_point:
                v.copy_(v * decay + msd[k] * (1.0 - decay))


def write_markdown_table(path: Path, rows: list[dict], title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "| run | mIoU | rock | crater | wall_s | peak_MB | error |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(
            "| {exp} | {miou} | {rock} | {crater} | {wall} | {mem} | {err} |".format(
                exp=r.get("exp") or r.get("run_name"),
                miou=_fmt(r.get("best_mIoU_scene5")),
                rock=_fmt(r.get("iou_rock")),
                crater=_fmt(r.get("iou_crater")),
                wall=_fmt(r.get("wall_sec")),
                mem=_fmt(r.get("peak_mem_mb")),
                err=("yes" if r.get("error") else ""),
            )
        )
    lines.append("")
    lines.append("注：本地 Scene5 验证，非官方测试成绩；未使用 Moon_2/4/9 标签。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(v):
    if v is None:
        return "-"
    try:
        return f"{float(v):.4f}" if abs(float(v)) < 10 else f"{float(v):.1f}"
    except Exception:
        return str(v)
