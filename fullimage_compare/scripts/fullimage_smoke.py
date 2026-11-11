#!/usr/bin/env python3
"""LuSNAR whole-image (native 1024x1024) smoke line -- fullimage_compare.

SEPARATE experiment line. Never mix its numbers with the crop-512 line
(tools/portable*), official_split_compare or P0. The control (FI_V_B1C) must be
re-run on the SAME machine as every module it is compared with.

Module code is REUSED BY IMPORT, never copied:
  tools/portable/_lusnar_common.py    -> 4090 crop-line implementations used for the
                                         V_B1C/ROW/BLV/SHAD/ABL/FADE/V_B1E results
                                         (RAM-safe CCBank, autocast-safe ABL BCE,
                                         combo recipe flags M_ROW_BLV etc.)
  tools/portable_b3/_lusnar_common.py -> batch-3 modules (DySample, SAPA-lite,
                                         hardness CE, boundary contrast), BoundaryMeter,
                                         ignore-safe metrics, upsampler-aware forward.
Differences vs. the crop line (by design, nothing else):
  * no RandomCrop: train and val at native 1024x1024 (val = whole image, val batch 1)
  * micro batch from CLI (default 1); optional --grad-accum (lr is NOT rescaled)
  * mosaic keeps the crop line's 4x(512 tiles) -> 1024 canvas, but the whole canvas is
    used instead of a random 512 crop of it (same pixel scale, 4x content per sample)
  * guards: VRAM probe, OOM fallback, Windows 1455 worker fallback, STOP file,
    resumable last.pth, module self-check at iter 1, UTF-8 ASCII-key summaries.
Scene5-only validation. Moon_2/4/9 labels are never read (assert_safe_label).
Never sets PYTORCH_CUDA_ALLOC_CONF. Single process, single GPU, no DDP, no ports.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import socket
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
FI_ROOT = SCRIPT.parents[1]                 # <project>/fullimage_compare
PROJECT_DEFAULT = SCRIPT.parents[2]         # <project>
PORTABLE = PROJECT_DEFAULT / "tools" / "portable"
PORTABLE_B3 = PROJECT_DEFAULT / "tools" / "portable_b3"


def _load(name: str, path: Path):
    """Import a module from an explicit file under a unique name (both helper
    files are called _lusnar_common.py). Runs at import time so DataLoader spawn
    workers on Windows re-create the same sys.modules entries before unpickling."""
    if name in sys.modules:
        return sys.modules[name]
    if not path.exists():
        raise FileNotFoundError(f"[fullimage] required helper missing: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


LC = _load("lusnar_common_portable", PORTABLE / "_lusnar_common.py")
LB = _load("lusnar_common_b3", PORTABLE_B3 / "_lusnar_common.py")

NUM_CLASSES = LC.NUM_CLASSES
IGNORE = LC.IGNORE
SEED = LC.SEED
MEAN, STD = LC.MEAN, LC.STD
CLASS_EN = ["regolith", "rock", "crater", "mountain", "sky"]
CLASS_ZH = list(LC.CLASS_NAMES)
assert len(CLASS_EN) == NUM_CLASSES
TRAIN_N, VAL_N = 1600, 350
NATIVE = 1024
EXIT_STOPPED = 3

# Experiment table. FI_<recipe>; flags come from the crop-line recipe_flags().
PORTABLE_RECIPES = (
    "V_B1C", "V_B1E", "M_ROW", "M_BLV", "M_SHAD", "M_ROW_BLV", "M_ROW_SHAD",
    "M_BLV_SHAD", "M_ROW_BLV_SHAD", "M_ABL", "M_INVFORM", "M_FADE",
)
B3_RECIPES = ("M_DYSAMPLE", "M_HARD", "M_CTRAST", "M_SAPA")
TRAIN_EXPS = tuple("FI_" + r for r in PORTABLE_RECIPES + B3_RECIPES)
EVAL_EXPS = ("FI_M_TTA", "FI_M_SEGFIX", "FI_M_MSINFER")
# default single-module queue (all crop-line training modules incl. rejected ones)
DEFAULT_QUEUE = (
    "FI_V_B1C", "FI_M_ROW_BLV", "FI_M_ROW", "FI_M_BLV", "FI_M_SHAD", "FI_M_ABL",
    "FI_M_INVFORM", "FI_V_B1E", "FI_M_HARD", "FI_M_DYSAMPLE", "FI_M_CTRAST",
    "FI_M_SAPA", "FI_M_FADE",
)
HEAVY_EXPS = ("FI_M_ROW_BLV", "FI_M_DYSAMPLE", "FI_M_CTRAST", "FI_M_SAPA")

METHOD_NOTE = {
    "V_B1C": "control: B1 (RCS+mosaic+CC paste+ACW) + crater RCS/ACW boost",
    "V_B1E": "legacy: B1 + edge-aux (no crater boost, exactly as crop line)",
    "M_ROW": "V_B1C + RowHeightAttn (HANet-style row bias on logits)",
    "M_BLV": "V_B1C + BLV logit variation sigma 0.5 (train only)",
    "M_SHAD": "V_B1C + shadow_augment (train only)",
    "M_ROW_BLV": "V_B1C + RowHeightAttn + BLV (current main recipe)",
    "M_ROW_SHAD": "V_B1C + RowHeightAttn + shadow_augment",
    "M_BLV_SHAD": "V_B1C + BLV + shadow_augment",
    "M_ROW_BLV_SHAD": "V_B1C + RowHeightAttn + BLV + shadow_augment",
    "M_ABL": "V_B1C + ABL-lite boundary BCE/Dice, weight 0.5, 20% ramp",
    "M_INVFORM": "stub: identical to ABL-lite (no InverseForm net) -> acts as an ABL repeat",
    "M_FADE": "V_B1C + FADE-lite residual refine of low-res logits with stage-1 feature",
    "M_DYSAMPLE": "V_B1C + DySample lp x4 groups4 on final logits",
    "M_HARD": "V_B1C + instantaneous hardness-weighted CE (alpha1 gamma2)",
    "M_CTRAST": "V_B1C + fused-feature boundary-pair InfoNCE (temp 0.1, w 0.1, cap 512)",
    "M_SAPA": "V_B1C + SAPA-B-lite 5x5 final-logit reassembly",
}


def recipe_of(exp: str) -> str:
    e = exp.strip().upper()
    if not e.startswith("FI_"):
        raise ValueError(f"experiment names must start with FI_: {exp}")
    return e[3:]


def fi_flags(exp: str) -> dict:
    r = recipe_of(exp)
    if r in PORTABLE_RECIPES:
        flags = dict(LC.recipe_flags(r))
        src = "tools/portable/_lusnar_common.recipe_flags"
    elif r in B3_RECIPES:
        flags = dict(LB.recipe_flags(r))
        src = "tools/portable_b3/_lusnar_common.recipe_flags"
    else:
        raise ValueError(f"unknown training exp {exp}; known: {TRAIN_EXPS + EVAL_EXPS}")
    for k in ("dysample", "hard", "ctrast", "sapa"):
        flags.setdefault(k, False)
    if flags.get("multi_scale") or flags.get("ema") or flags.get("photometric"):
        raise ValueError(f"{exp}: multi-scale/EMA/photometric recipes (C1) are not part of the FI line")
    if flags.get("eval_only"):
        raise ValueError(f"{exp} is eval-only in the crop line")
    flags["_flags_source"] = src
    return flags


# ---------------------------------------------------------------------------
# Paths / guards
# ---------------------------------------------------------------------------
_WIN_ABS = re.compile(r"^[A-Za-z]:[\\/]")


def check_native_path(p, what: str):
    s = str(p)
    if os.name == "nt":
        bad = (not _WIN_ABS.match(s)) or "/mnt/" in s.replace("\\", "/").lower() \
            or s.lower().startswith("\\\\wsl")
        if bad:
            raise SystemExit(f"[fullimage][FATAL] {what} must be a native Windows drive path "
                             f"(e.g. D:\\LuSNAR_Seg\\...), got: {s}")
    else:
        if s.startswith("/mnt/") and os.environ.get("FI_ALLOW_MNT") != "1":
            raise SystemExit(f"[fullimage][FATAL] {what} on /mnt (WSL 9p) is not allowed: {s}")


def stop_file() -> Path:
    return FI_ROOT / "STOP"


def stop_requested() -> bool:
    return stop_file().exists()


def host_tag() -> str:
    return (os.environ.get("COMPUTERNAME") or socket.gethostname() or "host").strip()


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    return str(o)


def write_json(path: Path, obj):
    """UTF-8 (no BOM), ensure_ascii=False, atomic replace. Keys are ASCII English."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def jf(x):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def log(msg):
    print(f"[fullimage {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Lists (copied per host into fullimage_compare/lists/<HOST>/)
# ---------------------------------------------------------------------------
LIST_FILES = ("train_sub.txt", "val_sub.txt", "meta.json", "rarity.json", "cc_bank_index.json")


def _scene_of(p: str):
    m = re.search(r"Moon_(\d+)", str(p).replace("\\", "/"))
    return int(m.group(1)) if m else None


def validate_lists(lists_dir: Path, check_files=True) -> dict:
    for f in LIST_FILES:
        if not (lists_dir / f).exists():
            raise SystemExit(f"[fullimage][FATAL] missing {lists_dir / f}; run --prepare-lists first")
    meta = read_json(lists_dir / "meta.json")
    tr = LC.load_pairs(lists_dir / "train_sub.txt")
    va = LC.load_pairs(lists_dir / "val_sub.txt")
    for ip, lp in tr + va:
        LB.assert_safe_label(lp)
        LB.assert_safe_label(ip)
    if len(tr) != TRAIN_N or len(va) != VAL_N or int(meta.get("seed", -1)) != SEED:
        raise SystemExit(f"[fullimage][FATAL] list protocol mismatch train={len(tr)} val={len(va)} "
                         f"seed={meta.get('seed')} (need {TRAIN_N}/{VAL_N}/seed {SEED})")
    bad_val = [ip for ip, _ in va if _scene_of(ip) != 5]
    bad_tr = [ip for ip, _ in tr if _scene_of(ip) not in (1, 3, 6, 7, 8)]
    if bad_val or bad_tr:
        raise SystemExit(f"[fullimage][FATAL] scene leak: val non-Scene5={bad_val[:3]} "
                         f"train outside 1/3/6/7/8={bad_tr[:3]}")
    bank = read_json(lists_dir / "cc_bank_index.json")
    bank_files = [p for v in bank.values() for p in v]
    rar = read_json(lists_dir / "rarity.json")
    if len(rar.get("items", [])) != TRAIN_N:
        raise SystemExit("[fullimage][FATAL] rarity.json items != 1600")
    if check_files:
        missing = []
        for ip, lp in tr + va:
            check_native_path(ip, "list image")
            check_native_path(lp, "list label")
            if not os.path.exists(ip):
                missing.append(ip)
            if not os.path.exists(lp):
                missing.append(lp)
        for p in bank_files:
            check_native_path(p, "cc bank file")
            if not os.path.exists(p):
                missing.append(p)
        if missing:
            raise SystemExit(f"[fullimage][FATAL] {len(missing)} listed files missing, e.g. {missing[:3]}")
    return {"train": len(tr), "val": len(va), "seed": meta.get("seed"), "cc_bank": len(bank_files),
            "scene_root": meta.get("scene_root")}


def prepare_lists(project: Path, lists_dir: Path, src: Path | None):
    src = src or (project / "data" / "smoke")
    check_native_path(src, "smoke source dir")
    lists_dir.mkdir(parents=True, exist_ok=True)
    man_path = lists_dir / "manifest.json"
    if man_path.exists():
        man = read_json(man_path)
        changed = [f for f in LIST_FILES if sha256(lists_dir / f) != man["sha256"].get(f)]
        if changed:
            raise SystemExit(f"[fullimage][FATAL] {lists_dir} was modified after copy: {changed}. "
                             "Move the folder away manually; lists are never overwritten.")
        log(f"lists already prepared at {lists_dir} (sha256 verified)")
        return man
    for f in LIST_FILES:
        if not (src / f).exists():
            raise SystemExit(f"[fullimage][FATAL] {src / f} missing; regenerate the crop-line smoke "
                             "assets on THIS machine (tools/portable/prepare_smoke_assets.py) first")
        shutil.copy2(src / f, lists_dir / f)
    info = validate_lists(lists_dir, check_files=True)
    man = {"source_dir": str(src), "host": host_tag(), "copied_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
           "sha256": {f: sha256(lists_dir / f) for f in LIST_FILES}, "counts": info,
           "note": "Copied from this machine's crop-line data/smoke (same 1600/350 seed-42 subset). "
                   "Scene5 val only; Moon_2/4/9 never listed."}
    write_json(man_path, man)
    log(f"lists prepared at {lists_dir}: {info}")
    return man


def crater_weights(lists_dir: Path):
    """crop-line crater-boost RCS weights (portable crater_boost_sampler_weights),
    cached per host because it reads 1600 labels."""
    cache = lists_dir / "crater_boost_weights.json"
    rsha = sha256(lists_dir / "rarity.json")
    if cache.exists():
        w = read_json(cache)
        if w.get("rarity_sha256") == rsha:
            return np.asarray(w["weights"], np.float64)
    rarity = read_json(lists_dir / "rarity.json")
    for item in rarity["items"]:
        LB.assert_safe_label(item["lab"])
    w = LC.crater_boost_sampler_weights(rarity["items"], boost=2.0)
    write_json(cache, {"rarity_sha256": rsha, "boost": 2.0, "weights": w.tolist()})
    return w


# ---------------------------------------------------------------------------
# Dataset (whole image)
# ---------------------------------------------------------------------------
class FullImageDataset(Dataset):
    """Crop-line SmokeDataset minus the crop. Train: mosaic(0.5) -> CC paste(0.7)
    -> shadow -> hflip(0.5); val: untouched whole image. `resize_to` is ONLY for
    the CPU dry check (never in real runs)."""

    def __init__(self, list_file, train=True, shadow_aug=False, cc_bank=None, mosaic=False,
                 seed=SEED, resize_to=None, limit=None):
        pairs = LC.load_pairs(Path(list_file))
        for _, lp in pairs:
            LB.assert_safe_label(lp)
        self.pairs = pairs[:limit] if limit else pairs
        self.train, self.shadow_aug, self.cc_bank, self.mosaic = train, shadow_aug, cc_bank, mosaic
        self.rng = random.Random(seed + (0 if train else 1))
        self.resize_to = resize_to

    def __len__(self):
        return len(self.pairs)

    def _read(self, idx):
        ip, lp = self.pairs[idx]
        LB.assert_safe_label(lp)
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        lab = cv2.imread(lp, cv2.IMREAD_COLOR)
        if img is None or lab is None:
            raise RuntimeError(f"bad pair {ip}")
        return img, LC.rgb_label_to_index(lab), ip

    def _mosaic4(self, idx):
        # identical to the crop line: 4 tiles resized to 512 -> 1024 canvas (kept whole here)
        idxs = [idx] + [self.rng.randrange(len(self.pairs)) for _ in range(3)]
        ti, tm = [], []
        for j in idxs:
            img, mask, _ = self._read(j)
            ti.append(cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR))
            tm.append(cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST))
        img = np.concatenate([np.concatenate(ti[:2], 1), np.concatenate(ti[2:], 1)], 0)
        mask = np.concatenate([np.concatenate(tm[:2], 1), np.concatenate(tm[2:], 1)], 0)
        return img, mask

    def __getitem__(self, idx):
        if self.train and self.mosaic and self.rng.random() < 0.5:
            img, mask = self._mosaic4(idx)
            ip = self.pairs[idx][0]
        else:
            img, mask, ip = self._read(idx)
        if self.train and self.cc_bank is not None and self.rng.random() < 0.7:
            img, mask = self.cc_bank.sample_paste(img, mask, self.rng, n_paste=self.rng.randint(1, 3))
        if self.train and self.shadow_aug:
            img = LC.shadow_augment(img, self.rng)
        if self.train and self.rng.random() < 0.5:
            img = img[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if img.shape[:2] != (NATIVE, NATIVE):
            raise RuntimeError(f"expected native {NATIVE}x{NATIVE}, got {img.shape[:2]} for {ip}")
        edge = LC.edge_map_from_mask(mask)
        if self.resize_to:
            s = int(self.resize_to)
            img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (s, s), interpolation=cv2.INTER_NEAREST)
            edge = cv2.resize(edge, (s, s), interpolation=cv2.INTER_NEAREST)
        img_f = (img.astype(np.float32) - MEAN) / STD
        return (torch.from_numpy(np.ascontiguousarray(img_f.transpose(2, 0, 1))).float(),
                torch.from_numpy(mask.astype(np.int64)),
                torch.from_numpy(edge.astype(np.float32)), ip)


# ---------------------------------------------------------------------------
# Loader with Windows 1455 / worker-crash fallback (2 -> 1 -> 0)
# ---------------------------------------------------------------------------
_WORKER_ERR = re.compile(r"1455|paging file|DataLoader worker|shared memory|"
                         r"_queue\.Empty|Couldn't open shared file|page file", re.I)


class ResilientLoader:
    def __init__(self, ds, batch, sampler, workers, pin, generator, events, drop_last=True):
        self.ds, self.batch, self.sampler, self.pin, self.gen = ds, batch, sampler, pin, generator
        self.workers, self.events, self.drop_last = int(workers), events, drop_last
        self._build()

    def _build(self):
        kw = dict(batch_size=self.batch, num_workers=self.workers, pin_memory=self.pin,
                  drop_last=self.drop_last)
        if self.sampler is not None:
            kw["sampler"] = self.sampler
        else:
            kw["shuffle"] = True
            kw["generator"] = self.gen
        if self.workers > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = 2
        self.loader = DataLoader(self.ds, **kw)
        self.it = iter(self.loader)

    def next(self):
        while True:
            try:
                return next(self.it)
            except StopIteration:
                self.it = iter(self.loader)
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                if self.workers > 0 and _WORKER_ERR.search(msg + " " + traceback.format_exc()):
                    new = {2: 1, 1: 0}.get(self.workers, max(0, self.workers // 2))
                    ev = {"type": "dataloader_worker_fallback", "from_workers": self.workers,
                          "to_workers": new, "error": msg[:400], "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                    self.events.append(ev)
                    log(f"[WARN] {ev}")
                    self.it = None
                    self.loader = None
                    gc.collect()
                    self.workers = new
                    self._build()
                    continue
                raise


# ---------------------------------------------------------------------------
# Model / train state
# ---------------------------------------------------------------------------
def is_oom(e: BaseException) -> bool:
    oom_t = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom_t is not None and isinstance(e, oom_t)) or "out of memory" in str(e).lower()


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_state(exp, flags, pretrain, device, lr, class_freq):
    """Same construction as the crop line's _run_exp_impl (portable_b3), single GPU."""
    model = LC.build_segformer_b2(pretrain).to(device)
    if flags.get("dysample"):
        model.method_upsampler = LB.DySample().to(device)
    if flags.get("sapa"):
        model.method_upsampler = LB.SAPALite().to(device)
    fused = {}
    hook = None
    if flags.get("ctrast"):
        hook = model.decode_head.fusion_conv.register_forward_hook(
            lambda m, i, o: fused.update(feature=o))
    edge_aux = LC.EdgeAux().to(device) if flags.get("eps") else None
    row_attn = LC.RowHeightAttn().to(device) if flags.get("row_attn") else None
    fade = LC.FadeLite().to(device) if flags.get("fade") else None
    params = [p for p in model.parameters() if p.requires_grad]
    for m in (edge_aux, row_attn, fade):
        if m is not None:
            params += list(m.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    cw = None
    if flags.get("acw"):
        freq = np.clip(class_freq, 1e-6, None)
        w = (1.0 / np.power(freq, 0.5)).astype(np.float32)
        w = w / w.mean()
        if flags.get("crater_boost"):
            w[2] *= 1.75
            w = w / w.mean()
        cw = torch.tensor(w, device=device)
    return dict(model=model, opt=opt, fused=fused, hook=hook, edge_aux=edge_aux, row_attn=row_attn,
                fade=fade, class_weights=cw, flags=flags, exp=exp)


def set_train(st, on=True):
    for k in ("model", "edge_aux", "row_attn", "fade"):
        if st.get(k) is not None:
            st[k].train(on)


def forward_loss(st, imgs, masks, edges, it, iters, args, class_freq, device, want_parts=False):
    """Loss composition identical to the crop line (CE[+BLV][or hardness] + Dice
    [+0.1 contrast][+0.4 edge-aux][+ABL ramp])."""
    flags, model = st["flags"], st["model"]
    st["fused"].clear()
    parts = {}
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = LB._forward_logits(model, imgs, row_attn=st["row_attn"], fade=st["fade"])
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        ce_logits = logits
        if flags.get("blv"):
            ce_logits = LC.blv_perturb_logits(logits, class_freq, sigma=args.blv_sigma)
        ce = F.cross_entropy(ce_logits.float(), masks, weight=st["class_weights"], ignore_index=IGNORE)
        if want_parts:
            with torch.no_grad():
                parts["ce_clean"] = float(F.cross_entropy(logits.float(), masks, weight=st["class_weights"],
                                                          ignore_index=IGNORE))
                parts["ce_used"] = float(ce)
        if flags.get("hard"):
            ce = LB.hardness_ce(ce_logits, masks, st["class_weights"], alpha=1., gamma=2.)
            if want_parts:
                parts["ce_hard"] = float(ce)
        dsc = LC.dice_loss(logits, masks)
        loss = ce + dsc
        contrastive = None
        if flags.get("ctrast"):
            contrastive = LB.boundary_contrast_loss(st["fused"]["feature"], masks, edges)
            loss = loss + .1 * contrastive
        if flags.get("eps") and st["edge_aux"] is not None:
            e_logits = st["edge_aux"](logits)
            valid = masks != IGNORE
            e_map = F.binary_cross_entropy_with_logits(e_logits.float(), edges.unsqueeze(1).float(),
                                                       reduction="none").squeeze(1)
            loss = loss + 0.4 * (e_map[valid].mean() if valid.any() else logits.new_zeros(()))
        if flags.get("abl"):
            ramp = min(1.0, float(it + 1) / max(1.0, 0.2 * float(iters)))
            a_loss = LC.abl_lite_loss(logits, masks, edges)
            loss = loss + float(args.abl_weight) * ramp * a_loss
            parts["abl"] = float(a_loss.detach())
    st["fused"].clear()
    parts.update(loss=float(loss.detach()), ce=float(ce.detach()), dice=float(dsc.detach()),
                 contrastive=None if contrastive is None else float(contrastive.detach()))
    return loss, parts


def _grad_l1(module):
    if module is None:
        return None
    s = 0.0
    for p in module.parameters():
        if p.grad is not None:
            s += float(p.grad.detach().float().abs().sum())
    return s


def module_self_check(st, parts):
    """ROW_BLV_EFFECT-style check after the first backward (before opt.step)."""
    f = st["flags"]
    out, bad = {}, []
    if f.get("row_attn"):
        g = _grad_l1(st["row_attn"].mlp[-1])
        out["row_grad_l1"] = g
        if not g or g <= 1e-6:
            bad.append("row(grad)")
    if f.get("blv"):
        d = abs(parts["ce_used"] - parts["ce_clean"])
        out["blv_ce_delta"] = d
        if d <= 1e-4:
            bad.append("blv(ce delta)")
    if f.get("hard"):
        d = abs(parts.get("ce_hard", 0.0) - parts["ce_used"])
        out["hard_ce_delta"] = d
        if d <= 1e-6:
            bad.append("hard(ce unchanged)")
    up = getattr(st["model"], "method_upsampler", None)
    if f.get("dysample") or f.get("sapa"):
        g = _grad_l1(up)
        out["upsampler"] = type(up).__name__ if up is not None else None
        out["upsampler_grad_l1"] = g
        if up is None or not g or g <= 0:
            bad.append("upsampler(grad)")
    if f.get("eps"):
        g = _grad_l1(st["edge_aux"])
        out["edge_aux_grad_l1"] = g
        if not g or g <= 0:
            bad.append("edge_aux(grad)")
    if f.get("fade"):
        g = _grad_l1(st["fade"])
        out["fade_grad_l1"] = g
        if not g or g <= 0:
            bad.append("fade(grad)")
    if f.get("abl"):
        out["abl_loss_iter1"] = parts.get("abl")
        if not parts.get("abl") or not math.isfinite(parts["abl"]):
            bad.append("abl(loss)")
    if f.get("ctrast"):
        out["contrastive_iter1"] = parts.get("contrastive")  # may be 0 on a pair-less batch; tracked to iter 50
    if f.get("shadow_aug"):
        probe = np.full((64, 64, 3), 128, np.uint8)
        changed = any(not np.array_equal(LC.shadow_augment(probe, random.Random(s)), probe) for s in range(5))
        out["shadow_aug_changes_image"] = bool(changed)
        if not changed:
            bad.append("shadow(no-op)")
    return out, bad


# ---------------------------------------------------------------------------
# Evaluation (whole image, batch 1; OOM -> sliding 512/256 fallback)
# ---------------------------------------------------------------------------
def _logits_whole(st, x):
    lg = LB._forward_logits(st["model"], x, row_attn=st.get("row_attn"), fade=st.get("fade"))
    if lg.shape[-2:] != x.shape[-2:]:
        lg = F.interpolate(lg, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return lg.float()


def _logits_slide(st, x, crop=512, stride=256):
    _, _, h, w = x.shape
    acc = torch.zeros((1, NUM_CLASSES, h, w), device=x.device, dtype=torch.float32)
    cnt = torch.zeros((1, 1, h, w), device=x.device, dtype=torch.float32)
    ys = sorted(set(list(range(0, max(h - crop, 0) + 1, stride)) + [max(0, h - crop)]))
    xs = sorted(set(list(range(0, max(w - crop, 0) + 1, stride)) + [max(0, w - crop)]))
    for y0 in ys:
        for x0 in xs:
            p = x[:, :, y0:y0 + crop, x0:x0 + crop]
            acc[:, :, y0:y0 + p.shape[2], x0:x0 + p.shape[3]] += _logits_whole(st, p)
            cnt[:, :, y0:y0 + p.shape[2], x0:x0 + p.shape[3]] += 1
    return acc / cnt.clamp(min=1)


def _logits_mode(st, x, mode, device):
    kind, scales = mode
    h, w = x.shape[-2:]
    outs = []
    for s in scales:
        xs = x if s == 1.0 else F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = _logits_whole(st, xs) if kind == "whole" else _logits_slide(st, xs)
        if lg.shape[-2:] != (h, w):
            lg = F.interpolate(lg, size=(h, w), mode="bilinear", align_corners=False)
        outs.append(lg.float())
    return torch.stack(outs).mean(0)


class _Stopped(Exception):
    pass


def en_keys(d):
    """BoundaryMeter keys are the Chinese CLASS_NAMES; summaries use ASCII English keys
    (old summaries with Chinese keys broke PowerShell ConvertFrom-Json)."""
    zh2en = dict(zip(CLASS_ZH, CLASS_EN))
    if isinstance(d, dict):
        return {zh2en.get(k, k): en_keys(v) for k, v in d.items()}
    return d


@torch.no_grad()
def evaluate(st, loader, device, mode=("whole", (1.0,)), segfix=False, allow_slide_fallback=True,
             events=None, check_stop=True):
    """Pooled Scene5 IoU (ignore-safe; classes absent from GT and pred excluded), confusion
    rates, boundary F1/IoU (portable_b3 BoundaryMeter). Returns (result, mode_used)."""
    for k in ("model", "edge_aux", "row_attn", "fade"):
        if st.get(k) is not None:
            st[k].eval()
    while True:
        inter, union = np.zeros(NUM_CLASSES), np.zeros(NUM_CLASSES)
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES))
        bm = LB.BoundaryMeter()
        bm_fix = LB.BoundaryMeter() if segfix else None
        inter_fix, union_fix = np.zeros(NUM_CLASSES), np.zeros(NUM_CLASSES)
        n, t_fwd = 0, 0.0
        try:
            for bi, b in enumerate(loader):
                if check_stop and bi % 10 == 0 and stop_requested():
                    raise _Stopped()
                x = b[0].to(device, non_blocking=True)
                gt = b[1].numpy()[0]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                lg = _logits_mode(st, x, mode, device)
                conf, pred_t = lg.softmax(1)[0].max(0)
                pred = pred_t.cpu().numpy().astype(np.uint8)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_fwd += time.perf_counter() - t0
                valid = gt != IGNORE
                for c in range(NUM_CLASSES):
                    p, g = (pred == c) & valid, gt == c
                    inter[c] += (p & g).sum()
                    union[c] += (p | g).sum()
                idx = pred[valid].astype(np.int64) * NUM_CLASSES + gt[valid].astype(np.int64)
                cm += np.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
                bm.update(pred, gt)
                if segfix:
                    pf = LB.segfix_lite(pred, conf.float().cpu().numpy())
                    for c in range(NUM_CLASSES):
                        p, g = (pf == c) & valid, gt == c
                        inter_fix[c] += (p & g).sum()
                        union_fix[c] += (p | g).sum()
                    bm_fix.update(pf, gt)
                n += 1
            break
        except _Stopped:
            raise
        except Exception as e:  # noqa: BLE001
            if is_oom(e) and allow_slide_fallback and mode[0] == "whole":
                free_cuda()
                ev = {"type": "val_oom_fallback", "from": f"whole{list(mode[1])}", "to": "slide512_s256",
                      "error": str(e)[:300], "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                if events is not None:
                    events.append(ev)
                log(f"[WARN] {ev}")
                mode = ("slide", mode[1])
                continue
            raise

    def pack(i, u):
        iou = [float(i[c] / u[c]) if u[c] > 0 else None for c in range(NUM_CLASSES)]
        good = [v for v in iou if v is not None]
        return (float(np.mean(good)) if good else None), iou

    miou, iou = pack(inter, union)
    res = {"mIoU": miou, "per_class_IoU": dict(zip(CLASS_EN, iou)),
           "confusion_rates": {k: jf(v) for k, v in LC.confusion_rates(cm).items()},
           "boundary_metrics": en_keys(bm.compute()), "n_images": n,
           "sec_per_image": t_fwd / max(n, 1), "mode": f"{mode[0]}{list(mode[1])}"}
    if segfix:
        m2, i2 = pack(inter_fix, union_fix)
        res["segfix"] = {"mIoU": m2, "per_class_IoU": dict(zip(CLASS_EN, i2)),
                         "boundary_metrics": en_keys(bm_fix.compute())}
    return res, mode


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def make_train_loader(lists_dir, flags, micro, workers, device, seed, events, resize_to=None, limit=None):
    cc = LC.CCBank(lists_dir / "cc_bank_index.json") if flags.get("cc_paste") else None
    ds = FullImageDataset(lists_dir / "train_sub.txt", train=True, shadow_aug=flags.get("shadow_aug", False),
                          cc_bank=cc, mosaic=flags.get("mosaic", False), seed=seed, resize_to=resize_to,
                          limit=limit)
    sampler = None
    if flags.get("rcs"):
        if flags.get("crater_boost"):
            w = crater_weights(lists_dir)
        else:
            w = np.asarray(read_json(lists_dir / "rarity.json")["weights"], np.float64)
        if limit:
            w = w[:limit]
        assert len(w) == len(ds)
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(ds),
                                        replacement=True, generator=torch.Generator().manual_seed(seed))
    return ResilientLoader(ds, micro, sampler, workers, device.type == "cuda",
                           torch.Generator().manual_seed(seed + 1), events)


def make_val_loader(lists_dir, workers, device, resize_to=None, limit=None):
    ds = FullImageDataset(lists_dir / "val_sub.txt", train=False, resize_to=resize_to, limit=limit)
    # val batch is ALWAYS 1 at full 1024 (no persistent workers: re-iterated per eval)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=min(1, int(workers)),
                      pin_memory=device.type == "cuda")


def cuda_peaks(device):
    if device.type != "cuda":
        return 0.0, 0.0
    return (torch.cuda.max_memory_allocated() / 2 ** 20, torch.cuda.max_memory_reserved() / 2 ** 20)


def device_info(device):
    if device.type != "cuda":
        return {"device": "cpu"}
    free, total = torch.cuda.mem_get_info()
    return {"device": torch.cuda.get_device_name(0), "total_gb": round(total / 2 ** 30, 2),
            "free_gb_at_start": round(free / 2 ** 30, 2),
            "used_by_others_gb_at_start": round((total - free) / 2 ** 30, 2),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


# ---------------------------------------------------------------------------
# Training (resume, STOP, OOM fallback)
# ---------------------------------------------------------------------------
class _OOMRetry(Exception):
    pass


def _ckpt_on(args):
    """Real runs always checkpoint; dry runs only with --dry-save (CPU test of resume/STOP)."""
    return (not args.dry_run) or bool(getattr(args, "dry_save", False))


def run_train(exp, args, project, lists_dir, pretrain, device):
    flags = fi_flags(exp)
    work = FI_ROOT / "work_dirs" / (exp + ("_dryrun" if args.dry_run else ""))
    work.mkdir(parents=True, exist_ok=True)
    summ_path = work / ("summary_dryrun.json" if args.dry_run else "summary.json")
    prog_path = work / "summary_progress.json"
    if not args.dry_run and summ_path.exists():
        if read_json(summ_path).get("status") == "done":
            log(f"{exp}: summary.json status=done -> skip (finished summaries are never overwritten)")
            return 0
    meta = read_json(lists_dir / "meta.json")
    class_freq = np.asarray(meta["class_freq"], np.float64)
    micro, accum = int(args.batch), int(args.grad_accum)
    eff = micro * accum
    fallbacks = []
    if (work / "last.pth").exists() and _ckpt_on(args) and not args.no_resume:
        try:  # resume keeps a previous OOM fallback's micro/accum
            ck = torch.load(work / "last.pth", map_location="cpu", weights_only=False)
            if ck.get("micro_batch") and ck["micro_batch"] * ck["accum"] == eff:
                micro, accum = int(ck["micro_batch"]), int(ck["accum"])
            del ck
        except Exception:  # noqa: BLE001
            pass
    while True:
        try:
            return _train_attempt(exp, flags, args, lists_dir, pretrain, device, work, summ_path, prog_path,
                                  class_freq, micro, accum, fallbacks, meta)
        except _OOMRetry as e:
            free_cuda()
            if micro <= 1:
                fallbacks.append({"type": "train_oom", "micro_batch": micro, "accum": accum,
                                  "error": str(e)[:300], "action": "abort: OOM at micro batch 1"})
                write_json(prog_path, {"exp": exp, "status": "error", "fallbacks": fallbacks,
                                       "error": "CUDA OOM at micro batch 1", "host": host_tag()})
                log(f"[FATAL] {exp}: OOM at micro batch 1")
                return 2
            new_micro = max(1, micro // 2)
            new_accum = int(math.ceil(eff / new_micro))
            rec = {"type": "train_oom_batch_halved", "from_micro": micro, "from_accum": accum,
                   "to_micro": new_micro, "to_accum": new_accum, "effective_batch": new_micro * new_accum,
                   "lr_unchanged": args.lr, "error": str(e)[:300], "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "note": "experiment restarted from iter 0 with the new micro batch"}
            fallbacks.append(rec)
            log(f"[WARN] {rec}")
            for p in (work / "last.pth", work / "best.pth", work / "train_log.jsonl"):
                if p.exists():
                    p.unlink()
            micro, accum = new_micro, new_accum


def _train_attempt(exp, flags, args, lists_dir, pretrain, device, work, summ_path, prog_path, class_freq,
                   micro, accum, fallbacks, meta):
    LC.set_seed(args.seed)
    events = []
    iters = min(args.iters, args.dry_iters) if args.dry_run else args.iters
    resize_to = args.dry_size if args.dry_run else None
    log(f"{exp}: micro={micro} accum={accum} eff={micro * accum} lr={args.lr} iters={iters} "
        f"workers={args.workers} flags_source={flags['_flags_source']} device={device}")
    train_loader = make_train_loader(lists_dir, flags, micro, args.workers, device, args.seed, events,
                                     resize_to=resize_to, limit=(16 if args.dry_run else None))
    val_loader = make_val_loader(lists_dir, args.workers, device, resize_to=resize_to,
                                 limit=(args.dry_val_n if args.dry_run else None))
    st = build_state(exp, flags, pretrain, device, args.lr, class_freq)
    it, best, best_res, last_res, resumed, val_mode = 0, -1.0, None, None, [], ("whole", (1.0,))
    module_effect, bad = None, []
    train_time, timed_iters = 0.0, 0
    ckpt_last = work / "last.pth"
    if ckpt_last.exists() and not args.no_resume and _ckpt_on(args):
        ck = torch.load(ckpt_last, map_location="cpu", weights_only=False)
        if (ck.get("micro_batch"), ck.get("accum"), ck.get("lr"), ck.get("seed"), ck.get("target_iters")) != \
                (micro, accum, args.lr, args.seed, iters):
            raise SystemExit(f"[fullimage][FATAL] {ckpt_last} has micro={ck.get('micro_batch')} "
                             f"accum={ck.get('accum')} lr={ck.get('lr')} iters={ck.get('target_iters')}; "
                             f"refusing to resume with micro={micro} accum={accum} lr={args.lr} iters={iters}. "
                             "Move the folder away (never deleted automatically).")
        st["model"].load_state_dict(ck["model"])
        st["opt"].load_state_dict(ck["opt"])
        for k in ("row_attn", "fade", "edge_aux"):
            if st[k] is not None and k in ck:
                st[k].load_state_dict(ck[k])
        it, best = int(ck["iter"]), float(ck.get("best", -1.0))
        best_res, last_res = ck.get("best_res"), ck.get("last_res")
        for f in ck.get("fallbacks", []):
            if f not in fallbacks:
                fallbacks.append(f)
        events.extend(ck.get("events", []))
        resumed = ck.get("resumed", []) + [it]
        val_mode = (ck.get("val_mode", ["whole", [1.0]])[0], tuple(ck.get("val_mode", ["whole", [1.0]])[1]))
        module_effect = ck.get("module_effect")
        train_time, timed_iters = ck.get("train_time", 0.0), ck.get("timed_iters", 0)
        random.setstate(ck["py_rng"])
        np.random.set_state(ck["np_rng"])
        torch.set_rng_state(ck["torch_rng"])
        del ck
        log(f"{exp}: RESUMED from {ckpt_last} at iter {it} (data order after resume is not bit-identical)")
    dinfo = device_info(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    peak_train = [0.0, 0.0]
    log_path = work / "train_log.jsonl"
    t_start = time.time()
    status, error = "running", None

    def base_summary():
        pa, pr = cuda_peaks(device)
        return {
            "exp": exp, "line": "fullimage_compare (whole-image 1024)", "host": host_tag(),
            "status": status, "iters": it, "target_iters": iters,
            "best_mIoU_scene5": jf(best) if best >= 0 else None,
            "best_iter": (best_res or {}).get("iter"),
            "last_mIoU_scene5": (last_res or {}).get("mIoU"),
            "per_class_IoU_best": (best_res or {}).get("per_class_IoU"),
            "per_class_IoU_last": (last_res or {}).get("per_class_IoU"),
            "iou_rock_best": ((best_res or {}).get("per_class_IoU") or {}).get("rock"),
            "iou_crater_best": ((best_res or {}).get("per_class_IoU") or {}).get("crater"),
            "boundary_metrics_best": (best_res or {}).get("boundary_metrics"),
            "boundary_metrics_last": (last_res or {}).get("boundary_metrics"),
            "confusion_rates_best": (best_res or {}).get("confusion_rates"),
            "class_names_en": CLASS_EN, "class_names_zh": CLASS_ZH,
            "micro_batch": micro, "grad_accum": accum, "effective_batch": micro * accum,
            "requested_batch": int(args.batch), "requested_grad_accum": int(args.grad_accum),
            "lr": args.lr, "lr_rescaled": False, "seed": args.seed,
            "workers_initial": int(args.workers), "workers_final": train_loader.workers,
            "input_size": [resize_to or NATIVE, resize_to or NATIVE], "crop": None,
            "val": {"scene": 5, "n_images": (last_res or {}).get("n_images"), "batch": 1,
                    "mode": f"{val_mode[0]}{list(val_mode[1])}", "every": args.val_every,
                    "sec_last_pass": (last_res or {}).get("val_sec")},
            "peak_mem_alloc_mb": round(pa, 1), "peak_mem_reserved_mb": round(pr, 1),
            "peak_train_alloc_mb_before_first_val": round(peak_train[0], 1),
            "peak_train_reserved_mb_before_first_val": round(peak_train[1], 1),
            "sec_per_iter_train": round(train_time / timed_iters, 3) if timed_iters else None,
            "sec_per_iter_note": "optimizer step incl. all accumulation micro-steps, excl. data wait/val, "
                                 "iters 1-20 excluded",
            "wall_sec_this_session": round(time.time() - t_start, 1),
            "fallbacks": fallbacks + events, "any_fallback": bool(fallbacks or events),
            "resumed_at_iters": resumed,
            "module_effect": module_effect, "flags": dict(flags),
            "method_note": METHOD_NOTE.get(recipe_of(exp), ""),
            "subset": {"train": meta.get("train_n"), "val": meta.get("val_n"), "seed": meta.get("seed")},
            "lists_dir": str(lists_dir), "pretrain": str(pretrain), "device_info": dinfo,
            "torch": torch.__version__, "dry_run": bool(args.dry_run), "error": error,
            "comparability": "Compare ONLY with FI_* runs on the same host with identical micro/accum/"
                             "effective batch/iters/val mode. Never with crop-line or cross-machine numbers.",
            "note": "Local Scene5 subset validation, not an official test score; Moon_2/4/9 labels never used.",
            "note_zh": "本地 Scene5 子集验证，非官方测试成绩；未使用 Moon_2/4/9 标签",
        }

    def save_last():
        ck = {"model": st["model"].state_dict(), "opt": st["opt"].state_dict(), "iter": it, "best": best,
              "best_res": best_res, "last_res": last_res, "micro_batch": micro, "accum": accum, "lr": args.lr,
              "seed": args.seed, "target_iters": iters, "fallbacks": fallbacks, "events": events,
              "resumed": resumed, "val_mode": [val_mode[0], list(val_mode[1])], "module_effect": module_effect,
              "train_time": train_time, "timed_iters": timed_iters, "flags": flags, "exp": exp,
              "py_rng": random.getstate(), "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}
        for k in ("row_attn", "fade", "edge_aux"):
            if st[k] is not None:
                ck[k] = st[k].state_dict()
        tmp = work / "last.pth.tmp"
        torch.save(ck, tmp)
        os.replace(tmp, ckpt_last)

    set_train(st, True)
    st["opt"].zero_grad(set_to_none=True)
    micro_count = 0
    first_val_done = it > 0
    ctrast_nonzero = False
    t_step = None
    try:
        while it < iters:
            if stop_requested():
                raise _Stopped()
            batch = train_loader.next()
            imgs, masks, edges = (t.to(device, non_blocking=True) for t in batch[:3])
            del batch
            if micro_count % accum == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_step = time.perf_counter()
            want = module_effect is None and micro_count == 0
            try:
                loss, parts = forward_loss(st, imgs, masks, edges, it, iters, args, class_freq, device, want)
                (loss / accum).backward()
            except Exception as e:  # noqa: BLE001
                if is_oom(e) and device.type == "cuda":
                    imgs = masks = edges = loss = None
                    st["opt"].zero_grad(set_to_none=True)
                    if st["hook"] is not None:
                        st["hook"].remove()
                    st.clear()
                    raise _OOMRetry(str(e)) from None
                raise
            if flags.get("ctrast") and parts.get("contrastive"):
                ctrast_nonzero = True
            if want:
                module_effect, bad = module_self_check(st, parts)
            micro_count += 1
            if micro_count % accum:
                continue
            try:
                st["opt"].step()
            except Exception as e:  # noqa: BLE001  (AdamW state alloc can OOM on step 1)
                if is_oom(e) and device.type == "cuda":
                    imgs = masks = edges = loss = None
                    if st["hook"] is not None:
                        st["hook"].remove()
                    st.clear()
                    raise _OOMRetry(str(e)) from None
                raise
            st["opt"].zero_grad(set_to_none=True)
            it += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            if it > 20 or args.dry_run:
                train_time += time.perf_counter() - t_step
                timed_iters += 1
            if module_effect is not None and "ok" not in module_effect:
                if flags.get("row_attn"):
                    last = st["row_attn"].mlp[-1]
                    moved = float(last.weight.detach().abs().sum() + last.bias.detach().abs().sum())
                    module_effect["row_weight_l1_after_step"] = moved
                    if moved <= 1e-8:
                        bad.append("row(no update)")
                module_effect["ok"] = not bad
                module_effect["failed"] = list(bad)
                print("[module] FI_EFFECT " + json.dumps(module_effect), flush=True)
                write_json(work / "module_effect.json", module_effect)
                if bad:
                    raise RuntimeError(f"module self-check failed (module not active): {bad} {module_effect}")
            if flags.get("ctrast") and it == min(50, iters) and module_effect is not None:
                module_effect["contrastive_nonzero_by_iter50"] = ctrast_nonzero
                if not ctrast_nonzero:
                    log("[WARN] contrastive loss was 0 for all of the first 50 iters (no boundary pairs?)")
            if it % 50 == 0 or it == 1 or args.dry_run:
                pa, pr = cuda_peaks(device)
                row = {"iter": it, **{k: v for k, v in parts.items() if v is not None},
                       "micro_batch": micro, "accum": accum, "workers": train_loader.workers,
                       "s_per_iter": round(train_time / timed_iters, 3) if timed_iters else None,
                       "peak_alloc_mb": round(pa, 1), "peak_reserved_mb": round(pr, 1),
                       "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                print(row, flush=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            do_eval = (it % args.val_every == 0) or it == iters
            if do_eval:
                if not first_val_done:
                    peak_train[:] = list(cuda_peaks(device))
                    first_val_done = True
                vt = time.time()
                res, val_mode = evaluate(st, val_loader, device, mode=val_mode, events=events)
                res["iter"] = it
                res["val_sec"] = round(time.time() - vt, 1)
                last_res = res
                print({"iter": it, "val_mIoU": res["mIoU"], "per_class": res["per_class_IoU"],
                       "val_mode": res["mode"]}, flush=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"iter": it, "val": res}, ensure_ascii=False) + "\n")
                if res["mIoU"] is not None and res["mIoU"] > best:
                    best, best_res = res["mIoU"], res
                    if _ckpt_on(args):
                        bk = {"model": st["model"].state_dict(), "iter": it, "miou": best, "flags": flags,
                              "exp": exp}
                        for k in ("row_attn", "fade", "edge_aux"):
                            if st[k] is not None:
                                bk[k] = st[k].state_dict()
                        torch.save(bk, work / "best.pth.tmp")
                        os.replace(work / "best.pth.tmp", work / "best.pth")
                set_train(st, True)
            if _ckpt_on(args) and (it % args.ckpt_every == 0 or do_eval):
                save_last()
                write_json(prog_path, base_summary())
        status = "done"
    except _OOMRetry:
        raise
    except _Stopped:
        status = "stopped"
        log(f"{exp}: STOP file seen at iter {it}; saving last.pth and exiting")
        if _ckpt_on(args) and it > 0:
            save_last()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        status, error = "error", traceback.format_exc()
        log(f"[ERROR] {exp}\n{error}")
        if _ckpt_on(args) and it > 0 and st.get("model") is not None:
            try:
                save_last()
            except Exception:  # noqa: BLE001
                pass
    finally:
        if st.get("hook") is not None:
            st["hook"].remove()
    summ = base_summary()
    if status == "done":
        write_json(summ_path, summ)
        if not args.dry_run:
            write_json(prog_path, summ)
        log(f"{exp}: DONE best={summ['best_mIoU_scene5']} s/iter={summ['sec_per_iter_train']} "
            f"peak_reserved={summ['peak_mem_reserved_mb']}MB fallbacks={len(summ['fallbacks'])}")
        return 0
    write_json(work / "summary_dryrun.json" if args.dry_run else prog_path, summ)
    return EXIT_STOPPED if status == "stopped" else 1


# ---------------------------------------------------------------------------
# VRAM probe (REAL train step + one whole-image val pass, batch 1/2/4)
# ---------------------------------------------------------------------------
def run_vram_probe(exps, args, project, lists_dir, pretrain, device):
    if device.type != "cuda":
        raise SystemExit("[fullimage][FATAL] --vram-probe needs CUDA")
    out_path = FI_ROOT / "results" / f"vram_probe_{host_tag()}.json"
    prev = read_json(out_path) if out_path.exists() else {}
    meta = read_json(lists_dir / "meta.json")
    class_freq = np.asarray(meta["class_freq"], np.float64)
    budget_mb = (args.mem_cap_gb - args.headroom_gb) * 1024
    res_all = dict(prev.get("exps", {}))
    batches = [int(b) for b in args.probe_batches.split(",")]
    stopped = False
    dinfo = device_info(device)
    for exp in exps:
        flags = fi_flags(exp)
        rows = []
        for b in batches:
            if stop_requested():
                log("STOP file seen during probe")
                stopped = True
                break
            free_cuda()
            torch.cuda.reset_peak_memory_stats()
            LC.set_seed(args.seed)
            events, row = [], {"micro_batch": b}
            st, loader = None, None
            try:
                st = build_state(exp, flags, pretrain, device, args.lr, class_freq)
                loader = make_train_loader(lists_dir, flags, b, args.workers, device, args.seed, events)
                set_train(st, True)
                t_steps = []
                for i in range(args.probe_iters):
                    if stop_requested():
                        raise _Stopped()
                    batch = loader.next()
                    imgs, masks, edges = (t.to(device, non_blocking=True) for t in batch[:3])
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    loss, _ = forward_loss(st, imgs, masks, edges, i, args.iters, args, class_freq, device)
                    loss.backward()
                    st["opt"].step()
                    st["opt"].zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    t_steps.append(time.perf_counter() - t0)
                pa_t, pr_t = cuda_peaks(device)
                row.update(train_iters=args.probe_iters, peak_train_alloc_mb=round(pa_t, 1),
                           peak_train_reserved_mb=round(pr_t, 1),
                           s_per_iter=round(float(np.mean(t_steps[5:] or t_steps)), 3))
                vl = make_val_loader(lists_dir, args.workers, device, limit=args.probe_val_n or None)
                vres, vmode = evaluate(st, vl, device, events=events, allow_slide_fallback=False)
                pa, pr = cuda_peaks(device)
                row.update(val_images=vres["n_images"], val_mode=f"{vmode[0]}{list(vmode[1])}",
                           val_sec_per_image=round(vres["sec_per_image"], 3),
                           peak_alloc_mb=round(pa, 1), peak_reserved_mb=round(pr, 1),
                           fits_budget=bool(pr <= budget_mb), status="ok",
                           workers_final=loader.workers, events=events)
            except _Stopped:
                row.update(status="stopped")
                stopped = True
            except Exception as e:  # noqa: BLE001
                if is_oom(e):
                    pa, pr = cuda_peaks(device)
                    row.update(status="oom", error=str(e)[:300], peak_alloc_mb=round(pa, 1),
                               peak_reserved_mb=round(pr, 1))
                else:
                    row.update(status="error", error=traceback.format_exc()[-1500:])
            finally:
                if st is not None and st.get("hook") is not None:
                    st["hook"].remove()
                st = loader = None
                imgs = masks = edges = loss = None
                free_cuda()
            rows.append(row)
            log(f"[probe] {exp} {row}")
            if row["status"] != "ok" or not row.get("fits_budget"):
                break
        ok = [r["micro_batch"] for r in rows if r["status"] == "ok" and r.get("fits_budget")]
        res_all[exp] = {"rows": rows, "recommended_micro_batch": max(ok) if ok else None,
                        "probed_at": time.strftime("%Y-%m-%d %H:%M:%S %z")}
        if stopped:
            break
    recs = [v["recommended_micro_batch"] for v in res_all.values()]
    common = min(recs) if recs and all(r is not None for r in recs) else None
    out = {"host": host_tag(), "device_info": dinfo, "torch": torch.__version__,
           "mem_cap_gb": args.mem_cap_gb, "headroom_gb": args.headroom_gb,
           "budget_reserved_mb": round(budget_mb, 1), "probe_iters": args.probe_iters,
           "probe_val_n": args.probe_val_n or VAL_N, "workers": args.workers, "stopped": stopped,
           "criterion": "largest micro batch whose peak torch.cuda.max_memory_reserved over >=30 real "
                        "train steps (fwd+bwd+AdamW, bf16, real loader) + one whole-image val pass "
                        "<= mem_cap_gb - headroom_gb",
           "exps": res_all, "recommended_common_micro_batch": common,
           "note": "Use the SAME micro batch / grad-accum for the control and every module "
                   "(common = min over probed exps). Late-training peaks run higher than early iters; "
                   "the headroom covers it."}
    write_json(out_path, out)
    log(f"probe written to {out_path}; common micro batch = {common}")
    if stopped:
        return EXIT_STOPPED
    return 0 if common else 1


# ---------------------------------------------------------------------------
# Eval-only modes on saved weights (never auto-train)
# ---------------------------------------------------------------------------
def run_eval_only(exp, args, project, lists_dir, pretrain, device):
    ckpt = Path(args.ckpt) if args.ckpt else FI_ROOT / "work_dirs" / "FI_V_B1C" / "best.pth"
    if not ckpt.is_absolute():
        ckpt = (project / ckpt).resolve()
    check_native_path(ckpt, "--ckpt")
    if "train_B1" in str(ckpt):
        raise SystemExit("[fullimage][FATAL] work_dirs/train_B1 is off-limits")
    if "fullimage_compare" not in str(ckpt):
        raise SystemExit("[fullimage][FATAL] eval-only modes only take FI weights under fullimage_compare")
    if not ckpt.exists():
        raise SystemExit(f"[fullimage][FATAL] checkpoint missing: {ckpt} (eval-only never auto-trains)")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    src_exp = str(ck.get("exp") or ckpt.parent.name).replace("_dryrun", "")
    meta = read_json(lists_dir / "meta.json")
    st = build_state(src_exp, fi_flags(src_exp), pretrain, device, args.lr,
                     np.asarray(meta["class_freq"], np.float64))
    st["model"].load_state_dict(ck["model"])
    for k in ("row_attn", "fade", "edge_aux"):
        if st[k] is not None and k in ck:
            st[k].load_state_dict(ck[k])
    tag = exp + ("" if src_exp == "FI_V_B1C" else f"__on_{src_exp}") + ("_dryrun" if args.dry_run else "")
    work = FI_ROOT / "work_dirs" / tag
    summ_path = work / ("summary_dryrun.json" if args.dry_run else "summary.json")
    if not args.dry_run and summ_path.exists() and read_json(summ_path).get("status") == "done":
        log(f"{tag}: done -> skip")
        return 0
    vl = make_val_loader(lists_dir, args.workers, device, resize_to=args.dry_size if args.dry_run else None,
                         limit=args.dry_val_n if args.dry_run else None)
    events = []
    scales = tuple(float(s) for s in args.tta_scales.split(","))
    out = {"exp": exp, "host": host_tag(), "ckpt": str(ckpt), "ckpt_exp": src_exp, "ckpt_iter": ck.get("iter"),
           "class_names_en": CLASS_EN, "class_names_zh": CLASS_ZH, "events": events,
           "note": "eval-only on saved FI weights; Scene5 val subset; whole-image domain; "
                   "compare against whole_1.0 of the SAME checkpoint only"}
    del ck
    try:
        out["whole_1.0"], _ = evaluate(st, vl, device, ("whole", (1.0,)), segfix=(exp == "FI_M_SEGFIX"),
                                       allow_slide_fallback=False, events=events)
        if exp == "FI_M_TTA":
            out["tta_whole"], _ = evaluate(st, vl, device, ("whole", scales), allow_slide_fallback=False,
                                           events=events)
            out["tta_scales"] = list(scales)
        elif exp == "FI_M_MSINFER":
            for s in (1.0, 1.25, 1.5):
                out[f"slide512_s256_x{s}"], _ = evaluate(st, vl, device, ("slide", (s,)), events=events)
        out["status"] = "done"
    except _Stopped:
        out["status"] = "stopped"
    write_json(summ_path if out["status"] == "done" else work / "summary_progress.json", out)
    log(f"{tag}: {out['status']}")
    return 0 if out["status"] == "done" else EXIT_STOPPED


# ---------------------------------------------------------------------------
# Check-only: every module config instantiates (no training, no data)
# ---------------------------------------------------------------------------
def run_check_only(exps, args, pretrain, device):
    rows = {}
    class_freq = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
    for exp in exps:
        try:
            flags = fi_flags(exp)
            st = build_state(exp, flags, pretrain, device, args.lr, class_freq)
            mods = [k for k in ("edge_aux", "row_attn", "fade") if st[k] is not None]
            up = getattr(st["model"], "method_upsampler", None)
            if up is not None:
                mods.append(type(up).__name__)
            if st["hook"] is not None:
                mods.append("fusion_conv_hook")
                st["hook"].remove()
            active = sorted(k for k, v in flags.items() if v is True)
            rows[exp] = {"ok": True, "extra_modules": mods, "active_flags": active,
                         "flags_source": flags["_flags_source"]}
            del st
            gc.collect()
        except Exception as e:  # noqa: BLE001
            rows[exp] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        log(f"[check] {exp}: {rows[exp]}")
    for exp in EVAL_EXPS:
        rows[exp] = {"ok": True, "eval_only": True,
                     "uses": "evaluate() whole/tta/slide modes + portable_b3 segfix_lite"}
    return rows


# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=str(PROJECT_DEFAULT))
    ap.add_argument("--lists-dir", default=None, help="default fullimage_compare/lists/<HOST>")
    ap.add_argument("--smoke-src", default=None, help="crop-line smoke assets to copy (default data/smoke)")
    ap.add_argument("--prepare-lists", action="store_true")
    ap.add_argument("--exps", default="FI_V_B1C", help="comma list or ALL (default queue)")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=1, help="micro batch per forward (default 1)")
    ap.add_argument("--grad-accum", type=int, default=1, help="micro-steps per optimizer step; lr NOT rescaled")
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers (auto 2->1->0 on 1455)")
    ap.add_argument("--pretrain", default=None)
    ap.add_argument("--blv-sigma", type=float, default=0.5)
    ap.add_argument("--abl-weight", type=float, default=0.5)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--vram-probe", action="store_true")
    ap.add_argument("--probe-batches", default="1,2,4")
    ap.add_argument("--probe-iters", type=int, default=30)
    ap.add_argument("--probe-val-n", type=int, default=0, help="0 = full 350-image val pass")
    ap.add_argument("--mem-cap-gb", type=float, default=14.0)
    ap.add_argument("--headroom-gb", type=float, default=1.5)
    ap.add_argument("--ckpt", default=None, help="eval-only: FI checkpoint (default work_dirs/FI_V_B1C/best.pth)")
    ap.add_argument("--tta-scales", default="0.75,1.0,1.25")
    ap.add_argument("--check-only", action="store_true", help="instantiate every module config, no training")
    ap.add_argument("--dry-run", action="store_true", help="tiny real fwd/bwd/opt + val; never writes summary.json")
    ap.add_argument("--dry-iters", type=int, default=2)
    ap.add_argument("--dry-save", action="store_true", help="dry run also writes/resumes last.pth+best.pth "
                    "in work_dirs/<EXP>_dryrun (CPU test of resume/STOP/eval-only)")
    ap.add_argument("--dry-val-n", type=int, default=2)
    ap.add_argument("--dry-size", type=int, default=None, help="CPU dry check only: resize to NxN")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    return ap.parse_args()


def main():
    args = parse_args()
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        log("[WARN] PYTORCH_CUDA_ALLOC_CONF is set in the environment; this line never sets it. "
            "The launcher removes it; unset it for manual runs.")
    if args.dry_size and not (args.dry_run or args.check_only):
        raise SystemExit("--dry-size is only allowed with --dry-run")
    if args.batch < 1 or args.grad_accum < 1:
        raise SystemExit("--batch/--grad-accum must be >= 1")
    if args.vram_probe and args.probe_iters < 30:
        raise SystemExit("--probe-iters must be >= 30 (late-training peak lesson)")
    torch.set_num_threads(max(1, args.threads))
    project = Path(args.project_root)
    check_native_path(project, "--project-root")
    if project.resolve() != PROJECT_DEFAULT.resolve():
        log(f"[WARN] project root {project} differs from script location {PROJECT_DEFAULT}")
    lists_dir = Path(args.lists_dir) if args.lists_dir else FI_ROOT / "lists" / host_tag()
    for sub in ("lists", "work_dirs", "results", "logs"):
        (FI_ROOT / sub).mkdir(parents=True, exist_ok=True)
    LC.ensure_mmcv_stub(PORTABLE / "_lusnar_common.py")
    if args.prepare_lists:
        prepare_lists(project, lists_dir, Path(args.smoke_src) if args.smoke_src else None)
        return 0
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise SystemExit("[fullimage][FATAL] exactly one visible GPU required; set CUDA_VISIBLE_DEVICES=0")
    pretrain = LC.find_pretrain(project, args.pretrain)
    check_native_path(pretrain, "pretrained mit_b2")
    exps = [e.strip().upper() for e in args.exps.split(",") if e.strip()]
    if exps == ["ALL"]:
        exps = list(DEFAULT_QUEUE)
    for e in exps:
        if e not in TRAIN_EXPS + EVAL_EXPS:
            raise SystemExit(f"unknown exp {e}; known: {TRAIN_EXPS + EVAL_EXPS}")
    if args.check_only:
        rows = run_check_only([e for e in exps if e not in EVAL_EXPS], args, pretrain, device)
        write_json(FI_ROOT / "logs" / f"check_only_{host_tag()}.json", rows)
        return 0 if all(r["ok"] for r in rows.values()) else 1
    info = validate_lists(lists_dir, check_files=True)
    log(f"lists OK {info} | device={device_info(device)}")
    if stop_requested():
        log(f"STOP file present ({stop_file()}); not starting")
        return EXIT_STOPPED
    if args.vram_probe:
        return run_vram_probe([e for e in exps if e not in EVAL_EXPS], args, project, lists_dir, pretrain, device)
    rc = 0
    for exp in exps:
        if stop_requested():
            return EXIT_STOPPED
        if exp in EVAL_EXPS:
            r = run_eval_only(exp, args, project, lists_dir, pretrain, device)
        else:
            r = run_train(exp, args, project, lists_dir, pretrain, device)
        free_cuda()
        if r == EXIT_STOPPED:
            return EXIT_STOPPED
        rc = rc or r
    return rc


if __name__ == "__main__":
    sys.exit(main())
