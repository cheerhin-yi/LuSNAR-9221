"""Windows/Conda SegFormer-B3 LuSNAR training. Run --help; see README.md.

Defaults follow https://github.com/zqyu9/LuSNAR-dataset, not inferred file contents.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_segformer import SegFormerB3
from augmentations import TrainingAugmentation, numpy_rng_state, restore_numpy_rng_state

NAMES = ['lunar_regolith', 'impact_crater', 'rock', 'mountain', 'sky']
COLORS = [[187, 70, 156], [120, 0, 200], [232, 250, 80], [173, 69, 31], [34, 201, 248]]
EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed(seed)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def device_for(value):
    device = torch.device(value)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Install a CUDA PyTorch build in lusnar; run inspect. CPU test: --device cpu')
        index = device.index if device.index is not None else torch.cuda.current_device()
        count = torch.cuda.device_count()
        if not 0 <= index < count:
            raise RuntimeError(f'Requested {device}, but only {count} CUDA device(s) are visible. '
                               'cuda:1 requires two visible GPUs. Check CUDA_VISIBLE_DEVICES or pass --device cuda:0.')
        torch.cuda.set_device(device)
        print('Selected device:', device, 'GPU:', torch.cuda.get_device_name(device),
              'VRAM GiB:', round(torch.cuda.get_device_properties(device).total_memory / 2**30, 2))
        # Fail early if the installed CUDA build does not support this GPU.
        x = torch.randn(16, 16, device=device)
        (x @ x).sum().item()
    return device


def schema_for(args):
    if args.schema:
        obj = json.loads(Path(args.schema).read_text(encoding='utf-8-sig'))
    else:
        obj = dict(names=NAMES, colors=COLORS, encoding='rgb', ignore_colors=[])
    if obj['encoding'] not in ('rgb', 'ids'): raise ValueError('encoding must be rgb or ids')
    n = len(obj['names'])
    if not 2 <= n <= 255: raise ValueError('Need 2..255 classes; 255 is reserved for ignore')
    if len(obj['colors']) != n: raise ValueError('Need one RGB color per class')
    if any(len(c) != 3 or any(not isinstance(v, int) or not 0 <= v <= 255 for v in c) for c in obj['colors']):
        raise ValueError('colors must be RGB integer triples in 0..255')
    values = obj['colors'] if obj['encoding'] == 'rgb' else obj['values']
    ignored = obj.get('ignore_colors' if obj['encoding'] == 'rgb' else 'ignore_values', [])
    keys = [tuple(v) if isinstance(v, list) else v for v in values]
    ignored_keys = [tuple(v) if isinstance(v, list) else v for v in ignored]
    if len(keys) != n or len(set(keys)) != n or set(keys) & set(ignored_keys):
        raise ValueError('Class label values must be unique, match class count and not overlap ignore values')
    return obj


def decode_mask(path, schema):
    with Image.open(path) as im:
        if schema['encoding'] == 'rgb':
            arr = np.asarray(im.convert('RGB'), dtype=np.uint32)
            raw = (arr[..., 0] << 16) | (arr[..., 1] << 8) | arr[..., 2]
            values = [(r << 16) | (g << 8) | b for r, g, b in schema['colors']]
            ignored = [(r << 16) | (g << 8) | b for r, g, b in schema.get('ignore_colors', [])]
        else:
            raw = np.asarray(im)
            if raw.ndim != 2: raise ValueError(f'Expected single channel indexed mask: {path}')
            values, ignored = schema['values'], schema.get('ignore_values', [])
    result = np.full(raw.shape, 255, dtype=np.uint8)
    known = np.isin(raw, values + ignored)
    if not known.all():
        unknown = np.unique(raw[~known])[:12].tolist()
        if schema['encoding'] == 'rgb': unknown = [f'#{v:06X}' for v in unknown]
        raise ValueError(f'Unknown label values {unknown} in {path}. Confirm RGB palette / use --schema. No automatic relabeling.')
    for idx, value in enumerate(values): result[raw == value] = idx
    return result


def collect(args):
    root = Path(args.data_root).resolve()
    if not root.is_dir(): raise FileNotFoundError(f'Data root unavailable: {root}')
    pairs = []
    rgb_dirs = sorted(p for p in root.rglob('*') if p.is_dir() and p.name.lower() == args.image_dir.lower())
    for rgb_dir in rgb_dirs:
        camera = rgb_dir.parent
        if args.camera != 'all' and camera.name.lower() != args.camera.lower(): continue
        if args.camera == 'all' and not (camera.name.lower().startswith('image') and camera.name[5:].isdigit()): continue
        masks = [p for p in camera.iterdir() if p.is_dir() and p.name.lower() == args.mask_dir.lower()]
        if len(masks) != 1: raise ValueError(f'Expected one {args.mask_dir} folder beside {rgb_dir}')
        # Scene is first Moon_* ancestor, preserving scene split for both cameras.
        ancestors = [p.name for p in camera.parents if p.name.lower().startswith('moon_')]
        if not ancestors: raise ValueError(f'Missing Moon_* scene ancestor: {rgb_dir}')
        scene = ancestors[0]
        by_stem = {}
        for mask in sorted(masks[0].rglob('*')):
            if mask.is_file() and mask.suffix.lower() in EXTS:
                key = mask.relative_to(masks[0]).with_suffix('').as_posix().lower()
                if key in by_stem: raise ValueError(f'Ambiguous mask stem: {mask}')
                by_stem[key] = mask
        seen = set()
        for image in sorted(rgb_dir.rglob('*')):
            if not image.is_file() or image.suffix.lower() not in EXTS: continue
            key = image.relative_to(rgb_dir).with_suffix('').as_posix().lower()
            if key in seen: raise ValueError(f'Duplicate image stem: {image}')
            seen.add(key)
            if key not in by_stem: raise ValueError(f'Missing same-stem label for {image}')
            pairs.append(dict(image=str(image), mask=str(by_stem[key]), scene=scene))
        extra = set(by_stem) - seen
        if extra: raise ValueError(f'{len(extra)} unpaired labels in {masks[0]}; first: {sorted(extra)[0]}')
    if not pairs:
        raise ValueError(f'No paired data. Expected Moon_*/{args.camera}/{args.image_dir}, {args.mask_dir}. Run inspect; adjust --camera / --image-dir / --mask-dir.')
    return pairs


def split_data(pairs, args):
    if args.train_all:
        if args.val_scenes is not None or args.test_scenes is not None or getattr(args, 'split_manifest', None):
            raise ValueError('--train-all does not hold out scenes. Use --scene-split with scene options or a split manifest.')
        if not pairs:
            raise ValueError('No paired training images found')
        print('ALL-DATA TRAINING: all collected scene/camera pairs are used for training; no validation/test set.')
        print('Training scenes:', sorted({p['scene'] for p in pairs}))
        return dict(train=list(pairs), val=[], test=[])
    if args.split_manifest:
        if args.val_scenes is not None or args.test_scenes is not None:
            raise ValueError('Use --split-manifest OR --val-scenes/--test-scenes')
        result = json.loads(Path(args.split_manifest).read_text(encoding='utf-8-sig'))
        if set(result) != {'train', 'val', 'test'} or not result['train'] or not result['val']:
            raise ValueError('Manifest must contain train/val/test, with nonempty train and val')
        available = {os.path.normcase(str(Path(p['image']).resolve())): p for p in pairs}
        used, scene_sets = set(), []
        for key in ('train', 'val', 'test'):
            scenes = set()
            for pair in result[key]:
                image = os.path.normcase(str(Path(pair['image']).resolve()))
                if image in used or image not in available or pair != available[image]:
                    raise ValueError(f'Duplicate / inconsistent / unavailable pair in manifest: {pair}')
                used.add(image)
                scenes.add(pair['scene'])
            if any(scenes & prior for prior in scene_sets):
                raise ValueError('Manifest mixes the same Moon scene across splits')
            scene_sets.append(scenes)
        if used != set(available): raise ValueError('Manifest must account for every collected image exactly once')
        print('Reusing exact split manifest:', args.split_manifest)
        print('Scene split:', {k: sorted({p['scene'] for p in v}) for k, v in result.items()})
        return result
    scenes = sorted({x['scene'] for x in pairs}, key=lambda s: (int(s[5:]) if s[5:].isdigit() else 10**9, s))
    available = set(scenes)
    if args.val_scenes is None and args.test_scenes is None:
        groups = dict(train={'Moon_1', 'Moon_3', 'Moon_5', 'Moon_6', 'Moon_7', 'Moon_8'},
                      val={'Moon_2'}, test={'Moon_4', 'Moon_9'})
        required = set.union(*groups.values())
        missing = required - available
        if missing:
            raise ValueError(f'Missing paired data for the fixed split: {", ".join(sorted(missing))}. '
                             f'Check --data-root and {args.camera}/color and label folders.')
        excluded = available - required
        if excluded:
            print('Scenes outside the fixed split:', ', '.join(sorted(excluded)))
    else:
        if len(scenes) < 2:
            raise ValueError('Need >=2 Moon scenes for independent validation; specify the actual data root')
        val = set(args.val_scenes if args.val_scenes is not None else [scenes[-1]])
        default_test = [scenes[-2]] if len(scenes) >= 3 else []
        test = set(args.test_scenes if args.test_scenes is not None else default_test)
        if not val or val & test or not (val | test) <= available:
            raise ValueError(f'Invalid/overlapping val/test scenes. Available: {scenes}')
        groups = dict(train=available - val - test, val=val, test=test)

    scene_group = {scene: key for key, group in groups.items() for scene in group}
    result = {key: [] for key in groups}
    for pair in pairs:
        key = scene_group.get(pair['scene'])
        if key is not None:
            result[key].append(pair)
    if not result['train']: raise ValueError('No training scenes remain')
    print('Scene split:', {k: sorted({x['scene'] for x in v}) for k, v in result.items()})
    return result


def validate_data(splits, schema):
    counts = {key: np.zeros(len(schema['names']), dtype=np.int64) for key in splits}
    for key, pairs in splits.items():
        for pair in tqdm(pairs, desc=f'Checking {key}'):
            mask = decode_mask(pair['mask'], schema)
            with Image.open(pair['image']) as im:
                if im.size != (mask.shape[1], mask.shape[0]):
                    raise ValueError(f'Image/mask size mismatch: {pair}')
                im.verify()
            valid = mask[mask != 255]
            if not valid.size: raise ValueError(f'All pixels ignored: {pair["mask"]}')
            counts[key] += np.bincount(valid, minlength=len(schema['names']))
    print('Pixel counts:', {k: dict(zip(schema['names'], v.tolist())) for k, v in counts.items()})
    if (counts['train'] == 0).any(): print('WARNING: Some classes are absent from training data')
    return {k: v.tolist() for k, v in counts.items()}


class LuSNAR(Dataset):
    def __init__(self, pairs, schema, size, train, augmentation=None):
        self.pairs, self.schema, self.size, self.train = pairs, schema, size, train
        self.augmentation = ((augmentation if augmentation is not None else TrainingAugmentation(size))
                             if train else None)
    def __len__(self): return len(self.pairs)
    def __getitem__(self, index):
        pair = self.pairs[index]
        with Image.open(pair['image']) as im: image = im.convert('RGB')
        mask = Image.fromarray(decode_mask(pair['mask'], self.schema))
        # Geometry is paired; only input RGB receives photometric changes.
        if self.train:
            image, mask = self.augmentation(image, mask)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.
        x = (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
        y = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        return x, y


def slide(model, x, crop, nclass):
    h, w = x.shape[-2:]
    stride = max(1, crop * 3 // 4)
    ys = sorted(set(list(range(0, max(h - crop, 0), stride)) + [max(h - crop, 0)]))
    xs = sorted(set(list(range(0, max(w - crop, 0), stride)) + [max(w - crop, 0)]))
    pred = torch.zeros((1, nclass, h, w), device=x.device)
    count = torch.zeros((1, 1, h, w), device=x.device)
    for top in ys:
        for left in xs:
            patch = x[..., top:top+crop, left:left+crop]
            ph, pw = patch.shape[-2:]
            padded = F.pad(patch, (0, max(0, crop-pw), 0, max(0, crop-ph)))
            logits = model(padded)[..., :ph, :pw]
            pred[..., top:top+ph, left:left+pw] += logits.float()
            count[..., top:top+ph, left:left+pw] += 1
    return pred / count


def scores(hist):
    hist = hist.double()
    union = hist.sum(1) + hist.sum(0) - hist.diag()
    iou = hist.diag() / union.clamp_min(1)
    present = union > 0
    return dict(miou=float(iou[present].mean()), pixel_accuracy=float(hist.diag().sum()/hist.sum().clamp_min(1)),
                iou=[float(v) if p else None for v, p in zip(iou, present)])


@torch.no_grad()
def evaluate(model, loader, device, crop, nclass, amp):
    model.eval()
    # Preserve training RNG while using a fixed evaluation seed.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    hist = torch.zeros(nclass, nclass, dtype=torch.int64, device=device)
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(12345)
        if device.type == 'cuda': torch.cuda.manual_seed(12345)
        for x, y in tqdm(loader, desc='Validation/test', leave=False):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = slide(model, x, crop, nclass)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite evaluation logits; no metrics/checkpoint selected for this epoch')
            pred = logits.argmax(1)
            valid = y != 255
            hist += torch.bincount(nclass*y[valid] + pred[valid], minlength=nclass*nclass).reshape(nclass, nclass)
    return scores(hist.cpu())


@torch.no_grad()
def save_previews(model, pairs, schema, device, crop, out, count):
    """Original image | label | prediction, for visual checks of the best model."""
    from PIL import ImageDraw
    model.eval()
    dataset = LuSNAR(pairs[:count], schema, crop, False)
    palette = np.asarray(schema['colors'] + [[0, 0, 0]] * (256-len(schema['colors'])), dtype=np.uint8)
    out.mkdir(parents=True, exist_ok=True)
    for index, pair in enumerate(pairs[:count]):
        x, y = dataset[index]
        logits = slide(model, x[None].to(device), crop, len(schema['names']))
        if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite preview logits')
        pred = logits.argmax(1)[0].cpu().numpy()
        with Image.open(pair['image']) as im: image = im.convert('RGB')
        label = Image.fromarray(palette[y.numpy()])
        prediction = Image.fromarray(palette[pred])
        panels = []
        for panel in (image, label, prediction):
            width = 384
            panels.append(panel.resize((width, max(1, round(panel.height*width/panel.width))), Image.Resampling.NEAREST))
        canvas = Image.new('RGB', (3*panels[0].width, panels[0].height+28), (255,255,255))
        draw = ImageDraw.Draw(canvas)
        for j, panel in enumerate(panels):
            canvas.paste(panel, (j*panel.width, 28))
            draw.text((j*panel.width+8, 8), ['Image', 'Ground truth', 'Prediction'][j], fill=(0,0,0))
        canvas.save(out/f'{index+1:02d}_{pair["scene"]}_{Path(pair["image"]).stem}.png')


def save_checkpoint(path, obj):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def backward_and_step(loss, model, optimizer, scaler):
    """Let GradScaler handle genuine AMP overflow without updating bad weights.

    Keep strict clipping: a nonfinite aggregate norm with individually finite
    gradients must not accidentally bypass GradScaler's per-element checks.
    """
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    try:
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    except RuntimeError:
        bad = [name for name, param in model.named_parameters()
               if param.grad is not None and not torch.isfinite(param.grad).all().item()]
        if not bad:
            # Could be aggregate norm overflow or another error; do not step.
            raise
        if not scaler.is_enabled():
            raise FloatingPointError(f'Nonfinite FP32 gradients in {bad[:6]}. '
                                     'Training stopped before optimizer update; inspect numerical stability.') from None
        old_scale = scaler.get_scale()
        # unscale_ has recorded these inf/NaN gradients; step() skips the update,
        # and update() reduces the scale. Clipping raised before mutating grads.
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        tqdm.write(f'AMP overflow: skipped batch; scale {old_scale:g} -> {scaler.get_scale():g}; '
                   f'first affected tensors: {bad[:3]}')
        return False
    scaler.step(optimizer)
    scaler.update()
    return True


def train(args):
    if args.size < 64 or args.size % 32: raise ValueError('--size must be >=64 and divisible by 32')
    if args.batch_size < 2: raise ValueError('Use batch-size >=2 for BatchNorm training')
    if args.epochs < 1 or args.workers < 0 or args.lr <= 0 or args.preview_count < 0: raise ValueError('Invalid training settings')
    augmentation = TrainingAugmentation.from_options(args.size, args.augmentation, args.augmentation_config)
    augmentation_config = augmentation.metadata()
    torch.set_num_threads(args.cpu_threads)
    device = device_for(args.device)
    seed_all(args.seed, device)
    if args.resume:
        print('Initialization: restore local last.pth; no pretrained download.')
    elif args.pretrained:
        print(f'Initialization: pretrained encoder explicitly requested: {args.pretrained}')
    else:
        print('Initialization: RANDOM / OFFLINE. No ImageNet pretrained weights will be loaded.')
    schema = schema_for(args)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'last.pth').exists() and not args.resume:
        raise ValueError('Output already contains a run. Use --resume or a new --output')
    splits = split_data(collect(args), args)
    counts = validate_data(splits, schema)
    has_validation = bool(splits['val'])
    identity = dict(augmentation=augmentation_config, train_all=args.train_all,
                    splits=splits, schema=schema, size=args.size, batch_size=args.batch_size,
                    lr=args.lr, epochs=args.epochs, seed=args.seed, no_amp=args.no_amp,
                    model='SegFormer-B3', implementation='transformers.SegformerForSemanticSegmentation')
    if args.resume:
        previous = json.loads((out/'run_identity.json').read_text(encoding='utf-8'))
        if previous != identity: raise ValueError('Resume requires identical data paths, schema, augmentation and training settings. Use a new --output for changed settings.')
    else:
        dump(out/'run_identity.json', identity)
    dump(out/'schema.json', schema)
    dump(out/'split_manifest.json', splits)
    dump(out/'pixel_counts.json', counts)
    dump(out/'args.json', vars(args))
    dump(out/'augmentation.json', augmentation_config)
    print('Training augmentation:', augmentation_config)
    print('Samples:', {k: len(v) for k, v in splits.items()})
    if args.check_only:
        print('Data check passed. No training performed.')
        return
    if len(splits['train']) < args.batch_size: raise ValueError('Not enough training pairs for batch-size')
    kwargs = dict(num_workers=args.workers, pin_memory=device.type == 'cuda', worker_init_fn=seed_worker)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(LuSNAR(splits['train'], schema, args.size, True, augmentation=augmentation),
                              batch_size=args.batch_size, shuffle=True, drop_last=not args.train_all, generator=generator, **kwargs)
    val_loader = (DataLoader(LuSNAR(splits['val'], schema, args.size, False), batch_size=1, **kwargs)
                  if has_validation else None)
    # Resume uses only local checkpoint tensors; never re-download pretrained weights.
    source = None if args.from_scratch or args.resume else args.pretrained
    model = SegFormerB3(schema['names'], pretrained=source,
                       local_files_only=args.local_files_only, revision=args.revision,
                       announce_initialization=not args.resume)
    if args.resume:
        metadata = json.loads((out/'initialization.json').read_text(encoding='utf-8'))
        model.initialization = metadata
    else:
        dump(out/'initialization.json', model.initialization)
    model.config.save_pretrained(out/'model_config')
    import importlib.metadata
    dump(out/'environment.json', dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
         transformers=importlib.metadata.version('transformers'), cuda=torch.version.cuda,
         gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None))
    model.to(device)
    # AdamW head multiplier and no decay on one-dimensional norm/bias terms.
    groups = {}
    for name, param in model.named_parameters():
        multiplier = 10. if name.startswith('network.decode_head.') else 1.
        decay = 0. if param.ndim == 1 or name.endswith('.bias') else .01
        groups.setdefault((multiplier, decay), []).append(param)
    optimizer = torch.optim.AdamW([dict(params=v, lr=args.lr*k[0], weight_decay=k[1], multiplier=k[0]) for k, v in groups.items()], betas=(.9, .999))
    amp = device.type == 'cuda' and not args.no_amp
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    print('Precision:', 'AMP FP16 with dynamic loss scaling' if amp else 'FP32 (AMP disabled)')
    steps = len(train_loader) * args.epochs
    warmup = min(1500, max(1, steps//10))
    start, best, global_step = 0, -1., 0
    best_train_loss = float('inf')
    if args.resume:
        ckpt = torch.load(out/'last.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        start, best, global_step = ckpt['epoch']+1, ckpt['best_miou'], ckpt['global_step']
        best_train_loss = ckpt.get('best_train_loss', float('inf'))
        random.setstate(ckpt['rng_python'])
        if 'rng_numpy' in ckpt: restore_numpy_rng_state(ckpt['rng_numpy'])
        torch.set_rng_state(ckpt['rng_torch'])
        if device.type == 'cuda':
            if ckpt.get('rng_cuda_device') is not None:
                torch.cuda.set_rng_state(ckpt['rng_cuda_device'], device)
            elif ckpt.get('rng_cuda'):
                old_states = ckpt['rng_cuda']
                old_index = device.index if device.index is not None else torch.cuda.current_device()
                torch.cuda.set_rng_state(old_states[old_index] if old_index < len(old_states) else old_states[0], device)
        generator.set_state(ckpt['rng_loader'])
        print('Resuming at epoch', start+1)
    with (out/'history.csv').open('a' if args.resume else 'w', newline='', encoding='utf-8') as stream:
        fields = (['epoch', 'train_loss', 'miou', 'pixel_accuracy', 'seconds'] if has_validation
                  else ['epoch', 'train_loss', 'seconds'])
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not args.resume: writer.writeheader()
        for epoch in range(start, args.epochs):
            model.train()
            total_loss, sample_count, begin = 0., 0, time.time()
            consecutive_overflows, skipped_batches = 0, 0
            progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
            for x, y in progress:
                factor = ((global_step+1)/warmup) if global_step < warmup else max(0., 1.-(global_step-warmup)/max(1, steps-warmup))
                for group in optimizer.param_groups: group['lr'] = args.lr * group['multiplier'] * factor
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y, ignore_index=255, reduction='sum') / (y != 255).sum().clamp_min(1)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss; inspect labels or use --no-amp')
                if not backward_and_step(loss, model, optimizer, scaler):
                    consecutive_overflows += 1
                    skipped_batches += 1
                    if consecutive_overflows >= 20:
                        raise FloatingPointError('20 consecutive AMP overflows; training stopped. '
                                                 'Retry with --no-amp and a new --output directory.')
                    continue
                consecutive_overflows = 0
                global_step += 1
                total_loss += loss.item()*x.shape[0]
                sample_count += x.shape[0]
                progress.set_postfix(loss=f'{loss.item():.4f}')
            if sample_count == 0:
                raise FloatingPointError('No successful optimizer updates this epoch. '
                                         'Retry with --no-amp and a new --output directory.')
            if skipped_batches:
                print(f'Epoch {epoch+1}: skipped {skipped_batches} AMP overflow batches; '
                      'loss average includes successful batches only.')
            train_loss = total_loss / sample_count
            if has_validation:
                metrics = evaluate(model, val_loader, device, args.size, len(schema['names']), amp)
                improved = metrics['miou'] > best
                best = max(best, metrics['miou'])
                best_name, criterion = 'best.pth', 'validation_miou'
            else:
                metrics = dict(train_loss=train_loss)
                improved = train_loss < best_train_loss
                best_train_loss = min(best_train_loss, train_loss)
                best_name, criterion = 'best_train_loss.pth', 'training_loss'
            row = dict(epoch=epoch+1, train_loss=train_loss, seconds=time.time()-begin)
            if has_validation:
                row.update(miou=metrics['miou'], pixel_accuracy=metrics['pixel_accuracy'])
            writer.writerow(row)
            stream.flush()
            print(row, 'best mIoU:' if has_validation else 'lowest training loss:',
                  best if has_validation else best_train_loss)
            if has_validation:
                print('Validation class IoU:', dict(zip(schema['names'], metrics['iou'])))
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                         epoch=epoch, best_miou=best, best_train_loss=best_train_loss,
                         selection_criterion=criterion, global_step=global_step, schema=schema, metrics=metrics,
                         model_config=model.config.to_dict(), initialization=model.initialization,
                         rng_python=random.getstate(), rng_numpy=numpy_rng_state(), rng_torch=torch.get_rng_state(),
                         rng_cuda_device=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                         rng_loader=generator.get_state())
            if improved:
                save_checkpoint(out/best_name, dict(model=model.state_dict(), schema=schema, epoch=epoch,
                                                   selection_criterion=criterion, metrics=metrics,
                                                   model_config=model.config.to_dict(), initialization=model.initialization))
                dump(out/'best_metrics.json', dict(epoch=epoch+1, classes=schema['names'], selection_criterion=criterion, **metrics))
            save_checkpoint(out/'last.pth', state)
            if improved and args.preview_count:
                preview_pairs = splits['val'] if has_validation else splits['train']
                preview_dir = out/('previews' if has_validation else 'training_previews')
                save_previews(model, preview_pairs, schema, device, args.size, preview_dir, args.preview_count)
    # Test only with the checkpoint selected using validation mIoU.
    if splits['test']:
        best_ckpt = torch.load(out/'best.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(best_ckpt['model'])
        test_loader = DataLoader(LuSNAR(splits['test'], schema, args.size, False), batch_size=1, **kwargs)
        test_metrics = evaluate(model, test_loader, device, args.size, len(schema['names']), amp)
        dump(out/'test_metrics.json', dict(classes=schema['names'], **test_metrics))
        print('Held-out test:', test_metrics)
    print('Saved selected checkpoint:', out/('best.pth' if has_validation else 'best_train_loss.pth'))


def inspect(args):
    print('Python:', sys.version, '\nExecutable:', sys.executable, '\nConda:', os.getenv('CONDA_DEFAULT_ENV'))
    print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda, 'CUDA available:', torch.cuda.is_available())
    import importlib.metadata
    try: print('transformers:', importlib.metadata.version('transformers'))
    except importlib.metadata.PackageNotFoundError: print('transformers: NOT INSTALLED; run pip install -r requirements.txt')
    for i in range(torch.cuda.device_count()): print(i, torch.cuda.get_device_name(i))
    try: subprocess.run(['nvidia-smi'], check=False)
    except FileNotFoundError: pass
    root = Path(args.data_root)
    print('DATA ROOT:', root, 'exists:', root.is_dir())
    if root.is_dir():
        for scene in sorted(root.iterdir())[:12]:
            if not scene.is_dir(): continue
            print(scene.name)
            for folder in sorted(scene.rglob('*')):
                if folder.is_dir() and len(folder.relative_to(scene).parts) <= 3:
                    samples = [p.name for p in folder.iterdir() if p.is_file()][:3]
                    print('  ', folder.relative_to(scene), samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__ + '\nDefaults: FP32, random initialization (offline), batch=4, crop=512, epochs=100.',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['inspect', 'train'])
    parser.add_argument('--data-root', default=r'D:\LuSNAR_Seg\data')
    parser.add_argument('--output', default=r'runs/segformer_b3_gpu0_scene_split_aug')
    parser.add_argument('--camera', default='image0', help='camera folder name; all includes image0 and image1')
    parser.add_argument('--image-dir', default='color')
    parser.add_argument('--mask-dir', default='label')
    parser.add_argument('--schema', help='custom class names, RGB palette / indexed IDs JSON')
    data_mode = parser.add_mutually_exclusive_group()
    data_mode.add_argument('--train-all', dest='train_all', action='store_true',
                           help='train on all collected scenes with no validation/test holdout')
    data_mode.add_argument('--scene-split', dest='train_all', action='store_false',
                           help='use the fixed scene split (default: validation Moon_2; test Moon_4 and Moon_9)')
    parser.set_defaults(train_all=False)
    parser.add_argument('--val-scenes', nargs='+', default=None)
    parser.add_argument('--test-scenes', nargs='*', default=None, help='pass without values to disable test split')
    parser.add_argument('--split-manifest', help='reuse a previous SegNeXt/SF split_manifest.json exactly')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=0, help='0 is Windows-safe; increase after checking RAM')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0', help='cuda:0 is the first CUDA-visible GPU')
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument('--pretrained', default=None,
                                help='optional MiT-B3 repo ID or local model directory; default is random initialization')
    initialization.add_argument('--from-scratch', action='store_true',
                                help='explicit random initialization (already the default), no weights download')
    parser.add_argument('--revision', help='optional Hugging Face commit/tag for pretrained weights')
    parser.add_argument('--local-files-only', action='store_true', help='load pretrained weights only from local files/cache')
    parser.add_argument('--resume', action='store_true', help='resume last.pth in --output, keeping original training settings')
    parser.add_argument('--check-only', action='store_true')
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument('--amp', dest='no_amp', action='store_false', help='opt into FP16 AMP (default is FP32)')
    precision.add_argument('--no-amp', dest='no_amp', action='store_true', help='FP32, already the default')
    parser.set_defaults(no_amp=True)
    parser.add_argument('--preview-count', type=int, default=3,
                        help='previews of validation samples in scene-split mode, or training samples in full-data mode')
    parser.add_argument('--augmentation', choices=['none', 'basic', 'standard'], default='standard')
    parser.add_argument('--augmentation-config', help='optional JSON overrides for augmentation parameters')
    args = parser.parse_args()
    inspect(args) if args.command == 'inspect' else train(args)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
