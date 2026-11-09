"""Paired segmentation augmentations using only Pillow and NumPy.

Geometry shares parameters between RGB input and decoded class-ID labels.
Only RGB inputs receive photometric changes. Label padding is always 255.
Randomness comes from Python/NumPy, seeded by the existing training/worker code.
"""
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def numpy_rng_state():
    """Serialize NumPy RNG as primitives accepted by torch.load(weights_only=True)."""
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return dict(name=name, keys=keys.tolist(), position=int(position),
                has_gauss=int(has_gauss), cached_gaussian=float(cached_gaussian))


def restore_numpy_rng_state(state):
    np.random.set_state((state['name'], np.asarray(state['keys'], dtype=np.uint32),
                         state['position'], state['has_gauss'], state['cached_gaussian']))


class TrainingAugmentation:
    VERSION = 1
    DEFAULTS = dict(scale_min=0.5, scale_max=2.0, hflip_prob=0.5,
                    rotation_prob=0.3, rotation_degrees=5.0,
                    photometric_prob=0.5, brightness=0.2, contrast=0.2,
                    gamma_min=0.9, gamma_max=1.1,
                    noise_prob=0.15, noise_std=2.0,
                    blur_prob=0.1, blur_radius=0.6)

    def __init__(self, size, mode='standard', overrides=None):
        if mode not in ('none', 'basic', 'standard'):
            raise ValueError('augmentation mode must be none, basic or standard')
        if type(size) is not int or size < 1:
            raise ValueError('augmentation size must be a positive integer')
        self.size, self.mode = size, mode
        overrides = {} if overrides is None else overrides
        if not isinstance(overrides, dict):
            raise ValueError('augmentation config must be a JSON object')
        unknown = set(overrides) - set(self.DEFAULTS)
        if unknown:
            raise ValueError('Unknown augmentation parameters: ' + str(sorted(unknown)))
        self.settings = dict(self.DEFAULTS, **overrides)
        cfg = self.settings
        if any(type(v) not in (int, float) or not np.isfinite(v) for v in cfg.values()):
            raise ValueError('Augmentation parameters must be finite numbers')
        for key in ('hflip_prob', 'rotation_prob', 'photometric_prob', 'noise_prob', 'blur_prob'):
            if not 0 <= cfg[key] <= 1:
                raise ValueError(key + ' must be between 0 and 1')
        if not 0 < cfg['scale_min'] <= cfg['scale_max']:
            raise ValueError('Need 0 < scale_min <= scale_max')
        if not 0 < cfg['gamma_min'] <= cfg['gamma_max']:
            raise ValueError('Need 0 < gamma_min <= gamma_max')
        if not 0 <= cfg['brightness'] < 1 or not 0 <= cfg['contrast'] < 1:
            raise ValueError('brightness and contrast must be between 0 (inclusive) and 1 (exclusive)')
        if not 0 <= cfg['rotation_degrees'] <= 15:
            raise ValueError('rotation_degrees must be between 0 and 15 for this lunar preset')
        if cfg['noise_std'] < 0 or cfg['blur_radius'] < 0:
            raise ValueError('noise_std and blur_radius must be nonnegative')

    @classmethod
    def from_options(cls, size, mode='standard', config_path=None):
        overrides = None
        if config_path:
            overrides = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
        return cls(size, mode, overrides)

    def metadata(self):
        return dict(version=self.VERSION, mode=self.mode, size=self.size,
                    parameters=self.settings, label_ignore_value=255,
                    image_fill_rgb=[124, 116, 104])

    def _pad(self, image, mask, centered=False):
        width, height = image.size
        pad_w, pad_h = max(0, self.size - width), max(0, self.size - height)
        left, top = (pad_w // 2, pad_h // 2) if centered else (0, 0)
        padding = (left, top, pad_w - left, pad_h - top)
        return (ImageOps.expand(image, padding, fill=(124, 116, 104)),
                ImageOps.expand(mask, padding, fill=255))

    def _photometric(self, image):
        cfg = self.settings
        if random.random() < cfg['photometric_prob']:
            operations = ['brightness', 'contrast', 'gamma']
            random.shuffle(operations)
            for operation in operations:
                if operation in ('brightness', 'contrast'):
                    strength = cfg[operation]
                    factor = random.uniform(1 - strength, 1 + strength)
                    enhancer = ImageEnhance.Brightness if operation == 'brightness' else ImageEnhance.Contrast
                    image = enhancer(image).enhance(factor)
                else:
                    gamma = random.uniform(cfg['gamma_min'], cfg['gamma_max'])
                    lookup = [round(255 * (value / 255) ** gamma) for value in range(256)]
                    image = image.point(lookup * 3)
        if random.random() < cfg['noise_prob'] and cfg['noise_std'] > 0:
            array = np.array(image, dtype=np.float32)
            array += np.random.normal(0., cfg['noise_std'], array.shape).astype(np.float32)
            image = Image.fromarray(np.rint(np.clip(array, 0, 255)).astype(np.uint8))
        if random.random() < cfg['blur_prob'] and cfg['blur_radius'] > 0:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.0, cfg['blur_radius'])))
        return image

    def __call__(self, image, mask):
        if image.size != mask.size:
            raise ValueError('Augmentation image/label dimensions differ')
        if mask.mode != 'L':
            raise ValueError('Augmentation requires decoded uint8 class-ID labels (PIL mode L)')
        image = image.convert('RGB')
        cfg = self.settings
        width, height = image.size
        if self.mode == 'none':
            # Deterministic letterbox: fixed tensor shape without random changes.
            scale = min(self.size / width, self.size / height)
        else:
            scale = random.uniform(cfg['scale_min'], cfg['scale_max'])
        dimensions = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(dimensions, Image.Resampling.BILINEAR)
        mask = mask.resize(dimensions, Image.Resampling.NEAREST)
        image, mask = self._pad(image, mask, centered=self.mode == 'none')
        if self.mode == 'none':
            return image, mask
        left = random.randint(0, image.width - self.size)
        top = random.randint(0, image.height - self.size)
        box = (left, top, left + self.size, top + self.size)
        image, mask = image.crop(box), mask.crop(box)
        if random.random() < cfg['hflip_prob']:
            image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
        if self.mode == 'standard':
            if random.random() < cfg['rotation_prob']:
                degrees = random.uniform(-cfg['rotation_degrees'], cfg['rotation_degrees'])
                image = image.rotate(degrees, resample=Image.Resampling.BILINEAR,
                                     fillcolor=(124, 116, 104))
                mask = mask.rotate(degrees, resample=Image.Resampling.NEAREST, fillcolor=255)
            image = self._photometric(image)
        return image, mask
