"""Standard Hugging Face SegFormer-B3, with optional NVIDIA MiT-B3 encoder weights.

No model architecture is reimplemented here. See SOURCES.md for provenance.
"""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

B3_ARCHITECTURE = dict(
    num_channels=3, num_encoder_blocks=4,
    depths=[3, 4, 18, 3], hidden_sizes=[64, 128, 320, 512],
    num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1],
    patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
    mlp_ratios=[4, 4, 4, 4], decoder_hidden_size=768,
    hidden_act='gelu', hidden_dropout_prob=0., attention_probs_dropout_prob=0.,
    classifier_dropout_prob=.1, drop_path_rate=.1, layer_norm_eps=1e-6,
    reshape_last_stage=True, semantic_loss_ignore_index=255,
)


class SegFormerB3(nn.Module):
    def __init__(self, names, pretrained=None, local_files_only=False, revision=None, announce_initialization=True):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerModel
        except ImportError as exc:
            raise ImportError('Install dependencies in lusnar: python -m pip install -r requirements.txt') from exc
        config = SegformerConfig(**B3_ARCHITECTURE, num_labels=len(names),
                                  id2label={i: name for i, name in enumerate(names)},
                                  label2id={name: i for i, name in enumerate(names)})
        self.network = SegformerForSemanticSegmentation(config)
        self.initialization = dict(pretrained_source=None, resolved_commit=None,
                                   type='random', implementation='transformers.SegformerForSemanticSegmentation')
        if pretrained:
            kwargs = dict(local_files_only=local_files_only)
            if revision: kwargs['revision'] = revision
            try:
                source_config = SegformerConfig.from_pretrained(pretrained, **kwargs)
                # Encoder fields must match B3. Classifier labels / decoder config
                # in a classification checkpoint are intentionally not reused.
                for key in ('depths', 'hidden_sizes', 'num_attention_heads', 'sr_ratios',
                            'patch_sizes', 'strides', 'mlp_ratios', 'num_channels'):
                    if getattr(source_config, key) != B3_ARCHITECTURE[key]:
                        raise ValueError(f'{pretrained} is not a compatible MiT-B3: {key}')
                backbone, info = SegformerModel.from_pretrained(
                    pretrained, config=source_config, output_loading_info=True, **kwargs)
                if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
                    raise ValueError(f'Incomplete pretrained encoder: {info}')
                unexpected = [k for k in info.get('unexpected_keys', [])
                              if not k.startswith(('classifier.', 'decode_head.'))]
                if unexpected: raise ValueError(f'Unexpected encoder weights: {unexpected[:8]}')
                self.network.segformer.load_state_dict(backbone.state_dict(), strict=True)
                self.initialization = dict(pretrained_source=str(pretrained),
                    resolved_commit=getattr(source_config, '_commit_hash', None),
                    type='pretrained_encoder_random_segmentation_head',
                    implementation='transformers.SegformerForSemanticSegmentation')
                del backbone
                print(f'Loaded pretrained MiT-B3 encoder: {pretrained}; new {len(names)}-class decoder')
            except Exception as exc:
                raise RuntimeError(
                    f'Unable to load complete MiT-B3 weights from {pretrained!r}. '
                    'No fallback to random initialization. Check network/cache or pass '
                    '--pretrained <local Hugging Face model directory>; '
                    'use --from-scratch only if you intend random initialization. '
                    f'Original error: {exc}') from exc
        elif announce_initialization:
            print('MiT-B3 + SegFormer decoder initialized randomly (no external downloads).')
        print('SegFormer-B3 parameters:', sum(p.numel() for p in self.parameters()))

    @property
    def config(self):
        return self.network.config

    def forward(self, x):
        logits = self.network(pixel_values=x, return_dict=True).logits
        return F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
