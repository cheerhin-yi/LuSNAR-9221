"""Hugging Face UPerNet + Swin-T, with optional ImageNet-only backbone initialization.

The library supplies both the Swin and UPerNet implementations. The wrapper
loads shared classification-backbone tensors and exposes the library's
main + 0.4 * auxiliary cross-entropy loss. See SOURCES.md.
"""
import torch
from torch import nn

SWIN_T_ARCHITECTURE = dict(
    image_size=224, patch_size=4, num_channels=3, embed_dim=96,
    depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
    mlp_ratio=4., qkv_bias=True, hidden_dropout_prob=0.,
    attention_probs_dropout_prob=0., drop_path_rate=.1,
    use_absolute_embeddings=False, layer_norm_eps=1e-5,
    out_features=['stage1', 'stage2', 'stage3', 'stage4'],
)


class UPerNetSwinTiny(nn.Module):
    def __init__(self, names, pretrained=None, local_files_only=False, revision=None, announce_initialization=True):
        super().__init__()
        try:
            from transformers import SwinConfig, SwinModel, UperNetConfig, UperNetForSemanticSegmentation
        except ImportError as exc:
            raise ImportError('Install dependencies: python -m pip install -r requirements.txt') from exc
        config = UperNetConfig(
            backbone_config=SwinConfig(**SWIN_T_ARCHITECTURE),
            hidden_size=512, pool_scales=[1, 2, 3, 6],
            use_auxiliary_head=True, auxiliary_in_channels=384,
            auxiliary_channels=256, auxiliary_num_convs=1,
            auxiliary_concat_input=False, auxiliary_loss_weight=.4,
            loss_ignore_index=255, num_labels=len(names),
            id2label={i: name for i, name in enumerate(names)},
            label2id={name: i for i, name in enumerate(names)},
        )
        self.network = UperNetForSemanticSegmentation(config)
        self.initialization = dict(pretrained_source=None, resolved_commit=None, type='random',
                                   implementation='transformers.UperNetForSemanticSegmentation')
        if pretrained:
            kwargs = dict(local_files_only=local_files_only)
            if revision: kwargs['revision'] = revision
            try:
                source_config = SwinConfig.from_pretrained(pretrained, **kwargs)
                for key in ('patch_size', 'num_channels', 'embed_dim', 'depths', 'num_heads',
                            'window_size', 'mlp_ratio', 'qkv_bias', 'use_absolute_embeddings'):
                    if getattr(source_config, key) != SWIN_T_ARCHITECTURE[key]:
                        raise ValueError(f'Not a compatible Swin-T checkpoint: {key}')
                source, info = SwinModel.from_pretrained(
                    pretrained, config=source_config, add_pooling_layer=False,
                    output_loading_info=True, **kwargs)
                if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
                    raise ValueError(f'Incomplete classification backbone: {info}')
                unexpected = [key for key in info.get('unexpected_keys', []) if not key.startswith('classifier.')]
                if unexpected: raise ValueError(f'Unexpected weights: {unexpected[:8]}')
                source_state = source.state_dict()
                target = self.network.backbone.state_dict()
                shared_keys = [key for key in target if not key.startswith('hidden_states_norms.')]
                bad = [key for key in shared_keys
                       if key not in source_state or source_state[key].shape != target[key].shape]
                if bad: raise ValueError(f'Missing or incompatible shared backbone tensors: {bad[:8]}')
                mapped = {key: source_state[key] for key in shared_keys}
                # Classification has one final norm; segmentation needs a norm
                # for each output stage. Reuse the final norm for stage4.
                for suffix in ('weight', 'bias'):
                    key = 'layernorm.' + suffix
                    target_key = 'hidden_states_norms.stage4.' + suffix
                    if key not in source_state or source_state[key].shape != target[target_key].shape:
                        raise ValueError(f'Missing final normalization tensor: {key}')
                    mapped[target_key] = source_state[key]
                result = self.network.backbone.load_state_dict(mapped, strict=False)
                expected_new = {f'hidden_states_norms.stage{i}.{suffix}'
                                for i in (1, 2, 3) for suffix in ('weight', 'bias')}
                if set(result.missing_keys) != expected_new or result.unexpected_keys:
                    raise ValueError(f'Unexpected transfer result: {result}')
                self.initialization = dict(
                    pretrained_source=str(pretrained), resolved_commit=getattr(source_config, '_commit_hash', None),
                    type='imagenet_backbone_new_upernet_and_auxiliary_heads',
                    new_stage_norms=sorted(expected_new),
                    implementation='transformers.UperNetForSemanticSegmentation')
                del source
                print(f'Loaded Swin-T ImageNet backbone: {pretrained}')
                print('UPerNet + auxiliary heads and stage1/2/3 output norms initialized for this task.')
            except Exception as exc:
                raise RuntimeError(
                    f'Unable to load complete Swin-T weights from {pretrained!r}. '
                    'No random fallback. Use --pretrained <local Hugging Face directory> '
                    '--local-files-only, or explicitly choose --from-scratch. '
                    f'Original error: {exc}') from exc
        elif announce_initialization:
            print('Swin-T + UPerNet initialized randomly (no pretrained weights; no weight download).')
        print('UPerNet-Swin-T parameters:', sum(p.numel() for p in self.parameters()))

    @property
    def config(self):
        return self.network.config

    def forward(self, x):
        return self.network(pixel_values=x, return_dict=True).logits

    def training_loss(self, x, labels):
        # Calling the library's built-in loss is essential: external CE on the
        # main logits alone would leave the auxiliary head untrained.
        return self.network(pixel_values=x, labels=labels, return_dict=True).loss
