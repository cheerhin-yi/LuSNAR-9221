"""Mask2Former (HF transformers) wrapper exposing the SegFormer-style API used by
lusnar_smoke_compare.py / _lusnar_common.py:

    feats  = model.extract_feat(imgs)          # identity: [imgs]
    logits = model.decode_head.forward(feats)  # B x C x H x W (input size)

Semantic map follows HF post_process_semantic_segmentation:
    seg = einsum("bqc,bqhw->bchw", softmax(class_logits)[..., :-1], sigmoid(mask_logits))
computed in fp32 at 1/4 resolution, bilinearly upsampled to the input size
(einsum and bilinear upsampling commute, so this equals upsample-then-einsum),
and returned as log(clamp(seg, 1e-6)) so CE/BLV/Dice/RowHeightAttn/argmax/TTA
logit averaging in the existing loop work unchanged.

Input: the existing loader's tensor (BGR order, normalised with the ImageNet
RGB-order MEAN/STD) is fed as-is, for parity with the B2/B5 smokes.
No Hungarian matching / mask loss / aux-layer loss (deviation from the
original Mask2Former recipe; documented in summary.json method_note).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class M2FHead(nn.Module):
    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.hf = hf_model

    def forward(self, feats):
        x = feats[0] if isinstance(feats, (list, tuple)) else feats
        out = self.hf(pixel_values=x)
        cls = out.class_queries_logits  # B Q K+1
        msk = out.masks_queries_logits  # B Q h w
        with torch.autocast(device_type=x.device.type, enabled=False):
            cls_p = F.softmax(cls.float(), dim=-1)[..., :-1]
            msk_p = torch.sigmoid(msk.float())
            seg = torch.einsum("bqc,bqhw->bchw", cls_p, msk_p)
            if seg.shape[-2:] != x.shape[-2:]:
                seg = F.interpolate(seg, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return torch.log(seg.clamp(min=1e-6))


class M2FSeg(nn.Module):
    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.decode_head = M2FHead(hf_model)

    def extract_feat(self, imgs):
        return [imgs]

    def forward(self, imgs):
        return self.decode_head(self.extract_feat(imgs))


def build_m2f(pretrain: Path, num_classes: int = 5) -> M2FSeg:
    from transformers import Mask2FormerForUniversalSegmentation

    id2label = {i: str(i) for i in range(num_classes)}
    hf = Mask2FormerForUniversalSegmentation.from_pretrained(
        str(pretrain),
        num_labels=num_classes,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
        ignore_mismatched_sizes=True,
    )
    print(f"[m2f] loaded HF Mask2Former from {pretrain} num_labels={num_classes}", flush=True)
    return M2FSeg(hf)
