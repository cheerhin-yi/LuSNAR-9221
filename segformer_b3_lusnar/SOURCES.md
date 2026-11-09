# Implementation and model provenance

- Implementation: Hugging Face Transformers `SegformerForSemanticSegmentation`, pinned to `transformers==4.57.1`.
  Documentation: https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/segformer
  Source: https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/segformer/modeling_segformer.py
  The installed Transformers code is used directly; no model source is vendored or rewritten in this package.
- Default initialization after the connection-timeout fix: RANDOM / OFFLINE, without external model requests.
- Optional encoder weights: NVIDIA `nvidia/mit-b3`, ImageNet-1k pretrained MiT-B3.
  https://huggingface.co/nvidia/mit-b3
  Its ImageNet classifier is discarded; the complete encoder is loaded strictly; the 5-class segmentation decoder starts randomly.
  Encoder configuration is checked against B3 before use. Classification weights do not include a pretrained segmentation decoder.
- Original SegFormer code / B3 segmentation configuration:
  https://github.com/NVlabs/SegFormer
  https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B3/segformer.b3.512x512.ade.160k.py
- LuSNAR data and five RGB label colors:
  https://github.com/zqyu9/LuSNAR-dataset
- This package supplies a custom LuSNAR training loop, dataset validation, scene split, evaluation, checkpointing and previews.
  It is not the original NVIDIA training pipeline and does not assert reproduction of published benchmark results.
- Pretrained weights are not bundled. Their applicable terms are linked from NVIDIA's model card / original project.
  The resolved model commit, when supplied by Hugging Face, is recorded in `initialization.json`.
- A requested pretrained model is never silently replaced with random weights. It must be
  explicitly requested with --pretrained; --from-scratch is mutually exclusive.
