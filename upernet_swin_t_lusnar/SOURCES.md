# Sources and model initialization

- Standard implementation: Hugging Face Transformers 4.57.1 `UperNetForSemanticSegmentation` with `SwinConfig`.
  https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/upernet
  https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/upernet/modeling_upernet.py
  https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/swin/modeling_swin.py
- Swin-T architecture: embed_dim=96; depths=[2,2,6,2]; heads=[3,6,12,24]; patch=4; window=7.
- UPerNet configuration: four backbone feature stages; PPM scales=[1,2,3,6]; main channels=512;
  auxiliary stage3 FCN input=384/channels=256; auxiliary cross-entropy loss weight=0.4.
  Reference configuration: https://huggingface.co/openmmlab/upernet-swin-tiny/blob/main/config.json
  The ADE20K-trained weights in that reference repo are not used by this package.
- Optional ImageNet-only pretrained backbone:
  https://huggingface.co/microsoft/swin-tiny-patch4-window7-224
  Classification encoder tensors are transferred after complete architecture and shape checks.
  The classification final normalization is reused as stage4 output normalization; stage1/2/3
  output normalizations and both segmentation heads start newly initialized.
  This transfer is explicit in `model_upernet_swin.py` and recorded in initialization.json.
- Original Swin code: https://github.com/microsoft/Swin-Transformer
- LuSNAR five-class RGB labels: https://github.com/zqyu9/LuSNAR-dataset
- Default is explicitly random initialization, because the target machine's connection to
  Hugging Face timed out in the supplied SegFormer log. There is no download and no silent
  fallback from pretrained to random. Optional pretrained-loading failure stops the run.
- This package uses library model code directly; custom code handles data, training,
  checkpointing, metrics and previews. It does not claim original-paper benchmark reproduction.
  Model weights are not included in this package; applicable terms are linked in model cards.
