# LuSNAR method smokes — portable SegFormer-B2

Official competition numbers remain the formal baseline. All numbers from these
scripts are **local Scene5 validation**, not official test scores. **Never read
Moon_2 / Moon_4 / Moon_9 labels.** Cached lists and rarity provenance are checked
before label access; training scenes are 1/3/6/7/8, validation is Scene5 only.

## Fixed cheerhin comparison protocol

- MMSegmentation **1.2.2**, SegFormer-B2; freshly rerun **V_B1C@2k on cheerhin**.
- train subset **1600**; Scene5 `val_sub` **350**; seed **42**; crop **512**.
- AdamW lr **6e-5**, **2000 optimizer steps**, validate every **500** steps.
- **Effective batch 16**. `--batch` defaults to16; method CLI rejects a different
  effective batch. `--accum-steps N` defaults to1 and must divide16.
- Data sampling uses a separate seeded generator, independent of module/loss RNG.
- Each train method inherits V_B1C RCS/mosaic/CC-paste/ACW/crater boost unchanged
  and adds exactly one method. No method stacking.
- `--seed`, `--lr`, `--iters`, `--val-every` are available for diagnostics; nonfixed
  settings are recorded with `config.fixed_protocol_matches=false`. Do not compare
  such runs against a fixed-protocol result.
- Compare only identical local split/seed/steps/accumulation and evaluation domain;
  cross-GPU/machine differences are **not** method gains. `|Δ| < 0.02` is noise.

## Recipes and exact scope

| Recipe | Runnable | Change from V_B1C / scope |
|---|---|---|
| V_B1C | yes | local control: B1 + crater RCS/ACW boost |
| M_ROW | yes | existing HANet-style row-height additive logit bias |
| M_BLV | yes | existing train-only simplified BLV, sigma0.5 |
| M_SHAD | yes | existing grayscale-friendly shadow augmentation |
| M_ABL | yes, lite | existing soft-boundary BCE/Dice, weight0.5 ramp |
| M_INVFORM | fallback only | warns and uses existing ABL-lite; **not** InverseForm |
| M_FADE | yes, lite | existing residual logits/stage1 refinement, **not** full FADE |
| M_DYSAMPLE | **yes** | DySample lp, scale4, groups4, fixed scope0.25; only final bilinear **logit** upsample replaced; train+eval |
| M_HARD | **yes, lite** | detached weight `1+(1-p_gt)^2`, valid mean normalized1; ACW CE weighted; Dice unchanged; **no EMA hardness map** |
| M_CTRAST | **yes, lite** | fused feature hook before `cls_seg`; boundary pixel InfoNCE, temp0.1, weight0.1; pairs3/1,4/3,2/0; no context hierarchy/memory bank |
| M_SAPA | **yes, lite** | final-logit 5x5 similarity-affiliation reassembly, learned 32d key/query, LayerNorm, softmax; stage1 guidance; pure torch chunks/checkpointing |
| M_MSINFER | **yes, eval only** | four independent full-image inference modes on V_B1C best; **does not auto-train** |
| M_TTA | yes, eval only | existing multiscale sliding average; auto-trains V_B1C if missing |
| M_SEGFIX | yes, lite eval only | existing boundary-majority postprocessing; needs checkpoint |
| T_B1_TTA | yes, eval only | existing TTA on `work_dirs/train_B1/best.pth`; fails if missing, no implicit20k |
| M_HRDA | **no (deliberate stub)** | records unimplemented; gate: TTA gain and FADE insufficient |

### New-method implementation caveats

**DySample:** five output classes are zero-padded to8 channels to meet groups4,
then sliced back to5. No second classifier or other network change. The released
lp coordinates and initial offsets are used, with 1x1 offset prediction,
`pixel_shuffle`, and `grid_sample`. Zero learned offsets equal bilinear resize.
Odd whole-image sizes use x4 output trimmed to input size. The train/eval module
is registered on the model; its parameters are in the optimizer/checkpoint.

**Contrastive:** hooks `SegformerHead.fusion_conv`, exactly before `cls_seg` in
MMSeg1.2.2. Uses class-stratified boundary-band pools, at most512 features per
micro-batch; self positives excluded. Negatives are the specified paired class
within8 fused-feature pixels in the same image and boundary band. Positives are
same-class boundary features in that image. If no valid positive/negative, zero
auxiliary loss. This is **Contextrast-inspired lite**, not the paper reproduction.

**SAPA:** preserves the core local encoder/decoder similarity -> normalized
kernel -> weighted decoder assembly. SegFormer stage1 is only1/4 resolution;
its guidance is bilinearly sampled to the target points, **not true full-resolution
encoder guidance**. Only final logit resize changes, not the paper's multi-stage
integration or gated SAPA-G. Uses32d projections,5x5 window,1024-position chunks,
activation checkpointing; slower than a custom CUDA kernel. Priority after DySample.

## Cheerhin RTX 5060 Ti 16GB — exact PowerShell commands

First copy `_lusnar_common.py`, `lusnar_smoke_compare.py`, `mmcv_stub.py`, and this
README from `/workspace/lusnar_portable/` into `D:\LuSNAR_Seg\tools\portable\`.
Existing ImageNet MiT-B2 pretrained weights are still required. Do not launch
multiple training processes on this single16GB GPU: the experiment list runs
**sequentially**. This file does not claim that GPU memory has been checked yet.

```powershell
cd D:\LuSNAR_Seg

# Real fwd/bwd/optimizer memory check: up to5 optimizer steps for each method.
& 'F:\Conda\envs\lusnar\python.exe' 'D:\LuSNAR_Seg\tools\portable\lusnar_smoke_compare.py' `
  --project-root D:\LuSNAR_Seg `
  --data-root D:\LuSNAR_Seg\data\raw --train-subdir LuSNAR_train `
  --exps V_B1C,M_DYSAMPLE,M_CTRAST --dry-run --batch 16

# Full local control + one-variable methods + inference diagnostic.
& 'F:\Conda\envs\lusnar\python.exe' 'D:\LuSNAR_Seg\tools\portable\lusnar_smoke_compare.py' `
  --project-root D:\LuSNAR_Seg `
  --data-root D:\LuSNAR_Seg\data\raw --train-subdir LuSNAR_train `
  --exps V_B1C,M_DYSAMPLE,M_MSINFER,M_HARD,M_CTRAST,M_SAPA `
  --iters 2000 --batch 16 --val-every 500 --seed 42 --lr 6e-5
```

Dry-run prints `[memory] ... effective=16 micro=16 accum=1 peak_mem_mb=...` and
writes **`work_dirs/smoke_<exp>_dryrun/summary.json`**. It does not overwrite the
normal control checkpoint/summary. CPU reports0MB CUDA allocation (not CPU RSS).
`--dry-run` on train methods does real forward/backward/optimizer steps but no
validation or best checkpoint. If cached assets are stale, add `--force-prepare`
to regenerate lists/rarity/CC-bank once, then omit it for subsequent runs.
Cached fixed-protocol assets must have1600/350 entries and matching seed.

### Memory fallback: keep effective16, halve micro-batch

If OOM, append **`--accum-steps 2`** to **both** commands above (micro8, effective16).
If necessary use4 (micro4),8 (micro2), or16 (micro1). Re-run **V_B1C and all methods
with the same factor**, even if only one method required it. Do not use `--batch 8`.
There is deliberately **no automatic OOM retry / `--auto-accum`**: silently changing
one method's micro-batch would compromise the control. Errors are saved and the
CLI exits nonzero after the experiment list. `peak_mem_mb` is peak CUDA allocated
memory, reset per experiment, not device-wide reserved/other-process usage.

Loss is divided byN and the optimizer/EMA advances everyN micro-batches. `iters`
counts optimizer steps, not micro-batches. BatchNorm and per-micro Dice/weighted-CE
normalization mean accumulation is **not mathematically identical** to a true16
sample forward pass; the note is recorded in every summary. Keep local control
and method accumulation identical. `--mem-fraction` remains optional, not needed
by default; it is not a substitute for memory validation.

## M_MSINFER: inference diagnosis, no training

Default checkpoint: **`work_dirs/smoke_V_B1C/best.pth`**; override with
`--ckpt work_dirs\smoke_V_B1C\best.pth`. Missing checkpoint is an error, never an
implicit training run. Four **independent** modes, no averaging across them:

1. `whole_1.0`: original image, padded to stride32 (not resized to crop512).
2. `slide_1.0`: original-size overlapping crop512/stride256.
3. `slide_1.25`: image enlarged1.25x then crop512/stride256.
4. `slide_1.5`: image enlarged1.5x then crop512/stride256.

Logits are resized back to original label shape. Every mode uses the same350
Scene5 images (first2 for inference dry-run). Reports mIoU, all five class IoUs,
Boundary F1/IoU and wall seconds/image. Timings synchronize CUDA and include
preprocessing/resizing/H2D/forward/stitch/D2H, excluding disk I/O and metric work;
no warmup, so first-mode cold overhead can matter. These are **full-image** results:
compare modes to `whole_1.0`, **not** against the center-crop training-validation
mIoU or4090's old0.8014. Legacy M_TTA keeps its existing center-crop vs full-image
comparison; use M_MSINFER for a clean, same-domain resolution diagnosis.

## Boundary evaluation and summaries

All common validation, detailed validation, TTA, SegFix before/after, and each
M_MSINFER mode compute:

- **Boundary F1**: per-class1-pixel inner contours, Euclidean distance transform
  matching with **3px tolerance**; dataset-pooled precision/recall -> F1.
- **Boundary IoU** (Cheng et al., CVPR2021): intersection/union of inner erosion
  bands, fixed **d=3px**, **not**0.02 times diagonal.
- Image exterior is background for contours; pixels within4px of ignored labels
  are excluded to avoid false padding/hole boundaries. Absent in both prediction
  and GT -> `null` and excluded from means; false-positive-only class ->0.
- Ignored GT pixels now do not count as IoU false positives; rerun local control
  if comparing against older summaries that counted them.

`summary.json` adds `config` (model/version/split/seed/crop/lr/steps/validation,
`batch_effective`, `micro_batch`, `accum_steps`, `peak_mem_mb` and accumulation
note), plus these batch/memory fields at top level. Training summaries store
`boundary_metrics_best` and `boundary_metrics_last`; eval-only summaries store
`boundary_metrics` and their detailed before/after/mode values. Existing eval
function tuple return arities are unchanged; optional `metrics_out` exposes new
metrics. The report is refreshed after each experiment:

```text
work_dirs/smoke_<exp>/{best.pth,iter_*.pth,summary.json,train_log.jsonl}
work_dirs/smoke_<exp>_dryrun/{summary.json,train_log.jsonl}
results/smoke_method_eval.md       # BF1/BIoU columns + MSINFER mode details
```

## Legacy entry points / knobs

A0/A1/B1/C1 and V_B1E remain available. Existing method flags remain:
`--blv-sigma 0.5`, `--abl-weight 0.5`,
`--tta-scales 0.75,1.0,1.25 --tta-crop 512 --tta-stride 256`,
`--ckpt work_dirs\train_B1\best.pth`, `--skip-prepare`, `--prepare-only`.
For4090 use its own Python and data root; rerun its local control too when adopting
new ignore/boundary metrics. Never overwrite a running worker's scripts mid-run.

Legacy pass/gate reminders: row sky->mountain<12% and mountain IoU+0.03; BLV crater
IoU+0.02 or crater->regolith<7%; shadow crater IoU+0.02; boundary changes must avoid
rock/crater trade-off collapse. New methods: inspect all five IoUs and boundaries,
retain only repeatable gain beyond smoke noise without large class regressions.
Full InverseForm, full FADE, full SAPA-G, HRDA and small-object size-bucket IoU are
not supplied. No fabricated paper-equivalent claim or official-test score.

## Implementation sources

- DySample released lp implementation: https://github.com/tiny-smart/dysample/blob/main/dysample.py
- SAPA formulation/variants: https://arxiv.org/abs/2209.12866
- Fused-feature hook location (MMSeg1.2.2): https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/models/decode_heads/segformer_head.py
