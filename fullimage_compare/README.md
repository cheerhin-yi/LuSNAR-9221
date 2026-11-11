# fullimage_compare — LuSNAR whole-image (1024×1024) comparison line

**Isolation rule:** this folder holds ONLY the whole-image experiment line. Its lists, weights,
logs and numbers are never mixed with the crop-512 line (`tools\portable*`, `work_dirs\smoke_*`,
`results\smoke_*`), `official_split_compare\`, or anything belonging to P0
(`D:\P0_journal_runs`, `11_Datasets`, `00_freeze`). Nothing here writes outside `fullimage_compare\`.

整图线独立目录：只放整图 1024 训练/评估的列表、权重、日志和结果；不与裁剪线、官方划分对照或 P0 混用。

## Purpose

The crop line (SegFormer-B2, crop 512, batch 16, 2k smokes) judged every module under random 512
crops and centre-crop validation. This line re-runs **every crop-line training module, including the
ones judged ineffective there**, under native whole-image training and whole-image validation, to
build a clean "crop gain vs whole-image gain" table. Same model, subset, seed, lr, steps and
metrics; only the input domain changes.

## Hard rules (also enforced in code)

- Validation = Scene5 only (350-image seed-42 subset). Moon_2/4/9 labels are never read
  (`assert_safe_label` on every path; lists are validated for scene leaks).
- Never set `PYTORCH_CUDA_ALLOC_CONF` (the launcher removes it from the environment).
- One process, one GPU (`CUDA_VISIBLE_DEVICES=0`), no DDP, no ports, no second concurrent train.
- Native Windows paths only (`D:\...`); no `/mnt/d`, no WSL 9p. Pretrained weights and every listed
  image/label/CC-bank file are checked to exist before training (fail fast).
- lr stays 6e-5. Gradient accumulation never rescales lr; it is recorded in the summary.
- On cheerhin, P0 has priority: the launcher refuses to start if a python process is on the GPU, or
  GPU memory used > `-BusyMemMiB` (3000) or utilisation > `-BusyUtilPct` (20 %). (On Windows,
  `nvidia-smi --query-compute-apps` also lists desktop apps and never lists WSL jobs, so the check is
  memory/utilisation based.) `STOP` must be created as soon as P0 needs the card.
- Nothing auto-launches: no scheduled task, no waiter, no follow-up script.

## Comparability rule

Compare an FI module **only** with `FI_V_B1C` re-run **on the same machine** with the same micro
batch, grad-accum, effective batch, iters and validation mode. Never compare FI numbers directly with
crop-line numbers (different training and validation domain: the crop line validated a 512 centre
crop) or with numbers from another machine. Report the crop-line Δ and FI Δ side by side (each vs its
own control). |Δ| < 0.02 is smoke noise. A run whose summary lists a fallback (`any_fallback=true`)
must be checked against the control's micro/accum/val mode before use.

## Files

| Path | What |
|---|---|
| `scripts\fullimage_smoke.py` | whole-image entry: prepare lists, VRAM probe, train, eval-only, check/dry |
| `scripts\run_fullimage_queue.ps1` | launcher: lists → probe → FI_V_B1C → modules one at a time |
| `lists\<HOST>\` | per-machine copy of that machine's `data\smoke` lists + manifest (sha256) |
| `work_dirs\FI_<RECIPE>\` | `last.pth` (every 500 it), `best.pth`, `summary.json`, `summary_progress.json`, `train_log.jsonl`, `module_effect.json` |
| `results\vram_probe_<HOST>.json` | probe table and recommended common micro batch |
| `logs\` | stdout/stderr per experiment, `queue_<HOST>.log` |
| `STOP` | create to stop (not present normally) |

## Module list (training; each = V_B1C + exactly that module, as in the crop line)

| FI exp | Crop-line source | Notes |
|---|---|---|
| FI_V_B1C | control | B1 (RCS+mosaic+CC paste+ACW) + crater RCS/ACW boost |
| FI_M_ROW_BLV | portable (combo) | current main recipe; queued right after the control |
| FI_M_ROW | portable | RowHeightAttn on logits |
| FI_M_BLV | portable | BLV σ0.5, train only |
| FI_M_SHAD | portable | shadow augmentation |
| FI_M_ABL | portable | ABL-lite (autocast-safe BCE version) |
| FI_M_INVFORM | portable | stub = ABL-lite again (acts as an ABL repeat / noise check) |
| FI_V_B1E | portable | legacy B1 + edge aux, **no** crater boost (exactly as crop line) |
| FI_M_HARD | portable_b3 | hardness-weighted CE |
| FI_M_DYSAMPLE | portable_b3 | DySample final-logit upsampler |
| FI_M_CTRAST | portable_b3 | boundary-pair contrastive loss |
| FI_M_SAPA | portable_b3 | SAPA-lite final-logit reassembly (slowest; 1024² output = 1024 checkpointed chunks) |
| FI_M_FADE | portable | FADE-lite (confirm it was part of the crop-line set; queued last) |

Also available, not queued (phase 2 combos): FI_M_ROW_SHAD, FI_M_BLV_SHAD, FI_M_ROW_BLV_SHAD.
Eval-only on saved FI weights (never auto-train): FI_M_TTA (whole-image 0.75/1.0/1.25 logit
average), FI_M_SEGFIX (SegFix-lite on whole-image prediction), FI_M_MSINFER (whole 1.0 vs sliding
512/256 at ×1.0/1.25/1.5). Default checkpoint `work_dirs\FI_V_B1C\best.pth`; `--ckpt` for others.

## Whole-image adaptations of 512-specific crop-line behaviour

- RandomCrop removed; train and val at native 1024×1024 (all LuSNAR images checked 1024×1024; a
  non-1024 image raises). Horizontal flip p=0.5 kept.
- Mosaic: crop line builds a 1024 canvas from four tiles resized to 512, then took a random 512
  crop of it (≈1 tile). Here the same canvas is used whole (4 half-scale tiles). Same pixel scale,
  more content per sample. p=0.5 unchanged.
- CC paste and shadow augmentation already ran on the full 1024 image before cropping; unchanged
  (in the crop line some pastes fell outside the crop, here all are visible).
- RCS/crater-boost sampler and ACW weights are per-image / per-class; unchanged.
- ROW: normalised row position now spans the whole image (crop line: relative to a random crop).
- CTRAST: feature sample cap 512 per micro-batch unchanged → sparser sampling at 1024.
- Validation: crop line validated a 512 **centre crop**; here the whole 1024 image, batch 1.
- Batch: crop line batch 16 @512 (≈4.2 M px/step). Whole-image micro batch comes from the probe
  (expected 1–2 on 16 GiB). Optional `-EffBatch 4` (≈ same pixels/step) via grad-accum; lr unchanged.
  Decide once, before the control; every FI exp must use the same value.

## Pitfall guards built in

1. **VRAM**: probe runs ≥30 real train steps (fwd+bwd+AdamW, bf16, real loader with
   mosaic/CC paste) **plus one full whole-image val pass**, per batch 1→2→4, for the control and the
   heavy modules (ROW_BLV, DYSAMPLE, CTRAST, SAPA). It records peak allocated/reserved and the free
   VRAM at start (desktop usage) and picks the largest batch with peak **reserved** ≤ cap − headroom
   (14.0 − 1.5 = 12.5 GB by default). The queue uses the **minimum over all probed exps** for every
   exp. Train-step OOM (including AdamW step 1): catch, free cache, restart that exp with micro
   halved and grad-accum doubled (effective batch and lr unchanged), record in `fallbacks`. OOM at
   micro 1 → error. Val OOM → sliding 512/256 eval on the 1024 image, recorded as `val_oom_fallback`
   and in `val.mode`.
2. **Windows 1455 / DataLoader**: workers default 2, `persistent_workers=True`, `pin_memory=True`;
   on a worker crash matching 1455 / paging file / shared memory, workers drop 2→1→0 and the loader is
   rebuilt (logged in `fallbacks`, `workers_final`). Val loader ≤1 worker, batch 1.
3. **Paths**: native `D:\` paths only, checked; pretrained `pretrained\mit_b2_20220624-66e8bf70.pth`
   resolved under the project root and checked; lists' files and CC-bank files checked.
4. **Resume**: `last.pth` (model+optimizer+aux+RNG+fallbacks) every 500 iters and at every val,
   plus `summary_progress.json`. STOP/crash → rerun resumes. `summary.json` is written only when
   an exp finishes (`status: done`); done exps are skipped and never overwritten. Resume refuses if
   micro/accum/lr/iters differ.
5. **Old pitfalls**: no waiter/auto-launch; single GPU; ROW_BLV_EFFECT-style self-check at
   iter 1 (`module_effect.json`): ROW grad>1e-6 and weights moved, BLV |CE delta|>1e-4, HARD CE
   changed, DySample/SAPA/edge-aux/FADE grads >0, ABL loss finite, shadow aug not a no-op
   (failure aborts the exp); CTRAST must be non-zero at least once by iter 50 (warning).
   JSON is UTF-8 without BOM, `ensure_ascii=False`, **ASCII English keys** (`regolith, rock,
   crater, mountain, sky`) with Chinese names separately in `class_names_zh` (old summaries with
   Chinese keys broke `ConvertFrom-Json`). In PowerShell read with `Get-Content -Raw -Encoding UTF8`.
6. `summary.json` records peak alloc/reserved (overall and train-only before first val), s/iter,
   workers initial/final, micro batch, grad_accum, effective batch, all fallbacks, resume points,
   device name/total/free-at-start, val mode and val seconds.

## Commands for cheerhin (only after P0's queued work is finished and the GPU is idle)

```powershell
cd D:\LuSNAR_Seg
# 0) CPU-only sanity (no GPU): every module config builds
$env:CUDA_VISIBLE_DEVICES='-1'
& 'F:\Conda\envs\lusnar\python.exe' fullimage_compare\scripts\fullimage_smoke.py --check-only --cpu --exps ALL
& 'F:\Conda\envs\lusnar\python.exe' fullimage_compare\scripts\fullimage_smoke.py --prepare-lists
& 'F:\Conda\envs\lusnar\python.exe' fullimage_compare\scripts\fullimage_smoke.py --dry-run --cpu --dry-size 256 --workers 2 --exps FI_V_B1C,FI_M_ROW_BLV
Remove-Item Env:CUDA_VISIBLE_DEVICES   # the launcher sets it to 0 itself

# 1) VRAM probe only (control + heavy modules), writes results\vram_probe_<HOST>.json
powershell -ExecutionPolicy Bypass -File fullimage_compare\scripts\run_fullimage_queue.ps1 -ProbeOnly

# 2) Baseline only (after reading the probe; micro from probe unless -Micro given)
powershell -ExecutionPolicy Bypass -File fullimage_compare\scripts\run_fullimage_queue.ps1 -SkipProbe -Exps FI_V_B1C

# 3) Full queue (skips finished exps; resumes a stopped one)
powershell -ExecutionPolicy Bypass -File fullimage_compare\scripts\run_fullimage_queue.ps1
#    optional: -EffBatch 4 (grad-accum, lr unchanged; decide BEFORE the control), -Workers 1, -IncludeEval
```

CPU-only test of resume/STOP/eval without the GPU: add `--dry-save` to a `--dry-run --cpu --dry-size 256`
command (writes only `work_dirs\<EXP>_dryrun\`). Dry runs never write `summary.json`.

Direct python (single exp): `fullimage_smoke.py --exps FI_M_ROW --batch <micro> --grad-accum <k>
--workers 2`. Probe a single heavy module: `fullimage_smoke.py --vram-probe --exps FI_M_SAPA`.

## Stop procedure (P0 reclaims the GPU)

```powershell
New-Item D:\LuSNAR_Seg\fullimage_compare\STOP -ItemType File
```
The running exp sees it within one iteration (or ≤10 val images), saves `last.pth` and
`summary_progress.json` (`status: stopped`) and exits 3; the queue ends. To continue later:
`Remove-Item D:\LuSNAR_Seg\fullimage_compare\STOP` and re-run the queue — finished exps are skipped,
the stopped one resumes from `last.pth`. Emergency only: kill the python PID (resume falls back to the
last 500-iter checkpoint).
