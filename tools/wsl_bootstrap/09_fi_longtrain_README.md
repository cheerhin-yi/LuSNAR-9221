# Full-image ROW_BLV longtrain (B2 protocol) - 09_fi_longtrain_wsl.sh

Written 2026-11-11 22:05 (Asia/Shanghai). Skipping FI_M_FADE approved by user (t295); protocol auto-execution per t288/t289.

## What runs
- Trainer: `tools/portable/lusnar_row_blv_ddp.py` (the unmodified B2 `train_ROW_BLV_20k` DDP trainer), `--recipe M_ROW_BLV --crop 1024`.
  LuSNAR images are 1024x1024, so crop=1024 means: train on the whole image (random offset is 0, hflip kept), val = whole Scene5 image.
  Recipe/model/loss = FI_M_ROW_BLV smoke (same `recipe_flags('M_ROW_BLV')` from tools/portable/_lusnar_common.py:
  SegFormer-B2, RowHeightAttn, BLV sigma 0.5, CE(ACW + crater boost 1.75) + Dice, RCS crater weights, mosaic 0.5, CC paste 0.7, hflip, bf16, AdamW 6e-5 wd 0.01).
  Data = full scene lists (6403 train images of scenes 1/3/6/7/8; 1599 Scene5 val), exactly like B2 longtrain, not the 1600/350 smoke subset.
- Per-GPU batch 8 x 2 GPUs (effective 16 whole images = 64 crop-512 equivalents; B2 was 48 crops x 2 = 96). lr 6e-5 not rescaled.
  Probe (GPU1 single, 40 iters, 22:03): peak alloc 41.0 GB at batch 8, 0.54 s/iter; batch 10 would exceed 48 GB.
- Workers 12 per rank. Val every 1000 (B2 cadence) on rank 0, val batch 4; `iter_<k>.pth` saved at every val (every 1000, superset of the 2000 rule), best.pth by Scene5 mIoU.
- Stage 1 also runs `--tta-final` (B2 TTA: sliding 512/256 x scales 0.75/1.0/1.25 on best.pth) -> tta_final.json.
- After each stage `_lt/lt_postprocess.py` copies iter_20000.pth -> last.pth and writes summary_full.json (full val curve with per-class IoU, train curve, trainer summary, TTA).

## Dirs / logs
- Stage 1: `work_dirs/LT_FI_ROW_BLV_20k` (train 1/3/6/7/8, val Scene5), list dir `/root/lusnar_data/lists_fi_lt`, port 29641, log `logs/LT_FI_ROW_BLV_20k_wsl.log`.
- Stage 2: `work_dirs/LT_FI_ROW_BLV_20k_plus5` (train 1/3/5/6/7/8, `--no-val`, deliverable last.pth), list dir `/root/lusnar_data/lists_fi_lt_plus5`, port 29651, log `logs/LT_FI_ROW_BLV_20k_plus5_wsl.log`.
  Chained automatically only when stage 1 exits 0 (CHAIN_PLUS5=1). Same hyperparameters, no Scene5 selection.
- Waiter log: `logs/LT_FI_ROW_BLV_waiter_wsl.log`. Work dirs live under project `work_dirs/` (not fullimage_compare/work_dirs) because the trainer hard-codes `<project>/work_dirs/<run>` and `results/train_report_<run>.md`.
- Both stages refuse to start if their work dir exists.

## Launch gate
Waits until GPU0 Mask2Former queue is finished (`ALL M2F 5k DONE` in logs/backbone_m2f_5k_wsl.log, or queue PID gone = crash case),
no fullimage_smoke.py / lusnar_smoke_compare.py / lusnar_row_blv_ddp.py process, and both GPUs < 4 GB used.

## Manual stage 2 (if not chained)
Same torch.distributed.run command as in train() with `--run-name LT_FI_ROW_BLV_20k_plus5 --train-scenes 1,3,5,6,7,8 --no-val --list-dir /root/lusnar_data/lists_fi_lt_plus5 --master_port 29651`.

## After stage 2 (not automatic)
Open-source Moon_2/4/9 eval of `LT_FI_ROW_BLV_20k_plus5/last.pth` with the same evaluator as B2 plus5 (tools/wsl_bootstrap/run_opensource_followup_wsl_plus5.py style), S and per scene; compare to B2 plus5 S=0.8751.
Note: B2 eval is sliding 512/256; for a full-image-trained model also report whole-image 1024 inference.