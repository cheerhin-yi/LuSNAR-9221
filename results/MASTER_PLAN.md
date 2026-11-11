# LuSNAR_Seg master plan (2026-11-10 10:40 Asia/Shanghai)
Deadline: 2026-11-20 23:59 (one official submission). Score S=(m2+m4+2*m9)/4. Val = Scene5 only; never use Moon_2/4/9 labels (except Stage 8, re-confirm first). Batch 48 default. No PYTORCH_CUDA_ALLOC_CONF. No reboot unless asked.

Current baseline: SegFormer-B2 + V_B1C + ROW + BLV, crop 512, best.pth iter17000 (work_dirs/train_ROW_BLV_20k). Converted S~0.8300 vs official baseline 0.8046; main gap = Moon_9 crater.

## Stage 1 - Inference phase (RUNNING, WSL, first-round weights, Scene5 only)
cache diag -> attribution -> I1 per-class scale fusion -> I2 overlap weighting/edge drop -> I3 small-object rezoom -> I4 whole-image(1024) for regolith/mountain/sky + tiles/rezoom for rock/crater, merged -> post-proc candidates (class logit bias, geometric/CC rules, temperature). Select on Scene5 half A, report half B once, paired test + Holm. Output results/inference_phase_report.md -> report to user.

## Stage 2 - plus5 (WSL ext4 DDP as of 2026-11-10 15:48; was Windows)
scenes 1/3/5/6/7/8, --no-val, last.pth, port 29621, workers=12, data /root/lusnar_data, lists /root/lusnar_data/lists_plus5, log logs/train_ROW_BLV_20k_plus5_wsl.log, resume from newest iter_*.pth via tools/wsl_bootstrap/03b_resume_plus5_wsl.sh. Do not relaunch on Windows.

## Stage 3 - Fine-tune recipe selection (first-round weights + Scene5 A), then freeze and apply to plus5 weights
T5 / T1 / T2 / depth head / T8 (hierarchical multi-scale attention head, 2k-4k iters + equal-length control).

## Stage 4 - Backbone smokes (after plus5; prep env lusnar_bb on CPU during plus5)
SegFormer-B5 and Mask2Former (HF Swin-B), parallel single-GPU. Winner -> 20k (batch 24 allowed if VRAM short).

## Stage 5 - Long trains with selected recipe; Stage 6 - multi-seed; ensemble/TTA final pick -> official submission (user submits).

## Stage 7 - Whole-image training line (SEPARATE experiment, after crop line results)
Port crop-line effective modules to 1024 whole-image training; own names/dirs (fullimage_compare / *_wholeimg_*), never mixed.

## Stage 8 - Official-split comparison (paper only, LAST, isolated in official_split_compare\, re-confirm before launch)
Train 1/2/4/6/8/9, test 3/5/7, compare with LuSNAR paper (SegFormer 89.96).

## Notes
Test set assumed whole 1024x1024 images; inference method is free. Paper: independent output (GRSL/RS/ASR baseline tier). Until end-Dec only GPU-automatic work.

## Stage 7 refinement (2026-11-10 10:43, user t205u)
Whole-image line re-smokes ALL crop-line modules, including ones rejected under crop (e.g. SHAD, CTRAST, SAPA, HARD, DYSAMPLE, V_B1E), because context/scale changes can flip results.
Protocol: 7a whole-image 1024 baseline (same B2, same 2k iters, batch reduced to fit, lr scaled; same seed) -> 7b single-module smoke matrix @2k vs that baseline, Scene5 val, same eval as crop line -> 7c pairwise/three-module stacks of whole-image winners -> 7d 20k long train of best stack -> 7e crop-vs-whole table (per module: crop delta vs whole delta). All under fullimage_compare\, never mixed.

## Time estimate (2026-11-10 10:50 Asia/Shanghai, t206u)
Measured crop DDP: ~2.2-2.5 s/iter (WSL). Historical 2k single-GPU smoke: ~2.5-3.5 h. plus5 on Windows: 12-16 h if workers 6-8; ~24-28 h if stuck at workers=4. B5/M2F 20k: 22-28 h.
- Contest path (must finish before 2026-11-20 23:59): remaining inference 0.5-1 d; plus5 12-16 h; fine-tunes incl. T8 1-1.5 d; backbone smokes ~0.5 d; winner 20k 22-28 h; selected long trains ~1.5 d; 2 extra seeds 1.5-2 d. Total ~7-9 days -> around 2026-11-17 to 11-19. Thin buffer of 1-3 days.
- Paper path (can run after submit, must not block GPU before submit): whole-image line (baseline + re-smoke ALL ~11 train modules including rejected SHAD/CTRAST/SAPA/HARD/DYSAMPLE/V_B1E/ABL + combos + 20k) 3-5 d, slower than crop; official-split 0.5-1 d.
- All-in from now: ~11-15 days, finishes after the contest deadline (around 11-21 to 11-25). Faults/reruns slip further.

## Speed record correction (2026-11-10 11:06, t209u)
Do not treat later Windows workers=4 (~5 s/iter after 1455) as the Windows best. Agreed measured segments (same recipe, batch 48, 2-card DDP) come from training logs. The old claim that early Windows ran at ~0.87 or 0.84 s/iter is NOT in any train log and must not be used as a target.
- First-round Windows start (iter 50-2050) and plus5 Windows pre-stall (~50-1300): ~3.4-3.7 s/iter. This is the measured Windows pace.
- Later Windows workers=4 fallback, and plus5 Windows auto-resume 1050-1100: ~4.9-5.2 s/iter.
- WSL /mnt/d: ~6 s/iter. WSL ext4 workers=8: ~2.5. WSL ext4 workers=12: ~2.18-2.2 (best verified stable).
Plan: plus5 was moved to WSL ext4 workers=12 on 2026-11-10 15:48 Asia/Shanghai, resumed from iter_1000.pth. Do not relaunch plus5 on Windows. Do not chase unverified 0.84.

## 2026-11-10 11:22 Asia/Shanghai user t213u: cheerhin whole-image smoke handoff (not started)
User confirmed: (1) data/code under the same D: project root on both machines, new folder only (planned D:\LuSNAR_Seg\fullimage_compare\); machines already git-synced, most content consistent. (2) VRAM smoke first on 5060 Ti (~16 GiB) to set batch; cannot copy 4090 batch 48 / crop 512 to 1024 whole-image. (3) one-smoke-at-a-time, pause immediately if P0 needs the GPU. cheerhin finishes already-scheduled P0 work first (WP2 seed 1 + queued DT val / WP2 evals), then smokes; no queue-jump, do not start now. Asked LuSNAR to confirm path / git coverage / missing data-weights-env / whole-image 2k scripts.

## 2026-11-10 12:31 Asia/Shanghai - user decision (t224u/t225u)
### Two splits (do not mix)  [corrected 12:32 t226u]
- Competition split: train Moon_1/3/5/6/7/8, val = Scene5 only; competition TEST = Moon_2/4/9; score S=(m2+m4+2*m9)/4; competition baseline 0.8046.
- Official LuSNAR PAPER split (already recorded 00:27): TRAIN = Moon_1/2/4/6/8/9, TEST = Moon_3/5/7; paper SegFormer baseline mIoU 89.96%. In this split Moon_2/4/9 are TRAINING scenes, not the test set.
### Stage 8 paper protocol
- Paper main results follow the official LuSNAR paper split and its baselines, isolated in official_split_compare\.
- Selection for the paper line follows the paper protocol only (not competition Scene5, and not competition Moon_2/4/9). Paper TEST Moon_3/5/7 is reported once on the final method; do not use 3/5/7 to pick the method.
- Competition line results are a SECOND, separate line of evidence: own section/table, not mixed into the main table.
- Main-table baselines: numbers from the official paper or reproduced under the official paper split; competition baseline 0.8046 only in the competition section.
### Competition line selection (changes the earlier "never use Moon_2/4/9 labels" rule for this line)
- After EACH long-train model finishes, score it on open-source Moon_2/4/9 GT (converted S) and use that to pick the best scheme for the official competition submission.
- In the competition line Moon_2/4/9 is a dev/second-val set, NOT a held-out test, and it is NOT the paper test set: never call these scores "test" in the paper, and never treat them as official-paper test numbers.
- Stage 8 paper line stays isolated and must not reuse competition-line choices made on Moon_2/4/9.

- 2026-11-10 13:53 (Asia/Shanghai): 论文（第八阶段，官方划分）推理方式与各方面实验全部重新做，不默认沿用 Scene5 选出的 P1P2 偏置加几何。比赛线当前按开源 Moon_2/4/9 选 I1a（S 0.8369）。
