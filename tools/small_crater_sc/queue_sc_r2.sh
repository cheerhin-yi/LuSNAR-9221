#!/usr/bin/env bash
# Small-crater SC round 2 queue. Two GPU lanes; score immediately after each train on the same GPU.
# Do not overwrite existing work dirs (sc_train.sh refuses if last.pth exists).
set -u
cd /mnt/d/LuSNAR_Seg
mkdir -p logs/small_crater_sc_r2
L=logs/small_crater_sc_r2/queue_sc_r2.log
T=tools/small_crater_sc
log(){ echo "[$(date '+%F %T')] $*"; echo "[$(date '+%F %T')] $*" >> "$L"; }

# GPU0: (a) stack residual+margin 2k, score; (b) stronger margin 2k, score. Compare to E0ft 2k.
(
  log "GPU0 lane start: sc_E3_both_2k train"
  "$T/sc_train.sh" 0 sc_E3_both_2k 2000 32 16 --cc-off --val-every 500 --log-every 50 --detail-residual --local-margin --local-margin-weight 0.1 --local-margin-m 0.5 > logs/small_crater_sc_r2/sc_E3_both_2k.log 2>&1
  log "sc_E3_both_2k train exit=$?"
  log "GPU0: score E3_both_2k"
  "$T/sc_score.sh" 0 work_dirs/sc_E3_both_2k/last.pth E3_both_2k > logs/small_crater_sc_r2/score_E3_both_2k.log 2>&1
  log "sc_E3_both_2k score exit=$?"
  log "GPU0: sc_E2s_lmargin_strong_2k train"
  "$T/sc_train.sh" 0 sc_E2s_lmargin_strong_2k 2000 32 16 --cc-off --val-every 500 --log-every 50 --local-margin --local-margin-weight 0.3 --local-margin-m 1.0 > logs/small_crater_sc_r2/sc_E2s_lmargin_strong_2k.log 2>&1
  log "sc_E2s_lmargin_strong_2k train exit=$?"
  log "GPU0: score E2s_lmargin_strong_2k"
  "$T/sc_score.sh" 0 work_dirs/sc_E2s_lmargin_strong_2k/last.pth E2s_lmargin_strong_2k > logs/small_crater_sc_r2/score_E2s_lmargin_strong_2k.log 2>&1
  log "sc_E2s_lmargin_strong_2k score exit=$?"
  log "GPU0 lane DONE"
) &

# GPU1: (c) stack 5k first, score; (d) no-module 5k control, score. Gate (c) vs (d).
(
  log "GPU1 lane start: sc_E3_both_5k train"
  "$T/sc_train.sh" 1 sc_E3_both_5k 5000 32 16 --cc-off --val-every 500 --log-every 50 --warmup 250 --detail-residual --local-margin --local-margin-weight 0.1 --local-margin-m 0.5 > logs/small_crater_sc_r2/sc_E3_both_5k.log 2>&1
  log "sc_E3_both_5k train exit=$?"
  log "GPU1: score E3_both_5k"
  "$T/sc_score.sh" 1 work_dirs/sc_E3_both_5k/last.pth E3_both_5k > logs/small_crater_sc_r2/score_E3_both_5k.log 2>&1
  log "sc_E3_both_5k score exit=$?"
  log "GPU1: sc_E0ft_ctrl_5k train"
  "$T/sc_train.sh" 1 sc_E0ft_ctrl_5k 5000 32 16 --cc-off --val-every 500 --log-every 50 --warmup 250 > logs/small_crater_sc_r2/sc_E0ft_ctrl_5k.log 2>&1
  log "sc_E0ft_ctrl_5k train exit=$?"
  log "GPU1: score E0ft_ctrl_5k"
  "$T/sc_score.sh" 1 work_dirs/sc_E0ft_ctrl_5k/last.pth E0ft_ctrl_5k > logs/small_crater_sc_r2/score_E0ft_ctrl_5k.log 2>&1
  log "sc_E0ft_ctrl_5k score exit=$?"
  log "GPU1 lane DONE"
) &

wait
log "QUEUE R2 DONE"
