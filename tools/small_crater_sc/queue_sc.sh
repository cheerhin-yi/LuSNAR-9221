#!/usr/bin/env bash
# Small-crater SC queue (own processes only). GPU0 used only after M2F 5k + TTA finished.
cd /mnt/d/LuSNAR_Seg
L=logs/small_crater_sc/queue_sc.log
T=tools/small_crater_sc
log(){ echo "[$(date '+%F %T')] $*" >> $L; }
gpu_mem(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1; }
(
  log "GPU0 lane: waiting for ALL M2F 5k DONE"
  until grep -q 'ALL M2F 5k DONE' logs/backbone_m2f_5k_wsl.log && [ "$(gpu_mem 0)" -lt 4000 ]; do sleep 30; done
  log "GPU0 free -> E0ft train"
  $T/sc_train.sh 0 sc_E0ft_ctrl_2k 2000 32 16 --cc-off --val-every 500 --log-every 50 > logs/small_crater_sc/sc_E0ft_ctrl_2k.log 2>&1
  log "E0ft train exit=$? -> score"
  $T/sc_score.sh 0 work_dirs/sc_E0ft_ctrl_2k/last.pth E0ft > logs/small_crater_sc/score_E0ft.log 2>&1
  log "E0ft score exit=$?"
) &
(
  log "GPU1 lane: waiting for E1 last.pth"
  until [ -f work_dirs/sc_E1_resid12_2k/summary.json ]; do sleep 30; done
  sleep 20
  log "E1 done -> E2 train + E1 score on GPU1"
  $T/sc_score.sh 1 work_dirs/sc_E1_resid12_2k/last.pth E1 > logs/small_crater_sc/score_E1.log 2>&1 &
  SP=$!
  $T/sc_train.sh 1 sc_E2_lmargin_2k 2000 32 16 --cc-off --local-margin --local-margin-weight 0.1 --val-every 500 --log-every 50 > logs/small_crater_sc/sc_E2_lmargin_2k.log 2>&1
  log "E2 train exit=$? -> score E2"
  $T/sc_score.sh 1 work_dirs/sc_E2_lmargin_2k/last.pth E2 > logs/small_crater_sc/score_E2.log 2>&1
  log "E2 score exit=$?"
  wait $SP; log "E1 score exit=$?"
) &
wait
log "QUEUE DONE"
