#!/usr/bin/env bash
# PREPARED ONLY - not launched. Small-crater ablation set, all identical to E0ft/E1/E2:
# init train_ROW_BLV_20k/best.pth, M_ROW_BLV, crop 512, batch 32, 2000 iters, lr 6e-5 warmup100+cosine(min 0.1),
# bf16, cc-off, seed 42 (SEED in _lusnar_common), scenes 1/3/6/7/8, score last.pth on Scene5 I1a.
# Rows already available from the main run: A0=E0ft (no module, main baseline), A1=E1 (resid 1+2), A2=E2 (margin).
# Usage: ablation_set.sh GPU_A GPU_B
set -u
cd /mnt/d/LuSNAR_Seg
T=tools/small_crater_sc; L=logs/small_crater_sc
GA=${1:-0}; GB=${2:-1}
run(){ gpu=$1; name=$2; tag=$3; shift 3
  $T/sc_train.sh $gpu $name 2000 32 16 --cc-off --val-every 500 --log-every 50 "$@" > $L/$name.log 2>&1 && \
  $T/sc_score.sh $gpu work_dirs/$name/last.pth $tag > $L/score_$tag.log 2>&1; }
( run $GA sc_A3_both_2k A3_both --detail-residual --local-margin --local-margin-weight 0.1 ) &
( run $GB sc_A4_resid_s1_2k A4_resid_s1 --detail-residual --resid-stages 1
  run $GB sc_A5_resid_s2_2k A5_resid_s2 --detail-residual --resid-stages 2 ) &
wait
echo ABLATION DONE
