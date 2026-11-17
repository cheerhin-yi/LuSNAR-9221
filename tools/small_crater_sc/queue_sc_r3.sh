#!/usr/bin/env bash
# Small-crater SC round 3 (DRIM-v2) queue.
# Two GPU lanes; score immediately after each train on the same GPU.
# Per-lane logs; do not overwrite existing work dirs (sc_train.sh refuses if last.pth exists).
#
# Usage:
#   bash tools/small_crater_sc/queue_sc_r3.sh           # all batches
#   bash tools/small_crater_sc/queue_sc_r3.sh batch1    # C0b || F1
#   bash tools/small_crater_sc/queue_sc_r3.sh from2     # F2||F2g, then F3||F4, then F5
set -u
cd /mnt/d/LuSNAR_Seg
unset PYTORCH_CUDA_ALLOC_CONF
export USE_LIBUV=0 PYTHONUNBUFFERED=1
mkdir -p logs/small_crater_sc_r3
T=tools/small_crater_sc
R3=logs/small_crater_sc_r3
MODE="${1:-all}"

logn() {
  local lane="$1"; shift
  local line="[$(date '+%F %T')] $*"
  echo "$line"
  echo "$line" >> "$R3/lane${lane}.log"
  echo "$line" >> "$R3/events.log"
}

run_one() {
  local gpu="$1" run="$2" tag="$3"; shift 3
  logn "$gpu" "TRAIN start $run gpu=$gpu"
  "$T/sc_train.sh" "$gpu" "$run" 2000 32 16 --cc-off --val-every 500 --log-every 50 "$@" > "$R3/${run}.log" 2>&1
  local te=$?
  echo "$te" > "$R3/${run}.train.exit"
  logn "$gpu" "TRAIN exit=$te $run"
  if [ "$te" -ne 0 ]; then
    logn "$gpu" "SKIP score $tag because train failed"
    echo "99" > "$R3/${tag}.score.exit"
    return "$te"
  fi
  logn "$gpu" "SCORE start $tag gpu=$gpu"
  "$T/sc_score.sh" "$gpu" "work_dirs/${run}/last.pth" "$tag" > "$R3/score_${tag}.log" 2>&1
  local se=$?
  echo "$se" > "$R3/${tag}.score.exit"
  logn "$gpu" "SCORE exit=$se $tag"
  return "$se"
}

# Batch 1
lane0_b1() {
  logn 0 "lane0 batch1 start: sc_E0ft_ctrl_2k_s2 (seed 2026)"
  run_one 0 sc_E0ft_ctrl_2k_s2 E0ft_ctrl_2k_s2 --seed 2026
  logn 0 "lane0 batch1 DONE"
}
lane1_b1() {
  logn 1 "lane1 batch1 start: sc_F1_sir_2k"
  run_one 1 sc_F1_sir_2k F1_sir_2k --size-instance-loss --sil-weight 0.2
  logn 1 "lane1 batch1 DONE"
}

# Batch 2
lane0_b2() {
  logn 0 "lane0 batch2 start: sc_F2_sgsdr_2k"
  run_one 0 sc_F2_sgsdr_2k F2_sgsdr_2k --detail-residual --shading-gate --sg-source feat
  logn 0 "lane0 batch2 DONE"
}
lane1_b2() {
  logn 1 "lane1 batch2 start: sc_F2g_sgsdr_gray_2k"
  run_one 1 sc_F2g_sgsdr_gray_2k F2g_sgsdr_gray_2k --detail-residual --shading-gate --sg-source gray
  logn 1 "lane1 batch2 DONE"
}

# Batch 3
lane0_b3() {
  logn 0 "lane0 batch3 start: sc_F3_sgsdr_slm_2k"
  run_one 0 sc_F3_sgsdr_slm_2k F3_sgsdr_slm_2k --detail-residual --shading-gate --sg-source feat --local-margin --local-margin-weight 0.1 --local-margin-m 0.5
  logn 0 "lane0 batch3 DONE"
}
lane1_b3() {
  logn 1 "lane1 batch3 start: sc_F4_drimv2_2k (full candidate)"
  run_one 1 sc_F4_drimv2_2k F4_drimv2_2k --detail-residual --shading-gate --sg-source feat --local-margin --local-margin-weight 0.1 --local-margin-m 0.5 --size-instance-loss --sil-weight 0.2
  logn 1 "lane1 batch3 DONE"
}

# Batch 4: first free GPU in the full queue is GPU0 after F3, but we also
# support running F5 alone on a specified GPU via env F5_GPU (default 0).
lane_b4() {
  local gpu="${F5_GPU:-0}"
  logn "$gpu" "lane${gpu} batch4 start: sc_F5_sgsdr_sir_2k"
  run_one "$gpu" sc_F5_sgsdr_sir_2k F5_sgsdr_sir_2k --detail-residual --shading-gate --sg-source feat --size-instance-loss --sil-weight 0.2
  logn "$gpu" "lane${gpu} batch4 DONE"
}

case "$MODE" in
  batch1)
    lane0_b1 &
    p0=$!
    lane1_b1 &
    p1=$!
    echo "$p0" > "$R3/lane0.pid"
    echo "$p1" > "$R3/lane1.pid"
    logn 0 "batch1 pids lane0=$p0 lane1=$p1"
    wait "$p0"
    e0=$?
    wait "$p1"
    e1=$?
    logn 0 "BATCH1 DONE lane0_exit=$e0 lane1_exit=$e1"
    ;;
  from2)
    lane0_b2 &
    p0=$!
    lane1_b2 &
    p1=$!
    echo "$p0" > "$R3/lane0.pid"
    echo "$p1" > "$R3/lane1.pid"
    wait "$p0"
    e0=$?
    wait "$p1"
    e1=$?
    logn 0 "BATCH2 DONE lane0_exit=$e0 lane1_exit=$e1"
    lane0_b3 &
    p0=$!
    lane1_b3 &
    p1=$!
    echo "$p0" > "$R3/lane0.pid"
    echo "$p1" > "$R3/lane1.pid"
    wait "$p0"
    e0=$?
    wait "$p1"
    e1=$?
    logn 0 "BATCH3 DONE lane0_exit=$e0 lane1_exit=$e1"
    # First free GPU: both just finished; use GPU0 for F5.
    F5_GPU=0 lane_b4
    logn 0 "BATCH4 DONE"
    echo "[$(date '+%F %T')] QUEUE R3 from2 DONE" | tee -a "$R3/events.log"
    ;;
  all)
    # GPU0: C0b, F2, F3, F5
    # GPU1: F1, F2g, F4
    (
      lane0_b1
      lane0_b2
      lane0_b3
      F5_GPU=0 lane_b4
      logn 0 "GPU0 lane ALL DONE"
    ) &
    p0=$!
    (
      lane1_b1
      lane1_b2
      lane1_b3
      logn 1 "GPU1 lane ALL DONE"
    ) &
    p1=$!
    echo "$p0" > "$R3/lane0.pid"
    echo "$p1" > "$R3/lane1.pid"
    wait "$p0"
    e0=$?
    wait "$p1"
    e1=$?
    echo "[$(date '+%F %T')] QUEUE R3 DONE lane0_exit=$e0 lane1_exit=$e1" | tee -a "$R3/events.log" | tee -a "$R3/queue_sc_r3.log"
    ;;
  *)
    echo "usage: $0 [all|batch1|from2]" >&2
    exit 2
    ;;
esac
