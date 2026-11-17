#!/usr/bin/env bash
# SegFormer-B5 crop longtrain, stage 1 (Scene5-val 20k). Written 2026-11-13 (Asia/Shanghai). User-authorized plan t331; crop chosen (FI I1a 0.8223 vs crop ~0.8837).
# Recipe = train_ROW_BLV_20k (M_ROW_BLV, crop 512, lr 6e-5, bf16, seed 42, train scenes 1/3/6/7/8, val Scene5 every 1k, best/last, --tta-final)
# Only deviations: backbone/head SegFormer-B5 (MiT-B5 ImageNet pretrain, pretrained/mit_b5_20220624-658746d9.pth); batch/gpu 24 (B2 used 48; B5 48 would OOM; 28 peaked 44.3GB).
# Trainer: tools/portable/lusnar_row_blv_ddp_b5.py (= lusnar_row_blv_ddp_fi.py + build_segformer(...,"b5")).
# Chain: 0) preflight gate  1) train LT_B5_crop_20k (DDP 2 GPU; trainer itself runs best.pth multi-scale TTA -> tta_final.json)
#        2) official I1a Scene5 scorer (B5 copy of lusnar_small_crater_score_sc.py, unchanged protocol) on last.pth (GPU0) and best.pth (GPU1) in parallel
#           -> results/small_crater_sc/LT_B5_crop_20k_last.json / LT_B5_crop_20k_best.json
# NOTE 2026-11-13 03:21: the live run's wrapper (pid 93807) predates the lt_postprocess line below; its tail was handed to 11b_b5_post_wsl.sh.
# Never: plus5 (NOT chained), Moon_2/4/9, official submit, PYTORCH_CUDA_ALLOC_CONF, overwriting existing work dirs.
set -uo pipefail
export USE_LIBUV=0 MASTER_ADDR=127.0.0.1 PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF || true
ROOT=/mnt/d/LuSNAR_Seg
export PYTHONPATH="$ROOT/tools/portable"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lusnar
cd "$ROOT"
RUN=LT_B5_crop_20k
BATCH="${LT_BATCH:-24}"
WORKERS="${LT_WORKERS:-12}"
ITERS=20000
PORT=29700
LIST=/root/lusnar_data/lists_b5_crop
LOG=$ROOT/logs/${RUN}_wsl.log
QLOG=$ROOT/logs/${RUN}_queue_wsl.log
B5PRE=$ROOT/pretrained/mit_b5_20220624-658746d9.pth
say() { echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }
say "queue start pid=$$ run=$RUN batch/gpu=$BATCH workers/rank=$WORKERS iters=$ITERS"
[ -e "$ROOT/work_dirs/$RUN" ] && { say "REFUSE: work_dirs/$RUN exists (never overwrite)"; exit 2; }
test -f "$B5PRE" || { say "missing $B5PRE"; exit 3; }
if pgrep -f 'lusnar_row_blv_ddp|lusnar_small_crater_score_sc|lusnar_smoke_compare.py|torch.distributed.run' >/dev/null; then say "REFUSE: training/scoring process running"; pgrep -af 'lusnar_row_blv_ddp|lusnar_small_crater_score_sc|torch.distributed.run' | tee -a "$QLOG"; exit 4; fi
if ! nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{x=0}{if($1+0>4000)x=1}END{exit x}'; then say "REFUSE: GPU busy: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"; exit 5; fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$PORT$"; then say "REFUSE: port $PORT busy"; exit 6; fi
mkdir -p "$LIST"; [ -f "$LIST/full_crater_weights.json" ] || cp -p /root/lusnar_data/lists/full_crater_weights.json "$LIST/"
say "LAUNCH $RUN port=$PORT list=$LIST"
echo "=== $RUN start $(date '+%F %T') trainer=lusnar_row_blv_ddp_b5.py batch/gpu=$BATCH x2 crop=512 lr=6e-5 workers=$WORKERS iters=$ITERS ===" >>"$LOG"
python -u -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.0.1 --master_port "$PORT" \
  tools/portable/lusnar_row_blv_ddp_b5.py \
  --recipe M_ROW_BLV --run-name "$RUN" --iters "$ITERS" --batch "$BATCH" --workers "$WORKERS" \
  --amp bf16 --crop 512 --lr 6e-5 --val-every 1000 --log-every 50 --tta-final \
  --pretrain "$B5PRE" \
  --project-root "$ROOT" --data-root /root/lusnar_data --list-dir "$LIST" --train-subdir train \
  >>"$LOG" 2>&1 < /dev/null
rc=$?
echo "=== $RUN exit $rc $(date '+%F %T') ===" >>"$LOG"
say "END $RUN exit=$rc"
sleep 20
python "$ROOT/tools/wsl_bootstrap/_lt/lt_postprocess.py" "$ROOT/work_dirs/$RUN" "$ITERS" >>"$QLOG" 2>&1 || say "postprocess failed"  # creates last.pth from iter_20000.pth (trainer only writes last.pth with --no-val)
score() {  # GPU CKPT TAG
  CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4 python -u tools/portable/lusnar_small_crater_score_sc_b5.py \
    --checkpoint "$2" --tag "$3" --project-root "$ROOT" --data-root /root/lusnar_data/train \
    --out-dir "$ROOT/results/small_crater_sc" > "$ROOT/logs/score_$3_wsl.log" 2>&1 < /dev/null
}
p0=""; p1=""
if [ -f "$ROOT/work_dirs/$RUN/last.pth" ]; then score 0 "work_dirs/$RUN/last.pth" "${RUN}_last" & p0=$!; say "SCORE last.pth started pid=$p0"; else say "no last.pth"; fi
if [ -f "$ROOT/work_dirs/$RUN/best.pth" ]; then score 1 "work_dirs/$RUN/best.pth" "${RUN}_best" & p1=$!; say "SCORE best.pth started pid=$p1"; else say "no best.pth"; fi
[ -n "$p0" ] && { wait $p0; say "SCORE ${RUN}_last exit=$?"; }
[ -n "$p1" ] && { wait $p1; say "SCORE ${RUN}_best exit=$?"; }
say "ALL $RUN DONE train=$rc (no plus5 chained; plus5 / Moon_2/4/9 eval are separate steps)"