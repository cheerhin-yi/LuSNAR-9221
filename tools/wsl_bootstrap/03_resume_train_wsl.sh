#!/usr/bin/env bash
# Resume ROW+BLV 20k on WSL from iter_4000.pth
# workers ladder: 12 -> 8 (DataLoader/shm OOM only; preferred 12; 16 rejected)
# Use setsid (not just nohup): torch.distributed.run registers a SIGHUP
# handler, so a dying login/wsl bash would otherwise kill the job.
set -euo pipefail
export USE_LIBUV=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29601}"
export NCCL_P2P_DISABLE=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="/mnt/d/LuSNAR_Seg/tools/portable${PYTHONPATH:+:$PYTHONPATH}"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lusnar

ROOT=/mnt/d/LuSNAR_Seg
cd "$ROOT"
mkdir -p logs work_dirs/train_ROW_BLV_20k

WORKERS="${WORKERS:-12}"
# Data on WSL ext4 (9p /mnt/d was data-bound: ~6 s/iter). Outputs stay on /mnt/d.
DATA_ROOT="${DATA_ROOT:-/root/lusnar_data}"
LIST_DIR="${LIST_DIR:-$DATA_ROOT/lists}"
test -d "$DATA_ROOT/train/Moon_5/color"
mkdir -p "$LIST_DIR"
# newest iter_*.pth unless CKPT given
CKPT="${CKPT:-$(ls -1 "$ROOT"/work_dirs/train_ROW_BLV_20k/iter_*.pth | sort -t_ -k2 -V | tail -n1)}"
# Reuse RCS weights (same pairs/order), only path prefix rewritten
if [ ! -f "$LIST_DIR/full_crater_weights.json" ] && [ -f "$ROOT/data/lists_portable/full_crater_weights.json" ]; then
  python - "$ROOT/data/lists_portable/full_crater_weights.json" "$LIST_DIR/full_crater_weights.json" "$ROOT/data/train" "$DATA_ROOT/train" <<'PY'
import json,sys
src,dst,a,b=sys.argv[1:]
d=json.load(open(src,encoding='utf-8'))
d['pairs']=[[x.replace(a,b,1) for x in p] for p in d['pairs']]
json.dump(d,open(dst,'w',encoding='utf-8'))
print('rcs weights rewritten',len(d['pairs']))
PY
fi
test -f "$CKPT"

if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${MASTER_PORT}$"; then
  echo "port $MASTER_PORT busy" >&2
  exit 1
fi

# Drop Windows-path RCS cache so WSL regenerates with /mnt/d paths
WFILE="$ROOT/data/lists_portable/full_crater_weights.json"
if [ -f "$WFILE" ] && grep -q 'D:\\LuSNAR_Seg' "$WFILE" 2>/dev/null; then
  mv -f "$WFILE" "${WFILE}.win_backup_$(date +%Y%m%d_%H%M%S)" || true
  echo "moved Windows-path RCS cache aside"
fi

LOG="$ROOT/logs/train_ROW_BLV_20k.log"
echo "=== WSL resume $(date -Is) port=$MASTER_PORT workers=$WORKERS (ladder 12->8; 16 rejected) data=$DATA_ROOT lists=$LIST_DIR ckpt=$CKPT ===" | tee -a "$LOG"

setsid python -u -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_addr 127.0.0.1 \
  --master_port "$MASTER_PORT" \
  tools/portable/lusnar_row_blv_ddp.py \
  --recipe M_ROW_BLV \
  --run-name train_ROW_BLV_20k \
  --iters 20000 \
  --batch 48 \
  --workers "$WORKERS" \
  --amp bf16 \
  --crop 512 \
  --lr 6e-5 \
  --val-every 1000 \
  --tta-final \
  --project-root "$ROOT" \
  --data-root "$DATA_ROOT" \
  --list-dir "$LIST_DIR" \
  --train-subdir train \
  --resume "$CKPT" \
  >>"$LOG" 2>&1 < /dev/null &

echo "TRAIN_LAUNCH_PID=$!"
echo "WORKERS=$WORKERS"
sleep 3
pgrep -af 'lusnar_row_blv_ddp|torch.distributed.run' || true
tail -n 20 "$LOG" || true