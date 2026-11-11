#!/usr/bin/env bash
set -uo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lusnar
LOG=/mnt/d/LuSNAR_Seg/logs/wsl_setup_lusnar.log
echo "=== pip extras $(date -Is) ===" | tee -a "$LOG"
pip install ftfy regex | tee -a "$LOG"
python - <<'PY'
import sys
sys.path.insert(0, "/mnt/d/LuSNAR_Seg/tools/portable")
import mmcv_stub  # noqa
missing = []
for name in ["ftfy", "regex", "mmseg", "mmcv", "mmengine"]:
    try:
        __import__(name)
        print("ok", name)
    except Exception as e:
        missing.append((name, repr(e)))
        print("FAIL", name, e)
try:
    from mmseg.utils import register_all_modules
    print("ok register_all_modules")
except Exception as e:
    print("FAIL register_all_modules", type(e), e)
    raise SystemExit(2)
print("IMPORT_OK")
PY
echo "=== extras end $(date -Is) ===" | tee -a "$LOG"