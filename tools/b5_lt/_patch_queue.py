from pathlib import Path
p = Path("/mnt/d/LuSNAR_Seg/tools/b5_lt/11_b5_crop_lt_queue_wsl.sh")
t = p.read_text()
a = 'say "END $RUN exit=$rc"\nsleep 20\n'
assert t.count(a) == 1
t = t.replace(a, a + 'python "$ROOT/tools/wsl_bootstrap/_lt/lt_postprocess.py" "$ROOT/work_dirs/$RUN" "$ITERS" >>"$QLOG" 2>&1 || say "postprocess failed"  # creates last.pth from iter_20000.pth (trainer only writes last.pth with --no-val)\n')
t = t.replace("# Never: plus5", "# NOTE 2026-11-13 03:21: the live run's wrapper (pid 93807) predates the lt_postprocess line below; its tail was handed to 11b_b5_post_wsl.sh.\n# Never: plus5", 1)
p.write_text(t); print("patched")