from pathlib import Path
import re

p = Path("/mnt/d/LuSNAR_Seg/tools/portable/lusnar_scene5_protocols.py")
text = p.read_text(encoding="utf-8")
# Fix BOM if any
if text.startswith("\ufeff"):
    text = text[1:]

old = '''    # B once for each I2/I3
    print("[I2/I3] B report-once", flush=True)
    results["I2a_gaussian_overlap"]["B"] = predict_custom(items_B, "gaussian", 0, False, i1_W)
    results["I2b_cosine_edge_drop"]["B"] = predict_custom(items_B, "cosine", 32, False, i1_W)
    results["I3_small_rezoom"]["B"] = predict_custom(items_B, wm, ed, True, i1_W)
    return results
'''

new = '''    # B once only for A-selected winners (keep cost down)
    print("[I2/I3] select on A then B report-once", flush=True)
    # load C0 A mIoU from cpu full if present
    c0_A = None
    fullp = Path(args.out_json).with_name("protocols_cpu_full.json")
    if fullp.exists():
        import json as _json
        c0_A = _json.loads(fullp.read_text(encoding="utf-8"))["candidates"]["C0_equal_tta"]["A"]["mIoU"]
    scores = {
        "I2a_gaussian_overlap": results["I2a_gaussian_overlap"]["A"]["mIoU"],
        "I2b_cosine_edge_drop": results["I2b_cosine_edge_drop"]["A"]["mIoU"],
        "I3_small_rezoom": results["I3_small_rezoom"]["A"]["mIoU"],
    }
    # always B-report best I2 and I3 if I3 >= best I2 - 0.002; skip clearly worse
    best_i2 = max(scores["I2a_gaussian_overlap"], scores["I2b_cosine_edge_drop"])
    to_b = set()
    to_b.add(i2_best)
    if scores["I3_small_rezoom"] >= best_i2 - 0.002:
        to_b.add("I3_small_rezoom")
    if c0_A is not None:
        to_b = {k for k in to_b if scores[k] >= c0_A - 0.001} or {max(to_b, key=lambda k: scores[k])}
    print(f"[I2/I3] B candidates: {sorted(to_b)}", flush=True)
    if "I2a_gaussian_overlap" in to_b:
        results["I2a_gaussian_overlap"]["B"] = predict_custom(items_B, "gaussian", 0, False, i1_W)
    if "I2b_cosine_edge_drop" in to_b:
        results["I2b_cosine_edge_drop"]["B"] = predict_custom(items_B, "cosine", 32, False, i1_W)
    if "I3_small_rezoom" in to_b:
        results["I3_small_rezoom"]["B"] = predict_custom(items_B, wm, ed, True, i1_W)
    return results
'''

if old not in text:
    raise SystemExit("block not found")
p.write_text(text.replace(old, new), encoding="utf-8")
compile(p.read_text(encoding="utf-8"), str(p), "exec")
print("GPU_B_SELECT_PATCH_OK")
