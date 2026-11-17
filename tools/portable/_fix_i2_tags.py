from pathlib import Path
p = Path("/mnt/d/LuSNAR_Seg/tools/portable/lusnar_scene5_protocols.py")
text = p.read_text(encoding="utf-8")
old = '''def _i2_worker(rank, items, weight_mode, edge_drop, rezoom, W, args_dict, out_path):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''
new = '''def _i2_worker(rank, items, weight_mode, edge_drop, rezoom, W, args_dict, out_path):
    set_seed(42)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
'''
if old not in text:
    raise SystemExit("worker header not found")
text = text.replace(old, new)
text = text.replace(
    '        "A": predict_custom(items_A, "gaussian", 0, False, i1_W),',
    '        "A": predict_custom(items_A, "gaussian", 0, False, i1_W, tag="I2aA"),',
)
text = text.replace(
    '        "A": predict_custom(items_A, "cosine", 32, False, i1_W),',
    '        "A": predict_custom(items_A, "cosine", 32, False, i1_W, tag="I2bA"),',
)
text = text.replace(
    '        "A": predict_custom(items_A, wm, ed, True, i1_W),',
    '        "A": predict_custom(items_A, wm, ed, True, i1_W, tag="I3A"),',
)
text = text.replace(
    '        results["I2a_gaussian_overlap"]["B"] = predict_custom(items_B, "gaussian", 0, False, i1_W)',
    '        results["I2a_gaussian_overlap"]["B"] = predict_custom(items_B, "gaussian", 0, False, i1_W, tag="I2aB")',
)
text = text.replace(
    '        results["I2b_cosine_edge_drop"]["B"] = predict_custom(items_B, "cosine", 32, False, i1_W)',
    '        results["I2b_cosine_edge_drop"]["B"] = predict_custom(items_B, "cosine", 32, False, i1_W, tag="I2bB")',
)
text = text.replace(
    '        results["I3_small_rezoom"]["B"] = predict_custom(items_B, wm, ed, True, i1_W)',
    '        results["I3_small_rezoom"]["B"] = predict_custom(items_B, wm, ed, True, i1_W, tag="I3B")',
)
p.write_text(text, encoding="utf-8")
compile(text, str(p), "exec")
print("TAGS_AND_DEVICE_OK")
