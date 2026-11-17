from pathlib import Path

p = Path("/mnt/d/LuSNAR_Seg/tools/portable/lusnar_scene5_protocols.py")
text = p.read_text(encoding="utf-8")
if text.startswith("\ufeff"):
    text = text[1:]

old_preload_l = '''def preload_scale_logits(logits_dir, items, scales=CORE_SCALES):
    data = []
    for stem, ip, lp in items:
        zs = load_scale_logits(logits_dir, stem, scales)
        if zs is None:
            continue
        mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
        data.append((stem, zs, mask))
    return data
'''

new_preload_l = '''def preload_scale_logits(logits_dir, items, scales=CORE_SCALES, max_items=None):
    """Keep float16 in RAM; cast on use. Optional max_items for search subsets."""
    data = []
    for stem, ip, lp in items:
        zs = load_scale_logits(logits_dir, stem, scales)
        if zs is None:
            continue
        # store float16 to cut RAM ~2x
        zs16 = [z.astype(np.float16, copy=False) for z in zs]
        mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
        data.append((stem, zs16, mask))
        if max_items is not None and len(data) >= max_items:
            break
    return data
'''

old_preload_p = '''def preload_probs(probs_dir, items):
    data = []
    for stem, ip, lp in items:
        pr = load_probs(probs_dir, stem)
        if pr is None:
            continue
        mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
        data.append((stem, pr, mask))
    return data
'''

new_preload_p = '''def preload_probs(probs_dir, items, max_items=None):
    data = []
    for stem, ip, lp in items:
        pr = load_probs(probs_dir, stem)
        if pr is None:
            continue
        mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
        data.append((stem, pr.astype(np.float16, copy=False), mask))
        if max_items is not None and len(data) >= max_items:
            break
    return data
'''

# In search_soft_weights, use subset for grid then full for final W score
text = text.replace(old_preload_l, new_preload_l)
text = text.replace(old_preload_p, new_preload_p)

text = text.replace(
    "packed = preload_scale_logits(logits_dir, items_A, scales)\n    print(f\"[I1b] preloaded {len(packed)} A images\", flush=True)",
    "packed_search = preload_scale_logits(logits_dir, items_A, scales, max_items=240)\n    print(f\"[I1b] search subset {len(packed_search)} A images\", flush=True)",
)
text = text.replace(
    """    for c in (1, 2, 3):
        best_c = equal
        best_c_iou = -1.0
        for w in candidates:
            inter = 0.0
            union = 0.0
            for stem, zs, mask in packed:
                Wc = np.stack([equal] * NUM_CLASSES, axis=0)
                Wc[c] = w
                fused = fuse_weighted(zs, Wc)
                pred = fused.argmax(0).astype(np.uint8)
                pp = pred == c
                gg = mask == c
                inter += np.logical_and(pp, gg).sum()
                union += np.logical_or(pp, gg).sum()
            iou = inter / union if union > 0 else 0.0
            if iou > best_c_iou:
                best_c_iou = iou
                best_c = w
        W[c] = best_c
        print(f"[I1b] class {EN[c]} best_iou={best_c_iou:.4f} w={best_c.tolist()}", flush=True)
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    per_img = []
    for stem, zs, mask in packed:
        fused = fuse_weighted(zs, W)
        pred = fused.argmax(0).astype(np.uint8)
        update_inter_union(inter, union, pred, mask)
        _, mi = per_image_iou(pred, mask)
        per_img.append(mi)
""",
    """    for c in (1, 2, 3):
        best_c = equal
        best_c_iou = -1.0
        for w in candidates:
            inter = 0.0
            union = 0.0
            for stem, zs, mask in packed_search:
                zs_f = [z.astype(np.float32) for z in zs]
                Wc = np.stack([equal] * NUM_CLASSES, axis=0)
                Wc[c] = w
                fused = fuse_weighted(zs_f, Wc)
                pred = fused.argmax(0).astype(np.uint8)
                pp = pred == c
                gg = mask == c
                inter += np.logical_and(pp, gg).sum()
                union += np.logical_or(pp, gg).sum()
            iou = inter / union if union > 0 else 0.0
            if iou > best_c_iou:
                best_c_iou = iou
                best_c = w
        W[c] = best_c
        print(f"[I1b] class {EN[c]} best_iou={best_c_iou:.4f} w={best_c.tolist()}", flush=True)
    # final score on full A via streaming (no giant RAM)
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    per_img = []
    for stem, ip, lp in items_A:
        zs = load_scale_logits(logits_dir, stem, scales)
        if zs is None:
            continue
        fused = fuse_weighted(zs, W)
        pred = fused.argmax(0).astype(np.uint8)
        mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
        update_inter_union(inter, union, pred, mask)
        _, mi = per_image_iou(pred, mask)
        per_img.append(mi)
""",
)

text = text.replace(
    "packed = preload_probs(probs_dir, items_A)\n    print(f\"[P1] preloaded {len(packed)} A probs\", flush=True)",
    "packed = preload_probs(probs_dir, items_A, max_items=240)\n    print(f\"[P1] search subset {len(packed)} A probs\", flush=True)",
)
text = text.replace(
    """        for stem, pr, mask in packed:
            out = apply_logit_bias(pr, b)
            pred = out.argmax(0).astype(np.uint8)
            update_inter_union(inter, union, pred, mask)
""",
    """        for stem, pr, mask in packed:
            out = apply_logit_bias(pr.astype(np.float32), b)
            pred = out.argmax(0).astype(np.uint8)
            update_inter_union(inter, union, pred, mask)
""",
)

p.write_text(text, encoding="utf-8")
compile(text, str(p), "exec")
print("MEM_PATCH_OK")
