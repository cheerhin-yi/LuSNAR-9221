import re
from pathlib import Path

p = Path("/mnt/d/LuSNAR_Seg/tools/portable/lusnar_scene5_protocols.py")
text = p.read_text(encoding="utf-8")

helper = '''
def preload_scale_logits(logits_dir, items, scales=CORE_SCALES):
    data = []
    for stem, ip, lp in items:
        zs = load_scale_logits(logits_dir, stem, scales)
        if zs is None:
            continue
        mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
        data.append((stem, zs, mask))
    return data


def preload_probs(probs_dir, items):
    data = []
    for stem, ip, lp in items:
        pr = load_probs(probs_dir, stem)
        if pr is None:
            continue
        mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
        data.append((stem, pr, mask))
    return data

'''

anchor = '''def load_probs(probs_dir: Path, stem: str):
    p = probs_dir / f"{stem}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)
'''

if "def preload_scale_logits" not in text:
    if anchor not in text:
        raise SystemExit("anchor missing")
    text = text.replace(anchor, anchor + "\n" + helper)

new_soft = '''def search_soft_weights(logits_dir, items_A, scales=CORE_SCALES):
    """I1b: coarse grid; preload A logits once."""
    S = len(scales)
    equal = np.ones(S, np.float32) / S
    candidates = []
    for pref in range(S):
        for alpha in (0.5, 0.7, 0.9):
            w = np.full(S, (1 - alpha) / (S - 1), np.float32)
            w[pref] = alpha
            candidates.append(w)
    candidates.append(equal)

    packed = preload_scale_logits(logits_dir, items_A, scales)
    print(f"[I1b] preloaded {len(packed)} A images", flush=True)
    W = np.stack([equal] * NUM_CLASSES, axis=0)
    for c in (1, 2, 3):
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
    miou, ious = iou_from_iu(inter, union)
    return W, {"mIoU": miou, "per_class": ious, "per_image_miou": per_img}


'''

new_bias = '''def search_logit_bias(probs_dir, items_A, step=0.1, span=0.8):
    """P1: bias search with preloaded probs."""
    grid = np.arange(-span, span + 1e-9, step)
    packed = preload_probs(probs_dir, items_A)
    print(f"[P1] preloaded {len(packed)} A probs", flush=True)
    bias = np.zeros(NUM_CLASSES, np.float32)

    def score(b):
        inter = np.zeros(NUM_CLASSES, np.float64)
        union = np.zeros(NUM_CLASSES, np.float64)
        for stem, pr, mask in packed:
            out = apply_logit_bias(pr, b)
            pred = out.argmax(0).astype(np.uint8)
            update_inter_union(inter, union, pred, mask)
        miou, _ = iou_from_iu(inter, union)
        return miou

    base = score(bias)
    for c in (3, 1, 2, 4):
        best_v = bias[c]
        best_s = base
        for v in grid:
            bias[c] = v
            s = score(bias)
            if s > best_s:
                best_s = s
                best_v = v
        bias[c] = best_v
        base = best_s
        print(f"[P1] class {EN[c]} bias={best_v:.2f} mIoU={best_s:.4f}", flush=True)
    return bias, base


'''

def replace_func(src, name, new_src):
    pattern = rf"def {name}\(.*?\n(?=def |\Z)"
    m = re.search(pattern, src, flags=re.S)
    if not m:
        raise SystemExit(f"func {name} not found")
    return src[: m.start()] + new_src + src[m.end():]

text2 = replace_func(text, "search_soft_weights", new_soft)
text2 = replace_func(text2, "search_logit_bias", new_bias)
p.write_text(text2, encoding="utf-8")
compile(text2, str(p), "exec")
print("PATCHED_OK")
