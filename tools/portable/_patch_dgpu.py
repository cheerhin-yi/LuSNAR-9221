from pathlib import Path

p = Path("/mnt/d/LuSNAR_Seg/tools/portable/lusnar_scene5_protocols.py")
text = p.read_text(encoding="utf-8")
if text.startswith("\ufeff"):
    text = text[1:]

old = '''@torch.no_grad()
def run_i2_i3_gpu(args, items_A, items_B, base_name, i1_W):
    """GPU protocols I2 and I3; select on A, score B once."""
    set_seed(42)
    project = resolve_project_root(SCRIPT, args.project_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net, row, ckpt = load_model(Path(args.checkpoint), project, device)
    scales = CORE_SCALES
    crop, stride = args.crop, args.stride

    def predict_custom(items, weight_mode, edge_drop, rezoom=False, W=None):
'''

# If already patched, skip
if "def _i2_worker" in text:
    print("already has worker")
    raise SystemExit(0)

new = '''def _merge_partial(parts):
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    per_img = []
    n_fwd = 0
    wall = 0.0
    for p in parts:
        inter += np.asarray(p["inter"], np.float64)
        union += np.asarray(p["union"], np.float64)
        per_img.extend(p["per_image_miou"])
        n_fwd += int(p["forward_windows"])
        wall = max(wall, float(p["wall_s"]))
    miou, ious = iou_from_iu(inter, union)
    return {
        "mIoU": miou,
        "per_class": {CLASS_NAMES[c]: ious[c] for c in range(NUM_CLASSES)},
        "per_class_en": {EN[c]: ious[c] for c in range(NUM_CLASSES)},
        "per_image_miou": per_img,
        "n": len(per_img),
        "forward_windows": int(n_fwd),
        "wall_s": round(wall, 2),
        "ms_per_img": round(1000 * wall / max(len(per_img), 1), 2),
    }


def _i2_worker(rank, items, weight_mode, edge_drop, rezoom, W, args_dict, out_path):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    project = resolve_project_root(SCRIPT, args_dict["project_root"])
    net, row, ckpt = load_model(Path(args_dict["checkpoint"]), project, device)
    scales = CORE_SCALES
    crop, stride = args_dict["crop"], args_dict["stride"]
    W = None if W is None else np.asarray(W, np.float32)
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    per_img = []
    n_fwd = 0
    t0 = time.time()
    for j, (stem, ip, lp) in enumerate(items):
        img = imread_bgr(ip)
        mask = rgb_label_to_index(cv2.imread(lp, cv2.IMREAD_COLOR))
        h0, w0 = mask.shape
        scale_logits = []
        for s in scales:
            nh, nw = max(1, int(round(h0 * s))), max(1, int(round(w0 * s)))
            img_s = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            logits_s, n_win = sliding_window_logits(
                net, img_s, device, crop=crop, stride=stride, row_attn=row,
                weight_mode=weight_mode, edge_drop=edge_drop,
            )
            n_fwd += n_win
            logits_up = F.interpolate(
                logits_s.unsqueeze(0), size=(h0, w0), mode="bilinear", align_corners=False
            )[0].cpu().numpy().astype(np.float32)
            scale_logits.append(logits_up)
        fused = fuse_equal(scale_logits) if W is None else fuse_weighted(scale_logits, W)
        pred = fused.argmax(0).astype(np.uint8)
        if rezoom:
            for c in (1, 2):
                nlab, labels, stats, _ = cv2.connectedComponentsWithStats(
                    (pred == c).astype(np.uint8), 8
                )
                for i in range(1, nlab):
                    area = int(stats[i, cv2.CC_STAT_AREA])
                    if area == 0 or area > 64 * 64:
                        continue
                    x, y, ww, hh = (int(stats[i, k]) for k in range(4))
                    pad = 32
                    x0 = max(0, x - pad)
                    y0 = max(0, y - pad)
                    x1 = min(w0, x + ww + pad)
                    y1 = min(h0, y + hh + pad)
                    crop_img = img[y0:y1, x0:x1]
                    if crop_img.size == 0:
                        continue
                    ch, cw = crop_img.shape[:2]
                    img_r = cv2.resize(
                        crop_img,
                        (max(1, int(cw * 1.5)), max(1, int(ch * 1.5))),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    use_crop = crop
                    if min(img_r.shape[:2]) < crop:
                        use_crop = min(crop, img_r.shape[0], img_r.shape[1])
                        use_crop = max(use_crop, 32)
                    logits_r, n_win = sliding_window_logits(
                        net, img_r, device, crop=use_crop, stride=min(stride, use_crop),
                        row_attn=row, weight_mode="gaussian", edge_drop=0,
                    )
                    n_fwd += n_win
                    logits_back = F.interpolate(
                        logits_r.unsqueeze(0), size=(ch, cw), mode="bilinear", align_corners=False
                    )[0].cpu().numpy()
                    local = fused[:, y0:y1, x0:x1].copy()
                    local = 0.5 * local + 0.5 * logits_back
                    pred[y0:y1, x0:x1] = local.argmax(0).astype(np.uint8)
                    fused[:, y0:y1, x0:x1] = local
        update_inter_union(inter, union, pred, mask)
        _, mi = per_image_iou(pred, mask)
        per_img.append(mi)
        if (j + 1) % 25 == 0 or (j + 1) == len(items):
            print(f"[gpu{rank} {weight_mode} ed{edge_drop} rz{int(rezoom)}] {j+1}/{len(items)}", flush=True)
    payload = {
        "inter": inter.tolist(),
        "union": union.tolist(),
        "per_image_miou": per_img,
        "forward_windows": int(n_fwd),
        "wall_s": round(time.time() - t0, 2),
    }
    Path(out_path).write_text(json.dumps(payload), encoding="utf-8")


def predict_custom_multi(items, weight_mode, edge_drop, rezoom=False, W=None, args=None, tag=""):
    import torch.multiprocessing as mp
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_gpu = min(n_gpu, 2)
    if n_gpu <= 1 or len(items) < 8:
        # single process fallback using worker with rank 0
        tmp = Path(args.out_json).with_name(f"_part_{tag}_0.json")
        args_dict = {
            "project_root": args.project_root,
            "checkpoint": args.checkpoint,
            "crop": args.crop,
            "stride": args.stride,
        }
        _i2_worker(0, items, weight_mode, edge_drop, rezoom, None if W is None else W.tolist(), args_dict, str(tmp))
        part = json.loads(tmp.read_text(encoding="utf-8"))
        return _merge_partial([part])
    chunks = [items[i::n_gpu] for i in range(n_gpu)]
    args_dict = {
        "project_root": args.project_root,
        "checkpoint": args.checkpoint,
        "crop": args.crop,
        "stride": args.stride,
    }
    tmp_paths = [str(Path(args.out_json).with_name(f"_part_{tag}_{r}.json")) for r in range(n_gpu)]
    ctx = mp.get_context("spawn")
    procs = []
    Wl = None if W is None else np.asarray(W, np.float32).tolist()
    for r in range(n_gpu):
        pr = ctx.Process(
            target=_i2_worker,
            args=(r, chunks[r], weight_mode, edge_drop, rezoom, Wl, args_dict, tmp_paths[r]),
        )
        pr.start()
        procs.append(pr)
    for pr in procs:
        pr.join()
        if pr.exitcode != 0:
            raise RuntimeError(f"I2 worker failed exit={pr.exitcode}")
    parts = [json.loads(Path(tp).read_text(encoding="utf-8")) for tp in tmp_paths]
    return _merge_partial(parts)


@torch.no_grad()
def run_i2_i3_gpu(args, items_A, items_B, base_name, i1_W):
    """GPU protocols I2 and I3; select on A, score B once. Uses up to 2 GPUs."""
    set_seed(42)
    scales = CORE_SCALES

    def predict_custom(items, weight_mode, edge_drop, rezoom=False, W=None, tag="x"):
        return predict_custom_multi(items, weight_mode, edge_drop, rezoom, W, args, tag=tag)

    # keep a dummy to satisfy any leftover references
    if False:
        project = resolve_project_root(SCRIPT, args.project_root)
        device = torch.device("cuda:0")
        net, row, ckpt = load_model(Path(args.checkpoint), project, device)
        crop, stride = args.crop, args.stride
        def _unused(items, weight_mode, edge_drop, rezoom=False, W=None):
'''

if old not in text:
    raise SystemExit("old run_i2 block not found")
text = text.replace(old, new, 1)

# The original inner function body is now under `if False: def _unused`.
# We need to close that unused def and restore the A/B evaluation that followed
# the inner def. Find the first "results = {}" after this and make sure indentation works.

# After replacement, the leftover body of predict_custom is still indented under _unused
# until "results = {}". That's fine if `_unused` is complete. Let's check structure by compile.

p.write_text(text, encoding="utf-8")
try:
    compile(text, str(p), "exec")
    print("COMPILE_OK")
except SyntaxError as e:
    print("SYNTAX", e)
    # show around error
    lines = text.splitlines()
    ln = e.lineno or 1
    for i in range(max(0, ln-8), min(len(lines), ln+8)):
        print(f"{i+1}: {lines[i]}")
