from pathlib import Path
import shutil
from datetime import datetime
p = Path(r"D:\LuSNAR_Seg\tools\portable\lusnar_row_blv_ddp.py")
bak = p.with_suffix(p.suffix + f".bak_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(p, bak)
text = p.read_text(encoding="utf-8")
old_arg = 'ap.add_argument("--tta-final", action="store_true")'
new_arg = '''ap.add_argument("--tta-final", action="store_true")
    ap.add_argument(
        "--resume",
        default=None,
        help="path to iter_*.pth / best.pth; restore weights+optimizer and continue from saved iter without wiping train_log.jsonl",
    )'''
if old_arg not in text:
    raise SystemExit("arg anchor missing")
if "--resume" in text and 'ap.add_argument(\n        "--resume"' not in text and 'ap.add_argument("--resume"' not in text:
    # may already have been patched differently
    pass
if 'ap.add_argument(\n        "--resume"' not in text and 'help="path to iter_*.pth' not in text:
    text = text.replace(old_arg, new_arg, 1)

old_init = '''    log_path = work / "train_log.jsonl"
    if is_main(rank) and log_path.exists() and not args.dry_run:
        log_path.unlink()

    peak_mem = 0.0
    it = 0
    t0 = time.time()
    best = -1.0
    best_ious = None
    last_ious = None
    error = None
    model.train()'''

new_init = '''    log_path = work / "train_log.jsonl"
    resume_ckpt = None
    if args.resume:
        resume_path = Path(args.resume).expanduser()
        if not resume_path.is_file():
            resume_path = work / args.resume
        if not resume_path.is_file():
            raise SystemExit(f"resume checkpoint not found: {args.resume}")
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(resume_ckpt["model"])
        if row_attn is not None:
            if not resume_ckpt.get("row_attn"):
                raise SystemExit("resume ckpt missing row_attn but recipe has RowHeightAttn")
            row_attn.load_state_dict(resume_ckpt["row_attn"])
        if resume_ckpt.get("optimizer"):
            opt.load_state_dict(resume_ckpt["optimizer"])
        if is_main(rank):
            print(
                {
                    "resume": str(resume_path),
                    "resume_iter": int(resume_ckpt.get("iter", 0)),
                    "resume_miou": resume_ckpt.get("miou"),
                    "resume_recipe": resume_ckpt.get("recipe"),
                },
                flush=True,
            )
    elif is_main(rank) and log_path.exists() and not args.dry_run:
        log_path.unlink()

    peak_mem = 0.0
    it = int(resume_ckpt.get("iter", 0)) if resume_ckpt is not None else 0
    t0 = time.time()
    best = -1.0
    best_ious = None
    last_ious = None
    error = None
    if resume_ckpt is not None and resume_ckpt.get("miou") is not None:
        best = float(resume_ckpt["miou"])
    best_path = work / "best.pth"
    if best_path.is_file():
        try:
            bck = torch.load(best_path, map_location="cpu", weights_only=False)
            if bck.get("miou") is not None and float(bck["miou"]) > best:
                best = float(bck["miou"])
                if bck.get("per_class_IoU"):
                    pass
        except Exception as exc:
            if is_main(rank):
                print(f"[resume] warn: could not read best.pth ({exc})", flush=True)
    if is_main(rank) and resume_ckpt is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "resume", "from_iter": it, "best": best}, ensure_ascii=False) + "\\n")
    model.train()'''

if old_init not in text:
    raise SystemExit("init anchor missing")
text = text.replace(old_init, new_init, 1)
p.write_text(text, encoding="utf-8")
print("patched", p)
print("backup", bak)
print("has_resume_arg", "--resume" in text and "path to iter_*.pth" in text)
