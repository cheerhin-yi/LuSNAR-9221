from pathlib import Path
p = Path(r"D:\LuSNAR_Seg\tools\portable\lusnar_row_blv_ddp.py")
t = p.read_text(encoding="utf-8")
changed = False
if "--resume-optimizer" not in t:
    old = '''ap.add_argument(
        "--resume",
        default=None,
        help="path to iter_*.pth / best.pth; restore weights+optimizer and continue from saved iter without wiping train_log.jsonl",
    )'''
    new = '''ap.add_argument(
        "--resume",
        default=None,
        help="path to iter_*.pth / best.pth; restore weights and continue from saved iter without wiping train_log.jsonl",
    )
    ap.add_argument("--resume-optimizer", action="store_true", help="also restore optimizer state from --resume ckpt")'''
    if old not in t:
        raise SystemExit("arg block missing")
    t = t.replace(old, new, 1)
    changed = True
old2 = '        if resume_ckpt.get("optimizer"):\n            opt.load_state_dict(resume_ckpt["optimizer"])'
new2 = '        if args.resume_optimizer and resume_ckpt.get("optimizer"):\n            opt.load_state_dict(resume_ckpt["optimizer"])\n        elif is_main(rank):\n            print({"resume_optimizer": False}, flush=True)'
if "args.resume_optimizer and resume_ckpt.get" not in t:
    if old2 not in t:
        raise SystemExit("optimizer load block missing")
    t = t.replace(old2, new2, 1)
    changed = True
old3 = '                it += 1\n                if device.type == "cuda":'
new3 = '''                it += 1
                if is_main(rank) and (it % 5 == 0 or (resume_ckpt is not None and it <= int(resume_ckpt.get("iter", 0)) + 5)):
                    print({"heartbeat_iter": it, "elapsed_s": round(time.time() - t0, 1)}, flush=True)
                if device.type == "cuda":'''
if "heartbeat_iter" not in t:
    if old3 not in t:
        raise SystemExit("it+=1 anchor missing")
    t = t.replace(old3, new3, 1)
    changed = True
p.write_text(t, encoding="utf-8")
print("changed=", changed)
