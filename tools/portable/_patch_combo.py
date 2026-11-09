# -*- coding: utf-8 -*-
"""Minimal combo-ablation patches for portable smoke scripts."""
from pathlib import Path

COMMON = Path(r"D:\LuSNAR_Seg\tools\portable\_lusnar_common.py")
SMOKE = Path(r"D:\LuSNAR_Seg\tools\portable\lusnar_smoke_compare.py")

# ---------- _lusnar_common.py ----------
c = COMMON.read_text(encoding="utf-8")

old_train = '''    train_method = (
        "M_ROW", "M_BLV", "M_SHAD", "M_ABL", "M_INVFORM", "M_FADE",
    )'''
new_train = '''    train_method = (
        "M_ROW", "M_BLV", "M_SHAD", "M_ROW_SHAD", "M_ABL", "M_INVFORM", "M_FADE",
    )'''
assert old_train in c, "train_method block missing"
c = c.replace(old_train, new_train, 1)

old_doc = "      M_SHAD: V_B1C + shadow_augment (train only)\n"
new_doc = (
    "      M_SHAD: V_B1C + shadow_augment (train only)\n"
    "      M_ROW_SHAD: V_B1C + RowHeightAttn + shadow_augment (combo train)\n"
)
assert old_doc in c, "M_SHAD doc missing"
if "M_ROW_SHAD:" not in c:
    c = c.replace(old_doc, new_doc, 1)

old_row = '        "row_attn": r == "M_ROW",\n'
new_row = '        "row_attn": r in ("M_ROW", "M_ROW_SHAD"),\n'
assert old_row in c, "row_attn flag missing"
c = c.replace(old_row, new_row, 1)

old_shad = '        "shadow_aug": r == "M_SHAD",\n'
new_shad = '        "shadow_aug": r in ("M_SHAD", "M_ROW_SHAD"),\n'
assert old_shad in c, "shadow_aug flag missing"
c = c.replace(old_shad, new_shad, 1)

COMMON.write_text(c, encoding="utf-8")
print("[ok] patched", COMMON)

# ---------- lusnar_smoke_compare.py ----------
s = SMOKE.read_text(encoding="utf-8")

# 1) run_exp: accept run_name and pass to TTA / training work dir
old_sig = '''def run_exp(
    exp: str,
    project: Path,
    smoke_dir: Path,
    pretrain: Path,
    iters: int,
    batch: int,
    lr: float,
    dry_run: bool = False,
    mem_fraction: float | None = None,
    blv_sigma: float = 0.5,
    tta_scales=(0.75, 1.0, 1.25),
    tta_crop: int = 512,
    tta_stride: int = 256,
    ckpt: str | None = None,
    abl_weight: float = 0.5,
):'''
new_sig = '''def run_exp(
    exp: str,
    project: Path,
    smoke_dir: Path,
    pretrain: Path,
    iters: int,
    batch: int,
    lr: float,
    dry_run: bool = False,
    mem_fraction: float | None = None,
    blv_sigma: float = 0.5,
    tta_scales=(0.75, 1.0, 1.25),
    tta_crop: int = 512,
    tta_stride: int = 256,
    ckpt: str | None = None,
    abl_weight: float = 0.5,
    run_name: str | None = None,
):'''
assert old_sig in s, "run_exp signature missing"
s = s.replace(old_sig, new_sig, 1)

# After flags = recipe_flags(exp), set run_tag
old_flags = "    set_seed(SEED)\n    flags = recipe_flags(exp)\n"
new_flags = (
    "    set_seed(SEED)\n"
    "    flags = recipe_flags(exp)\n"
    "    run_tag = (run_name or exp).strip() or exp\n"
)
assert old_flags in s, "set_seed/flags block missing"
s = s.replace(old_flags, new_flags, 1)

# HRDA work dir uses run_tag
s = s.replace(
    '        work = project / "work_dirs" / f"smoke_{exp}"\n'
    "        work.mkdir(parents=True, exist_ok=True)\n"
    "        summary = {\n"
    '            "exp": exp,\n'
    '            "run_name": f"smoke_{exp}",\n'
    '            "error": (\n'
    '                "M_HRDA not implemented for portable smoke. "',
    '        work = project / "work_dirs" / f"smoke_{run_tag}"\n'
    "        work.mkdir(parents=True, exist_ok=True)\n"
    "        summary = {\n"
    '            "exp": exp,\n'
    '            "run_name": f"smoke_{run_tag}",\n'
    '            "error": (\n'
    '                "M_HRDA not implemented for portable smoke. "',
    1,
)

# TTA dispatch: pass run_name=run_tag, keep exp_name as recipe
old_tta = '''    if flags["tta"]:
        return run_tta_exp(
            project=project,
            smoke_dir=smoke_dir,
            pretrain=pretrain,
            iters=iters,
            batch=batch,
            lr=lr,
            dry_run=dry_run,
            mem_fraction=mem_fraction,
            blv_sigma=blv_sigma,
            tta_scales=tta_scales,
            tta_crop=tta_crop,
            tta_stride=tta_stride,
            ckpt_override=ckpt,
            exp_name="M_TTA",
            require_ckpt=False,
            default_ckpt=project / "work_dirs" / "smoke_V_B1C" / "best.pth",
            auto_train_vb1c=True,
        )'''
new_tta = '''    if flags["tta"]:
        return run_tta_exp(
            project=project,
            smoke_dir=smoke_dir,
            pretrain=pretrain,
            iters=iters,
            batch=batch,
            lr=lr,
            dry_run=dry_run,
            mem_fraction=mem_fraction,
            blv_sigma=blv_sigma,
            tta_scales=tta_scales,
            tta_crop=tta_crop,
            tta_stride=tta_stride,
            ckpt_override=ckpt,
            exp_name="M_TTA",
            run_name=run_tag,
            require_ckpt=False,
            default_ckpt=project / "work_dirs" / "smoke_V_B1C" / "best.pth",
            auto_train_vb1c=True,
        )'''
assert old_tta in s, "M_TTA dispatch missing"
s = s.replace(old_tta, new_tta, 1)

old_tb1 = '''    if flags.get("t_b1_tta"):
        return run_tta_exp(
            project=project,
            smoke_dir=smoke_dir,
            pretrain=pretrain,
            iters=iters,
            batch=batch,
            lr=lr,
            dry_run=dry_run,
            mem_fraction=mem_fraction,
            blv_sigma=blv_sigma,
            tta_scales=tta_scales,
            tta_crop=tta_crop,
            tta_stride=tta_stride,
            ckpt_override=ckpt,
            exp_name="T_B1_TTA",
            require_ckpt=True,
            default_ckpt=project / "work_dirs" / "train_B1" / "best.pth",
            auto_train_vb1c=False,
        )'''
new_tb1 = '''    if flags.get("t_b1_tta"):
        return run_tta_exp(
            project=project,
            smoke_dir=smoke_dir,
            pretrain=pretrain,
            iters=iters,
            batch=batch,
            lr=lr,
            dry_run=dry_run,
            mem_fraction=mem_fraction,
            blv_sigma=blv_sigma,
            tta_scales=tta_scales,
            tta_crop=tta_crop,
            tta_stride=tta_stride,
            ckpt_override=ckpt,
            exp_name="T_B1_TTA",
            run_name=run_tag,
            require_ckpt=True,
            default_ckpt=project / "work_dirs" / "train_B1" / "best.pth",
            auto_train_vb1c=False,
        )'''
assert old_tb1 in s, "T_B1_TTA dispatch missing"
s = s.replace(old_tb1, new_tb1, 1)

# Training work dir uses run_tag (first occurrence after segfix block)
old_work = '''    work = project / "work_dirs" / f"smoke_{exp}"
    work.mkdir(parents=True, exist_ok=True)
    log_path = work / "train_log.jsonl"'''
new_work = '''    work = project / "work_dirs" / f"smoke_{run_tag}"
    work.mkdir(parents=True, exist_ok=True)
    log_path = work / "train_log.jsonl"'''
assert old_work in s, "train work dir missing"
s = s.replace(old_work, new_work, 1)

# method_note for M_ROW_SHAD
old_note = '''        "M_SHAD": "V_B1C + shadow_augment (train-only)",
        "M_ABL": "V_B1C + ABL-lite boundary BCE/Dice (train-only)",'''
new_note = '''        "M_SHAD": "V_B1C + shadow_augment (train-only)",
        "M_ROW_SHAD": "V_B1C + RowHeightAttn + shadow_augment (combo)",
        "M_ABL": "V_B1C + ABL-lite boundary BCE/Dice (train-only)",'''
assert old_note in s, "method_note block missing"
s = s.replace(old_note, new_note, 1)

# summary run_name in training uses run_tag
# Careful: there may be multiple "run_name": f"smoke_{exp}" — change the one after method_note / training summary
# We'll change the training summary one specifically by context
old_sum_run = '''    summary = {
        "exp": exp,
        "run_name": f"smoke_{exp}",
        "iters": int(it),
        "target_iters": int(iters),'''
new_sum_run = '''    summary = {
        "exp": exp,
        "run_name": f"smoke_{run_tag}",
        "iters": int(it),
        "target_iters": int(iters),'''
assert old_sum_run in s, "train summary run_name missing"
s = s.replace(old_sum_run, new_sum_run, 1)

# 2) run_tta_exp: add run_name param, load row_attn, use run_tag for work dir
old_tta_sig = '''def run_tta_exp(
    project: Path,
    smoke_dir: Path,
    pretrain: Path,
    iters: int,
    batch: int,
    lr: float,
    dry_run: bool = False,
    mem_fraction: float | None = None,
    blv_sigma: float = 0.5,
    tta_scales=(0.75, 1.0, 1.25),
    tta_crop: int = 512,
    tta_stride: int = 256,
    ckpt_override: str | None = None,
    exp_name: str = "M_TTA",
    require_ckpt: bool = False,
    default_ckpt: Path | None = None,
    auto_train_vb1c: bool = True,
):'''
new_tta_sig = '''def run_tta_exp(
    project: Path,
    smoke_dir: Path,
    pretrain: Path,
    iters: int,
    batch: int,
    lr: float,
    dry_run: bool = False,
    mem_fraction: float | None = None,
    blv_sigma: float = 0.5,
    tta_scales=(0.75, 1.0, 1.25),
    tta_crop: int = 512,
    tta_stride: int = 256,
    ckpt_override: str | None = None,
    exp_name: str = "M_TTA",
    run_name: str | None = None,
    require_ckpt: bool = False,
    default_ckpt: Path | None = None,
    auto_train_vb1c: bool = True,
):'''
assert old_tta_sig in s, "run_tta_exp signature missing"
s = s.replace(old_tta_sig, new_tta_sig, 1)

old_tta_work = '''    exp = exp_name
    work = project / "work_dirs" / f"smoke_{exp}"
    work.mkdir(parents=True, exist_ok=True)'''
new_tta_work = '''    exp = exp_name
    run_tag = (run_name or exp).strip() or exp
    work = project / "work_dirs" / f"smoke_{run_tag}"
    work.mkdir(parents=True, exist_ok=True)'''
assert old_tta_work in s, "run_tta_exp work dir missing"
s = s.replace(old_tta_work, new_tta_work, 1)

# Fix missing-ckpt summary run_name to use run_tag (first occurrence in run_tta_exp)
# Replace the two "run_name": f"smoke_{exp}" inside run_tta_exp carefully.
# After our change, exp still exists; we want run_tag in summaries.
# Do a targeted replace for the block right after work.mkdir in TTA missing path —
# easier: replace all remaining smoke_{exp} run_name in TTA function by editing the known blocks.

# Block 1: missing ckpt summary
old_miss = '''            summary = {
                "exp": exp,
                "run_name": f"smoke_{exp}",
                "error": (
                    f"checkpoint missing: {ckpt_path}. "
                    "T_B1_TTA will not auto-train 20k; place best.pth or pass --ckpt."
                    if exp.upper() == "T_B1_TTA" or require_ckpt
                    else f"checkpoint missing: {ckpt_path}"
                ),'''
new_miss = '''            summary = {
                "exp": exp,
                "run_name": f"smoke_{run_tag}",
                "error": (
                    f"checkpoint missing: {ckpt_path}. "
                    "T_B1_TTA will not auto-train 20k; place best.pth or pass --ckpt."
                    if exp.upper() == "T_B1_TTA" or require_ckpt
                    else f"checkpoint missing: {ckpt_path}"
                ),'''
assert old_miss in s, "TTA missing-ckpt summary missing"
s = s.replace(old_miss, new_miss, 1)

# Load row_attn like SegFix
old_load = '''    model.load_state_dict(state, strict=False)
    # TTA recipes use plain SegFormer (no RowHeightAttn / Fade) unless present in ckpt
    fade = None
    if isinstance(ckpt, dict) and "fade" in ckpt:
        fade = FadeLite().to(device)
        fade.load_state_dict(ckpt["fade"], strict=False)
        print(f"[{exp}] loaded FadeLite from ckpt", flush=True)
    print(f"[{exp}] loaded {ckpt_path}", flush=True)'''
new_load = '''    model.load_state_dict(state, strict=False)
    # Load optional extras from ckpt (mirror SegFix branch)
    row_attn = None
    if isinstance(ckpt, dict) and "row_attn" in ckpt:
        row_attn = RowHeightAttn().to(device)
        row_attn.load_state_dict(ckpt["row_attn"], strict=False)
        row_attn.eval()
        print(f"[{run_tag}] loaded RowHeightAttn from ckpt", flush=True)
    fade = None
    if isinstance(ckpt, dict) and "fade" in ckpt:
        fade = FadeLite().to(device)
        fade.load_state_dict(ckpt["fade"], strict=False)
        fade.eval()
        print(f"[{run_tag}] loaded FadeLite from ckpt", flush=True)
    print(f"[{run_tag}] loaded {ckpt_path}", flush=True)'''
assert old_load in s, "TTA load block missing"
s = s.replace(old_load, new_load, 1)

old_eval = '''        miou_s, ious_s, _cm_s, rates_s = evaluate_detailed(
            model, val_loader, device, fade=fade
        )
        print(
            {"single_mIoU": jfloat(miou_s), "per_class": dict(zip(CLASS_NAMES, ious_s))},
            flush=True,
        )
        miou_t, ious_t, _cm_t, rates_t, tta_meta = evaluate_tta(
            model,
            pairs_eval,
            device,
            scales=tta_scales,
            crop=tta_crop,
            stride=tta_stride,
            max_images=max_images,
            fade=fade,
        )'''
new_eval = '''        miou_s, ious_s, _cm_s, rates_s = evaluate_detailed(
            model, val_loader, device, row_attn=row_attn, fade=fade
        )
        print(
            {"single_mIoU": jfloat(miou_s), "per_class": dict(zip(CLASS_NAMES, ious_s))},
            flush=True,
        )
        miou_t, ious_t, _cm_t, rates_t, tta_meta = evaluate_tta(
            model,
            pairs_eval,
            device,
            scales=tta_scales,
            crop=tta_crop,
            stride=tta_stride,
            max_images=max_images,
            row_attn=row_attn,
            fade=fade,
        )'''
assert old_eval in s, "TTA evaluate calls missing"
s = s.replace(old_eval, new_eval, 1)

old_tta_sum = '''    summary = {
        "exp": exp,
        "run_name": f"smoke_{exp}",
        "iters": 0,
        "target_iters": 0,
        "best_mIoU_scene5": jfloat(miou_t) if miou_t is not None else jfloat(miou_s),'''
new_tta_sum = '''    summary = {
        "exp": exp,
        "run_name": f"smoke_{run_tag}",
        "iters": 0,
        "target_iters": 0,
        "best_mIoU_scene5": jfloat(miou_t) if miou_t is not None else jfloat(miou_s),'''
assert old_tta_sum in s, "TTA success summary missing"
s = s.replace(old_tta_sum, new_tta_sum, 1)

# Improve method_note when custom run_name / ckpt
old_mnote = '''        "method_note": (
            "TTA on train_B1 best (no retrain)"
            if exp.upper() == "T_B1_TTA"
            else "TTA diagnostic on V_B1C weights (no retrain of M_TTA itself)"
        ),'''
new_mnote = '''        "method_note": (
            "TTA on train_B1 best (no retrain)"
            if exp.upper() == "T_B1_TTA"
            else (
                f"TTA diagnostic on {ckpt_path} (run_name={run_tag}; "
                f"row_attn={'yes' if row_attn is not None else 'no'})"
            )
        ),'''
assert old_mnote in s, "TTA method_note missing"
s = s.replace(old_mnote, new_mnote, 1)

# METHOD_EXPS add M_ROW_SHAD
old_mex = '''METHOD_EXPS = (
    "V_B1C",
    "M_ROW",
    "M_BLV",
    "M_SHAD",
    "M_ABL",
    "M_INVFORM",
    "M_FADE",
    "M_TTA",
    "M_SEGFIX",
    "T_B1_TTA",
    "M_HRDA",
)'''
new_mex = '''METHOD_EXPS = (
    "V_B1C",
    "M_ROW",
    "M_BLV",
    "M_SHAD",
    "M_ROW_SHAD",
    "M_ABL",
    "M_INVFORM",
    "M_FADE",
    "M_TTA",
    "M_SEGFIX",
    "T_B1_TTA",
    "M_HRDA",
)'''
assert old_mex in s, "METHOD_EXPS missing"
s = s.replace(old_mex, new_mex, 1)

# argparse description + --run-name
old_desc = '''            "Recipes: A0/A1/B1/C1 + V_B1C/V_B1E + M_ROW/M_BLV/M_SHAD/M_ABL/"
            "M_INVFORM/M_FADE + M_TTA/M_SEGFIX/T_B1_TTA; M_HRDA stub."'''
new_desc = '''            "Recipes: A0/A1/B1/C1 + V_B1C/V_B1E + M_ROW/M_BLV/M_SHAD/M_ROW_SHAD/M_ABL/"
            "M_INVFORM/M_FADE + M_TTA/M_SEGFIX/T_B1_TTA; M_HRDA stub."'''
assert old_desc in s, "argparse desc missing"
s = s.replace(old_desc, new_desc, 1)

old_ckpt_arg = '''    ap.add_argument(
        "--ckpt",
        default=None,
        help="optional checkpoint for M_SEGFIX / M_TTA / T_B1_TTA (project-relative or absolute)",
    )'''
new_ckpt_arg = '''    ap.add_argument(
        "--ckpt",
        default=None,
        help="optional checkpoint for M_SEGFIX / M_TTA / T_B1_TTA (project-relative or absolute)",
    )
    ap.add_argument(
        "--run-name",
        default=None,
        help=(
            "optional work_dirs/smoke_<run-name> override so consecutive TTA/train "
            "runs do not overwrite (e.g. M_SHAD_TTA, V_B1C_noise)"
        ),
    )'''
assert old_ckpt_arg in s, "--ckpt arg missing"
s = s.replace(old_ckpt_arg, new_ckpt_arg, 1)

old_call = '''        run_exp(
            exp,
            project,
            smoke_dir,
            pretrain,
            iters=args.iters,
            batch=args.batch,
            lr=args.lr,
            dry_run=args.dry_run,
            mem_fraction=args.mem_fraction,
            blv_sigma=args.blv_sigma,
            tta_scales=tta_scales,
            tta_crop=args.tta_crop,
            tta_stride=args.tta_stride,
            ckpt=args.ckpt,
            abl_weight=args.abl_weight,
        )'''
new_call = '''        run_exp(
            exp,
            project,
            smoke_dir,
            pretrain,
            iters=args.iters,
            batch=args.batch,
            lr=args.lr,
            dry_run=args.dry_run,
            mem_fraction=args.mem_fraction,
            blv_sigma=args.blv_sigma,
            tta_scales=tta_scales,
            tta_crop=args.tta_crop,
            tta_stride=args.tta_stride,
            ckpt=args.ckpt,
            abl_weight=args.abl_weight,
            run_name=args.run_name,
        )'''
assert old_call in s, "main run_exp call missing"
s = s.replace(old_call, new_call, 1)

SMOKE.write_text(s, encoding="utf-8")
print("[ok] patched", SMOKE)
print("done")
