#!/usr/bin/env python3
"""Build lusnar_row_blv_ddp_sc.py from lusnar_row_blv_ddp.py (original untouched)."""
from pathlib import Path
src = Path("/mnt/d/LuSNAR_Seg/tools/portable/lusnar_row_blv_ddp.py")
dst = Path("/mnt/d/LuSNAR_Seg/tools/portable/lusnar_row_blv_ddp_sc.py")
assert not dst.exists(), "refuse to overwrite"
t = src.read_text(encoding="utf-8")

def rep(old, new, count=1):
    global t
    n = t.count(old)
    assert n == count, (old[:80], n)
    t = t.replace(old, new)

rep('"""LuSNAR SegFormer-B2 formal training', '"""[SC copy for small-crater rows E0ft/E1/E2/E3; original lusnar_row_blv_ddp.py untouched]\nLuSNAR SegFormer-B2 formal training')
rep("import json\n", "import json\nimport math\n")

# ---- new modules: stage1+stage2 residual and per-instance local margin ----
new_mod = r'''

class DetailResidual12(torch.nn.Module):
    """Spec module B, plain (G=1): H = H0 + alpha * Proj(F_detail).

    F_detail = [stage1 (64ch, 1/4), stage2 (128ch, 1/8) bilinear-upsampled to 1/4]
    (P0 probe: 16-256px craters separable at stage2 AUC~0.95 and stage1 ~0.85-0.89,
    lost after fusion). Proj = 1x1 conv per stage, summed. alpha init 0.
    stage2 is captured with a forward hook on core.backbone so the shared
    _decode_logits(residual(f1, h0)) call path is reused unchanged.
    """

    def __init__(self, c1: int = 64, c2: int = 128, out_ch: int = 256):
        super().__init__()
        self.proj1 = torch.nn.Conv2d(c1, out_ch, kernel_size=1, bias=False)
        self.proj2 = torch.nn.Conv2d(c2, out_ch, kernel_size=1, bias=False)
        for m in (self.proj1, self.proj2):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        self.alpha = torch.nn.Parameter(torch.zeros(1))
        self._f2 = None

    def attach(self, core):
        def hook(_m, _i, out):
            self._f2 = out[1]
        return core.backbone.register_forward_hook(hook)

    def forward(self, f1: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        f2 = self._f2
        if f2 is None:
            raise RuntimeError("DetailResidual12: stage2 hook did not fire")
        self._f2 = None
        if f1.shape[-2:] != h0.shape[-2:]:
            f1 = F.interpolate(f1, size=h0.shape[-2:], mode="bilinear", align_corners=False)
        f2 = F.interpolate(f2, size=h0.shape[-2:], mode="bilinear", align_corners=False)
        return h0 + self.alpha * (self.proj1(f1) + self.proj2(f2))


def local_margin_loss_inst(logits: torch.Tensor, masks: torch.Tensor, margin: float = 0.5,
                           ring_px: float = 8.0, band_px: float = 8.0):
    """Spec module C, per instance, on the post-augmentation 512 label.

    Instances = 8-connected crater components. Positives: all pixels of components with
    area < 1024 and short side < 32, plus (any component) crater pixels within 8 px of
    the crater edge. Negatives: regolith pixels within 8 px outside the crater whose
    nearest crater pixel belongs to that instance (other classes and 255 excluded).
    L = mean_i softplus(m - (mu_i+ - mu_i-)), d = z_crater - z_regolith. 0 if no instance.
    """
    from scipy import ndimage

    crater_id, regolith_id = 2, 0
    B, _, H, W = logits.shape
    d = (logits[:, crater_id] - logits[:, regolith_id]).float().reshape(-1)
    m_np = masks.detach().cpu().numpy()
    pi, pk, ni, nk = [], [], [], []
    off = 0
    for b in range(B):
        m = m_np[b]
        crater = (m == crater_id).astype(np.uint8)
        if int(crater.sum()) == 0:
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats(crater, connectivity=8)
        area = st[:, cv2.CC_STAT_AREA]
        short = np.minimum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
        small = (area < 1024) & (short < 32)
        small[0] = False
        din = cv2.distanceTransform(crater, cv2.DIST_L2, 3)
        pos = (crater > 0) & (small[lab] | (din <= band_px))
        dout, (iy, ix) = ndimage.distance_transform_edt(crater == 0, return_indices=True)
        near = lab[iy, ix]
        neg = (m == regolith_id) & (dout <= ring_px) & (dout > 0)
        base = b * H * W
        p_flat = np.flatnonzero(pos)
        n_flat = np.flatnonzero(neg)
        pi.append(p_flat + base); pk.append(lab.reshape(-1)[p_flat] + off)
        ni.append(n_flat + base); nk.append(near.reshape(-1)[n_flat] + off)
        off += n
    if off == 0:
        return logits.new_zeros(()).float(), 0
    dev = d.device
    pi = torch.from_numpy(np.concatenate(pi)).to(dev); pk = torch.from_numpy(np.concatenate(pk)).to(dev)
    ni = torch.from_numpy(np.concatenate(ni)).to(dev); nk = torch.from_numpy(np.concatenate(nk)).to(dev)
    sp = torch.zeros(off, device=dev).index_add(0, pk, d[pi])
    cp = torch.zeros(off, device=dev).index_add(0, pk, torch.ones_like(pk, dtype=torch.float32))
    sn = torch.zeros(off, device=dev).index_add(0, nk, d[ni])
    cn = torch.zeros(off, device=dev).index_add(0, nk, torch.ones_like(nk, dtype=torch.float32))
    ok = (cp > 0) & (cn > 0)
    if not bool(ok.any()):
        return logits.new_zeros(()).float(), 0
    gap = sp[ok] / cp[ok] - sn[ok] / cn[ok]
    return F.softplus(margin - gap).mean(), int(ok.sum())


class _SegLogits(torch.nn.Module):'''
rep("\n\nclass _SegLogits(torch.nn.Module):", new_mod)

# ---- CC bank: index holds Windows paths (silently no-op under WSL); remap ----
rep("    cc_bank = CCBank(smoke_dir / \"cc_bank_index.json\") if (smoke_dir and flags[\"cc_paste\"]) else None\n",
    "    cc_bank = CCBank(smoke_dir / \"cc_bank_index.json\") if (smoke_dir and flags[\"cc_paste\"]) else None\n"
    "    if cc_bank is not None:\n"
    "        cc_dir = Path(args.cc_dir) if args.cc_dir else smoke_dir / \"cc_bank\"\n"
    "        cc_bank.items = [str(cc_dir / p.replace(\"\\\\\", \"/\").split(\"/\")[-1]) for p in cc_bank.items]\n"
    "        n_ok = sum(Path(p).is_file() for p in cc_bank.items)\n"
    "        if is_main(rank):\n"
    "            print(f\"[cc_bank] remapped to {cc_dir}: {n_ok}/{len(cc_bank.items)} files exist\", flush=True)\n"
    "        if n_ok != len(cc_bank.items):\n"
    "            raise SystemExit(\"cc_bank files missing after remap\")\n")

# ---- args ----
rep('    ap.add_argument("--reset-optimizer"',
    '    ap.add_argument("--init-from", default=None, help="fine-tune: load model+row_attn weights only, iter starts at 0")\n'
    '    ap.add_argument("--detail-residual", action="store_true", help="E1: stage1+stage2 zero-init residual (spec module B, G=1)")\n'
    '    ap.add_argument("--warmup", type=int, default=0)\n'
    '    ap.add_argument("--cosine", action="store_true", help="cosine lr decay to lr*min_lr_ratio after warmup")\n'
    '    ap.add_argument("--min-lr-ratio", type=float, default=0.1)\n'
    '    ap.add_argument("--new-lr-mult", type=float, default=10.0, help="lr multiplier for newly added residual params")\n'
    '    ap.add_argument("--cc-dir", default=None)\n'
    '    ap.add_argument("--reset-optimizer"')

# ---- model build ----
rep("    residual = PlainStage1Residual().to(device) if args.plain_residual else None\n",
    "    residual = PlainStage1Residual().to(device) if args.plain_residual else None\n"
    "    if args.detail_residual:\n"
    "        if residual is not None:\n"
    "            raise SystemExit(\"--plain-residual and --detail-residual are exclusive\")\n"
    "        residual = DetailResidual12().to(device)\n"
    "        residual.attach(core)\n")
rep("    params = [p for p in model.parameters() if p.requires_grad]\n    if edge_aux is not None:\n        params += list(edge_aux.parameters())\n    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)\n",
    "    new_ids = {id(p) for p in residual.parameters()} if residual is not None else set()\n"
    "    params = [p for p in model.parameters() if p.requires_grad and id(p) not in new_ids]\n"
    "    if edge_aux is not None:\n"
    "        params += list(edge_aux.parameters())\n"
    "    groups = [{\"params\": params, \"lr_mult\": 1.0}]\n"
    "    if new_ids:\n"
    "        groups.append({\"params\": [p for p in residual.parameters()], \"lr_mult\": args.new_lr_mult})\n"
    "    for g in groups:\n"
    "        g[\"lr\"] = args.lr * g[\"lr_mult\"]\n"
    "    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.01)\n"
    "    def lr_factor(step):\n"
    "        f = 1.0\n"
    "        if args.warmup > 0 and step < args.warmup:\n"
    "            f = (step + 1) / args.warmup\n"
    "        elif args.cosine:\n"
    "            prog = (step - args.warmup) / max(1, args.iters - args.warmup)\n"
    "            f = args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))\n"
    "        return f\n")

# ---- init-from (fresh iter counter) ----
rep("    elif is_main(rank) and log_path.exists() and not args.dry_run:\n        log_path.unlink()\n",
    "    elif is_main(rank) and log_path.exists() and not args.dry_run:\n        log_path.unlink()\n"
    "    if args.init_from:\n"
    "        if args.resume:\n"
    "            raise SystemExit(\"--init-from and --resume are exclusive\")\n"
    "        ick = torch.load(args.init_from, map_location=device, weights_only=False)\n"
    "        raw_model.load_state_dict(ick[\"model\"])\n"
    "        if row_attn is not None:\n"
    "            row_attn.load_state_dict(ick[\"row_attn\"])\n"
    "        if is_main(rank):\n"
    "            print({\"init_from\": args.init_from, \"init_iter\": int(ick.get(\"iter\", 0)), \"init_miou\": ick.get(\"miou\")}, flush=True)\n"
    "        del ick\n")

# ---- lr schedule per step + margin loss ----
rep("                opt.zero_grad(set_to_none=True)\n",
    "                fac = lr_factor(it)\n"
    "                for g in opt.param_groups:\n"
    "                    g[\"lr\"] = args.lr * g.get(\"lr_mult\", 1.0) * fac\n"
    "                opt.zero_grad(set_to_none=True)\n")
rep("                    if args.local_margin:\n                        loss = loss + args.local_margin_weight * local_margin_loss(logits.float(), masks)\n",
    "                    lm_val, lm_n = 0.0, 0\n"
    "                    if args.local_margin:\n"
    "                        lm, lm_n = local_margin_loss_inst(logits.float(), masks)\n"
    "                        lm_val = float(lm.detach())\n"
    "                        loss = loss + args.local_margin_weight * lm\n")
rep("                        \"dice\": float(dsc.item()),\n",
    "                        \"dice\": float(dsc.item()),\n"
    "                        \"lmargin\": lm_val,\n"
    "                        \"lmargin_n\": lm_n,\n"
    "                        \"lr\": opt.param_groups[0][\"lr\"],\n"
    "                        \"alpha\": float(residual.alpha.detach()) if residual is not None else None,\n")

# ---- checkpoints: store residual under a type-specific key; always write last.pth ----
ck_old = "\"plain_residual\": residual.state_dict() if residual is not None else None, "
ck_new = "\"plain_residual\": residual.state_dict() if (residual is not None and not args.detail_residual) else None, \"detail_residual\": residual.state_dict() if (residual is not None and args.detail_residual) else None, "
rep(ck_old, ck_new, count=2)
rep("                    if miou > best:\n",
    "                    if it == iters:\n"
    "                        torch.save({\"model\": net.state_dict(), \"row_attn\": row_attn.state_dict() if row_attn is not None else None, " + ck_new + "\"iter\": it, \"miou\": miou, \"recipe\": args.recipe}, work / \"last.pth\")\n"
    "                    if miou > best:\n")
# drop optimizer state from iter_* checkpoints to save disk (fine-tune rows only)
rep("\"optimizer\": opt.state_dict(), \"iter\": it, \"miou\": miou", "\"iter\": it, \"miou\": miou", count=2)
rep("            \"tta_final\": args.tta_final,\n",
    "            \"tta_final\": args.tta_final,\n"
    "            \"init_from\": args.init_from,\n"
    "            \"detail_residual\": bool(args.detail_residual),\n"
    "            \"local_margin\": bool(args.local_margin),\n"
    "            \"local_margin_weight\": args.local_margin_weight,\n"
    "            \"warmup\": args.warmup, \"cosine\": bool(args.cosine), \"min_lr_ratio\": args.min_lr_ratio,\n"
    "            \"new_lr_mult\": args.new_lr_mult,\n"
    "            \"cc_paste_remapped\": cc_bank is not None,\n")
dst.write_text(t, encoding="utf-8")
print("wrote", dst, len(t))
