from pathlib import Path
p = Path("/mnt/d/LuSNAR_Seg/tools/portable/lusnar_row_blv_ddp_sc.py")
t = p.read_text(encoding="utf-8")
def rep(o, n, c=1):
    global t
    assert t.count(o) == c, (o[:60], t.count(o)); t = t.replace(o, n)
rep('''    def __init__(self, c1: int = 64, c2: int = 128, out_ch: int = 256):
        super().__init__()
        self.proj1 = torch.nn.Conv2d(c1, out_ch, kernel_size=1, bias=False)
        self.proj2 = torch.nn.Conv2d(c2, out_ch, kernel_size=1, bias=False)
        for m in (self.proj1, self.proj2):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")''',
'''    def __init__(self, c1: int = 64, c2: int = 128, out_ch: int = 256, stages: str = "12"):
        super().__init__()
        # stages: "12" (default, E1), "1" (stage1-only ablation), "2" (stage2-only ablation)
        self.stages = stages
        self.proj1 = torch.nn.Conv2d(c1, out_ch, kernel_size=1, bias=False) if "1" in stages else None
        self.proj2 = torch.nn.Conv2d(c2, out_ch, kernel_size=1, bias=False) if "2" in stages else None
        for m in (self.proj1, self.proj2):
            if m is not None:
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")''')
rep('''        f2 = self._f2
        if f2 is None:
            raise RuntimeError("DetailResidual12: stage2 hook did not fire")
        self._f2 = None
        if f1.shape[-2:] != h0.shape[-2:]:
            f1 = F.interpolate(f1, size=h0.shape[-2:], mode="bilinear", align_corners=False)
        f2 = F.interpolate(f2, size=h0.shape[-2:], mode="bilinear", align_corners=False)
        return h0 + self.alpha * (self.proj1(f1) + self.proj2(f2))''',
'''        f2 = self._f2
        if f2 is None:
            raise RuntimeError("DetailResidual12: stage2 hook did not fire")
        self._f2 = None
        r = 0
        if self.proj1 is not None:
            if f1.shape[-2:] != h0.shape[-2:]:
                f1 = F.interpolate(f1, size=h0.shape[-2:], mode="bilinear", align_corners=False)
            r = r + self.proj1(f1)
        if self.proj2 is not None:
            f2 = F.interpolate(f2, size=h0.shape[-2:], mode="bilinear", align_corners=False)
            r = r + self.proj2(f2)
        return h0 + self.alpha * r''')
rep('        residual = DetailResidual12().to(device)\n',
    '        residual = DetailResidual12(stages=args.resid_stages).to(device)\n')
rep('    ap.add_argument("--cc-off", action="store_true")\n',
    '    ap.add_argument("--cc-off", action="store_true")\n    ap.add_argument("--resid-stages", default="12", choices=["12", "1", "2"])\n')
o = '"detail_residual": residual.state_dict() if (residual is not None and args.detail_residual) else None, '
n = o + '"detail_residual_stages": args.resid_stages, '
rep(o, n, c=3)
rep('            "detail_residual": bool(args.detail_residual),\n',
    '            "detail_residual": bool(args.detail_residual),\n            "resid_stages": args.resid_stages,\n')
p.write_text(t, encoding="utf-8")
s = Path("/mnt/d/LuSNAR_Seg/tools/portable/lusnar_small_crater_score_sc.py")
u = s.read_text(encoding="utf-8")
o = '        residual = DetailResidual12().to(device)\n'
assert u.count(o) == 1
u = u.replace(o, '        residual = DetailResidual12(stages=ckpt.get("detail_residual_stages") or "12").to(device)\n')
s.write_text(u, encoding="utf-8")
print("ok")
