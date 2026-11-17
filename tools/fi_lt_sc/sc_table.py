import json, os, glob
D = "/mnt/d/LuSNAR_Seg/results/small_crater_sc"
C1 = dict(m=0.876516, c=0.892485, r=0.757514, t=0.751820)
C2 = dict(m=0.884229, c=0.902002, r=0.776408, t=0.761198)
def load(f):
    j = json.load(open(f, encoding="utf-8"))
    pc = j["per_class_IoU"]
    b = {(x["lo"], x["hi"]): x for x in j["crater_instances_by_area"]}
    return dict(m=j["mIoU"], c=pc["\u649e\u51fb\u5751"], r=pc["\u5ca9\u77f3"], t=pc["\u5c71\u4f53"],
                h0=b[(0, 16)]["hit50"], n0=b[(0, 16)]["n"], h1=b[(16, 64)]["hit50"], n1=b[(16, 64)]["n"], ck=j.get("checkpoint"))
def gate(x, C):
    dm, dc, dr, dt = x["m"]-C["m"], x["c"]-C["c"], x["r"]-C["r"], x["t"]-C["t"]
    ok = dict(crater=dc >= 0.01, miou=dm >= -0.005, rock=dr >= -0.01, mount=dt >= -0.01)
    return "dm=%+.4f dc=%+.4f dr=%+.4f dt=%+.4f pass=%s %s" % (dm, dc, dr, dt, all(ok.values()), {k: v for k, v in ok.items()})
for f in sorted(glob.glob(D + "/*.json"), key=os.path.getmtime):
    t = os.path.basename(f)[:-5]
    try:
        x = load(f)
    except Exception as e:
        print(t, "ERR", e); continue
    print("%-28s mIoU=%.6f crater=%.6f rock=%.6f mount=%.6f 0-16=%d/%d 16-64=%d/%d" % (t, x["m"], x["c"], x["r"], x["t"], x["h0"], x["n0"], x["h1"], x["n1"]))
    print("   vs E0ft s42 :", gate(x, C1))
    print("   vs E0ft s2026:", gate(x, C2))