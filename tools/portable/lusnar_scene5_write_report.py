#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write results/inference_phase_report.md and append HANDOFF / migrate status.

Looks up class scores via per_class_en (regolith/rock/crater/mountain/sky).
Chinese keys, if present, are mapped by official name or by class order.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Shanghai")
PROJECT = Path("/mnt/d/LuSNAR_Seg")
PHASE = PROJECT / "results" / "inference_phase"
EN = ["regolith", "rock", "crater", "mountain", "sky"]
ZH = ["\u6708\u58e4", "\u5ca9\u77f3", "\u649e\u51fb\u5751", "\u5c71\u4f53", "\u5929\u7a7a"]
ZH_BY_EN = dict(zip(EN, ZH))
EN_BY_ZH = dict(zip(ZH, EN))


def fmt(x, nd=4):
    if x is None:
        return "\u2014"
    try:
        xf = float(x)
        if xf != xf:
            return "\u2014"
        return f"{xf:.{nd}f}"
    except Exception:
        return str(x)


def load(name):
    p = PHASE / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def lookup_class(src, key_en: str):
    """IoU for one English class name from a split dict or a raw per-class map."""
    if not isinstance(src, dict) or not key_en:
        return None
    enmap = src.get("per_class_en")
    if isinstance(enmap, dict) and key_en in enmap:
        return enmap[key_en]
    pc = src.get("per_class") if "per_class" in src or "mIoU" in src else src
    if not isinstance(pc, dict):
        return None
    if key_en in pc:
        return pc[key_en]
    zh = ZH_BY_EN.get(key_en)
    if zh and zh in pc:
        return pc[zh]
    keys = list(pc.keys())
    if len(keys) == 5:
        known = [EN_BY_ZH.get(k, k if k in EN else None) for k in keys]
        if all(k is None for k in known):
            return list(pc.values())[EN.index(key_en)]
    return None


def class_vec(src):
    return [lookup_class(src, k) for k in EN]


def frozen_from(proto):
    if not proto:
        return {}
    fr = proto.get("frozen_recipe")
    if isinstance(fr, dict) and fr.get("name"):
        return fr
    cands = proto.get("candidates") or {}
    name = proto.get("winner")
    if not name or name not in cands:
        # derive: max A mIoU among candidates that already have a record
        ranked = []
        for n, cand in cands.items():
            if not isinstance(cand, dict) or "A" not in cand:
                continue
            if n.endswith("_ref") or n.startswith("P2_"):
                continue
            ranked.append((float((cand.get("A") or {}).get("mIoU") or -1.0), n))
        ranked.sort(reverse=True)
        name = ranked[0][1] if ranked else None
    if not name or name not in cands:
        return {"name": "n/a"}
    A = cands[name].get("A") or {}
    B = cands[name].get("B") or {}
    return {
        "name": name,
        "A_mIoU": A.get("mIoU"),
        "B_mIoU": B.get("mIoU"),
        "B_per_class": B.get("per_class_en") or B.get("per_class"),
        "note": "Derived from winner/candidates (no frozen_recipe key).",
    }


def fmt_pc_row(src):
    return " / ".join(f"{ZH[i]} {fmt(v)}" for i, v in enumerate(class_vec(src)))


def main():
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    cache = load("cache_diag.json")
    audit = load("t0_scale_audit.json")
    attrib = load("scene5_error_attrib.json")
    proto = load("protocols.json") or load("protocols_cpu.json")
    split = json.loads(
        (PROJECT / "data" / "lists_portable" / "scene5_ab" / "split_meta.json").read_text(
            encoding="utf-8"
        )
    )

    lines = []
    lines += [
        "# Scene5 \u63a8\u7406\u9636\u6bb5\u62a5\u544a\uff08I1\u2013I4 \u4e0e\u540e\u5904\u7406\uff09",
        "",
        f"- \u751f\u6210\u65f6\u95f4\uff08Asia/Shanghai\uff09\uff1a{now}",
        "- \u6743\u91cd\uff1a`work_dirs/train_ROW_BLV_20k/best.pth`\uff08iter 17000\uff0cSegFormer-B2 + V_B1C + ROW + BLV\uff0ccrop 512\uff09",
        "- \u8303\u56f4\uff1a\u4ec5 Scene5\u3002Moon_2/4/9 \u6807\u7b7e\u672a\u7528\u4e8e\u9009\u578b\u3002A \u534a\u96c6\u9009\u578b\uff0cB \u534a\u96c6\u53ea\u62a5\u4e00\u6b21\u3002",
        f"- A/B \u5212\u5206\uff1aseed={split['seed']}\uff0cA={split['n_A']}\uff0cB={split['n_B']}\uff0c\u6e05\u5355 `data/lists_portable/scene5_ab/`",
        "",
        "## 1. \u8bca\u65ad",
        "",
    ]

    if cache:
        ft = cache["fused_tta"]
        lines += [
            "### 1.1 \u7b49\u6743 TTA\uff08\u5c3a\u5ea6 0.75/1.0/1.25\uff0c\u6ed1\u7a97 512/256\uff09",
            "",
            f"- \u5168 Scene5 mIoU\uff1a**{fmt(ft['mIoU'])}**",
            f"- \u5206\u7c7b IoU\uff1a{fmt_pc_row(ft)}",
            f"- \u6df7\u6dc6\u7387\uff1amountain\u2192rock={fmt((ft.get('rates') or {}).get('mountain_to_rock'))}\uff0c"
            f"crater\u2192regolith={fmt((ft.get('rates') or {}).get('crater_to_regolith'))}\uff0c"
            f"sky\u2192mountain={fmt((ft.get('rates') or {}).get('sky_to_mountain'))}",
            f"- \u8017\u65f6 {fmt(cache.get('wall_s'), 1)} s\uff0c\u7a97\u53e3\u524d\u5411 {cache.get('forward_windows')} \u6b21",
            "",
            "### 1.2 \u5355\u5c3a\u5ea6 mIoU \u4e0e\u5206\u7c7b IoU",
            "",
            "| scale | mIoU | \u6708\u58e4 | \u5ca9\u77f3 | \u649e\u51fb\u5751 | \u5c71\u4f53 | \u5929\u7a7a |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for sk, row in cache["per_scale"].items():
            pc = class_vec(row)
            lines.append(
                "| "
                + " | ".join([str(sk), fmt(row.get("mIoU"))] + [fmt(v) for v in pc])
                + " |"
            )
        lines.append("")

    if audit:
        lines += ["### 1.3 T0 \u5c3a\u5ea6\u5ba1\u8ba1\uff08GT \u8fde\u901a\u57df\u5c3a\u5bf8\uff09", ""]
        for cls in ("rock", "crater"):
            d = audit["classes"][cls]
            lines.append(
                f"- **{cls}**\uff1a\u8fde\u901a\u57df {d['n_cc']} \u4e2a\uff0cside p50={fmt(d['side_p50'], 1)} px\uff0c"
                f"p90={fmt(d['side_p90'], 1)}\uff0c<32px \u5360\u6bd4 {fmt(d['frac_side_lt_32'], 3)}"
            )
        lines += ["", f"- \u5efa\u8bae\uff1a{audit.get('recommendation', '')}", ""]

    if attrib:
        lines += [
            "### 1.4 \u8bef\u5dee\u5f52\u56e0\uff08Scene5 \u7b49\u6743 TTA \u9884\u6d4b\uff09",
            "",
            f"- \u56fe\u6570 {attrib['n_images']}\uff0cmIoU {fmt(attrib['mIoU'])}",
            "",
            "| \u7c7b\u522b | IoU | \u4e3b\u56e0 | \u8bc1\u636e |",
            "|---|---:|---|---|",
        ]
        for row in attrib["classes"]:
            en = row.get("class") or ""
            name = row.get("name") or ZH_BY_EN.get(en, en)
            prim = row["primary"] or "\u2014"
            ev = "\uff1b".join(f.get("evidence", "") for f in (row.get("flags") or [])) or "\u2014"
            lines.append(f"| {name} ({en}) | {fmt(row['iou'])} | {prim} | {ev} |")
        lines.append("")

    lines += ["## 2. \u9884\u6ce8\u518c\u65b9\u6848\uff08\u542b I2 / I3 / I4\uff09", ""]
    if proto:
        preg = proto.get("preregistered") or list((proto.get("candidates") or {}).keys())
        lines.append("\u9884\u6ce8\u518c\uff1a`" + "`, `".join(preg) + "`")
        if proto.get("selection_rule"):
            lines.append(f"\u9009\u578b\u89c4\u5219\uff1a{proto['selection_rule']}")
        lines.append("")
        lines += [
            "| \u65b9\u6848 | A mIoU | B mIoU | rock A/B | crater A/B | mountain A/B | ms/\u56fe |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        # keep preregistered order, then any extra candidates (e.g. P2 ref)
        cand_order = list(preg)
        for name in (proto.get("candidates") or {}):
            if name not in cand_order:
                cand_order.append(name)
        for name in cand_order:
            cand = (proto.get("candidates") or {}).get(name)
            if not isinstance(cand, dict) or "A" not in cand:
                continue
            A, B = cand["A"], cand.get("B") or {}
            lines.append(
                f"| {name} | {fmt(A.get('mIoU'))} | {fmt(B.get('mIoU'))} | "
                f"{fmt(lookup_class(A, 'rock'))}/{fmt(lookup_class(B, 'rock'))} | "
                f"{fmt(lookup_class(A, 'crater'))}/{fmt(lookup_class(B, 'crater'))} | "
                f"{fmt(lookup_class(A, 'mountain'))}/{fmt(lookup_class(B, 'mountain'))} | "
                f"{fmt(A.get('ms_per_img'), 1)} |"
            )
        lines.append("")

        i4 = (proto.get("candidates") or {}).get("I4_hybrid_full_tiles") or {}
        if i4:
            fo = i4.get("full_only_A") or {}
            lines += [
                "I4 \u8bf4\u660e\uff1a"
                + str(proto.get("i4_note") or ""),
                f"- \u9009\u4e2d\u7684\u5408\u5e76\u89c4\u5219\uff08\u4ec5\u6309 A\uff09\uff1a`{i4.get('merge_rule')}`\uff1bB \u53ea\u62a5\u4e00\u6b21\u3002",
                f"- \u7eaf 1024 \u6574\u56fe\uff08\u8bca\u65ad\uff0c\u4e0d\u53c2\u4e0e\u9009\u578b\uff09A mIoU={fmt(fo.get('mIoU'))}\uff0c"
                f"rock={fmt(lookup_class(fo, 'rock'))}\uff0cmountain={fmt(lookup_class(fo, 'mountain'))}\u3002",
                "",
            ]
        i3 = (proto.get("candidates") or {}).get("I3_small_rezoom") or {}
        if i3:
            i3_base = i3.get("base") or "\u2014"
            lines += [
                "I3 \u57fa\u7ebf\uff1a`" + str(i3_base) + "`\u3002"
                "I2b \u4e0e I3 \u5728 A \u4e0a\u660e\u663e\u4f4e\u4e8e I2a / I1a\uff0c\u6309\u9009\u578b\u89c4\u5219\u672a\u518d\u62a5 B\u3002",
                "",
            ]

        tests = proto.get("paired_tests_vs_C0_on_A") or {}
        if tests:
            lines += [
                "### \u4e0e C0 \u7684\u914d\u5bf9\u68c0\u9a8c\uff08A \u96c6\u9010\u56fe mIoU\uff09",
                "",
                "| \u65b9\u6848 | \u0394mean | Wilcoxon p | Holm p | bootstrap 95% CI |",
                "|---|---:|---:|---:|---|",
            ]
            for name, t in tests.items():
                ci = t.get("bootstrap_ci95") or [None, None]
                lines.append(
                    f"| {name} | {fmt(t.get('delta_mean'))} | {fmt(t.get('wilcoxon_p'), 4)} | "
                    f"{fmt(t.get('holm_p'), 4)} | [{fmt(ci[0])}, {fmt(ci[1])}] |"
                )
            lines.append("")
        fr = frozen_from(proto)
        bpc = fr.get("B_per_class") or {}
        bpc_txt = ", ".join(
            f"{ZH_BY_EN.get(k, k)}={fmt(v)}" for k, v in (
                zip(EN, class_vec({"per_class_en": bpc} if any(x in (bpc or {}) for x in EN) else {"per_class": bpc}))
                if bpc else []
            )
        )
        if bpc and not bpc_txt:
            # class_vec already handles ZH keys via per_class
            vals = class_vec({"per_class": bpc, "per_class_en": bpc if any(k in bpc for k in EN) else {}})
            # simpler:
            vals = class_vec({"per_class": bpc})
            if all(v is None for v in vals):
                vals = class_vec({"per_class_en": bpc})
            bpc_txt = ", ".join(f"{ZH[i]}={fmt(v)}" for i, v in enumerate(vals))
        lines += [
            "## 3. \u51bb\u7ed3\u65b9\u6848",
            "",
            f"- **\u9009\u4e2d**\uff1a`{fr.get('name', 'n/a')}`",
            f"- A mIoU={fmt(fr.get('A_mIoU'))}\uff0cB \u4e00\u6b21\u6027 mIoU={fmt(fr.get('B_mIoU'))}",
            f"- B \u5206\u7c7b IoU\uff1a{bpc_txt or fmt(None)}",
            f"- \u8bf4\u660e\uff1a{fr.get('note', '')}",
            "",
        ]
    else:
        lines += ["\u672a\u627e\u5230 protocols.json\u3002", ""]

    lines += [
        "## 4. \u7f13\u5b58\u8def\u5f84",
        "",
        "- TTA \u6982\u7387 float16\uff1a`/root/lusnar_data/cache/scene5_tta_probs/`\uff08`stem.npy`\uff0c\u5f62\u72b6 5\u00d71024\u00d71024\uff09",
        "- \u5355\u5c3a\u5ea6 logits float16\uff1a`/root/lusnar_data/cache/scene5_scale_logits/`\uff08`stem_s{scale}.npy`\uff09",
        "- \u9884\u6d4b\u56fe\uff1a`/root/lusnar_data/cache/scene5_preds/`",
        "- \u8fc7\u7a0b JSON\uff1a`results/inference_phase/`",
        "",
        "## 5. \u4e0b\u4e00\u6b65",
        "",
        "- **\u5c1a\u672a\u542f\u52a8 plus5**\u3002",
        "- \u63a8\u7406\u9636\u6bb5\u62a5\u544a\u843d\u5730\u540e\uff0c\u6309\u8ba1\u5212\u518d\u8fdb\u5165 Windows plus5\u3002",
        "",
    ]

    report = PROJECT / "results" / "inference_phase_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("REPORT_OK", report)

    note = (
        f"\n\n## Inference phase report written ({now} Asia/Shanghai)\n"
        f"- Scene5 A/B split seed42: A={split['n_A']} B={split['n_B']}; lists in data/lists_portable/scene5_ab/\n"
        f"- Diagnostics + I1-I4 + postproc on first-round best.pth done; report: results/inference_phase_report.md\n"
        f"- TTA probs: /root/lusnar_data/cache/scene5_tta_probs/\n"
        f"- Frozen: {frozen_from(proto).get('name', 'n/a')}; "
        f"B mIoU={frozen_from(proto).get('B_mIoU', 'n/a')}\n"
        f"- HARD STOP: no plus5 launched by this report script.\n"
    )
    for rel in ("HANDOFF.md", "logs/wsl_migrate_status.md"):
        path = PROJECT / rel
        if path.exists():
            prev = path.read_text(encoding="utf-8")
            path.write_text(prev + note, encoding="utf-8")
            print("APPENDED", path, "bytes_before", len(prev.encode('utf-8')), "bytes_after", path.stat().st_size)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(note, encoding="utf-8")
            print("CREATED", path)


if __name__ == "__main__":
    main()
