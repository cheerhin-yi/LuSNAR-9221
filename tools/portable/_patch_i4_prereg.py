from pathlib import Path

# strip BOM + syntax check i4
p = Path("/mnt/d/LuSNAR_Seg/tools/portable/lusnar_scene5_i4.py")
b = p.read_bytes()
if b.startswith(b"\xef\xbb\xbf"):
    p.write_bytes(b[3:])
    print("stripped i4 BOM")
compile(p.read_text(encoding="utf-8"), str(p), "exec")
print("I4_SYNTAX_OK")

# patch protocols PREREGISTERED
q = Path("/mnt/d/LuSNAR_Seg/tools/portable/lusnar_scene5_protocols.py")
t = q.read_text(encoding="utf-8")
if t.startswith("\ufeff"):
    t = t[1:]
old = '''# P1  class logit bias on fused probs
# P1P2 P1 + geometric/CC rules
PREREGISTERED = [
    "C0_equal_tta",
    "I1a_best_scale_per_class",
    "I1b_soft_scale_weights",
    "I2a_gaussian_overlap",
    "I2b_cosine_edge_drop",
    "I3_small_rezoom",
    "P1_logit_bias",
    "P1P2_bias_plus_geo",
]
'''
new = '''# P1  class logit bias on fused probs
# P1P2 P1 + geometric/CC rules
# I4  hybrid full-image 1024 (large classes) + tiled C0 (rock/crater)
#     9th candidate, added before any B scores were used
PREREGISTERED = [
    "C0_equal_tta",
    "I1a_best_scale_per_class",
    "I1b_soft_scale_weights",
    "I2a_gaussian_overlap",
    "I2b_cosine_edge_drop",
    "I3_small_rezoom",
    "P1_logit_bias",
    "P1P2_bias_plus_geo",
    "I4_hybrid_full_tiles",
]
'''
if old not in t:
    raise SystemExit("PREREGISTERED block not found")
t = t.replace(old, new)
t = t.replace(
    '"""Scene5 inference protocols I1/I2/I3 + postproc on first-round cached logits/probs.\n\nPre-registered candidates (<=8), select on A only, report B once.',
    '"""Scene5 inference protocols I1/I2/I3/I4 + postproc on first-round cached logits/probs.\n\nPre-registered candidates (8+I4), select on A only, report B once.',
)
q.write_text(t, encoding="utf-8")
compile(t, str(q), "exec")
print("PROTOCOLS_PREREG_OK")
