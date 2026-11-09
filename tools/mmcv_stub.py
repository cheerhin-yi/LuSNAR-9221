"""Stub mmcv._ext so mmseg 1.2.2 imports on mmcv-lite (no CUDA ops). Models used (MiT/MSCAN/Swin) need no custom ops; any call to a stubbed op raises."""
import sys, types
class _Ext(types.ModuleType):
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        def f(*a, **k): raise RuntimeError(f"mmcv op {n} unavailable (mmcv-lite)")
        return f
sys.modules.setdefault("mmcv._ext", _Ext("mmcv._ext"))
