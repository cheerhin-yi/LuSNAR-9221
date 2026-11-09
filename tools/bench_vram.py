import mmcv_stub, os, sys, time, json, csv, torch, torch.nn.functional as F
import mmseg
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
register_all_modules()
CFG = os.path.join(os.path.dirname(mmseg.__file__), '.mim', 'configs')
MODELS_ = {
 'SegFormer-B2': 'segformer/segformer_mit-b2_8xb2-160k_ade20k-512x512.py',
 'SegFormer-B3': 'segformer/segformer_mit-b3_8xb2-160k_ade20k-512x512.py',
 'SegNeXt-S': 'segnext/segnext_mscan-s_1xb16-adamw-160k_ade20k-512x512.py',
 'SegNeXt-B': 'segnext/segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py',
 'UPerNet-Swin-T': 'swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py',
 'DeepLabV3+-R50': 'deeplabv3plus/deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512.py',
}
def fix(o):
    if isinstance(o, dict):
        if o.get('type') == 'SyncBN': o['type'] = 'BN'
        if 'num_classes' in o: o['num_classes'] = 5
        for v in o.values(): fix(v)
    elif isinstance(o, list):
        for v in o: fix(v)
def build(name):
    cfg = Config.fromfile(os.path.join(CFG, MODELS_[name]))
    m = cfg.model.to_dict(); fix(m); m.pop('data_preprocessor', None)
    init = m.get('backbone', {}).get('init_cfg') or m.get('pretrained')
    model = MODELS.build(m); model.init_weights()
    return model, str(init)
def step(model, x, y, opt):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        feats = model.extract_feat(x)
        loss = 0
        heads = [model.decode_head] + (list(model.auxiliary_head) if isinstance(getattr(model,'auxiliary_head',None), torch.nn.ModuleList) else ([model.auxiliary_head] if getattr(model,'auxiliary_head',None) is not None else []))
        for h in heads:
            o = F.interpolate(h(feats), size=x.shape[-2:], mode='bilinear', align_corners=False)
            loss = loss + F.cross_entropy(o.float(), y)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
if __name__ == '__main__':
    names = sys.argv[1:] or list(MODELS_)
    free,total = torch.cuda.mem_get_info(); FRAC = min(0.95,(free-0.5*2**30)/total); torch.cuda.set_per_process_memory_fraction(FRAC); print('mem cap GB', FRAC*total/2**30, flush=True)
    out = open(r'D:\LuSNAR_Seg\results\vram_bench.csv', 'a', newline='')
    w = csv.writer(out)
    for name in names:
        model, src = build(name); model.cuda().train()
        params = sum(p.numel() for p in model.parameters())/1e6
        print(name, 'params(M)=%.1f'%params, 'init=', src, flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=6e-5)
        for res in [512, 768, 896, 1024]:
            for bs in ([2,4,8] if name in ('UPerNet-Swin-T','DeepLabV3+-R50') else [1,2,4,8]):
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                try:
                    x = torch.randn(bs,3,res,res,device='cuda'); y = torch.randint(0,5,(bs,res,res),device='cuda')
                    for _ in range(3): step(model,x,y,opt)
                    torch.cuda.synchronize(); t=time.time(); n=10
                    for _ in range(n): step(model,x,y,opt)
                    torch.cuda.synchronize(); dt=(time.time()-t)/n
                    pk = torch.cuda.max_memory_allocated()/2**30
                    r=[name,res,bs,'OK','%.2f'%pk,'%.2f'%(1/dt),'%.1f'%(bs/dt),'%.1f'%params,src]
                except torch.cuda.OutOfMemoryError:
                    r=[name,res,bs,'OOM','','','','%.1f'%params,src]
                finally:
                    x=y=None; opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                print(r, flush=True); w.writerow(r); out.flush()
                if r[3]=='OOM': break
        del model, opt; torch.cuda.empty_cache()
