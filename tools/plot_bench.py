import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=pd.read_csv(r'D:\LuSNAR_Seg\results\vram_bench.csv'); ok=d[d.status=='OK']
fig,ax=plt.subplots(1,2,figsize=(13,5))
for m,g in ok.groupby('model',sort=False):
    b=g.loc[g.groupby('res').bs.idxmax()]
    ax[0].plot(b.res,b.bs,'o-',label=m); ax[1].plot(b.res,b.img_per_s,'o-',label=m)
ax[0].set(title='Max batch size (BF16 AMP, cap 14.3GB)',xlabel='crop',ylabel='batch'); ax[1].set(title='Train throughput at max batch',xlabel='crop',ylabel='img/s')
for a in ax: a.grid(alpha=.3); a.set_xticks([512,768,896,1024])
ax[1].legend(); plt.tight_layout(); plt.savefig(r'D:\LuSNAR_Seg\results\vram_bench.png',dpi=110)
