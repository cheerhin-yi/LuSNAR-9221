import cv2, os
from pathlib import Path
for s in [2,4,9]:
    d = Path(rf'D:\LuSNAR_Seg\data\test\Moon_{s}\color')
    files = sorted(d.glob('*.png'))[:3]
    print(f'Moon_{s} n={len(list(d.glob("*.png")))} samples:')
    for f in files:
        im = cv2.imread(str(f), cv2.IMREAD_COLOR)
        print(' ', f.name, None if im is None else im.shape)
