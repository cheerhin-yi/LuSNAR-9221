# ROW+BLV opensource test compare (non-official, one-shot)

Both runs use the locked oneshot protocol (equal TTA). Moon_2/4/9 were not used to train.

| run | train scenes | ckpt | overall mIoU | macro | weighted |
|---|---|---|---:|---:|---:|
| no Moon_5 | 1,3,6,7,8 | Scene5 best.pth | 0.8299 | 0.8592 | 0.8447 |
| plus Moon_5 | 1,3,5,6,7,8 | plus5 final | 0.8684 | 0.8919 | 0.8808 |

Moon_5 delta overall mIoU +0.0385

per-scene mIoU:

| scene | no Moon_5 | plus Moon_5 | delta |
|---|---:|---:|---:|
| Moon_2 | 0.9215 | 0.9369 | +0.0155 |
| Moon_4 | 0.9136 | 0.9355 | +0.0219 |
| Moon_9 | 0.7425 | 0.8033 | +0.0608 |

