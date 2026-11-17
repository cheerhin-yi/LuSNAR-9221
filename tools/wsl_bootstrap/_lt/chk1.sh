cd /mnt/d/LuSNAR_Seg
ls fullimage_compare/work_dirs/FI_M_SAPA fullimage_compare/logs
tail -n 5 fullimage_compare/logs/queue_DESKTOP-ORQLB8F-WSL.log
wc -l /root/lusnar_data/lists/*.txt /root/lusnar_data/lists_plus5/*.txt
ls -la /root/lusnar_data/lists /root/lusnar_data/lists_plus5
grep -h peak_mem work_dirs/train_ROW_BLV_20k/summary.json work_dirs/train_ROW_BLV_20k_plus5/summary.json
grep -E 'mem_mb' logs/train_ROW_BLV_20k_plus5_wsl.log | tail -2
grep -E 'val_mIoU' work_dirs/train_ROW_BLV_20k/train_log.jsonl | tail -2
grep -hE 'val_sec|sec_per_iter|peak_mem' fullimage_compare/work_dirs/FI_M_ROW_BLV/summary.json