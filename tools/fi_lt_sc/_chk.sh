cd /mnt/d/LuSNAR_Seg/tools/fi_lt_sc
cp -p 10_fi_lt_sc_queue_wsl.sh 10_fi_lt_sc_queue_wsl.sh.v1_failed_nccl_cpu
sed -i -e 's/^RUN0=LT_FI_dry_E3sc$/RUN0=LT_FI_dry_E3sc_v2/' \
  -e 's/^RUN1=LT_FI_ROW_BLV_20k$/RUN1=LT_FI_ROW_BLV_20k_v2/' \
  -e 's/^RUN2=LT_FI_ROW_BLV_20k_E3sc$/RUN2=LT_FI_ROW_BLV_20k_E3sc_v2/' \
  -e 's/train lusnar_row_blv_ddp_sc.py/train lusnar_row_blv_ddp_sc_fi.py/g' \
  -e 's/train lusnar_row_blv_ddp.py/train lusnar_row_blv_ddp_fi.py/g' \
  -e 's/ports_v1//' 10_fi_lt_sc_queue_wsl.sh
sed -i -e 's/ 29671 / 29672 /; s/ 29641 / 29642 /; s/ 29661 / 29662 /' 10_fi_lt_sc_queue_wsl.sh
sed -i '2a # v2 (14:15): v1 died at iter 1 (NCCL all_gather of CPU tensor in the DDP sync check). Now uses *_fi.py trainer copies (only that fix).' 10_fi_lt_sc_queue_wsl.sh
sed -i 's/\r$//' 10_fi_lt_sc_queue_wsl.sh
bash -n 10_fi_lt_sc_queue_wsl.sh && echo OK
grep -n -E '^RUN|train lusnar|v2' 10_fi_lt_sc_queue_wsl.sh
ls -d /mnt/d/LuSNAR_Seg/work_dirs/*_v2 2>&1