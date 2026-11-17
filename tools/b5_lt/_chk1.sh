cd /mnt/d/LuSNAR_Seg
whoami
grep -vE '^\s*#' tools/wsl_bootstrap/03_resume_train_wsl.sh | grep -E 'LIST|list|DATA|batch|workers|--'
ls -la /root/lusnar_data /root/lusnar_data/lists* 2>&1 | head -60
grep -nE 'scales|1\.25|stride|bf16|weight' tools/portable/lusnar_small_crater_score_sc.py | head -20
df -h /mnt/d /root | cat; nproc; free -g