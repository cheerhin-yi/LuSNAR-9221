$path = 'D:\LuSNAR_Seg\tools\wsl_bootstrap\03_resume_train_wsl.sh'
$c = Get-Content -Raw $path
$c = $c -replace 'workers ladder: 8 -> 6 -> 4','workers ladder: 16 -> 12 -> 8 (DataLoader/shm OOM only; preferred 16)'
$c = $c -replace 'WORKERS="\$\{WORKERS:-8\}"','WORKERS="${WORKERS:-16}"'
$c = $c -replace 'workers=\$WORKERS \(ladder 8->6->4\)','workers=$WORKERS (ladder 16->12->8)'
Set-Content -Path $path -Value $c -Encoding utf8NoBOM -NoNewline
Write-Output 'patched 03_resume_train_wsl.sh'
Select-String -Path $path -Pattern 'WORKERS|ladder' | ForEach-Object { $_.Line }
