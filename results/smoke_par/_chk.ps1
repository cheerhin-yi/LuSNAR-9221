Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "python.exe" } | ForEach-Object { Write-Host ("PID={0} PPID={1}" -f $_.ProcessId, $_.ParentProcessId); Write-Host $_.CommandLine; Write-Host "----" }
Write-Host "==== launcher ps ===="
Get-CimInstance Win32_Process | Where-Object { $_.Name -match "powershell|pwsh" -and $_.CommandLine -match "smoke|LuSNAR|run_smoke" } | ForEach-Object { Write-Host ("PID={0} PPID={1}" -f $_.ProcessId, $_.ParentProcessId); Write-Host $_.CommandLine; Write-Host "----" }
Write-Host "==== train logs ===="
Get-Item D:\LuSNAR_Seg\work_dirs\smoke_M_ROW\train_log.jsonl,D:\LuSNAR_Seg\work_dirs\smoke_V_B1E\train_log.jsonl | Format-List FullName,Length,LastWriteTime
Get-Content D:\LuSNAR_Seg\work_dirs\smoke_M_ROW\train_log.jsonl -Tail 5
Write-Host "---"
Get-Content D:\LuSNAR_Seg\work_dirs\smoke_V_B1E\train_log.jsonl -Tail 5
Write-Host "==== err sizes ===="
Get-ChildItem D:\LuSNAR_Seg\results\smoke_par\*.err.log | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
Write-Host "==== now ===="
Get-Date -Format "yyyy-MM-dd HH:mm:ss"
