$ErrorActionPreference = 'Continue'
$root = 'D:\LuSNAR_Seg'
$py = 'F:\Conda\envs\lusnar\python.exe'
Set-Location $root
$scripts = @(
  'tools\portable\run_opensource_followup.py',
  'tools\portable\run_error_attrib_after.py'
)
foreach ($script in $scripts) {
  $old = Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -and ($_.CommandLine -match [regex]::Escape($script)) }
  foreach ($o in @($old)) { try { Stop-Process -Id $o.ProcessId -Force -ErrorAction SilentlyContinue } catch {} }
}
Start-Sleep -Seconds 1
$followCmd = 'set USE_LIBUV=0&& set PYTHONUNBUFFERED=1&& "' + $py + '" -u tools\portable\run_opensource_followup.py >> "D:\LuSNAR_Seg\logs\opensource_followup.stdout.log" 2>&1'
$fp = Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c', $followCmd) -WorkingDirectory $root -WindowStyle Hidden -PassThru
Write-Host "followup_CMD_PID=$($fp.Id)"
Start-Sleep -Seconds 4
$attribCmd = 'set USE_LIBUV=0&& set PYTHONUNBUFFERED=1&& "' + $py + '" -u tools\portable\run_error_attrib_after.py >> "D:\LuSNAR_Seg\logs\error_attrib_waiter.log" 2>&1'
$ap = Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c', $attribCmd) -WorkingDirectory $root -WindowStyle Hidden -PassThru
Write-Host "attrib_CMD_PID=$($ap.Id)"
Start-Sleep -Seconds 8
Write-Host '==== TIME ===='
Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host '==== WAITER PROCS ===='
Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -and ($_.CommandLine -match 'followup|error_attrib') } | Format-List ProcessId,ParentProcessId,CommandLine
Write-Host '==== FOLLOWUP LOG ===='
if (Test-Path 'D:\LuSNAR_Seg\logs\opensource_followup.log') { Get-Content 'D:\LuSNAR_Seg\logs\opensource_followup.log' -Tail 10 } else { Write-Host 'missing' }
Write-Host '==== ATTRIB LOG ===='
Get-Content 'D:\LuSNAR_Seg\logs\error_attrib_waiter.log' -Tail 8
Write-Host '==== TRAIN STILL ===='
Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -and ($_.CommandLine -match 'lusnar_row_blv_ddp.py') } | Format-Table ProcessId,ParentProcessId
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
Write-Host '==== JSONL ===='
Get-Content 'D:\LuSNAR_Seg\work_dirs\train_ROW_BLV_20k\train_log.jsonl' -Tail 6
