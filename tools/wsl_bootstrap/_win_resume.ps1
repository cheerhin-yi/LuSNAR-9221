$ErrorActionPreference = 'Continue'
$root = 'D:\LuSNAR_Seg'
$py = 'F:\Conda\envs\lusnar\python.exe'
$log = 'D:\LuSNAR_Seg\logs\train_ROW_BLV_20k.log'
$ckpt = 'D:\LuSNAR_Seg\work_dirs\train_ROW_BLV_20k\iter_4000.pth'
Set-Location $root
function Get-TrainProcs {
  Get-CimInstance Win32_Process -Filter "name='python.exe'" |
    Where-Object { $_.CommandLine -and ($_.CommandLine -match 'lusnar_row_blv_ddp|torch.distributed.run') } |
    Select-Object ProcessId, ParentProcessId, CommandLine
}
Write-Host "==== TIME ===="
Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host "==== EXISTING TRAIN ===="
$existing = @(Get-TrainProcs)
if ($existing.Count -gt 0) {
  Write-Host 'WINDOWS_TRAIN_ALREADY_ALIVE'
  $existing | Format-List | Out-String | Write-Host
  exit 0
}
Write-Host 'no windows train'
Write-Host "==== GPU ===="
& nvidia-smi.exe --query-gpu=index,utilization.gpu,memory.used --format=csv
Write-Host "==== LAUNCH TRAIN ===="
$env:USE_LIBUV = '0'
$stamp = "=== Windows resume $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') workers=4 ckpt=$ckpt ==="
Add-Content -Path $log -Value $stamp -Encoding utf8
$trainCmd = 'set USE_LIBUV=0&& set PYTHONUNBUFFERED=1&& "' + $py + '" -u -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 29601 tools\portable\lusnar_row_blv_ddp.py --recipe M_ROW_BLV --run-name train_ROW_BLV_20k --iters 20000 --batch 48 --workers 4 --amp bf16 --crop 512 --lr 6e-5 --val-every 1000 --tta-final --project-root D:\LuSNAR_Seg --data-root D:\LuSNAR_Seg\data --train-subdir train --resume "' + $ckpt + '" >> "' + $log + '" 2>&1'
$p = Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c', $trainCmd) -WorkingDirectory $root -WindowStyle Hidden -PassThru
Write-Host "TRAIN_LAUNCH_PID=$($p.Id)"
Start-Sleep -Seconds 4
Write-Host "==== LAUNCH WAITERS ===="
$waiters = @(
  @{ name='followup'; script='tools\portable\run_opensource_followup.py'; log='D:\LuSNAR_Seg\logs\opensource_followup.log' },
  @{ name='attrib'; script='tools\portable\run_error_attrib_after.py'; log='D:\LuSNAR_Seg\logs\error_attrib_waiter.log' }
)
foreach ($pair in $waiters) {
  $old = Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -and ($_.CommandLine -match [regex]::Escape($pair.script)) }
  foreach ($o in @($old)) { try { Stop-Process -Id $o.ProcessId -Force -ErrorAction SilentlyContinue } catch {} }
  $wcmd = 'set USE_LIBUV=0&& set PYTHONUNBUFFERED=1&& "' + $py + '" -u ' + $pair.script + ' >> "' + $pair.log + '" 2>&1'
  $wp = Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c', $wcmd) -WorkingDirectory $root -WindowStyle Hidden -PassThru
  Write-Host "$($pair.name)_CMD_PID=$($wp.Id)"
}
Write-Host "==== WAIT 150s ===="
Start-Sleep -Seconds 150
Write-Host "==== TIME2 ===="
Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host "==== TRAIN PROCS ===="
Get-TrainProcs | Format-List | Out-String | Write-Host
Write-Host "==== GPU2 ===="
& nvidia-smi.exe --query-gpu=index,utilization.gpu,memory.used --format=csv
& nvidia-smi.exe --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
Write-Host "==== JSONL ===="
Get-Content 'D:\LuSNAR_Seg\work_dirs\train_ROW_BLV_20k\train_log.jsonl' -Tail 8
Write-Host "==== LOG TAIL ===="
Get-Content $log -Tail 25 | ForEach-Object { if ($_ -notmatch 'mmengine|PretrainedInit|torch.Size') { if ($_.Length -gt 240) { $_.Substring(0,240) } else { $_ } } }
