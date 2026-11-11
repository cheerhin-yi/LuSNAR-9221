# Stop ROW+BLV DDP gracefully then reboot
$ErrorActionPreference = 'Continue'
$Root = 'D:\LuSNAR_Seg'
$Handoff = Join-Path $Root 'HANDOFF.md'
$WorkDir = Join-Path $Root 'work_dirs\train_ROW_BLV_20k'
$Stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

function Write-Step($m) { Write-Host ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m) }

Write-Step 'Snapshot ckpts before stop'
Get-ChildItem (Join-Path $WorkDir '*.pth') -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | ForEach-Object {
  Write-Host ("CKPT {0} {1} {2}" -f $_.Name, $_.Length, $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
}

Write-Step 'Find train / waiter PIDs'
$trainPids = @()
$waiterPids = @()
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine } | ForEach-Object {
  $cl = $_.CommandLine
  if ($cl -match 'lusnar_row_blv_ddp|torch\.distributed\.run.*29601|master_port 29601') {
    $trainPids += $_.ProcessId
    Write-Host ("TRAIN PID={0} PPID={1} {2}" -f $_.ProcessId, $_.ParentProcessId, $cl.Substring(0, [Math]::Min(180,$cl.Length)))
  }
  if ($cl -match 'run_opensource_followup\.py|run_error_attrib_after\.py') {
    $waiterPids += $_.ProcessId
    Write-Host ("WAITER PID={0} PPID={1} {2}" -f $_.ProcessId, $_.ParentProcessId, $cl.Substring(0, [Math]::Min(180,$cl.Length)))
  }
}
$trainPids = @($trainPids | Select-Object -Unique)
$waiterPids = @($waiterPids | Select-Object -Unique)
Write-Host ("trainPids=" + ($trainPids -join ','))
Write-Host ("waiterPids=" + ($waiterPids -join ','))

foreach ($known in @(3208,42796,58232)) {
  if ((Get-Process -Id $known -ErrorAction SilentlyContinue) -and ($trainPids -notcontains $known)) {
    $trainPids += $known
  }
}

Write-Step 'Graceful stop: taskkill without /F on train tree'
foreach ($tp in $trainPids) {
  cmd /c ("taskkill /PID {0}" -f $tp) 2>&1 | ForEach-Object { Write-Host $_ }
}

Write-Step 'Wait up to 60s for train PIDs to exit'
$deadline = (Get-Date).AddSeconds(60)
while ((Get-Date) -lt $deadline) {
  $alive = @()
  foreach ($tp in $trainPids) {
    if (Get-Process -Id $tp -ErrorAction SilentlyContinue) { $alive += $tp }
  }
  if ($alive.Count -eq 0) { break }
  Write-Host ("still alive: " + ($alive -join ','))
  Start-Sleep -Seconds 5
}

$still = @()
foreach ($tp in $trainPids) {
  if (Get-Process -Id $tp -ErrorAction SilentlyContinue) { $still += $tp }
}
if ($still.Count -gt 0) {
  Write-Step 'Force kill remaining train PIDs'
  foreach ($tp in $still) {
    cmd /c ("taskkill /F /T /PID {0}" -f $tp) 2>&1 | ForEach-Object { Write-Host $_ }
  }
  Start-Sleep -Seconds 3
}

Write-Step 'Confirm train dead / port 29601'
$alive2 = @()
foreach ($tp in $trainPids) {
  if (Get-Process -Id $tp -ErrorAction SilentlyContinue) { $alive2 += $tp }
}
Write-Host ("alive_after=" + ($alive2 -join ','))
netstat -ano | findstr 29601 | ForEach-Object { Write-Host $_ }

Write-Step 'Newest ckpt after stop'
$newest = Get-ChildItem (Join-Path $WorkDir '*.pth') -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
if ($newest) {
  Write-Host ("NEWEST_CKPT={0} SIZE={1} MTIME={2}" -f $newest.FullName, $newest.Length, $newest.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
} else {
  Write-Host 'NO_CKPT'
}

$jsonl = Join-Path $WorkDir 'train_log.jsonl'
$lastIter = 'unknown'
$lastLoss = 'unknown'
if (Test-Path $jsonl) {
  $tail = Get-Content $jsonl -Tail 5
  $tail | ForEach-Object { Write-Host ("LOG " + $_) }
  $last = $tail[-1]
  if ($last -match '"iter"\s*:\s*(\d+)') { $lastIter = $Matches[1] }
  if ($last -match '"loss"\s*:\s*([0-9.eE+-]+)') { $lastLoss = $Matches[1] }
}

Write-Step 'Ensure .wslconfig'
$src = Join-Path $Root 'tools\wsl_bootstrap\dot_wslconfig'
$dst = 'C:\Users\Administrator\.wslconfig'
if (Test-Path $src) {
  Copy-Item -Force $src $dst
  Write-Host 'Copied dot_wslconfig -> .wslconfig'
}
Get-Content $dst | ForEach-Object { Write-Host ("WSLCFG " + $_) }

Write-Step 'Update HANDOFF.md'
$waiterStr = ($waiterPids -join ', ')
$block = @"

## 用户指令：立即重启迁移 WSL（$Stamp Asia/Shanghai）

用户（t121u）明确要求：**现在这轮训练完，直接重启**。不再等 iter_5000.pth。

- 已优雅停止 FIRST-ROUND ROW+BLV DDP（原 launcher/workers 约 PID 3208 / 42796 / 58232；port 29601）。
- 停训时最后已知 iter/loss：**$lastIter / $lastLoss**（以 train_log.jsonl 末行为准）。
- 最新 checkpoint：见脚本输出 NEWEST_CKPT（可能仍是 iter_4000.pth — 用户允许）。
- Waiter PIDs（重启会杀掉，WSL 环境就绪后需重拉）：$waiterStr
  - ``tools/portable/run_opensource_followup.py``
  - ``tools/portable/run_error_attrib_after.py``
- ``.wslconfig``：memory=64GB processors=24 swap=32GB（已确认/刷新）。
- WSL MSI 已是 2.7.14.0；重启后虚拟化生效再装 Ubuntu，再跑 ``setup_lusnar_inside_wsl.sh``、挂载 ``D:\LuSNAR_Seg``。
- **迁移门已开（用户批准）**。重启后下一步：Ubuntu + 环境 + remount；优先从最新 ckpt 在 WSL 续训第一轮，否则 Windows 上续完再上 WSL；plus5 在 WSL；waiters 环境好后在 WSL 重拉。
- 不要动 ``work_dirs\train_B1``；不要碰 Moon_2/4/9 训练；batch 保持 >=48；不要改配方。

"@
Add-Content -Path $Handoff -Value $block -Encoding UTF8
Write-Host 'HANDOFF updated'

Write-Step 'Schedule reboot in 30s'
$rebootCmd = 'shutdown /r /t 30 /c "LuSNAR WSL migrate"'
Write-Host ("REBOOT_CMD=" + $rebootCmd)
cmd /c $rebootCmd
Write-Host 'REBOOT_SCHEDULED'
Write-Host ("WAITER_PIDS_FOR_HANDOFF=" + ($waiterPids -join ','))
Write-Host ("LAST_ITER=" + $lastIter)
Write-Host ("LAST_LOSS=" + $lastLoss)
Write-Host 'DONE'
