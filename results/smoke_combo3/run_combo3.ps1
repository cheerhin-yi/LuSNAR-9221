# Three-train-module smoke: M_ROW_BLV_SHAD then M_ROW_BLV_SHAD_TTA. GPU0 only. No batch-3.
$ErrorActionPreference = 'Continue'
$py  = 'F:\Conda\envs\lusnar\python.exe'
$scr = 'D:\LuSNAR_Seg\tools\portable\lusnar_smoke_compare.py'
$res = 'D:\LuSNAR_Seg\results\smoke_combo3'
$wd  = 'D:\LuSNAR_Seg\work_dirs'
$proj = 'D:\LuSNAR_Seg'
New-Item -ItemType Directory -Force -Path $res | Out-Null

$q0 = [System.Collections.ArrayList]@(
  @{ Name='M_ROW_BLV_SHAD';     Exp='M_ROW_BLV_SHAD'; Ckpt=$null; RunName=$null; Dep=$null },
  @{ Name='M_ROW_BLV_SHAD_TTA'; Exp='M_TTA';          Ckpt="$wd\smoke_M_ROW_BLV_SHAD\best.pth"; RunName='M_ROW_BLV_SHAD_TTA'; Dep='M_ROW_BLV_SHAD' }
)
$queues = @{ 0 = $q0 }
$slots = 1
$running = @{ 0 = @() }

function L($m) {
  $line = "{0} {1}" -f (Get-Date -Format o), $m
  Add-Content -Path "$res\launcher.log" -Value $line -Encoding UTF8
  Write-Host $line
}
function Ready($job) {
  if ($null -eq $job.Dep -or $job.Dep -eq '') { return $true }
  $sum = "$wd\smoke_$($job.Dep)\summary.json"
  if (-not (Test-Path $sum)) { return $false }
  $dep = [string]$job.Dep
  if ($dep -match '_TTA$' -or $dep -eq 'M_SEGFIX') { return $true }
  return (Test-Path "$wd\smoke_$dep\best.pth")
}
function CkptOk($job) {
  if ($null -ne $job.Dep -and $job.Dep -ne '') { return $true }
  if ($null -eq $job.Ckpt -or $job.Ckpt -eq '') { return $true }
  return (Test-Path $job.Ckpt)
}
function Start-JobSpec($job, $g) {
  $outName = if ($job.RunName) { $job.RunName } else { $job.Exp }
  $log = "$res\$outName.log"; $err = "$res\$outName.err.log"
  $args = "--exps $($job.Exp) --iters 2000 --batch 16 --lr 6e-5 --skip-prepare --project-root $proj --data-root $proj\data --train-subdir train"
  if ($job.RunName) { $args += " --run-name $($job.RunName)" }
  if ($job.Ckpt) {
    $rel = $job.Ckpt
    if ($rel.StartsWith("$wd\", [StringComparison]::OrdinalIgnoreCase)) {
      $rel = "work_dirs\" + $rel.Substring($wd.Length + 1)
    }
    $args += " --ckpt `"$rel`""
  }
  $cmd = "set CUDA_VISIBLE_DEVICES=$g&& set OMP_NUM_THREADS=8&& set MKL_NUM_THREADS=8&& set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True&& `"$py`" `"$scr`" $args > `"$log`" 2> `"$err`""
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = 'cmd.exe'; $psi.Arguments = "/c $cmd"; $psi.WorkingDirectory = $proj
  $psi.UseShellExecute = $false; $psi.CreateNoWindow = $true
  $p = [Diagnostics.Process]::Start($psi)
  L "start $($job.Name) gpu$g exp=$($job.Exp) runName=$($job.RunName) ckpt=$($job.Ckpt) cmdPID=$($p.Id)"
  return $p
}

L "=== combo3 launcher: M_ROW_BLV_SHAD (train) -> M_ROW_BLV_SHAD_TTA on GPU0 ==="

foreach ($g in @(0)) {
  $keep = [System.Collections.ArrayList]@()
  foreach ($job in @($queues[$g])) {
    if (-not (CkptOk $job)) { L "SKIP $($job.Name) — missing source ckpt $($job.Ckpt)"; continue }
    [void]$keep.Add($job)
  }
  $queues[$g] = $keep
}

while ($true) {
  foreach ($g in @(0)) {
    $running[$g] = @($running[$g] | Where-Object { -not $_.HasExited })
    while ($running[$g].Count -lt $slots -and $queues[$g].Count -gt 0) {
      $next = $queues[$g][0]
      if (-not (Ready $next)) { break }
      $queues[$g].RemoveAt(0)
      $outName = if ($next.RunName) { $next.RunName } else { $next.Exp }
      Remove-Item "$wd\smoke_$outName\summary.json","$wd\smoke_$outName\train_log.jsonl" -Force -ErrorAction SilentlyContinue
      $running[$g] += ,(Start-JobSpec $next $g)
      Start-Sleep 15
    }
  }
  if ($queues[0].Count -eq 0 -and $running[0].Count -eq 0) {
    L 'ALL DONE'
    break
  }
  Start-Sleep 30
}
