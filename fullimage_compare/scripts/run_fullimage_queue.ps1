<#
 LuSNAR whole-image (1024) smoke queue -- fullimage_compare line ONLY.
 Manual start only. Nothing here schedules, waits for, or auto-launches anything else.

 Order: prepare lists -> VRAM probe (control + heavy modules) -> FI_V_B1C (control)
        -> every module one at a time (skip exps whose summary.json says done)
        -> optional eval-only modes (-IncludeEval) on FI_V_B1C\best.pth.
 Stop:  create D:\LuSNAR_Seg\fullimage_compare\STOP  (checked every iteration inside python,
        every 10 val images, and between experiments). The running exp saves last.pth and
        exits with code 3; the queue then ends. Delete STOP and re-run this script to resume.
 Single GPU (CUDA_VISIBLE_DEVICES=0), no DDP, no ports. PYTORCH_CUDA_ALLOC_CONF is removed,
 never set. ASCII-only file (Windows PowerShell 5.1 safe).
#>
param(
  [string]$Root = 'D:\LuSNAR_Seg',
  [string]$Python = 'F:\Conda\envs\lusnar\python.exe',
  [string[]]$Exps = @('FI_V_B1C','FI_M_ROW_BLV','FI_M_ROW','FI_M_BLV','FI_M_SHAD','FI_M_ABL',
                      'FI_M_INVFORM','FI_V_B1E','FI_M_HARD','FI_M_DYSAMPLE','FI_M_CTRAST',
                      'FI_M_SAPA','FI_M_FADE'),
  [string[]]$ProbeExps = @('FI_V_B1C','FI_M_ROW_BLV','FI_M_DYSAMPLE','FI_M_CTRAST','FI_M_SAPA'),
  [int]$Micro = 0,        # 0 = take recommended_common_micro_batch from the VRAM probe
  [int]$EffBatch = 0,     # 0 = no accumulation (effective = micro); else grad-accum = EffBatch/Micro, lr unchanged
  [int]$Workers = 2,
  [int]$Iters = 2000,
  [double]$MemCapGB = 14.0,
  [double]$HeadroomGB = 1.5,
  [switch]$SkipProbe,
  [switch]$ProbeOnly,
  [switch]$IncludeEval,
  [int]$BusyMemMiB = 3000,   # desktop alone uses ~1.5-2 GB; more than this = someone else is on the GPU
  [int]$BusyUtilPct = 20,
  [switch]$IgnoreBusyGpu
)
$ErrorActionPreference = 'Stop'
if ($Root -notmatch '^[A-Za-z]:\\') { throw "Root must be a native Windows drive path, got $Root" }
$FI = Join-Path $Root 'fullimage_compare'
$Script = Join-Path $FI 'scripts\fullimage_smoke.py'
$Logs = Join-Path $FI 'logs'
$StopFile = Join-Path $FI 'STOP'
$HostTag = $env:COMPUTERNAME
New-Item -ItemType Directory -Force -Path $Logs, (Join-Path $FI 'results'), (Join-Path $FI 'work_dirs') | Out-Null
$QLog = Join-Path $Logs "queue_$HostTag.log"
function L([string]$m) { $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m; Write-Host $line; Add-Content -Path $QLog -Value $line -Encoding UTF8 }

foreach ($p in @($Python, $Script, (Join-Path $Root 'tools\portable\_lusnar_common.py'), (Join-Path $Root 'tools\portable_b3\_lusnar_common.py'), (Join-Path $Root 'pretrained\mit_b2_20220624-66e8bf70.pth'))) {
  if (-not (Test-Path $p)) { throw "missing required file: $p" }
}
if (Test-Path $StopFile) { L "STOP file present ($StopFile); not starting. Delete it to run."; exit 3 }

# single instance: lock file + no other fullimage_smoke.py python running
$Lock = Join-Path $Logs 'queue.lock'
if (Test-Path $Lock) {
  $old = (Get-Content $Lock -Raw).Trim()
  if ($old -and (Get-Process -Id ([int]$old) -ErrorAction SilentlyContinue)) { L "another queue (PID $old) is running; exit"; exit 4 }
}
$others = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*fullimage_smoke.py*' })
if ($others.Count -gt 0) { L "fullimage_smoke.py already running (PID $($others.ProcessId -join ',')); exit"; exit 4 }

# GPU must be free (P0 has priority on cheerhin)
# On Windows (WDDM) query-compute-apps also lists desktop apps (dwm, explorer, browsers) with [N/A]
# memory, and WSL jobs are not listed at all -> judge by python processes AND by used VRAM / util.
if (-not $IgnoreBusyGpu) {
  $py = @(& nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>$null | Where-Object { $_ -match 'python' })
  $q = (& nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
  $memUsed = 0; $util = 0
  if ($q) { $f = $q.Split(','); $memUsed = [int]$f[1].Trim(); $util = [int]$f[2].Trim() }
  if ($py.Count -gt 0 -or $memUsed -gt $BusyMemMiB -or $util -gt $BusyUtilPct) {
    L "GPU busy, not starting (P0 has priority): python=[$($py -join ' | ')] mem_used=${memUsed}MiB util=${util}% (limits ${BusyMemMiB}MiB / ${BusyUtilPct}%)"; exit 5
  }
  L "GPU idle check ok: mem_used=${memUsed}MiB util=${util}%"
}
Set-Content -Path $Lock -Value $PID -Encoding ASCII

$env:CUDA_VISIBLE_DEVICES = '0'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue

function Run-FI([string]$name, [string[]]$pyArgs) {
  $out = Join-Path $Logs "$name.log"; $err = Join-Path $Logs "$name.err.log"
  $all = @("`"$Script`"") + $pyArgs
  L "start $name : $Python $($all -join ' ')"
  $p = Start-Process -FilePath $Python -ArgumentList $all -WorkingDirectory $Root -NoNewWindow -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
  $null = $p.Handle   # cache the handle so ExitCode is populated (PS 5.1 + redirect quirk)
  $p.WaitForExit()
  $code = $p.ExitCode
  L "end   $name exit=$code"
  return $code
}
function Read-Json([string]$path) { Get-Content -Path $path -Raw -Encoding UTF8 | ConvertFrom-Json }

try {
  L "=== fullimage queue start host=$HostTag workers=$Workers iters=$Iters ==="
  $rc = Run-FI 'prepare_lists' @('--prepare-lists')
  if ($rc -ne 0) { throw "prepare-lists failed (exit $rc); see $Logs\prepare_lists.err.log" }

  $ProbeJson = Join-Path $FI "results\vram_probe_$HostTag.json"
  if (-not $SkipProbe) {
    $need = $true
    if (Test-Path $ProbeJson) {
      $pj = Read-Json $ProbeJson
      $missing = @($ProbeExps | Where-Object { -not ($pj.exps.PSObject.Properties.Name -contains $_) })
      $need = ($missing.Count -gt 0) -or (-not $pj.recommended_common_micro_batch)
    }
    if ($need) {
      $rc = Run-FI 'vram_probe' @('--vram-probe', '--exps', ($ProbeExps -join ','), '--workers', "$Workers",
                                 '--mem-cap-gb', "$MemCapGB", '--headroom-gb', "$HeadroomGB", '--probe-iters', '30')
      if ($rc -eq 3) { L 'STOP during probe; queue ends'; exit 3 }
    } else { L "probe already done: $ProbeJson" }
  }
  if ($ProbeOnly) { L 'ProbeOnly: done'; exit 0 }
  if ($Micro -le 0) {
    if (-not (Test-Path $ProbeJson)) { throw "no probe result $ProbeJson and -Micro not given" }
    $Micro = [int]((Read-Json $ProbeJson).recommended_common_micro_batch)
    if ($Micro -le 0) { throw "probe found no micro batch within budget; inspect $ProbeJson" }
  }
  $Accum = 1
  if ($EffBatch -gt 0) {
    if ($EffBatch % $Micro -ne 0) { throw "EffBatch $EffBatch not divisible by micro $Micro" }
    $Accum = [int]($EffBatch / $Micro)
  }
  L "using micro=$Micro grad_accum=$Accum effective=$($Micro*$Accum) lr=6e-5 (never rescaled) for ALL exps"

  foreach ($e in $Exps) {
    if (Test-Path $StopFile) { L 'STOP file seen between experiments; queue ends'; exit 3 }
    $s = Join-Path $FI "work_dirs\$e\summary.json"
    if ((Test-Path $s) -and ((Read-Json $s).status -eq 'done')) { L "skip $e (summary.json done)"; continue }
    $rc = Run-FI $e @('--exps', $e, '--batch', "$Micro", '--grad-accum', "$Accum", '--iters', "$Iters",
                      '--workers', "$Workers", '--lr', '6e-5', '--seed', '42', '--val-every', '500', '--ckpt-every', '500')
    if ($rc -eq 3) { L "STOP during $e (last.pth saved); queue ends"; exit 3 }
    if ($rc -ne 0) { L "WARN $e exit=$rc; see logs\$e.log / work_dirs\$e\summary_progress.json; continuing" }
  }

  if ($IncludeEval) {
    if (Test-Path (Join-Path $FI 'work_dirs\FI_V_B1C\best.pth')) {
      foreach ($e in @('FI_M_TTA', 'FI_M_SEGFIX', 'FI_M_MSINFER')) {
        if (Test-Path $StopFile) { L 'STOP; queue ends'; exit 3 }
        $rc = Run-FI $e @('--exps', $e, '--workers', "$Workers")
        if ($rc -eq 3) { exit 3 }
      }
    } else { L 'skip eval-only modes: FI_V_B1C\best.pth missing (eval never auto-trains)' }
  }
  L '=== fullimage queue finished ==='
}
finally {
  Remove-Item $Lock -ErrorAction SilentlyContinue
}
