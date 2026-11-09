$ErrorActionPreference = 'Stop'
$py = 'F:\Conda\envs\lusnar\python.exe'
$root = 'D:\LuSNAR_Seg'
$logdir = Join-Path $root 'work_dirs\smoke_logs'
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
Set-Location $root

Write-Host "=== prepare assets ==="
& $py tools\prepare_smoke_assets.py 2>&1 | Tee-Object -FilePath (Join-Path $logdir 'prepare.log')
if ($LASTEXITCODE -ne 0) { throw "prepare failed $LASTEXITCODE" }

foreach ($exp in @('A0','A1','B1','C1')) {
  Write-Host "=== EXP $exp ==="
  $out = Join-Path $logdir "$exp.log"
  & $py tools\train_smoke_matrix.py --exp $exp --iters 1000 --batch 8 2>&1 | Tee-Object -FilePath $out
  if ($LASTEXITCODE -ne 0) {
    Write-Host "EXP $exp failed rc=$LASTEXITCODE (continuing)"
  }
}
Write-Host "=== ALL DONE ==="
