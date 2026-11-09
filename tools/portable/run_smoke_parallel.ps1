param()
$ErrorActionPreference = 'Continue'
$py   = 'F:\Conda\envs\lusnar\python.exe'
$scr  = 'D:\LuSNAR_Seg\tools\portable\lusnar_smoke_compare.py'
$res  = 'D:\LuSNAR_Seg\results\smoke_par'
New-Item -ItemType Directory -Force -Path $res | Out-Null
function Start-Exp([string]$exp, [int]$gpu) {
  $log = Join-Path $res "$exp.log"; $err = Join-Path $res "$exp.err.log"
  $cmd = "set CUDA_VISIBLE_DEVICES=$gpu&& set OMP_NUM_THREADS=6&& set MKL_NUM_THREADS=6&& `"$py`" `"$scr`" --exps $exp --iters 2000 --batch 16 --lr 6e-5 --skip-prepare --project-root D:\LuSNAR_Seg --data-root D:\LuSNAR_Seg\data --train-subdir train > `"$log`" 2> `"$err`""
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = 'cmd.exe'; $psi.Arguments = "/c $cmd"; $psi.WorkingDirectory = 'D:\LuSNAR_Seg'
  $psi.UseShellExecute = $false; $psi.CreateNoWindow = $true
  $p = [System.Diagnostics.Process]::Start($psi)
  "{0} start {1} gpu{2} cmdPID={3}" -f (Get-Date -Format o), $exp, $gpu, $p.Id | Add-Content (Join-Path $res 'launcher.log')
  return $p
}
# Wave 1: six independent trainings, 3 per GPU
$w1 = @(
  (Start-Exp 'V_B1C' 0), (Start-Exp 'M_ROW' 0), (Start-Exp 'M_BLV' 0),
  (Start-Exp 'V_B1E' 1), (Start-Exp 'M_SHAD' 1), (Start-Exp 'M_ABL' 1)
)
# Wave 2: wait for V_B1C, then dependent eval-style runs
$vb = $w1[0]; $vb.WaitForExit()
"{0} V_B1C exit={1}" -f (Get-Date -Format o), $vb.ExitCode | Add-Content (Join-Path $res 'launcher.log')
$a = Start-Exp 'M_TTA' 0
$b = Start-Exp 'T_B1_TTA' 1
$a.WaitForExit()
$c = Start-Exp 'M_SEGFIX' 0
$b.WaitForExit(); $c.WaitForExit()
foreach ($p in $w1) { $p.WaitForExit() }
"{0} ALL DONE" -f (Get-Date -Format o) | Add-Content (Join-Path $res 'launcher.log')
