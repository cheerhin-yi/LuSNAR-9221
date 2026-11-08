$py='F:\Conda\envs\lusnar\python.exe'; $scr='D:\LuSNAR_Seg\tools\portable\lusnar_smoke_compare.py'; $res='D:\LuSNAR_Seg\results\smoke_par'; $wd='D:\LuSNAR_Seg\work_dirs'
$queues=@{ 0=[System.Collections.ArrayList]@('V_B1C','M_ROW','M_SHAD','M_ABL','M_TTA','M_SEGFIX'); 1=[System.Collections.ArrayList]@('M_BLV','V_B1E','M_INVFORM') }
$dep=@('M_TTA','M_SEGFIX'); $slots=1; $running=@{0=@();1=@()}
function L($m){ "{0} {1}" -f (Get-Date -Format o),$m | Add-Content "$res\launcher.log" }
function Ready($e){ if($dep -notcontains $e){return $true}; $s="$wd\smoke_V_B1C\summary.json"; (Test-Path $s) -and (Test-Path "$wd\smoke_V_B1C\best.pth") }
function Start-Exp($e,$g){ $cmd="set CUDA_VISIBLE_DEVICES=$g&& set OMP_NUM_THREADS=8&& set MKL_NUM_THREADS=8&& set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True&& `"$py`" `"$scr`" --exps $e --iters 2000 --batch 16 --lr 6e-5 --skip-prepare --project-root D:\LuSNAR_Seg --data-root D:\LuSNAR_Seg\data --train-subdir train > `"$res\$e.log`" 2> `"$res\$e.err.log`""
 $psi=New-Object System.Diagnostics.ProcessStartInfo; $psi.FileName='cmd.exe'; $psi.Arguments="/c $cmd"; $psi.WorkingDirectory='D:\LuSNAR_Seg'; $psi.UseShellExecute=$false; $psi.CreateNoWindow=$true
 $p=[Diagnostics.Process]::Start($psi); L "start $e gpu$g cmdPID=$($p.Id)"; return $p }
L "=== queue launcher v3 (1 slot/GPU, commit-limit safe) ==="
while($true){
 foreach($g in 0,1){
  $running[$g]=@($running[$g] | ? { -not $_.HasExited })
  while($running[$g].Count -lt $slots -and $queues[$g].Count -gt 0){
   $next=$null; foreach($e in $queues[$g]){ if(Ready $e){$next=$e;break} }
   if(-not $next){break}
   $queues[$g].Remove($next); Remove-Item "$wd\smoke_$next\summary.json","$wd\smoke_$next\train_log.jsonl" -Force -EA SilentlyContinue
   $running[$g]+= ,(Start-Exp $next $g); Start-Sleep 20 }
 }
 if($queues[0].Count -eq 0 -and $queues[1].Count -eq 0 -and $running[0].Count -eq 0 -and $running[1].Count -eq 0){ L 'ALL DONE'; break }
 Start-Sleep 30 }

