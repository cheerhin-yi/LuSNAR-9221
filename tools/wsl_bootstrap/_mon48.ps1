$g=@(); 1..12 | % { $g += (nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) -join '/'; Start-Sleep 5 }
"GPU util samples (gpu0/gpu1, 5s): " + ($g -join '  ')
$os = Get-CimInstance Win32_OperatingSystem
$pf = Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit','\Processor(_Total)\% Processor Time' -SampleInterval 2 -MaxSamples 1
$pf.CounterSamples | % { "{0} = {1:N1}" -f $_.Path.Split('\')[-1], ($(if($_.Path -match 'bytes|limit'){$_.CookedValue/1GB}else{$_.CookedValue})) }
"FreePhysGB = {0:N1}" -f ($os.FreePhysicalMemory/1MB)
$ev = Get-WinEvent -FilterHashtable @{LogName='System';Id=2004;StartTime=(Get-Date).AddHours(-1)} -ErrorAction SilentlyContinue
"event2004 last 1h: " + @($ev).Count
nvidia-smi --query-gpu=index,memory.used,power.draw --format=csv,noheader