# Run in elevated PowerShell AFTER reboot (features already Enabled).
# Does not touch Windows train processes beyond what reboot already stopped.
$ErrorActionPreference='Stop'
Write-Host '=== WSL post-reboot bootstrap ==='
wsl --set-default-version 2
# Install Ubuntu if missing
$distros = (wsl -l -q 2>$null)
if (-not ($distros -match 'Ubuntu')) {
  Write-Host 'Installing Ubuntu (default)...'
  wsl --install -d Ubuntu --no-launch
}
# Import may need first-run user; create via root if needed later
wsl -l -v
Write-Host 'Next: run setup_lusnar_inside_wsl.sh via: wsl -d Ubuntu -u root -- bash /mnt/d/LuSNAR_Seg/tools/wsl_bootstrap/setup_lusnar_inside_wsl.sh'
