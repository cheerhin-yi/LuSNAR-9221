$ErrorActionPreference = 'Continue'
Write-Host '=== Install Ubuntu non-interactive ==='
wsl --set-default-version 2
$distros = @(wsl -l -q 2>$null)
$hasUbuntu = $false
foreach ($d in $distros) {
  if ($d -match 'Ubuntu') { $hasUbuntu = $true }
}
if (-not $hasUbuntu) {
  Write-Host 'Installing Ubuntu with --no-launch ...'
  # Prefer Microsoft Store / wsl --install
  wsl --install -d Ubuntu --no-launch 2>&1 | Out-String | Write-Host
  Start-Sleep -Seconds 3
  $distros2 = @(wsl -l -q 2>$null)
  foreach ($d in $distros2) { Write-Host ("DISTRO:[$d]") }
  $hasUbuntu = $false
  foreach ($d in $distros2) { if ($d -match 'Ubuntu') { $hasUbuntu = $true } }
}
if (-not $hasUbuntu) {
  Write-Host 'Fallback: wsl --install Ubuntu'
  wsl --install Ubuntu --no-launch 2>&1 | Out-String | Write-Host
}
Write-Host '=== wsl -l -v ==='
wsl -l -v 2>&1 | Out-String | Write-Host
