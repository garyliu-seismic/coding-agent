param()
Set-StrictMode -Version Latest

$repo = Split-Path -Parent $PSScriptRoot
$out = Join-Path $env:TEMP 'bench\img_run.txt'
$img = Join-Path $env:TEMP 'claude\c--project-new-app-livedoc-service\44e3b134-dc86-4f06-9e3d-66ec4dfcf15a\images\21.png'
if(-not (Test-Path (Split-Path $out -Parent))){ New-Item -ItemType Directory -Path (Split-Path $out -Parent) -Force | Out-Null }
Write-Host "Running image test. Image exists: " (Test-Path $img)
Set-Location $repo
& .\scripts\run_image_test.ps1 -Image $img -Question 'What error message is shown in red? Quote it exactly.' *> $out
Get-Content $out -Encoding UTF8 | Where-Object { $_ -match '⚙|Finished|429|Error|finish' } | ForEach-Object { if ($_.Length -gt 200) { $_.Substring(0,200) } else { $_ } } | Select-Object -First 25
