param(
    [string]$Root = "C:\out",
    [string]$LogFile = "2026-09-29.1158-e6e709c-rel.txt",
    [int]$TimeoutSec = 300
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Python venv not found: $python" }
if (-not (Test-Path (Join-Path $Root $LogFile))) { throw "Log file not found: $(Join-Path $Root $LogFile)" }

$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$task = "Analyze the log file $LogFile. It is the log of a failed QA deployment. Find which Gatekeeper policy (constraint/template name and the exact rule violated) blocked the deployment. Report the rejected resource (kind/name), the original error text, the root cause and a fix suggestion. Every conclusion must quote the original log text and line number."

Write-Host "Root=$Root  Timeout=${TimeoutSec}s  Started=$(Get-Date -Format HH:mm:ss)" -ForegroundColor Cyan

Set-Location $repo
$argLine = "-m coding_agent --root `"$Root`" --read-only run `"$task`""
$proc = Start-Process -FilePath $python -ArgumentList $argLine -NoNewWindow -PassThru

if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
    Write-Host "`nTIMEOUT after ${TimeoutSec}s - killing agent (agent did not converge or a model call stalled)." -ForegroundColor Red
    Stop-Process -Id $proc.Id -Force
    exit 2
}

Write-Host "`nFinished exit=$($proc.ExitCode) at $(Get-Date -Format HH:mm:ss)" -ForegroundColor Green
exit $proc.ExitCode
