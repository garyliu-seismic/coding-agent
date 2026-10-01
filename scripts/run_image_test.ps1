param(
    [string]$Image = "",
    [string]$Question = "Describe what is shown in this image. Quote any visible text exactly.",
    [int]$TimeoutSec = 180,
    [string]$Vision = "on"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Python venv not found: $python" }
if (-not $Image) { throw "Usage: .\scripts\run_image_test.ps1 -Image C:\path\to\picture.png [-Question '...'] [-TimeoutSec 180]" }
if (-not (Test-Path $Image)) { throw "Image not found: $Image" }

# Isolated sandbox root so the agent sees only this one image.
$work = Join-Path $env:TEMP ("agent-img-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $work | Out-Null
$name = Split-Path -Leaf $Image
Copy-Item $Image (Join-Path $work $name)

$env:PYTHONIOENCODING = "utf-8"
$env:CODING_AGENT_VISION = $Vision
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$task = "There is an image file named $name in the project root. Use the view_image tool to look at it, then call finish with your answer. Question: $Question"

Write-Host "Image=$name  Sandbox=$work  Vision=$Vision  Timeout=${TimeoutSec}s  Started=$(Get-Date -Format HH:mm:ss)" -ForegroundColor Cyan
Set-Location $repo
$sw = [Diagnostics.Stopwatch]::StartNew()
$argLine = "-m coding_agent --root `"$work`" --read-only run `"$task`""
$proc = Start-Process -FilePath $python -ArgumentList $argLine -NoNewWindow -PassThru

if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
    Write-Host "`nTIMEOUT after ${TimeoutSec}s - killing agent (did not converge or a model call stalled)." -ForegroundColor Red
    Stop-Process -Id $proc.Id -Force
    exit 2
}

$null = $proc.Handle
$proc.WaitForExit()
Write-Host "`nFinished exit=$($proc.ExitCode) in $([int]$sw.Elapsed.TotalSeconds)s" -ForegroundColor Green
Write-Host "Check above: a view_image tool call should appear, followed by finish with an answer about the image." -ForegroundColor Yellow
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
exit $proc.ExitCode
