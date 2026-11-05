<#
.SYNOPSIS
  Starts the PersonaPlex realtime voice server on an AMD GPU (Windows).

.DESCRIPTION
  Serves the web UI at http://localhost:8998 (open it in Chrome or Edge, allow the
  microphone, use headphones to avoid the model hearing itself).
  The first start downloads ~17 GB of weights and tunes GEMM kernels for your GPU
  (a few minutes); later starts take ~30 s.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File rocm\run-server.ps1
#>
param(
    [string]$Venv = (Join-Path (Split-Path $PSScriptRoot -Parent) ".venv"),
    [int]$Port = 8998,
    [string]$HostName = "localhost",
    [string]$HfRepo = "",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra
)
$ErrorActionPreference = "Stop"
$py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No venv at $Venv. Run rocm\install.ps1 first." }

$serverArgs = @("-m", "moshi.server", "--host", $HostName, "--port", "$Port")
if ($HfRepo) { $serverArgs += @("--hf-repo", $HfRepo) }
if ($Extra) { $serverArgs += $Extra }

Write-Host "Starting PersonaPlex on http://localhost:$Port  (Ctrl+C to stop)" -ForegroundColor Cyan
& $py @serverArgs
