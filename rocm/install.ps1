# SPDX-FileCopyrightText: Copyright (c) 2026 HrMarcussen
# SPDX-License-Identifier: MIT
<#
.SYNOPSIS
  Installs PersonaPlex for AMD Radeon on native Windows (ROCm 7.2.1 PyTorch wheels).

.DESCRIPTION
  Creates a Python 3.12 virtual environment (default: .venv next to this repo),
  installs AMD's ROCm SDK + PyTorch wheels for Windows and the patched moshi package.
  Requirements: Windows 11, Radeon RX 7000/9000 (RDNA3/RDNA4), AMD Adrenalin driver
  26.2.2 or newer, Python 3.12 (python.org installer, with the "py" launcher).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File rocm\install.ps1
#>
param(
    [string]$Venv = (Join-Path (Split-Path $PSScriptRoot -Parent) ".venv"),
    [string]$RocmRelease = "7.2.1"
)
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent

Write-Host "== PersonaPlex ROCm installer ==" -ForegroundColor Cyan
$gpu = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "Radeon" } | Select-Object -First 1)
if ($gpu) { Write-Host "GPU: $($gpu.Name)  driver $($gpu.DriverVersion)" } else { Write-Warning "No AMD Radeon GPU detected." }

if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    Write-Host "Creating Python 3.12 venv at $Venv"
    & py -3.12 -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.12 not found. Install it from https://www.python.org/downloads/ (3.12.x)." }
}
$py = Join-Path $Venv "Scripts\python.exe"
& $py -m pip install --upgrade pip

$base = "https://repo.radeon.com/rocm/windows/rocm-rel-$RocmRelease"
Write-Host "Installing ROCm SDK $RocmRelease wheels (large download)..."
& $py -m pip install --no-cache-dir `
    "$base/rocm_sdk_core-$RocmRelease-py3-none-win_amd64.whl" `
    "$base/rocm_sdk_devel-$RocmRelease-py3-none-win_amd64.whl" `
    "$base/rocm_sdk_libraries_custom-$RocmRelease-py3-none-win_amd64.whl" `
    "$base/rocm-$RocmRelease.tar.gz"
if ($LASTEXITCODE -ne 0) { throw "ROCm SDK install failed" }

Write-Host "Installing PyTorch for ROCm..."
& $py -m pip install --no-cache-dir `
    "$base/torch-2.9.1%2Brocm$RocmRelease-cp312-cp312-win_amd64.whl" `
    "$base/torchaudio-2.9.1%2Brocm$RocmRelease-cp312-cp312-win_amd64.whl"
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }

Write-Host "Installing PersonaPlex (patched moshi package)..."
& $py -m pip install -e (Join-Path $repo "moshi") accelerate
if ($LASTEXITCODE -ne 0) { throw "moshi install failed" }

Write-Host "Checking GPU + building HIP kernels (first run compiles them, ~10 s)..."
& $py -c "import moshi, torch; from moshi import kernels; print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0), '| custom kernels:', kernels.available())"
if ($LASTEXITCODE -ne 0) { throw "GPU check failed" }

Write-Host ""
Write-Host "Done. Next:" -ForegroundColor Green
Write-Host "  1) Accept the model license: https://huggingface.co/nvidia/personaplex-7b-v1"
Write-Host "  2) Log in once:  & '$Venv\Scripts\huggingface-cli.exe' login"
Write-Host "  3) Start:        powershell -ExecutionPolicy Bypass -File rocm\run-server.ps1"
