$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
Push-Location $taskRoot
try {
    $env:UV_CACHE_DIR = Join-Path $taskRoot '.cache/uv'
    if (-not (Test-Path .fingertip-venv/Scripts/python.exe)) {
        uv venv .fingertip-venv --python 3.13
        if ($LASTEXITCODE) { throw 'FingerTip environment creation failed' }
    }
    uv pip install --python .fingertip-venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu
    if ($LASTEXITCODE) { throw 'CPU PyTorch installation failed' }
    $fingerRequirements = 'scripts/requirements-fingertip.txt'
    if (Test-Path scripts/requirements-fingertip.lock.txt) { $fingerRequirements = 'scripts/requirements-fingertip.lock.txt' }
    uv pip install --python .fingertip-venv/Scripts/python.exe -r $fingerRequirements
    if ($LASTEXITCODE) { throw 'FingerTip dependency installation failed' }
    uv pip check --python .fingertip-venv/Scripts/python.exe
    if ($LASTEXITCODE) { throw 'FingerTip dependencies are incompatible' }
} finally { Pop-Location }
