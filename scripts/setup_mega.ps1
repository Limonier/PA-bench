$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
Push-Location $taskRoot
try {
    $env:UV_CACHE_DIR = Join-Path $taskRoot '.cache/uv'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $taskRoot '.python'
    uv python install 3.11.7
    if ($LASTEXITCODE) { throw 'Python installation failed' }
    if (-not (Test-Path .mega-venv/Scripts/python.exe)) {
        uv venv .mega-venv --python 3.11.7
        if ($LASTEXITCODE) { throw 'Environment creation failed' }
    }
    if (-not (Test-Path .mega-tools-venv/Scripts/python.exe)) {
        uv venv .mega-tools-venv --python 3.11.7
        if ($LASTEXITCODE) { throw 'Build tool environment creation failed' }
    }
    $env:VIRTUAL_ENV = Join-Path $taskRoot '.mega-venv'
    $env:POETRY_CACHE_DIR = Join-Path $taskRoot '.cache/poetry'
    uv pip install --python .mega-tools-venv/Scripts/python.exe poetry==2.5.1
    if ($LASTEXITCODE) { throw 'Poetry installation failed' }
    & .mega-tools-venv/Scripts/poetry.exe -C upstream/Twin-2K-500-Mega-Study-main install --only main --no-root --no-interaction
    if ($LASTEXITCODE) { throw 'Official locked dependency installation failed' }
    git apply --reverse --check --directory=upstream/Twin-2K-500-Mega-Study-main scripts/mega_upstream_fixes.patch 2>$null
    if ($LASTEXITCODE -ne 0) {
        git apply --check --directory=upstream/Twin-2K-500-Mega-Study-main scripts/mega_upstream_fixes.patch
        if ($LASTEXITCODE) { throw 'Upstream source differs from the compatibility patch' }
        git apply --directory=upstream/Twin-2K-500-Mega-Study-main scripts/mega_upstream_fixes.patch
        if ($LASTEXITCODE) { throw 'Compatibility patch failed' }
    }
} finally { Pop-Location }
