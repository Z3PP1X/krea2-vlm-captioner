# ==============================================================================
# Windows PowerShell Installation & Virtual Environment Setup Script
# Krea 2 LoRA End-to-End Data Pipeline
# ==============================================================================
param (
    [switch]$Gpu,
    [switch]$Cpu,
    [switch]$Recreate,
    [string]$VenvPath = ".venv",
    [switch]$NoTest
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "     Krea 2 LoRA Pipeline: Windows Installation & Setup               " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Python check
Write-Host "[1/6] Checking Python version..." -ForegroundColor Yellow
$pyVer = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Found Python version: $pyVer" -ForegroundColor Green

# 2. Virtual environment
Write-Host "[2/6] Setting up virtual environment at '$VenvPath'..." -ForegroundColor Yellow
if ($Recreate -and (Test-Path $VenvPath)) {
    Write-Host "Removing existing virtual environment..." -ForegroundColor DarkYellow
    Remove-Item -Recurse -Force $VenvPath
}

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Green
    python -m venv $VenvPath
} else {
    Write-Host "Reusing existing virtual environment." -ForegroundColor Green
}

# Activate
$activateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
}

Write-Host "[3/6] Upgrading pip, setuptools, wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel --quiet

# 3. Dependencies
Write-Host "[4/6] Installing dependencies..." -ForegroundColor Yellow
python -m pip install "numpy<2" --quiet

if ($Gpu) {
    Write-Host "Installing PyTorch (CUDA) & GPU requirements..." -ForegroundColor Green
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet
    python -m pip install -r requirements-gpu.txt --quiet
} else {
    Write-Host "Installing standard pipeline requirements..." -ForegroundColor Green
    python -m pip install -r requirements.txt --quiet
}

# 4. Pipeline editable install
Write-Host "[5/6] Installing pipeline in editable mode ('pip install -e .')..." -ForegroundColor Yellow
python -m pip install -e . --quiet

# 5. Directories
New-Item -ItemType Directory -Force -Path "data\raw", "data\qc", "data\processed", "data\export", "logs", "config" | Out-Null

# 6. Tests
if (-not $NoTest) {
    Write-Host "[6/6] Running pytest verification..." -ForegroundColor Yellow
    python -m pytest tests/ -q
} else {
    Write-Host "[6/6] Skipping pytest (--NoTest specified)." -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "             Installation Completed Successfully!                     " -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "To activate the environment in PowerShell:"
Write-Host "  .$VenvPath\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Commands:"
Write-Host "  pipeline status"
Write-Host "  pipeline crawl --url `"https://scrolller.com/...`""
Write-Host "======================================================================" -ForegroundColor Cyan
