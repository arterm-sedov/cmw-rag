# Test setup script for RAG engine (Windows PowerShell)
# Run from project root: .\rag_engine\tests\scripts\setup_and_test.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== RAG Engine Test Setup Script ===" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "1. Checking Python version..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: python not found" -ForegroundColor Red
    exit 1
}

# Create venv if it doesn't exist
if (!(Test-Path ".venv")) {
    Write-Host "2. Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "2. Virtual environment exists, reusing it" -ForegroundColor Yellow
}

# Activate venv
Write-Host "3. Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "4. Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# Install dependencies
Write-Host "5. Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r rag_engine\requirements.txt --quiet

Write-Host ""
Write-Host "=== Dependencies installed! ===" -ForegroundColor Green
Write-Host ""

# Run linter
Write-Host "6. Running linter (Ruff)..." -ForegroundColor Yellow
ruff check rag_engine\ --fix

Write-Host ""

# Run tests
Write-Host "7. Running tests..." -ForegroundColor Yellow
python -m pytest rag_engine\tests\ -v --tb=short

Write-Host ""
Write-Host "=== Test setup and execution complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment manually:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "To run tests only:"
Write-Host "  python -m pytest rag_engine\tests\ -v"

