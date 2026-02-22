# Run BELT Verification Script with Specific Virtual Environment
# This script activates the virtual environment and runs comprehensive verification

$ErrorActionPreference = "Stop"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "BELT IMPLEMENTATION VERIFICATION" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Virtual environment path
$VenvPath = "C:\Users\n1sha\Desktop\Mtech Research\EEGtoWord\.venv"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

# Check if virtual environment exists
if (-not (Test-Path $PythonExe)) {
    Write-Host "ERROR: Python executable not found at: $PythonExe" -ForegroundColor Red
    Write-Host "Please ensure the virtual environment exists at the specified location." -ForegroundColor Yellow
    exit 1
}

Write-Host "Using virtual environment: $VenvPath" -ForegroundColor Green

# Check Python is available
Write-Host "`nChecking Python..." -ForegroundColor Yellow
$PythonVersion = & $PythonExe --version 2>&1
Write-Host "Python version: $PythonVersion" -ForegroundColor Cyan

# Check required packages
Write-Host "`nChecking required packages..." -ForegroundColor Yellow

$RequiredPackages = @(
    "torch",
    "numpy",
    "pyyaml",
    "transformers"
)

foreach ($Package in $RequiredPackages) {
    $CheckPackage = & $PythonExe -c "import $Package; print($Package.__version__)" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ $Package : $CheckPackage" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $Package : NOT FOUND" -ForegroundColor Red
        Write-Host "`nInstalling missing package: $Package" -ForegroundColor Yellow
        & $PythonExe -m pip install $Package
    }
}

# Change to model directory
$ModelDir = "C:\Users\n1sha\Desktop\model_custom"
Set-Location $ModelDir

Write-Host "`nWorking directory: $ModelDir" -ForegroundColor Cyan

# Run verification script
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "RUNNING BELT VERIFICATION..." -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

& $PythonExe verify_belt_implementation.py

$ExitCode = $LASTEXITCODE

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "VERIFICATION COMPLETE" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

if ($ExitCode -eq 0) {
    Write-Host "✓ Verification completed successfully!" -ForegroundColor Green
} else {
    Write-Host "✗ Verification completed with errors (Exit code: $ExitCode)" -ForegroundColor Red
    Write-Host "Please review the output above and fix any issues." -ForegroundColor Yellow
}

exit $ExitCode
