# Start BELT Training
# Uses venv to avoid dependency issues
# Fixed to use GPU, not CPU

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  BELT Model Training - Fixed Version" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Changes made to prevent CPU overload:" -ForegroundColor Yellow
Write-Host "  - num_workers set to 0 (no multiprocessing on Windows)" -ForegroundColor Yellow
Write-Host "  - pin_memory set to False (avoid memory issues)" -ForegroundColor Yellow
Write-Host "  - Will use GPU properly now" -ForegroundColor Yellow
Write-Host ""

# Activate venv
& 'C:\Users\n1sha\Desktop\Mtech Research\EEGtoWord\.venv\Scripts\Activate.ps1'

Write-Host "Choose which model to train:" -ForegroundColor Green
Write-Host "  1) BELT-Exact (replica, ~31% expected)" -ForegroundColor White
Write-Host "  2) BELT-Enhanced (~37-40% expected)" -ForegroundColor White
Write-Host "  3) Both (in sequence)" -ForegroundColor White
Write-Host ""
$choice = Read-Host "Enter choice (1-3)"

if ($choice -eq "1") {
    Write-Host "`nStarting BELT-Exact training..." -ForegroundColor Cyan
    python experiments/model_with_bootstrapping.py
}
elseif ($choice -eq "2") {
    Write-Host "`nStarting BELT-Enhanced training..." -ForegroundColor Green
    python experiments/model_enhanced.py
}
elseif ($choice -eq "3") {
    Write-Host "`nStarting BELT-Exact training..." -ForegroundColor Cyan
    python experiments/model_with_bootstrapping.py
    Write-Host "`nStarting BELT-Enhanced training..." -ForegroundColor Green
    python experiments/model_enhanced.py
}
else {
    Write-Host "Invalid choice" -ForegroundColor Red
}

Write-Host "`nTraining complete!" -ForegroundColor Green
