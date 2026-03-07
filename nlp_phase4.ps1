# nlp_phase4.ps1
# Script to run Phase 4 of the Custom Summarization NLP Pipeline

$DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$TRAIN_SCRIPT = Join-Path $DIR "model_training\train.py"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " Running NLP Phase 4: Model Training" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Check if Python is available
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Host "[!] Python is not found in PATH." -ForegroundColor Red
    exit 1
}

# Run the training script
python $TRAIN_SCRIPT

if ($LASTEXITCODE -eq 0) {
    Write-Host "[+] Phase 4 executed successfully." -ForegroundColor Green
} else {
    Write-Host "[-] Phase 4 failed with exit code $LASTEXITCODE." -ForegroundColor Red
}
