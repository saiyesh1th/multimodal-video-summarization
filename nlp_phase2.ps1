# nlp_phase2.ps1
# Script to run Phase 2 of the Custom Summarization NLP Pipeline

$DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$DATASET_SCRIPT = Join-Path $DIR "model_training\dataset.py"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " Running NLP Phase 2: PyTorch DataLoader" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Check if Python is available
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Host "[!] Python is not found in PATH." -ForegroundColor Red
    exit 1
}

# Run the dataset verification script
python $DATASET_SCRIPT

if ($LASTEXITCODE -eq 0) {
    Write-Host "[+] Phase 2 executed successfully." -ForegroundColor Green
} else {
    Write-Host "[-] Phase 2 failed with exit code $LASTEXITCODE." -ForegroundColor Red
}
