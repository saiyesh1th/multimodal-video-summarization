# nlp_phase3.ps1
# Script to run Phase 3 of the Custom Summarization NLP Pipeline

$DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$MODEL_SCRIPT = Join-Path $DIR "model_training\model.py"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " Running NLP Phase 3: Transformer Check" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Check if Python is available
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Host "[!] Python is not found in PATH." -ForegroundColor Red
    exit 1
}

# Run the dataset verification script
python $MODEL_SCRIPT

if ($LASTEXITCODE -eq 0) {
    Write-Host "[+] Phase 3 executed successfully." -ForegroundColor Green
} else {
    Write-Host "[-] Phase 3 failed with exit code $LASTEXITCODE." -ForegroundColor Red
}
