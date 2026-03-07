# nlp_phase5.ps1
# Script to run Phase 5 of the Custom Summarization NLP Pipeline
# (Autoregressive Inference Engine verification)

$DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$INFER_SCRIPT = Join-Path $DIR "model_training\inference.py"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " Running NLP Phase 5: Inference Engine" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Check if Python is available
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Host "[!] Python is not found in PATH." -ForegroundColor Red
    exit 1
}

# Run the inference script
python $INFER_SCRIPT

if ($LASTEXITCODE -eq 0) {
    Write-Host "[+] Phase 5 executed successfully." -ForegroundColor Green
} else {
    Write-Host "[-] Phase 5 failed with exit code $LASTEXITCODE." -ForegroundColor Red
}
