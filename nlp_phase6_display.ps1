# nlp_phase6.ps1
# Script to run Phase 6 of the Custom Summarization NLP Pipeline
# (Generative Output Display)

$DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$DISPLAY_SCRIPT = Join-Path $DIR "model_training\display_summaries.py"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " Running NLP Phase 6: Summary Display" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Check if Python is available
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Host "[!] Python is not found in PATH." -ForegroundColor Red
    exit 1
}

# Run the display script
python $DISPLAY_SCRIPT

if ($LASTEXITCODE -eq 0) {
    Write-Host "[+] Phase 6 executed successfully." -ForegroundColor Green
} else {
    Write-Host "[-] Phase 6 failed with exit code $LASTEXITCODE." -ForegroundColor Red
}
