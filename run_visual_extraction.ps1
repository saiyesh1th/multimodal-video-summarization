# run_visual_extraction.ps1
# Phase 7: Visual Keyframe Extraction (Multimodal)

$ErrorActionPreference = "Stop"

Write-Host "[-] Starting Phase 7: Visual Extraction..." -ForegroundColor Cyan

# ---------------------------------------------------------
# 1. INSTALL DEPENDENCIES (OpenCV)
# ---------------------------------------------------------
Write-Host "    [1/3] Installing Computer Vision libraries..." -ForegroundColor Yellow

# Install dependencies (ignoring the root user warning, which is fine in Docker)
docker exec -u 0 spark pip install opencv-python-headless numpy --quiet --root-user-action=ignore

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Failed to install OpenCV!" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------
# 2. SUBMIT SPARK JOB
# ---------------------------------------------------------
Write-Host "    [2/3] Submitting Extraction Job..." -ForegroundColor Yellow

# FIX: Use absolute path '/opt/spark/bin/spark-submit'
docker exec spark /opt/spark/bin/spark-submit `
  --master local[*] `
  /opt/spark_jobs/visual_extraction.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Extraction Job Failed!" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------
# 3. VERIFY OUTPUT ON HOST
# ---------------------------------------------------------
Write-Host "    [3/3] Verifying Local Output..." -ForegroundColor Yellow

if (Test-Path "data/keyframes") {
    $count = (Get-ChildItem "data/keyframes" -Filter *.jpg).Count
    if ($count -gt 0) {
        Write-Host "[-] Success! Extracted $count Keyframes." -ForegroundColor Green
        Write-Host "[-] Check the folder: data\keyframes" -ForegroundColor Cyan
    } else {
        Write-Host "[!] Job finished, but 'data/keyframes' is empty." -ForegroundColor Red
    }
} else {
    Write-Host "[!] 'data/keyframes' folder does not exist!" -ForegroundColor Red
}