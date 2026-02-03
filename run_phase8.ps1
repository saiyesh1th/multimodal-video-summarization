# run_phase8.ps1
# Phase 8: Final Video Reconstruction

$ErrorActionPreference = "Stop"

Write-Host "[-] Starting Phase 8: Multimodal Reconstruction..." -ForegroundColor Cyan

# ---------------------------------------------------------
# 1. SUBMIT SPARK JOB
# ---------------------------------------------------------
Write-Host "    [1/2] Generating Video Summaries..." -ForegroundColor Yellow

# Use absolute path to spark-submit
docker exec spark /opt/spark/bin/spark-submit `
  --master local[*] `
  /opt/spark_jobs/stitch_summaries.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Stitching Job Failed!" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------
# 2. EXTRACT VIDEOS TO HOST
# ---------------------------------------------------------
Write-Host "    [2/2] Extracting Videos to Host..." -ForegroundColor Yellow

$local_dest = "results/summaries"
if (-not (Test-Path $local_dest)) {
    New-Item -ItemType Directory -Force -Path $local_dest | Out-Null
}

# Copy from Container (/opt/data/final_summaries) -> Host
# Since /opt/data is mounted to 'data', we can actually check there directly!
# But let's verify.

if (Test-Path "data/final_summaries") {
    Copy-Item "data/final_summaries/*.mp4" -Destination $local_dest -Force
    $count = (Get-ChildItem $local_dest -Filter *.mp4).Count
    
    if ($count -gt 0) {
        Write-Host "[-] Success! $count Summary Videos are ready." -ForegroundColor Green
        Write-Host "[-] Location: $local_dest" -ForegroundColor Cyan
    } else {
        Write-Host "[!] Job finished, but no videos found." -ForegroundColor Red
    }
} else {
    Write-Host "[!] Output folder missing in container volume." -ForegroundColor Red
}