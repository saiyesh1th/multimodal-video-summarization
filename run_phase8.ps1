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
# ---------------------------------------------------------
# 2. EXTRACT VIDEOS TO HOST (Hardened Version)
# ---------------------------------------------------------
Write-Host "    [2/2] Extracting Videos to Host..." -ForegroundColor Yellow

$local_dest = "results/summaries"
if (-not (Test-Path $local_dest)) { New-Item -ItemType Directory -Force -Path $local_dest }

# Try Docker CP first (most reliable way to get files out of a volume)
docker cp spark:/opt/data/final_summaries/. $local_dest

$count = (Get-ChildItem $local_dest -Filter *.mp4).Count

if ($count -gt 0) {
    Write-Host "[-] Success! $count Summary Videos are ready." -ForegroundColor Green
} else {
    Write-Host "[!] Critical Error: No .mp4 files found in container or local summaries folder." -ForegroundColor Red
}