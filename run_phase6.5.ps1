# run_phase6_5.ps1
# Phase 6.5: Time Series Smoothing (Gaussian Convolution)

$ErrorActionPreference = "Stop"

Write-Host "[-] Starting Phase 6.5: Time Series Smoothing..." -ForegroundColor Cyan

# ---------------------------------------------------------
# EXECUTE SPARK JOB
# ---------------------------------------------------------
Write-Host "    [1/2] Submitting Smoothing Job..." -ForegroundColor Yellow

docker exec spark /opt/spark/bin/spark-submit `
  --master local[*] `
  /opt/spark_jobs/smoothing.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Smoothing Job Failed!" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------
# VERIFY OUTPUT
# ---------------------------------------------------------
Write-Host "    [2/2] Verifying Output..." -ForegroundColor Yellow

$csv_files = docker exec namenode hdfs dfs -ls /project/output/summaries_smoothed | Select-String ".csv"

if (-not $csv_files) {
    Write-Host "[!] Job finished, but no output found!" -ForegroundColor Red
    exit 1
}

Write-Host "[-] Phase 6.5 Complete: Scores smoothed." -ForegroundColor Green