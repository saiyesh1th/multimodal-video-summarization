# run_phase6_6.ps1
# Phase 6.6: Scene Density Grouping
$ErrorActionPreference = "Stop"
Write-Host "[-] Starting Phase 6.6: Scene Grouping..." -ForegroundColor Cyan

docker exec spark /opt/spark/bin/spark-submit `
  --master local[*] `
  /opt/spark_jobs/smoothing.py

if ($LASTEXITCODE -ne 0) { Write-Host "[!] Scene Grouping Failed!" -ForegroundColor Red; exit 1 }
Write-Host "[-] Phase 6.6 Complete." -ForegroundColor Green