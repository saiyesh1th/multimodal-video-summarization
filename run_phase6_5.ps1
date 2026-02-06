# run_phase6_5.ps1
# Phase 6.5: Hybrid Attention Scoring
$ErrorActionPreference = "Stop"
Write-Host "[-] Starting Phase 6.5: Attention Scoring..." -ForegroundColor Cyan

docker exec spark /opt/spark/bin/spark-submit `
  --master local[*] `
  /opt/spark_jobs/attention_scoring.py

if ($LASTEXITCODE -ne 0) { Write-Host "[!] Attention Job Failed!" -ForegroundColor Red; exit 1 }
Write-Host "[-] Phase 6.5 Complete." -ForegroundColor Green