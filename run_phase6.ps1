# run_phase6.ps1
# Master Script for Phase 6: Semantic Analytics

$ErrorActionPreference = "Stop"

Write-Host "[-] Starting Phase 6: Semantic Analysis Engine..." -ForegroundColor Cyan

# ---------------------------------------------------------
# EXECUTE SPARK JOB
# ---------------------------------------------------------
Write-Host "    [1/2] Submitting Spark Job (K-Means + TF-IDF)..." -ForegroundColor Yellow

docker exec spark /opt/spark/bin/spark-submit `
  --master local[*] `
  --driver-memory 4G `
  --executor-memory 2G `
  /opt/spark_jobs/semantic_analysis.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Spark Job Failed!" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------
# VERIFY OUTPUT (The Robust Way)
# ---------------------------------------------------------
Write-Host "    [2/2] Verifying Output..." -ForegroundColor Yellow

# FIX 2: Check for actual CSV data files, not just _SUCCESS
$csv_files = docker exec namenode hdfs dfs -ls /project/output/summaries | Select-String ".csv"

if (-not $csv_files) {
    Write-Host "[!] Job finished, but no CSV files found in output!" -ForegroundColor Red
    exit 1
}

Write-Host "[-] Phase 6 Complete: Summaries generated." -ForegroundColor Green

# Preview
Write-Host "[-] Preview of Generated Summaries:" -ForegroundColor Cyan
docker exec namenode hdfs dfs -cat /project/output/summaries/*.csv | Select-Object -First 10