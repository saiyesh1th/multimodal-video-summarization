# run_phase5.ps1
# Master Script for Phase 5: MFCC Feature Extraction (From Scratch)

$ErrorActionPreference = "Stop"

Write-Host "[-] Starting Phase 5: MFCC Extraction..." -ForegroundColor Cyan

# ---------------------------------------------------------
# EXECUTE SPARK JOB
# ---------------------------------------------------------
Write-Host "    [1/2] Submitting Spark Job..." -ForegroundColor Yellow

docker exec spark /opt/spark/bin/spark-submit `
  --master local[*] `
  --driver-memory 4G `
  --executor-memory 2G `
  /opt/spark_jobs/mfcc_extraction.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Spark Job Failed!" -ForegroundColor Red
    exit 1  
}

# ---------------------------------------------------------
# VERIFY OUTPUT
# ---------------------------------------------------------
Write-Host "    [2/2] Verifying HDFS Output..." -ForegroundColor Yellow

# Check for the _SUCCESS file
$success_file = docker exec namenode hdfs dfs -ls /project/features/mfcc_vectors | Select-String "_SUCCESS"

if (-not $success_file) {
    Write-Host "[!] Job finished, but no _SUCCESS file found in HDFS!" -ForegroundColor Red
    exit 1
}

# List the files so you can see the result
docker exec namenode hdfs dfs -ls /project/features/mfcc_vectors

Write-Host "[-] Phase 5 Complete: MFCC Vectors stored in HDFS." -ForegroundColor Green