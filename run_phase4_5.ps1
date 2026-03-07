# run_phase4_5.ps1
# Phase 4.5: Audio Transcription (ASR with Whisper)

$ErrorActionPreference = "Stop"

Write-Host "[-] Starting Phase 4.5: ASR Transcription..." -ForegroundColor Cyan

# ---------------------------------------------------------
# EXECUTE SPARK JOB
# ---------------------------------------------------------
Write-Host "    [0/2] Waiting for HDFS to leave Safe Mode..." -ForegroundColor Yellow
docker exec namenode hdfs dfsadmin -safemode wait

Write-Host "    [1/2] Submitting Spark Job (Whisper ASR)..." -ForegroundColor Yellow

# Note: We might need more memory for the driver/executor since Whisper models can be large
# But 'base' model is ~140MB so 2G/2G should be fine for now.
docker exec spark /opt/spark/bin/spark-submit `
  --master local[*] `
  --driver-memory 4G `
  --executor-memory 4G `
  /opt/spark_jobs/transcription.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Transcription Job Failed!" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------
# VERIFY OUTPUT
# ---------------------------------------------------------
Write-Host "    [2/2] Verifying Transcripts in HDFS..." -ForegroundColor Yellow

# Check for success
$success_check = docker exec namenode hdfs dfs -ls /project/transcripts/_SUCCESS 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Job finished, but output directory is missing or incomplete!" -ForegroundColor Red
    exit 1
}

Write-Host "[-] Phase 4.5 Complete: Transcripts generated." -ForegroundColor Green

# ---------------------------------------------------------
# EXPORT TO LOCAL
# ---------------------------------------------------------
Write-Host "    [3/3] Exporting Transcripts to Local Disk..." -ForegroundColor Yellow

$output_dir = "results"
if (-not (Test-Path $output_dir)) {
    New-Item -ItemType Directory -Force -Path $output_dir | Out-Null
}

# 1. Merge/Get from HDFS to Namenode Local
docker exec namenode rm -rf /tmp/transcripts_export > $null 2>&1
docker exec namenode mkdir -p /tmp/transcripts_export
# Copy all CSV parts to a single file (or folder)
docker exec namenode bash -c "hdfs dfs -getmerge /project/transcripts /tmp/transcripts_export/transcripts.csv"

# 2. Copy from Namenode to Host
docker cp namenode:/tmp/transcripts_export/transcripts.csv ./$output_dir/transcripts.csv

if (Test-Path "./$output_dir/transcripts.csv") {
    Write-Host "[-] Export Complete: Saved to .\results\transcripts.csv" -ForegroundColor Green
    
    # Preview (First 5 lines)
    Get-Content "./$output_dir/transcripts.csv" -Head 5
} else {
    Write-Host "[!] Failed to export transcripts to local disk." -ForegroundColor Red
}
