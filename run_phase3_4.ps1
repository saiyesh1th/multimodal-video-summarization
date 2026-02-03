# run_phase3_4.ps1
# Master Orchestration Script for Audio Processing & Spark Ingestion

$ErrorActionPreference = "Stop" # Stop on any error immediately

Write-Host "[-] Starting Phase 3: Audio Preprocessing Pipeline..." -ForegroundColor Cyan

# ---------------------------------------------------------
# STEP 1: Audio Conversion
# ---------------------------------------------------------
Write-Host "    [1/4] Converting Video to WAV (inside Spark container)..."

# Use the shell script inside the container
docker exec spark bash /opt/data/convert.sh

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Audio conversion failed!" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------
# STEP 2: HDFS Ingestion
# ---------------------------------------------------------
Write-Host "    [2/4] Uploading Audio to HDFS..."

# 1. Clean HDFS destination
docker exec namenode hdfs dfs -rm -r -f /project/audio > $null 2>&1
docker exec namenode hdfs dfs -mkdir -p /project/audio

# --- THE FIX IS HERE ---
# We must open permissions so the 'spark' user can write to this folder later.
docker exec namenode hdfs dfs -chmod -R 777 /project
# -----------------------

# 2. PRE-CLEANUP: Clean Namenode temp
docker exec namenode rm -rf /tmp/audio_upload

# 3. THE BRIDGE: Spark -> Host -> Namenode
Write-Host "          Bridging files: Spark -> Host..." -ForegroundColor DarkGray
docker cp spark:/opt/data/audio ./_audio_tmp

Write-Host "          Bridging files: Host -> Namenode..." -ForegroundColor DarkGray
docker cp ./_audio_tmp namenode:/tmp/audio_upload

Remove-Item -Recurse -Force ./_audio_tmp

# 4. Push from Namenode local -> HDFS distributed
docker exec namenode bash -c "hdfs dfs -put -f /tmp/audio_upload/*.wav /project/audio/"

# 5. Cleanup Namenode temp
docker exec namenode rm -rf /tmp/audio_upload

# 6. Verify Upload
$hdfs_files = docker exec namenode hdfs dfs -ls /project/audio | Select-String ".wav"
if (-not $hdfs_files) {
    Write-Host "[!] HDFS Upload failed (No WAV files found in /project/audio)!" -ForegroundColor Red
    exit 1
}
Write-Host "          HDFS Ingestion Complete." -ForegroundColor Green

# ---------------------------------------------------------
# STEP 3: Spark Chunking Job
# ---------------------------------------------------------
Write-Host "    [3/4] Submitting Spark Job (Audio Chunking)..." -ForegroundColor Cyan

# Using absolute path to spark-submit
docker exec spark /opt/spark/bin/spark-submit `
  --master local[*] `
  --driver-memory 2G `
  --executor-memory 2G `
  /opt/spark_jobs/audio_chunking.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Spark Job Failed!" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------
# STEP 4: Final Verification
# ---------------------------------------------------------
Write-Host "    [4/4] Verifying Output..." -ForegroundColor Cyan
docker exec namenode hdfs dfs -ls /project/features/audio_chunks

Write-Host "[-] Pipeline Finished Successfully." -ForegroundColor Green