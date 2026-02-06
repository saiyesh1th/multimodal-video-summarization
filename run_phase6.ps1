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
# 2. VERIFY OUTPUT (Parquet Style)
# ---------------------------------------------------------
Write-Host "    [2/2] Verifying Output..." -ForegroundColor Yellow

# Check for the _SUCCESS flag
$success_check = docker exec namenode hdfs dfs -ls /project/output/summaries/_SUCCESS 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Job finished, but Parquet output directory is missing or incomplete!" -ForegroundColor Red
    exit 1
}

Write-Host "[-] Phase 6 Complete: Parquet dataset generated." -ForegroundColor Green

# Correct Preview: Running a one-liner via pyspark
Write-Host "[-] Previewing Data (First 5 rows):" -ForegroundColor Cyan
# docker exec spark /opt/spark/bin/spark-submit -e "spark.read.parquet('/project/output/summaries').show(5)" 2>$null
docker exec spark /opt/spark/bin/pyspark --master local[*] --command "spark.read.parquet('/project/output/summaries').show(5); exit()"