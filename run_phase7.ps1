# run_phase7.ps1
# Final Phase: Extract and Display Results (Fixed Formatting)

$ErrorActionPreference = "Stop"

Write-Host "[-] Starting Phase 7: Result Extraction..." -ForegroundColor Cyan

# 1. Prepare Destination
$output_dir = "results"
if (-not (Test-Path $output_dir)) {
    New-Item -ItemType Directory -Force -Path $output_dir | Out-Null
}

# 2. Extract from HDFS
Write-Host "    [1/2] Extracting Scene-Aware Summaries..." -ForegroundColor Yellow

docker exec namenode rm -rf /tmp/summary_export > $null 2>&1
docker exec namenode mkdir -p /tmp/summary_export
docker exec namenode bash -c "hdfs dfs -get /project/output/summaries_smoothed/*.csv /tmp/summary_export/summary_raw.csv"
docker cp namenode:/tmp/summary_export/summary_raw.csv ./$output_dir/final_summary.csv

if (-not (Test-Path "./$output_dir/final_summary.csv")) {
    Write-Host "[!] Failed to extract CSV file!" -ForegroundColor Red
    exit 1
}

# 3. Visualize the Narrative
Write-Host "    [2/2] Reconstructing Narrative Timeline..." -ForegroundColor Yellow
$data = Import-Csv "./$output_dir/final_summary.csv"

# Sort by File then Time
$timeline = $data | Sort-Object file, {[double]$_.start_time}

Write-Host "`n=======================================================" -ForegroundColor White
Write-Host "         MULTIMODAL SUMMARY: NARRATIVE TIMELINE        " -ForegroundColor White
Write-Host "=======================================================" -ForegroundColor White

# Group and Display
$groups = $timeline | Group-Object file

foreach ($group in $groups) {
    Write-Host "`nVIDEO ID: $($group.Name)" -ForegroundColor Cyan
    Write-Host "-------------------------------------------------------"
    # Fixed formatting line:
    $header = "{0,-10} {1,-12} {2,-10}" -f "Start (s)", "End (s)", "Score"
    Write-Host $header -ForegroundColor Gray
    Write-Host "-------------------------------------------------------"
    
    foreach ($row in $group.Group) {
        $start = [math]::Round([double]$row.start_time, 1)
        $end = $start + 5.0
        $score = [math]::Round([double]$row.tfidf_score, 4)
        
        $line = "{0,-10} {1,-12} {2,-10}" -f $start, $end, $score
        
        # Highlight extremely relevant scenes
        if ($score -gt 0.8) {
            Write-Host "$line [HIGH RELEVANCE]" -ForegroundColor Green
        } else {
            Write-Host $line
        }
    }
    Write-Host "-------------------------------------------------------"
}

Write-Host "`n[-] Extraction Complete. Results saved to .\results\final_summary.csv" -ForegroundColor Green