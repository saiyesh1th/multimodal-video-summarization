# upload_videos_to_hdfs.ps1
$ErrorActionPreference = "Stop"

Write-Host "[-] Uploading raw videos to HDFS..." -ForegroundColor Cyan

# Ensure the /project/raw_videos directory exists in HDFS
docker exec namenode hdfs dfs -mkdir -p /project/raw_videos
docker exec namenode hdfs dfs -chmod -R 777 /project

# Create a temporary directory in namenode
docker exec namenode bash -c "rm -rf /tmp/video_upload"
docker exec namenode bash -c "mkdir -p /tmp/video_upload"

Write-Host "    Bridging files: Host -> Namenode..." -ForegroundColor DarkGray
# Copy raw_videos from host to namenode /tmp
docker cp ./data/raw_videos namenode:/tmp/video_upload_parent

Write-Host "    Pushing from Namenode local -> HDFS distributed..." -ForegroundColor DarkGray
docker exec namenode bash -c "hdfs dfs -put -f /tmp/video_upload_parent/*.mp4 /project/raw_videos/"

Write-Host "    Cleaning up Namenode temp..." -ForegroundColor DarkGray
docker exec namenode bash -c "rm -rf /tmp/video_upload_parent"

# Verify Upload
$hdfs_files = docker exec namenode hdfs dfs -ls /project/raw_videos | Select-String ".mp4"
if (-not $hdfs_files) {
    Write-Host "[!] HDFS Upload failed (No MP4 files found in /project/raw_videos)!" -ForegroundColor Red
    exit 1
}
Write-Host "[-] HDFS Video Upload Complete." -ForegroundColor Green
