#!/bin/bash
# Move to the videos directory
cd /opt/data/raw_videos || exit

# Create output folder
mkdir -p /opt/data/audio
echo "[-] Starting processing..."

# Loop and convert
for f in *.mp4; do
    [ -e "$f" ] || continue
    
    filename=$(basename "$f" .mp4)
    output="/opt/data/audio/$filename.wav"
    
    if [ ! -f "$output" ]; then
        echo "[-] Converting: $filename"
        # -y forces overwrite, -v error silences logs
        ffmpeg -v error -y -i "$f" -ac 1 -ar 16000 "$output"
    else
        echo "[-] Skipping $filename (Already exists)"
    fi
done

echo "[-] Audio extraction finished."