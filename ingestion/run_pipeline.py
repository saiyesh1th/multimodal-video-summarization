import pandas as pd
import json
import os
import subprocess

# CONFIG
RAW_VIDEO_DIR = "data/raw_videos"
MANIFEST_DIR = "data/manifests"
CSV_PATH = os.path.join(MANIFEST_DIR, "HowTo100M_v1.csv")
JSON_PATH = os.path.join(MANIFEST_DIR, "caption.json")
OUTPUT_CAPTIONS_CSV = os.path.join(MANIFEST_DIR, "processed_captions.csv")
TARGET_GB = 5.0 

def get_dir_size_gb(path):
    total = 0
    if not os.path.exists(path): return 0
    for f in os.scandir(path):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024**3)

def phase_a_metadata_sampling():
    print("[-] Phase A: Metadata Sampling...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing {CSV_PATH}. Run bootstrap first.")
    df = pd.read_csv(CSV_PATH)
    subset = df[df['category_1'] == 'Food and Entertaining']
    print(f"    [+] Found {len(subset)} candidate videos.")
    return subset

def phase_b_video_retrieval(subset_df):
    print("[-] Phase B: Video Retrieval...")
    os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
    downloaded_ids = []
    
    for _, row in subset_df.iterrows():
        if get_dir_size_gb(RAW_VIDEO_DIR) >= TARGET_GB:
            print(f"    [!] Storage limit {TARGET_GB}GB reached.")
            break
            
        vid_id = str(row['video_id']).strip()
        path = f"{RAW_VIDEO_DIR}/{vid_id}.mp4"
        
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print(f"    [-] {vid_id} already exists. Skipping.")
            downloaded_ids.append(vid_id)
            continue
            
        print(f"    [-] Downloading {vid_id}...")
        try:
            # --- THE FIX IS HERE ---
            # We switched to 'best[ext=mp4]' which downloads a single pre-merged file.
            # This bypasses the need for FFmpeg on the host machine.
            cmd = [
                "yt-dlp", 
                "-f", "best[ext=mp4]", 
                "-o", path, 
                "--max-filesize", "300M",
                "--no-warnings", # Removed --quiet so we can see errors if they happen
                f"https://www.youtube.com/watch?v={vid_id}"
            ]
            
            # Allow verbose output to stdout so you can see what's happening
            subprocess.run(cmd, check=True, timeout=180)
            
            if os.path.exists(path) and os.path.getsize(path) > 1000:
                downloaded_ids.append(vid_id)
                print(f"    [+] Success.")
            else:
                print(f"    [x] Download failed (File missing or empty).")
                
        except subprocess.CalledProcessError as e:
            # If yt-dlp fails, it will now print the error code
            print(f"    [x] Failed to download {vid_id}. Error code: {e.returncode}")
        except Exception as e:
            print(f"    [x] Unexpected error: {e}")
            if os.path.exists(path): os.remove(path)

    return downloaded_ids

def phase_c_caption_extraction(downloaded_ids):
    print("[-] Phase C: Caption Extraction...")
    if not os.path.exists(JSON_PATH):
        print("    [!] Missing caption.json. Skipping.")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
        
    processed_rows = []
    for vid in downloaded_ids:
        if vid in data:
            record = data[vid]
            full_text = " ".join(record['text'])
            has_caption = 1 if full_text.strip() else 0

            processed_rows.append({
                'video_id': vid,
                'caption_text': full_text,
                'has_caption': has_caption,
                'start_time': record['start'][0] if record['start'] else 0,
                'end_time': record['end'][-1] if record['end'] else 0
            })
            
    pd.DataFrame(processed_rows).to_csv(OUTPUT_CAPTIONS_CSV, index=False)
    print(f"    [+] Saved {len(processed_rows)} processed captions to {OUTPUT_CAPTIONS_CSV}")

if __name__ == "__main__":
    candidates = phase_a_metadata_sampling()
    valid_ids = phase_b_video_retrieval(candidates)
    phase_c_caption_extraction(valid_ids)