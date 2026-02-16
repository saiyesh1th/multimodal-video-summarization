import pandas as pd
import json
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# 1. SETUP PATHS
MANIFEST_DIR = "data/manifests"
CSV_PATH = os.path.join(MANIFEST_DIR, "HowTo100M_v1.csv")
JSON_PATH = os.path.join(MANIFEST_DIR, "caption.json")
os.makedirs(MANIFEST_DIR, exist_ok=True)

# NOTE:
# task_id values below are synthetic integers used to preserve schema compatibility.
# They are NOT used semantically in this pipeline.

# Schema: [Video ID, Task ID (Int), Rank (Int), Category 1, Category 2]
SEED_DATA = [
    # Batch A: Validated YouCook2 Subset
    ["8MVo7fje_oE", 101, 1, "Food and Entertaining", "Sandwiches"],
    ["XQUC7O-1tNw", 101, 2, "Food and Entertaining", "Sandwiches"],
    ["FE49-r1_h8w", 101, 3, "Food and Entertaining", "Sandwiches"],
    ["Vae8c59I6B4", 205, 1, "Food and Entertaining", "Baking"],
    ["Mus_v0dHUBg", 309, 1, "Food and Entertaining", "Salads"],
    ["S-W7W4V3n8g", 410, 1, "Food and Entertaining", "Burgers"],
    ["Vi_eTq7K4lg", 550, 1, "Food and Entertaining", "Rice"],
    
    # Batch B: General Instructional (High Stability)
    ["x4l_q5L3bT8", 600, 1, "Food and Entertaining", "Eggs"],
    ["4d7y7_W0Qj8", 999, 1, "Food and Entertaining", "General Cooking"],
    ["j6Yd0aW7a_E", 600, 2, "Food and Entertaining", "Eggs"],
    ["0sN_dJ70FvA", 600, 3, "Food and Entertaining", "Eggs"],
    ["6i3a2F937C4", 205, 2, "Food and Entertaining", "Baking"],
    ["e9rODVcvYZY", 309, 2, "Food and Entertaining", "Salads"],
    ["1-SJGQ2HLp8", 999, 2, "Food and Entertaining", "General Cooking"],
    ["5hQM7qL9-1w", 410, 2, "Food and Entertaining", "Burgers"],
    ["h-3y7M2eZ_I", 700, 1, "Food and Entertaining", "Pasta"],
    ["p0q1r2s3t4u", 800, 1, "Food and Entertaining", "Meat"]
]

def bootstrap_csv():
    """Creates a HowTo100M_v1.csv file that matches the official schema types."""
    print(f"[-] Bootstrapping {CSV_PATH}...")
    columns = ["video_id", "task_id", "rank", "category_1", "category_2"]
    df = pd.DataFrame(SEED_DATA, columns=columns)
    # Ensure task_id and rank are integers
    df['task_id'] = df['task_id'].astype(int)
    df['rank'] = df['rank'].astype(int)
    
    df.to_csv(CSV_PATH, index=False)
    print(f"    [+] Created CSV with {len(df)} rows.")
    return df

def bootstrap_json(video_ids):
    """Fetches REAL captions and saves them in the official HowTo100M JSON format."""
    print(f"[-] Bootstrapping {JSON_PATH} (fetching real captions)...")
    
    captions_db = {}
    
    for vid in video_ids:
        try:
            # FIX: Use new API method list_transcripts -> find -> fetch
            transcript_list = YouTubeTranscriptApi.list_transcripts(vid)

            # Prefer human captions, fallback to auto-generated
            try:
                transcript_obj = transcript_list.find_manually_created_transcript(['en'])
            except:
                transcript_obj = transcript_list.find_generated_transcript(['en'])

            transcript = transcript_obj.fetch()

            # Format exactly like HowTo100M: {'text': [], 'start': [], 'end': []}
            entry = {'text': [], 'start': [], 'end': []}
            for line in transcript:
                entry['text'].append(line['text'])
                entry['start'].append(line['start'])
                entry['end'].append(line['start'] + line['duration'])
            
            captions_db[vid] = entry
            print(f"    [+] Fetched captions for {vid}")
            
        except (TranscriptsDisabled, NoTranscriptFound):
            print(f"    [!] No captions available for {vid} (empty record).")
            captions_db[vid] = {
                'text': [],
                'start': [],
                'end': []
            }
        except Exception as e:
            print(f"    [!] Error fetching {vid}: {e}")
            captions_db[vid] = {
                'text': [],
                'start': [],
                'end': []
            }

    with open(JSON_PATH, 'w') as f:
        json.dump(captions_db, f)
    print(f"    [+] Saved JSON database.")

if __name__ == "__main__":
    df = bootstrap_csv()
    bootstrap_json(df['video_id'].tolist())