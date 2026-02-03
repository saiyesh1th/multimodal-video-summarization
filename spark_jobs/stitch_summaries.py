from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import subprocess
import os
import shutil
import sys

def main():
    spark = SparkSession.builder \
        .appName("Phase8_Stitching") \
        .getOrCreate()

    # ==========================================
    # 0. SAFETY CHECKS
    # ==========================================
    if shutil.which("ffmpeg") is None:
        print("[Fatal] FFmpeg not found!")
        sys.exit(1)

    # ==========================================
    # 1. READ & FILTER METADATA
    # ==========================================
    print("[-] Reading Visual Metadata from HDFS...")
    # FIX: Port 8020
    df = spark.read.option("header", "true") \
        .csv("hdfs://namenode:8020/project/output/visual_metadata")
    
    # Cast columns
    df = df.withColumn("start_time", col("start_time").cast("float")) \
           .withColumn("blur_score", col("blur_score").cast("float"))

    # FILTER: "The Multimodal Gate"
    # We drop frames that are too blurry (Score < 50)
    print("[-] Filtering out blurry segments (Blur Score < 50)...")
    clean_df = df.filter(col("blur_score") > 50.0) \
                 .orderBy("file", "start_time")
    
    # Collect work items
    rows = clean_df.collect()
    work_plan = {}
    
    for row in rows:
        vid = row['file']
        # Safety: Ensure extension is mp4
        if vid.endswith(".wav"): vid = vid.replace(".wav", ".mp4")
        
        if vid not in work_plan:
            work_plan[vid] = []
        work_plan[vid].append(row['start_time'])

    print(f"[-] Ready to summarize {len(work_plan)} videos.")

    # ==========================================
    # 2. STITCHING ENGINE
    # ==========================================
    output_base = "/opt/data/final_summaries"
    if os.path.exists(output_base):
        shutil.rmtree(output_base)
    os.makedirs(output_base)
    
    temp_dir = "/opt/data/temp_chunks"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    for vid_file, timestamps in work_plan.items():
        if not timestamps: 
            continue

        print(f"[-] Processing {vid_file} ({len(timestamps)} chunks)...")
        
        input_path = f"/opt/data/raw_videos/{vid_file}"
        if not os.path.exists(input_path):
            print(f"    [Skip] Raw video not found: {input_path}")
            continue

        concat_list_path = f"{temp_dir}/list_{vid_file}.txt"
        valid_chunks = []
        
        # A. Cut Segments
        for i, start_t in enumerate(timestamps):
            chunk_name = f"chunk_{vid_file}_{i}.mp4"
            chunk_path = f"{temp_dir}/{chunk_name}"
            
            # Cut 5 seconds, re-encoding for concat safety
            cmd_cut = [
                "ffmpeg", "-y",
                "-ss", str(start_t),
                "-t", "5",
                "-i", input_path,
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac",
                "-loglevel", "error",
                chunk_path
            ]
            
            res = subprocess.run(cmd_cut)
            if res.returncode == 0 and os.path.exists(chunk_path):
                valid_chunks.append(chunk_path)
        
        if not valid_chunks:
            print(f"    [Warn] No valid chunks generated for {vid_file}")
            continue

        # B. Create Concat List
        with open(concat_list_path, 'w') as f:
            for chunk in valid_chunks:
                f.write(f"file '{chunk}'\n")
        
        # C. Stitch
        final_output = f"{output_base}/Summary_{vid_file}"
        
        cmd_stitch = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            "-loglevel", "error",
            final_output
        ]
        
        subprocess.run(cmd_stitch)
        
        if os.path.exists(final_output):
            print(f"    [SUCCESS] Created {final_output}")
        
        # Cleanup temp list
        if os.path.exists(concat_list_path): os.remove(concat_list_path)

    # Final cleanup
    shutil.rmtree(temp_dir)
    print("[-] Phase 8 Complete. Check /opt/data/final_summaries")

if __name__ == "__main__":
    main()