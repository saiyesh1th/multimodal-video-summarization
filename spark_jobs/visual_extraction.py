from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import cv2
import numpy as np
import subprocess
import os
import shutil
import sys

# ==========================================
# 1. COMPUTER VISION KERNEL (Blur Check)
# ==========================================
def calculate_blur_score(image_path):
    """
    Computes the Variance of Laplacian.
    High Variance (>100) = Sharp Edges.
    Low Variance (<100) = Blurry.
    """
    try:
        # Read image as grayscale
        image = cv2.imread(image_path)
        if image is None: return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Convolve with Laplacian Kernel (2nd Derivative)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Calculate variance
        score = laplacian.var()
        return float(score)
    except Exception as e:
        # Fail gracefully on corrupt images
        return 0.0

def main():
    spark = SparkSession.builder \
        .appName("Phase7_VisualExtraction") \
        .getOrCreate()

    # ==========================================
    # 0. SAFETY CHECKS
    # ==========================================
    if shutil.which("ffmpeg") is None:
        print("[Fatal] FFmpeg not found in container! Install it or check your Docker image.")
        sys.exit(1)

    # ==========================================
    # 2. READ SMOOTHED CANDIDATES
    # ==========================================
    print("[-] Reading Smoothed Summaries from HDFS...")
    # FIX: Port 8020
    df = spark.read.option("header", "true") \
        .csv("hdfs://namenode:8020/project/output/summaries_smoothed")
    
    candidates = df.collect()
    print(f"[-] Processing {len(candidates)} visual candidates...")

    # ==========================================
    # 3. EXTRACTION LOOP
    # ==========================================
    output_dir = "/opt/data/keyframes"
    
    # Clean output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    results = []

    for row in candidates:
        vid_file = row['file']
        
        # --- CRITICAL FIX: Handle Extension Mismatch ---
        # The pipeline passed "video.wav", but we need "video.mp4"
        vid_file = vid_file.replace(".wav", ".mp4")
        
        # Parse timestamp
        try:
            start_time = float(row['start_time'])
            # Target middle of chunk (Start + 2.5s)
            target_time = start_time + 2.5
        except (ValueError, TypeError):
            continue 
        
        # Path to raw video (mapped volume)
        input_path = f"/opt/data/raw_videos/{vid_file}"
        
        if not os.path.exists(input_path):
            print(f"    [Skip] Video file not found: {input_path}")
            continue

        # Image Filename: videoID_timestamp.jpg
        img_name = f"{vid_file.replace('.mp4','')}_{int(start_time)}.jpg"
        img_path = f"{output_dir}/{img_name}"

        # A. FFmpeg Extraction
        cmd = [
            "ffmpeg", "-y", 
            "-ss", str(target_time),
            "-i", input_path,
            "-vframes", "1",
            "-q:v", "2", 
            "-loglevel", "error", 
            img_path
        ]
        
        subprocess.run(cmd)

        # B. Computer Vision Quality Check
        if os.path.exists(img_path):
            blur_score = calculate_blur_score(img_path)
            
            status = "SHARP" if blur_score > 100 else "BLURRY"
            print(f"    [Extracted] {img_name} | Score: {blur_score:.1f} ({status})")
            
            results.append((vid_file, start_time, img_name, blur_score))
        else:
            print(f"    [Error] Frame extraction failed for {img_name}")

    # ==========================================
    # 4. SAVE METADATA
    # ==========================================
    print("[-] Saving Visual Metadata to HDFS...")
    
    if results:
        results_df = spark.createDataFrame(results, ["file", "start_time", "image_name", "blur_score"])
        
        results_df.write.mode("overwrite").option("header", "true") \
            .csv("hdfs://namenode:8020/project/output/visual_metadata")
    else:
        print("[!] No frames were extracted.")

    print("[-] Phase 7 Complete. Images saved to data/keyframes")

if __name__ == "__main__":
    main()