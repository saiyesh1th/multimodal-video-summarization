from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace
import subprocess, os, shutil

def get_video_duration(path):
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path]
        return float(subprocess.check_output(cmd).decode().strip())
    except: return 0.0

def main():
    spark = SparkSession.builder.appName("Phase8_Dynamic_Budget_Director").getOrCreate()
    spark.catalog.clearCache()

    output_base, temp_dir, raw_dir = "/opt/data/final_summaries", "/opt/data/temp_chunks", "/opt/data/raw_videos"
    for d in [output_base, temp_dir]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)

    df_v = spark.read.option("header", "true").csv("hdfs://namenode:8020/project/output/visual_metadata").withColumn("start_time", col("start_time").cast("double"))
    df_a = spark.read.option("header", "true").csv("hdfs://namenode:8020/project/output/summaries_smoothed").withColumn("start_time", col("start_time").cast("double"))

    df_v = df_v.withColumn("base", regexp_replace(col("file"), "\\.mp4$", ""))
    df_a = df_a.withColumn("base", regexp_replace(col("file"), "\\.wav$", ""))

    fused_rows = df_v.join(df_a, ["base", "start_time"]) \
                     .select(df_v["file"].alias("vid"), "start_time", df_a["tfidf_score"].alias("score")) \
                     .collect()

    video_data = {}
    for row in fused_rows:
        v = row['vid']
        if v not in video_data:
            dur = get_video_duration(f"{raw_dir}/{v}")
            # ENFORCE 20% Budget
            video_data[v] = {"chunks": [], "limit": max(30.0, dur * 0.20), "orig_dur": dur}
        video_data[v]["chunks"].append({"t": float(row['start_time']), "s": float(row['score'])})

    work_plan = {}
    for vid, data in video_data.items():
        # Rank by importance
        sorted_chunks = sorted(data["chunks"], key=lambda x: x['s'], reverse=True)
        keep_count = int(data["limit"] / 5.0)
        elite_chunks = sorted(sorted_chunks[:keep_count], key=lambda x: x['t'])
        
        # 0.5s Bridge logic
        work_plan[vid] = []
        for chunk in elite_chunks:
            t = chunk['t']
            if work_plan[vid] and t <= (work_plan[vid][-1][1] + 1.0):
                work_plan[vid][-1] = (work_plan[vid][-1][0], t + 5.0)
            else:
                work_plan[vid].append((t, t + 5.0))

        # HARD EDITORIAL CAP: Max 1 segment per 45s of video
        MAX_SEGMENTS = max(3, int(data["orig_dur"] / 45))
        if len(work_plan[vid]) > MAX_SEGMENTS:
            print(f"[Cap] {vid}: Trimming to {MAX_SEGMENTS} elite segments.")
            work_plan[vid] = sorted(work_plan[vid], key=lambda x: x[1]-x[0], reverse=True)[:MAX_SEGMENTS]
            work_plan[vid] = sorted(work_plan[vid], key=lambda x: x[0])

    # 3. ASSEMBLY
    for vid, segments in work_plan.items():
        print(f"[-] Assembling {vid}: {len(segments)} segments planned.")
        input_path = f"{raw_dir}/{vid}"
        valid_chunks = []
        for i, (s_t, e_t) in enumerate(segments):
            chunk = f"{temp_dir}/scene_{i}_{vid}"
            
            # THE NEGATIVE SHIELD
            actual_end = max(s_t + 1.0, e_t)
            p_start, p_dur = max(0, s_t - 0.25), (actual_end - s_t) + 0.5
            
            cmd = ["ffmpeg", "-y", "-ss", str(p_start), "-t", str(p_dur), "-i", input_path, 
                   "-c:v", "libx264", "-crf", "26", "-preset", "veryfast", "-c:a", "aac", "-loglevel", "error", chunk]
            if subprocess.run(cmd).returncode == 0: valid_chunks.append(chunk)

        if valid_chunks:
            list_f = f"{temp_dir}/list_{vid}.txt"
            with open(list_f, "w") as f:
                for c in valid_chunks: f.write(f"file '{c}'\n")
            subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_f, "-c", "copy", "-loglevel", "error", f"{output_base}/summary_{vid}"])

    print("[-] Phase 8 Complete. Victory!")

if __name__ == "__main__":
    main()