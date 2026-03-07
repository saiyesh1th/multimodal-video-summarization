from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import subprocess
import os

# --------------------------------------------------------
# PATH CONFIG
# --------------------------------------------------------

SEGMENTS_PATH = "/opt/data/final_segments.csv"

SCORES_PATH = "hdfs://namenode:8020/project/output/summaries_smoothed"

RAW_VIDEO_DIR = "/opt/data/raw_videos"

OUTPUT_PATH = "/opt/data/summary_evaluation_results.csv"


# --------------------------------------------------------
# GET VIDEO DURATION
# --------------------------------------------------------

def get_video_duration(path):

    try:

        cmd = [
            "ffprobe",
            "-v","error",
            "-show_entries","format=duration",
            "-of","default=noprint_wrappers=1:nokey=1",
            path
        ]

        duration = float(subprocess.check_output(cmd).decode().strip())

        return duration

    except:

        return 0.0


# --------------------------------------------------------
# TEMPORAL F1 FUNCTION
# --------------------------------------------------------

def temporal_f1_score(summary_segments, important_segments):

    overlap = 0

    for s_start, s_end in summary_segments:

        for i_start, i_end in important_segments:

            inter_start = max(s_start, i_start)

            inter_end = min(s_end, i_end)

            if inter_end > inter_start:

                overlap += inter_end - inter_start

    summary_length = sum(e - s for s, e in summary_segments)

    important_length = sum(e - s for s, e in important_segments)

    precision = overlap / summary_length if summary_length > 0 else 0

    recall = overlap / important_length if important_length > 0 else 0

    if precision + recall == 0:

        f1 = 0

    else:

        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------

def main():

    spark = SparkSession.builder.appName("VideoSummaryEvaluation").getOrCreate()

    print("\nLoading datasets...\n")

    segments_df = pd.read_csv(SEGMENTS_PATH)

    scores_df = spark.read.option("header","true").csv(SCORES_PATH).toPandas()

    scores_df["start_time"] = scores_df["start_time"].astype(float)
    scores_df["tfidf_score"] = scores_df["tfidf_score"].astype(float)

    videos = segments_df["video"].unique()

    results = []


    # ----------------------------------------------------
    # LOOP THROUGH VIDEOS
    # ----------------------------------------------------

    for video in videos:

        seg = segments_df[segments_df["video"] == video]

        summary_duration = seg["duration"].sum()

        segments = len(seg)

        start = seg["start_time"].min()

        end = seg["end_time"].max()

        timeline_span = end - start

        diversity_score = segments / summary_duration if summary_duration > 0 else 0


        # ------------------------------------------------
        # ORIGINAL VIDEO DURATION
        # ------------------------------------------------

        video_path = os.path.join(RAW_VIDEO_DIR, video)

        original_duration = get_video_duration(video_path)

        compression_ratio = summary_duration / original_duration if original_duration > 0 else 0


        # ------------------------------------------------
        # TIMELINE COVERAGE
        # ------------------------------------------------

        timeline_coverage = timeline_span / original_duration if original_duration > 0 else 0


        # ------------------------------------------------
        # IMPORTANCE METRICS
        # ------------------------------------------------

        base = video.split(".")[0]

        video_scores = scores_df[scores_df["file"].str.contains(base)]

        total_importance = video_scores["tfidf_score"].sum()

        selected_scores = video_scores[
            video_scores["start_time"].isin(seg["start_time"])
        ]

        selected_importance = selected_scores["tfidf_score"].sum()

        importance_coverage = selected_importance / total_importance if total_importance > 0 else 0

        avg_importance = selected_scores["tfidf_score"].mean() if len(selected_scores) > 0 else 0


        # ------------------------------------------------
        # HIGHLIGHT DENSITY
        # ------------------------------------------------

        highlight_density = segments / original_duration if original_duration > 0 else 0


        # ------------------------------------------------
        # REDUNDANCY
        # ------------------------------------------------

        redundancy = 0

        if len(seg) > 1:

            overlaps = []

            seg_sorted = seg.sort_values("start_time")

            for i in range(len(seg_sorted)-1):

                a_end = seg_sorted.iloc[i]["end_time"]

                b_start = seg_sorted.iloc[i+1]["start_time"]

                overlap = max(0, a_end - b_start)

                overlaps.append(overlap)

            redundancy = np.mean(overlaps) if overlaps else 0


        # ------------------------------------------------
        # TEMPORAL F1
        # ------------------------------------------------

        threshold = video_scores["tfidf_score"].quantile(0.80)

        important_rows = video_scores[
            video_scores["tfidf_score"] >= threshold
        ]

        important_segments = [
            (row["start_time"], row["start_time"] + 5)
            for _, row in important_rows.iterrows()
        ]

        summary_segments = [
            (row["start_time"], row["end_time"])
            for _, row in seg.iterrows()
        ]

        precision, recall, temporal_f1 = temporal_f1_score(
            summary_segments,
            important_segments
        )


        # ------------------------------------------------
        # STORE RESULTS
        # ------------------------------------------------

        results.append({

            "video": video,

            "original_duration_sec": round(original_duration,2),

            "summary_duration_sec": round(summary_duration,2),

            "compression_ratio": round(compression_ratio,3),

            "segments": segments,

            "timeline_span_sec": round(timeline_span,2),

            "timeline_coverage": round(timeline_coverage,3),

            "diversity_score": round(diversity_score,3),

            "highlight_density": round(highlight_density,4),

            "importance_coverage": round(importance_coverage,3),

            "avg_importance": round(avg_importance,3),

            "redundancy_score": round(redundancy,3),

            "temporal_precision": round(precision,3),

            "temporal_recall": round(recall,3),

            "temporal_f1": round(temporal_f1,3)

        })


    results_df = pd.DataFrame(results)

    print("\n===== VIDEO SUMMARY EVALUATION =====\n")

    print(results_df)

    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nEvaluation saved to:", OUTPUT_PATH)


if __name__ == "__main__":

    main()