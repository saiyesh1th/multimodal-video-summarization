from pyspark.sql import SparkSession
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------
# PATHS
# ---------------------------------------

SCORES_PATH = "hdfs://namenode:8020/project/output/summaries_smoothed"
SEGMENTS_PATH = "/opt/data/final_segments.csv"
OUTPUT_DIR = "/opt/data/plots"


# ---------------------------------------
# MAIN
# ---------------------------------------

def main():

    spark = SparkSession.builder.appName("SummaryVisualization").getOrCreate()

    scores_df = spark.read.option("header","true").csv(SCORES_PATH).toPandas()

    segments_df = pd.read_csv(SEGMENTS_PATH)

    scores_df["start_time"] = scores_df["start_time"].astype(float)
    scores_df["tfidf_score"] = scores_df["tfidf_score"].astype(float)

    videos = segments_df["video"].unique()

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for video in videos:

        base = video.split(".")[0]

        video_scores = scores_df[scores_df["file"].str.contains(base)]

        video_segments = segments_df[segments_df["video"] == video]

        plt.figure(figsize=(14,4))

        # -----------------------------------
        # IMPORTANCE CURVE
        # -----------------------------------

        plt.plot(
            video_scores["start_time"],
            video_scores["tfidf_score"],
            color="blue",
            label="Importance Score",
            linewidth=2
        )

        # -----------------------------------
        # SUMMARY SEGMENTS
        # -----------------------------------

        for _, row in video_segments.iterrows():

            plt.axvspan(
                row["start_time"],
                row["end_time"],
                color="red",
                alpha=0.35
            )

        plt.title(f"Video Summary Timeline — {video}")

        plt.xlabel("Time (seconds)")
        plt.ylabel("Importance Score")

        plt.legend()

        plt.tight_layout()

        save_path = f"{OUTPUT_DIR}/{video}_timeline.png"

        plt.savefig(save_path)

        plt.close()

        print(f"Saved plot → {save_path}")


if __name__ == "__main__":
    main()