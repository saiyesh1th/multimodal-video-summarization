from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, lead, coalesce, lit
from pyspark.sql.window import Window

def main():
    spark = SparkSession.builder \
        .appName("Phase6.5_TimeSeriesSmoothing") \
        .getOrCreate()

    # 1. Load Raw Scores (From Phase 6)
    print("[-] Reading Raw TF-IDF Scores from HDFS...")
    # FIX: Port 8020
    df = spark.read.option("header", "true") \
        .csv("hdfs://namenode:8020/project/output/summaries") \
        .select(
            col("file"), 
            col("start_time").cast("float"), 
            col("tfidf_score").cast("float"),
            col("token_id")
        )

    # 2. Define Time Window
    # Partition by file, Order by time
    window_spec = Window.partitionBy("file").orderBy("start_time")

    # 3. Convolution (Gaussian Kernel [0.25, 0.5, 0.25])
    print("[-] Applying Gaussian Smoothing Kernel...")
    
    # Get neighbors
    # COALESCE FIX: If prev_score is NULL (start of video), use current score. 
    # This prevents the score from artificially dropping to 0 at the edges.
    df_neighbors = df.withColumn("prev_score", coalesce(lag("tfidf_score", 1).over(window_spec), col("tfidf_score"))) \
                     .withColumn("next_score", coalesce(lead("tfidf_score", 1).over(window_spec), col("tfidf_score")))

    # Apply Math: 25% Prev + 50% Curr + 25% Next
    df_smoothed = df_neighbors.withColumn(
        "smoothed_score",
        (col("prev_score") * 0.25) + 
        (col("tfidf_score") * 0.50) + 
        (col("next_score") * 0.25)
    )

    # 4. Save Smoothed Scores
    print("[-] Saving Smoothed Data to HDFS...")
    
    # We rename 'smoothed_score' back to 'tfidf_score' so Phase 7 works without changes
    output_df = df_smoothed.select(
        "file", "start_time", "token_id", 
        col("smoothed_score").alias("tfidf_score") 
    )
    
    output_df.write.mode("overwrite").option("header", "true") \
        .csv("hdfs://namenode:8020/project/output/summaries_smoothed")
        
    print("[-] Time Series Smoothing Complete.")

if __name__ == "__main__":
    main()