from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, lead, expr, percentile_approx
from pyspark.sql.window import Window

def main():
    spark = SparkSession.builder.appName("Phase6.6_Budgetary_Scissors").getOrCreate()
    spark.catalog.clearCache()

    df = spark.read.option("header", "true").csv("hdfs://namenode:8020/project/output/summaries_hybrid") \
              .withColumn("start_time", col("start_time").cast("double")) \
              .withColumn("score", col("tfidf_score").cast("double"))

    window_time = Window.partitionBy("file").orderBy("start_time")
    
    # Smooth the scores to avoid "jittery" single-chunk selections
    df_smooth = df.withColumn("p1", lag("score", 1, 0.0).over(window_time)) \
                  .withColumn("n1", lead("score", 1, 0.0).over(window_time)) \
                  .withColumn("smooth_score", (col("p1") * 0.2) + (col("score") * 0.6) + (col("n1") * 0.2))

    # FORCED VALLEYS: Only keep chunks in the top 40% of the video's scores.
    window_file = Window.partitionBy("file")
    df_threshold = df_smooth.withColumn("threshold", percentile_approx("smooth_score", 0.60).over(window_file))
    
    # We filter for high-quality peaks only
    final_chunks = df_threshold.filter(col("smooth_score") >= col("threshold")) \
                               .filter(col("smooth_score") > 0.35)

    final_chunks.select("file", "start_time", col("smooth_score").alias("tfidf_score")) \
                .write.mode("overwrite").option("header", "true").csv("hdfs://namenode:8020/project/output/summaries_smoothed")

    print("[-] Phase 6.6 Complete: High-pass budgetary filter applied.")

if __name__ == "__main__":
    main()