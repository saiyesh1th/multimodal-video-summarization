from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as _sum, log, lit, desc, row_number, udf
from pyspark.sql.window import Window
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType

def main():
    spark = SparkSession.builder \
        .appName("Phase6_SemanticAnalysis") \
        .config("spark.sql.shuffle.partitions", "20") \
        .getOrCreate()

    # ==========================================
    # 1. LOAD DATA
    # ==========================================
    print("[-] Loading MFCC vectors from HDFS...")
    df = spark.read.parquet("hdfs://namenode:8020/project/features/mfcc_vectors")
    
    # Convert Array<Float> to MLlib DenseVector
    to_vector = udf(lambda x: Vectors.dense(x), VectorUDT())
    df_vec = df.withColumn("features_raw", to_vector(col("mfcc")))

    # ==========================================
    # 2. NORMALIZATION (Optimized)
    # ==========================================
    print("[-] Normalizing Features (StandardScaler)...")
    # FIX 1: withMean=False prevents dense matrix explosion.
    # We rely on withStd=True to handle the scaling differences between coefficients.
    scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled", 
                            withStd=True, withMean=False)
    scaler_model = scaler.fit(df_vec)
    df_scaled = scaler_model.transform(df_vec)

    # ==========================================
    # 3. TOKENIZATION (K-Means Clustering)
    # ==========================================
    print("[-] Training K-Means Model (k=50)...")
    kmeans = KMeans(featuresCol="features_scaled", k=50, seed=42)
    model = kmeans.fit(df_scaled)
    
    # Assign every chunk to a cluster (The 'Acoustic Word')
    df_clustered = model.transform(df_scaled).select(
        col("file"), 
        col("start_time"), 
        col("prediction").alias("token_id")
    )

    # ==========================================
    # 4. TF-IDF SCORING
    # ==========================================
    print("[-] Computing TF-IDF Scores...")
    
    # A. Term Frequency (TF)
    tf_df = df_clustered.groupBy("file", "token_id").agg(count("token_id").alias("tf"))
    
    # B. Document Frequency (DF)
    df_doc = df_clustered.select("file", "token_id").distinct() \
        .groupBy("token_id").agg(count("file").alias("df"))
    
    # C. Total Documents (N)
    total_docs = df_clustered.select("file").distinct().count()
    
    # D. Calculate Score
    ranking_df = tf_df.join(df_doc, on="token_id", how="inner") \
        .withColumn("idf", log(lit(total_docs) / (col("df") + 1))) \
        .withColumn("tfidf_score", col("tf") * col("idf"))

    # Join scores back to chunks
    df_scored = df_clustered.join(ranking_df, on=["file", "token_id"], how="inner")

    # ==========================================
    # 5. GENERATE 30s SUMMARIES
    # ==========================================
    print("[-] Selecting Top Chunks for Summary...")
    
    SUMMARY_DURATION = 30 
    CHUNK_DURATION = 5    
    TOP_K = int(SUMMARY_DURATION / CHUNK_DURATION) 
    
    # Rank by Importance (TF-IDF)
    window_rank = Window.partitionBy("file").orderBy(col("tfidf_score").desc())
    
    df_top_k = df_scored.withColumn("rank", row_number().over(window_rank)) \
        .filter(col("rank") <= TOP_K)
    
    # Sort by Time (Chronological Playback)
    window_sort = Window.partitionBy("file").orderBy("start_time")
    
    final_output = df_top_k.select("file", "start_time", "tfidf_score", "token_id") \
        .withColumn("final_order", row_number().over(window_sort))

    # ==========================================
    # 6. OUTPUT
    # ==========================================
    print("[-] Saving Summaries to HDFS...")
    final_output.coalesce(1).write.mode("overwrite").option("header", "true") \
        .csv("hdfs://namenode:8020/project/output/summaries")
    
    print("[-] Semantic Analysis Complete.")

if __name__ == "__main__":
    main()