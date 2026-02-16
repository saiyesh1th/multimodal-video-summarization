from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as _sum, log, lit, udf
from pyspark.sql.window import Window
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark import StorageLevel
import math

def main():
    spark = SparkSession.builder \
        .appName("Phase6_SemanticScoring_Elite") \
        .config("spark.sql.shuffle.partitions", "20") \
        .getOrCreate()

    # ==========================================
    # 1. LOAD & PERSIST BEFORE FIT
    # ==========================================
    print("[-] Loading MFCC features...")
    df = spark.read.parquet("hdfs://namenode:8020/project/features/mfcc_vectors")
    
    to_vector = udf(lambda x: Vectors.dense(x), VectorUDT())
    # OPTIMIZATION: Persist here to avoid re-running UDF during fit() and transform()
    df_vec = df.withColumn("features_raw", to_vector(col("mfcc"))).persist(StorageLevel.MEMORY_AND_DISK)

    # ==========================================
    # 2. SCALING (fit triggers one scan, transform triggers another)
    # ==========================================
    print("[-] Scaling features...")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled", 
                            withStd=True, withMean=False)
    scaler_model = scaler.fit(df_vec) # Uses persisted df_vec
    df_scaled = scaler_model.transform(df_vec).persist(StorageLevel.MEMORY_AND_DISK)
    
    # Cleanup df_vec as it's no longer needed for later stages
    df_vec.unpersist()

    # ==========================================
    # 3. ADAPTIVE K-MEANS
    # ==========================================
    total_chunks = df_scaled.count()
    k_value = min(max(int(math.sqrt(total_chunks / 2)), 20), 80)
    print(f"[-] Clustering with adaptive K={k_value}...")
    
    kmeans = KMeans(featuresCol="features_scaled", k=k_value, seed=42)
    model = kmeans.fit(df_scaled)
    
    df_clustered = model.transform(df_scaled).select(
        col("file"), 
        col("start_time"), 
        col("mfcc"),
        col("prediction").alias("token_id")
    ).persist(StorageLevel.MEMORY_AND_DISK)

    # ==========================================
    # 4. TF-IDF SCORING (Defensible Standard Version)
    # ==========================================
    print("[-] Computing Global TF-IDF...")
    
    tf_df = df_clustered.groupBy("file", "token_id").agg(count("token_id").alias("tf"))
    df_doc = df_clustered.select("file", "token_id").distinct().groupBy("token_id").agg(count("file").alias("df"))
    total_docs = df_clustered.select("file").distinct().count()
    
    # We stick with the standard log(N/df+1) for simplicity and defense
    ranking_weights = tf_df.join(df_doc, on="token_id", how="inner") \
        .withColumn("idf", log(lit(total_docs) / (col("df") + 1))) \
        .withColumn("tfidf_score", col("tf") * col("idf"))

    final_output = df_clustered.join(ranking_weights.select("file", "token_id", "tfidf_score"), 
                                    on=["file", "token_id"], how="inner") \
        .select("file", "start_time", "tfidf_score", "token_id", "mfcc")

    # ==========================================
    # 5. OUTPUT (Interoperable CSV)
    # ==========================================

    # ==========================================
    # 5. OUTPUT (Switching to Parquet for Array Support)
    # ==========================================
    print("[-] Saving final semantic map as Parquet...")
    
    # We remove .option("header", "true") because Parquet handles schema internally
    final_output.write.mode("overwrite") \
        .parquet("hdfs://namenode:8020/project/output/summaries")
    
    df_scaled.unpersist()
    df_clustered.unpersist()
    print("[-] Phase 6 Complete: Parquet output generated.")

    # print("[-] Saving final semantic map...")
    # # CSV is chosen here for easier loading in Phase 6.5's Pandas logic
    # final_output.write.mode("overwrite").option("header", "true") \
    #     .csv("hdfs://namenode:8020/project/output/summaries")
    
    # df_scaled.unpersist()
    # df_clustered.unpersist()
    # print("[-] Phase 6 Complete: Results are production-optimized.")

if __name__ == "__main__":
    main()