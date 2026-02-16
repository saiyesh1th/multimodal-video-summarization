from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark import StorageLevel
import pandas as pd
import numpy as np

# ==========================================
# 1. NARRATIVE-AWARE MATH KERNEL
# ==========================================
def compute_linear_attention(pdf):
    pdf = pdf.sort_values("start_time")
    pdf = pdf.dropna(subset=['mfcc'])
    n = len(pdf)
    
    if n < 5: 
        pdf['hybrid_score'] = pdf['tfidf_score']
        return pdf[['file', 'start_time', 'hybrid_score']]

    timestamps = pdf['start_time'].values
    duration = timestamps[-1] - timestamps[0]
    ALPHA = np.clip(0.55 - 0.0015 * duration, 0.35, 0.55)
    
    TAU, WINDOW_SEC = 45.0, 90.0
    features = np.vstack(pdf['mfcc'].values).astype(float)
    feats = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
    
    # --- DYNAMIC PEAK ANCHOR (Refinement 1) ---
    intro_search_end = max(1, int(n * 0.15))
    intro_scores = pdf['tfidf_score'].values[:intro_search_end]
    
    # If intro is pure fluff (low scores), skip it entirely for the anchor
    if np.max(intro_scores) < 0.05:
        intro_peak_idx = intro_search_end
    else:
        intro_peak_idx = np.argmax(intro_scores)

    global_indices = [intro_peak_idx, n // 2, n - 1]
    global_feats = feats[global_indices]
    raw_attention = np.zeros(n)

    # O(N) Sliding Window
    left = 0
    right = 0
    for i in range(n):
        while timestamps[i] - timestamps[left] > WINDOW_SEC: left += 1
        while right + 1 < n and timestamps[right + 1] - timestamps[i] <= WINDOW_SEC: right += 1
        local_sim = np.dot(feats[i], feats[left : right + 1].T)
        local_decay = np.exp(-np.abs(timestamps[i] - timestamps[left : right + 1]) / TAU)
        raw_attention[i] = np.sum(local_sim * local_decay) + np.mean(np.dot(feats[i], global_feats.T))

    def robust_norm(arr):
        mi, ma = np.min(arr), np.max(arr)
        return 0.1 + (0.9 * (arr - mi) / (ma - mi)) if ma != mi else np.full(len(arr), 0.5)

    tfidf_part = ALPHA * robust_norm(pdf['tfidf_score'].values)
    attn_part = (1 - ALPHA) * robust_norm(raw_attention)
    combined_score = tfidf_part + attn_part

    # --- INTRO SQUELCH (Refinement 2: Soft Ramp) ---
    penalty_mask = np.ones_like(timestamps)
    intro_limit = min(30.0, 0.1 * duration)
    intro_mask = timestamps < intro_limit
    # intro_mask = timestamps < 30.0
    num_intro_chunks = np.sum(intro_mask)
    if num_intro_chunks > 0:
        penalty_mask[intro_mask] = np.linspace(0.7, 1.0, num_intro_chunks)

    pdf['hybrid_score'] = combined_score * penalty_mask
    return pdf[['file', 'start_time', 'hybrid_score']]

# ==========================================
# 2. SPARK ORCHESTRATION (Parquet Aligned)
# ==========================================
def main():
    spark = SparkSession.builder.appName("Phase6.5_ParquetAttention").getOrCreate()

    print("[-] Loading All Features from Parquet...")
    # Now reading from the Parquet output of your elite Phase 6
    input_df = spark.read.parquet("hdfs://namenode:8020/project/output/summaries") \
                    .withColumn("start_time", col("start_time").cast("double")) \
                    .withColumn("tfidf_score", col("tfidf_score").cast("double")) \
                    .persist(StorageLevel.MEMORY_AND_DISK)

    print("[-] Computing Hybrid Temporal Attention via applyInPandas...")
    
    schema = StructType([
        StructField("file", StringType(), True),
        StructField("start_time", DoubleType(), True),
        StructField("hybrid_score", DoubleType(), True)
    ])

    df_hybrid = input_df.groupBy("file").applyInPandas(
        compute_linear_attention,
        schema=schema
    ).persist(StorageLevel.MEMORY_AND_DISK)

    print("[-] Saving Hybrid Scores to Summaries_Hybrid...")
    # Saving as CSV is fine here because we are only saving 3 flat columns (no arrays)
    df_hybrid.select("file", "start_time", col("hybrid_score").alias("tfidf_score")) \
             .write.mode("overwrite").option("header", "true") \
             .csv("hdfs://namenode:8020/project/output/summaries_hybrid")
            
    input_df.unpersist()
    df_hybrid.unpersist()
    print("[-] Phase 6.5 Complete.")

if __name__ == "__main__":
    main()