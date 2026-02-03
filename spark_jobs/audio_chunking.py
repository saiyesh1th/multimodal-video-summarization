from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, ArrayType
import io
import soundfile as sf
import numpy as np

def process_wav_file(record):
    """
    Input: (hdfs_path, binary_content)
    Output: List of (filename, start_time, sample_array)
    """
    path, binary_content = record
    filename = path.split("/")[-1]
    
    try:
        # 1. Read Bytes into Audio Array
        with io.BytesIO(binary_content) as b:
            data, sr = sf.read(b)
            
        # 2. Define Chunk Size (5 seconds * 16000 Hz)
        CHUNK_DURATION = 5 
        chunk_size = int(CHUNK_DURATION * sr)
        total_samples = len(data)
        chunks = []
        
        # 3. Slice and Dice
        for i in range(0, total_samples, chunk_size):
            segment = data[i : i + chunk_size]
            
            # Pad if short
            if len(segment) < chunk_size:
                padding = np.zeros(chunk_size - len(segment))
                segment = np.concatenate((segment, padding))
                
            start_time = float(i) / sr
            
            # Convert to list for Spark serialization
            chunks.append((filename, start_time, segment.tolist()))
            
        return chunks
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return []

def main():
    spark = SparkSession.builder.appName("Phase4_AudioChunking").getOrCreate()

    # 1. Read from HDFS (Port 8020)
    print("[-] Reading WAVs from HDFS...")
    rdd_raw = spark.sparkContext.binaryFiles("hdfs://namenode:8020/project/audio/*.wav")

    # 2. Map & Flatten
    print("[-] Chunking audio...")
    rdd_chunks = rdd_raw.flatMap(process_wav_file)

    # 3. Define Schema
    schema = StructType([
        StructField("file", StringType(), True),
        StructField("start_time", FloatType(), True),
        StructField("samples", ArrayType(FloatType()), True)
    ])

    # 4. Save to Parquet
    print("[-] Writing to HDFS...")
    df = spark.createDataFrame(rdd_chunks, schema)
    df.write.mode("overwrite").parquet("hdfs://namenode:8020/project/features/audio_chunks")
    
    print("[-] Success. Data stored in /project/features/audio_chunks")

if __name__ == "__main__":
    main()