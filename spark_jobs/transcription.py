from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import io
import soundfile as sf
import numpy as np
import torch
import whisper
import os

def transcribe_audio_partition(iterator):
    """
    Process a partition of audio files using a local Whisper model.
    Loading the model inside the partition allows us to load it once per executor/task 
    rather than for every single file.
    """
    # Load model once per partition/executor to save overhead
    # 'tiny' or 'base' are recommended for CPU inference
    try:
        # FIX: Redirect Whisper cache to /tmp because 'spark' user cannot write to /home/spark
        os.environ["XDG_CACHE_HOME"] = "/tmp"
        model = whisper.load_model("base")
    except Exception as e:
        # Fallback or error logging
        yield ("ERROR_LOADING_MODEL", 0.0, str(e))
        return

    for record in iterator:
        path, binary_content = record
        filename = path.split("/")[-1]
        
        try:
            # 1. Read Audio from Bytes
            with io.BytesIO(binary_content) as b:
                # Whisper expects 16kHz audio
                data, sr = sf.read(b)
            
            # Simple assumption: Input is mono 16kHz from convert.sh
            # If stereo or different, whisper usually handles raw numpy array well if it's float32
            
            result = model.transcribe(data.astype(np.float32))
            text = result['text'].strip()
            
            yield (filename, 0.0, text) 
            
        except Exception as e:
            yield (filename, 0.0, f"ERROR: {str(e)}")

def main():
    spark = SparkSession.builder.appName("Phase4.5_AudioTranscription").getOrCreate()

    # 1. Read from HDFS
    print("[-] Reading WAVs from HDFS...")
    rdd_raw = spark.sparkContext.binaryFiles("hdfs://namenode:8020/project/audio/*.wav")

    # 2. MapPartitions to Transcribe
    print("[-] Transcribing with Whisper...")
    # mapPartitions is more efficient for heavy model loading
    rdd_transcripts = rdd_raw.mapPartitions(transcribe_audio_partition)

    # 3. Define Schema for DataFrame
    # (schema matching expected output structure)
    schema = StructType([
        StructField("file", StringType(), True),
        StructField("start_time", FloatType(), True), 
        StructField("text", StringType(), True)
    ])

    # 4. Save to HDFS
    print("[-] Writing transcripts to HDFS...")
    df = spark.createDataFrame(rdd_transcripts, schema)
    
    # Save as CSV for inspection
    df.write.mode("overwrite").option("header", "true").csv("hdfs://namenode:8020/project/transcripts")
    
    print("[-] Success. Transcripts stored in /project/transcripts")

if __name__ == "__main__":
    main()
