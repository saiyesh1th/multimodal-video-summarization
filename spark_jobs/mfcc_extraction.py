from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
import numpy as np

# ==========================================
# CORE MATH KERNEL (Pure NumPy - No SciPy)
# ==========================================
def calculate_mfcc_scratch(signal_list):
    """
    Computes MFCCs using only NumPy (Standard Library).
    Includes Orthonormal DCT scaling for mathematical correctness.
    """
    try:
        signal = np.array(signal_list, dtype=float)
        if len(signal) == 0: return [0.0] * 13

        # Constants
        SAMPLE_RATE = 16000
        PRE_EMPHASIS = 0.97
        FRAME_SIZE = 0.025
        FRAME_STRIDE = 0.01
        N_FILTERS = 26
        N_MFCC = 13
        N_FFT = 512

        # 1. Pre-Emphasis
        emphasized_signal = np.append(signal[0], signal[1:] - PRE_EMPHASIS * signal[:-1])

        # 2. Framing (Fixed: Handle short signals gracefully)
        frame_length = int(round(FRAME_SIZE * SAMPLE_RATE))
        frame_step = int(round(FRAME_STRIDE * SAMPLE_RATE))
        signal_length = len(emphasized_signal)
        
        # FIX 1: Ensure at least 1 frame exists
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
        num_frames = max(1, num_frames)

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized_signal, z)
        
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
                  np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        # 3. Windowing (Hamming)
        frames *= np.hamming(frame_length)

        # 4. FFT & Power Spectrum
        mag_frames = np.absolute(np.fft.rfft(frames, N_FFT))
        pow_frames = ((1.0 / N_FFT) * ((mag_frames) ** 2))

        # 5. Mel Filterbank
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (SAMPLE_RATE / 2) / 700))
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_FILTERS + 2)
        hz_points = (700 * (10**(mel_points / 2595) - 1))
        bin = np.floor((N_FFT + 1) * hz_points / SAMPLE_RATE)

        fbank = np.zeros((N_FILTERS, int(np.floor(N_FFT / 2 + 1))))
        for m in range(1, N_FILTERS + 1):
            f_m_minus = int(bin[m - 1])
            f_m = int(bin[m])
            f_m_plus = int(bin[m + 1])

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)

        # 6. DCT - Manual Implementation with Orthonormal Scaling (FIX 2)
        # Type-II DCT formula with scaling
        num_ceps = N_MFCC
        dct_matrix = np.zeros((num_ceps, N_FILTERS))
        for k in range(num_ceps):
            for n in range(N_FILTERS):
                dct_matrix[k, n] = np.cos(np.pi * k * (2 * n + 1) / (2 * N_FILTERS))
        
        # Apply Ortho-normalization (Matches scipy.fftpack.dct(norm='ortho'))
        dct_matrix[0, :] *= 1.0 / np.sqrt(N_FILTERS)
        dct_matrix[1:, :] *= np.sqrt(2.0 / N_FILTERS)
        
        mfcc = np.dot(filter_banks, dct_matrix.T)
        
        # Average over time to get a single vector for the chunk
        avg_mfcc = np.mean(mfcc, axis=0)
        return avg_mfcc.tolist()

    except Exception as e:
        # Fallback for corrupt chunks to avoid crashing the job
        return [0.0] * 13

def main():
    spark = SparkSession.builder.appName("Phase5_MFCC_Scratch").getOrCreate()

    # 1. Read Chunks
    print("[-] Reading Audio Chunks from HDFS...")
    df = spark.read.parquet("hdfs://namenode:8020/project/features/audio_chunks")

    # 2. Register UDF 
    manual_mfcc_udf = udf(calculate_mfcc_scratch, ArrayType(FloatType()))

    # 3. Apply Transformation
    print("[-] Computing MFCCs (Pure NumPy with Ortho DCT)...")
    df_features = df.withColumn("mfcc", manual_mfcc_udf(df["samples"]))

    # 4. Select Final Columns
    df_final = df_features.select("file", "start_time", "mfcc")

    # 5. Save
    print("[-] Saving MFCC vectors...")
    df_final.write.mode("overwrite").parquet("hdfs://namenode:8020/project/features/mfcc_vectors")
    print("[-] Phase 5 Complete.")

if __name__ == "__main__":
    main()