# Multimodal Video Summarization

A Big Data and Deep Learning pipeline designed to process videos, extract features, and generate abstractive summaries. This project leverages Hadoop HDFS for distributed storage, Apache Spark for distributed data processing, and PyTorch for custom NLP summarization models.

## Architecture

The project relies on a containerized Big Data cluster managed via `docker-compose`.

- **Hadoop (HDFS)**: `namenode` and `datanode` containers handle distributed file storage.
- **Apache Spark**: A `spark` master and `spark-worker` architecture handles distributed feature extraction, transcription, and inference.
- **MongoDB**: Used as a NoSQL datastore for metadata and intermediate records.
- **Model Training / NLP**: Custom PyTorch code implementing a Seq2Seq Transformer for abstractive summarization. Transcription is handled via Whisper ASR within Spark jobs.

## Prerequisites

- **Docker & Docker Compose**: To run the Hadoop/Spark cluster.
- **Python 3.10+**: For local model training and testing.
- **PowerShell**: Used for pipeline orchestration scripts.

*Note: Ensure you have sufficient RAM and Disk space, as Whisper models, Spark workers, and HDFS nodes require substantial resources.*

## Project Structure

- `data/`: Contains raw videos, extracted audio, and datasets.
- `docker/`: Custom Dockerfiles (e.g., Spark with required Python dependencies).
- `docker-compose.yml`: Defines the local Big Data cluster.
- `ingestion/`: Scripts for downloading datasets (e.g., HowTo100M).
- `model_training/`: PyTorch implementation of the custom summarization Transformer, tokenizers, and dataset loaders.
- `spark_jobs/`: PySpark scripts for distributed processing (audio chunking, transcription, visual extraction, summarization).
- `.ps1` scripts: PowerShell scripts to orchestrate the multi-phase pipeline.

## Pipeline Execution

The project is divided into several orchestrated phases.

### 1. Data Processing and Feature Extraction

Orchestrated using the `run_phase*.ps1` scripts. They handle moving data between the host, Spark containers, and HDFS.

- **Phase 3 & 4** (`run_phase3_4.ps1`): Audio preprocessing. Converts videos to WAV, uploads to HDFS, and submits a Spark job (`audio_chunking.py`) to chunk the audio.
- **Phase 4.5** (`run_phase4_5.ps1`): Audio transcription. Runs a distributed Spark job (`transcription.py`) using Whisper ASR to generate transcripts from audio chunks, then exports the results to local CSV.

### 2. Custom NLP Summarization Pipeline

Once transcripts are generated, the NLP pipeline (`nlp_phase*.ps1`) trains and applies the summarization model.

- **Phase 1** (`nlp_phase1.ps1`): Train tokenizer (`train_tokenizer.py`) on the dataset.
- **Phase 2** (`nlp_phase2.ps1`): PyTorch DataLoader verification.
- **Phase 3** (`nlp_phase3.ps1`): Train the PyTorch Seq2Seq Summarization Model.
- **Phase 4** (`nlp_phase4.ps1`): Run model evaluation / local inference.
- **Phase 5** (`nlp_phase5.ps1`): Distributed PySpark Summarization. Applies the trained summarization model across the Hadoop cluster to process transcripts at scale (`spark_summarization.py`).
- **Phase 6** (`nlp_phase6_display.ps1`): Display generative output results.

## Setup Instructions

1. **Start the Cluster**:
   ```powershell
   docker-compose up -d
   ```
   *Wait for the cluster to initialize and HDFS to exit safe mode.*

2. **Prepare Environment**:
   Create a virtual environment and install dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt # (If available) or install PyTorch, PySpark, Transformers, etc.
   ```

3. **Run Pipeline**:
   Execute the PowerShell scripts sequentially, starting from data ingestion/preprocessing to the final NLP phases.

## Notes
- To view Spark UI: http://localhost:8080 or http://localhost:4040
- To view Hadoop Namenode UI: http://localhost:9870
