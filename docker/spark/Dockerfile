# File: docker/spark/Dockerfile

# Switch to the OFFICIAL Apache Spark image
FROM apache/spark:3.5.0

USER root

# Install FFmpeg and build tools
# We use -y and --no-install-recommends to keep it clean
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    python3-dev \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python libraries
# We upgrade pip first to avoid compatibility issues
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    numpy \
    scipy \
    soundfile \
    scikit-learn

# Switch back to the default Spark user (185 is the standard ID for this image)
USER 185