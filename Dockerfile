FROM ollama/ollama:latest

# Create model directory
RUN mkdir -p /models

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && pip3 install --break-system-packages --no-cache-dir runpod requests

# Download the GGUF model (APEX - 23.9GB, faster than Q8_0)
RUN wget --tries=3 --timeout=300 -q \
    -O /models/qwen3.6-35b-uncensored.gguf \
    "https://huggingface.co/LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V6-GGUF/resolve/main/Hermes3.6-35B-A3B-Uncensored-Genesis-V6-APEX.gguf"

# Verify model exists
RUN ls -lh /models/qwen3.6-35b-uncensored.gguf

# Copy handler
COPY handler.py /handler.py
COPY Modelfile /Modelfile

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:11434/api/tags || exit 1

# -u for unbuffered output
CMD ["python3", "-u", "/handler.py"]
