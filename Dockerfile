FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install minimal dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Install llama.cpp via pre-built binary
RUN wget -q https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-server-cuda -O /usr/local/bin/llama-server \
    && chmod +x /usr/local/bin/llama-server

# Create model directory
RUN mkdir -p /models

# Download the GGUF model (Q4_K_M - smaller, faster)
RUN wget --tries=3 --timeout=120 -q \
    -O /models/qwen3.6-35b-uncensored.gguf \
    "https://huggingface.co/LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V6-GGUF/resolve/main/Hermes3.6-35B-A3B-Uncensored-Genesis-V6-APEX.gguf" \
    || wget --tries=3 --timeout=120 -q \
    -O /models/qwen3.6-35b-uncensored.gguf \
    "https://huggingface.co/LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V6-GGUF/resolve/main/Hermes3.6-35B-A3B-Uncensored-Genesis-V6-Q8_0.gguf"

# Verify model exists
RUN ls -lh /models/qwen3.6-35b-uncensored.gguf

# Copy handler
COPY handler.py /handler.py
COPY requirements.txt /requirements.txt

# Install Python deps
RUN pip3 install --no-cache-dir -r /requirements.txt

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# -u for unbuffered output
CMD ["python3", "-u", "/handler.py"]
