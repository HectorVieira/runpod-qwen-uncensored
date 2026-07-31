FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# Create model directory
RUN mkdir -p /models

# Download the GGUF model (Q4_K_M - smaller, faster startup)
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
RUN pip install --no-cache-dir -r /requirements.txt

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# -u for unbuffered output
CMD ["python3", "-u", "/handler.py"]
