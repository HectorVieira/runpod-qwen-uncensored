FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y curl wget && rm -rf /var/lib/apt/lists/*

# Download pre-built llama-server (CUDA)
RUN wget -q https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-b10201-bin-ubuntu-x64.tar.gz -O /tmp/llama.tar.gz \
    && tar -xzf /tmp/llama.tar.gz -C /tmp/ \
    && cp /tmp/llama-b10201-bin-ubuntu-x64/bin/llama-server /usr/local/bin/ \
    && chmod +x /usr/local/bin/llama-server \
    && rm -rf /tmp/llama* \
    && /usr/local/bin/llama-server --version

# Install Python deps
RUN python3.10 -m pip install --no-cache-dir runpod requests

# Copy handler
COPY handler.py /handler.py

CMD ["python3.10", "-u", "/handler.py"]
