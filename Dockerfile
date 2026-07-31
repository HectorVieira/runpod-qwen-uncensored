FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Clone and build llama.cpp
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp
WORKDIR /opt/llama.cpp
RUN cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF \
    && cmake --build build --config Release -j$(nproc) \
    && cp build/bin/llama-server /usr/local/bin/ \
    && chmod +x /usr/local/bin/llama-server

# Create model directory
RUN mkdir -p /models

# Download the GGUF model (Q8_0 quantization)
RUN wget --tries=3 --timeout=120 -q \
    -O /models/qwen3.6-35b-uncensored.gguf \
    "https://huggingface.co/LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V6-GGUF/resolve/main/Hermes3.6-35B-A3B-Uncensored-Genesis-V6-Q8_0.gguf"

# Verify model exists
RUN ls -lh /models/qwen3.6-35b-uncensored.gguf

# Copy handler
COPY handler.py /handler.py
COPY requirements.txt /requirements.txt

# Install Python deps
RUN pip3 install --no-cache-dir -r /requirements.txt

# Expose port for RunPod
ENV HTTP_PORT=8080

# -u for unbuffered output
CMD ["python3", "-u", "/handler.py"]
