FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y curl wget build-essential cmake && rm -rf /var/lib/apt/lists/*

# Build llama.cpp with CUDA
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp \
    && cd /opt/llama.cpp \
    && cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF \
    && cmake --build build --config Release -j$(nproc) \
    && cp build/bin/llama-server /usr/local/bin/ \
    && chmod +x /usr/local/bin/llama-server

# Verify llama-server works
RUN /usr/local/bin/llama-server --help | head -5

# Install Python deps
RUN python3.10 -m pip install --no-cache-dir runpod requests fastapi uvicorn

# Copy files
COPY handler.py /handler.py

CMD ["python3.10", "-u", "/handler.py"]
