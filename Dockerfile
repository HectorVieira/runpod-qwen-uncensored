FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Check architecture and install Ollama
RUN ARCH=$(uname -m) && echo "Architecture: $ARCH" && \
    if [ "$ARCH" = "x86_64" ]; then \
        curl -L -o /usr/local/bin/ollama https://ollama.com/download/ollama-linux-amd64 && chmod +x /usr/local/bin/ollama; \
    elif [ "$ARCH" = "aarch64" ]; then \
        curl -L -o /usr/local/bin/ollama https://ollama.com/download/ollama-linux-arm64 && chmod +x /usr/local/bin/ollama; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    /usr/local/bin/ollama --version

# Install Python deps
RUN python3.10 -m pip install --no-cache-dir runpod requests

# Copy files
COPY handler.py /handler.py
COPY Modelfile /Modelfile

CMD ["python3.10", "-u", "/handler.py"]
