FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Override entrypoint
ENTRYPOINT []

# Install Ollama binary directly (no systemd, robust in containers)
RUN curl -L -o /usr/bin/ollama https://ollama.com/download/ollama-linux-amd64 && \
    chmod +x /usr/bin/ollama && \
    ollama --version
RUN pip install --no-cache-dir requests runpod

# Model will be loaded from network volume at runtime
ENV OLLAMA_HOST="0.0.0.0:11434"
ENV HF_HUB_OFFLINE="1"

# Copy handler
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
