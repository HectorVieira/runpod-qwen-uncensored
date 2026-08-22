FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Override entrypoint
ENTRYPOINT []

# Install Ollama (serves GGUF natively) and deps
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN pip install --no-cache-dir requests runpod

# Model will be loaded from network volume at runtime
ENV OLLAMA_HOST="0.0.0.0:11434"
ENV HF_HUB_OFFLINE="1"

# Copy handler
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
