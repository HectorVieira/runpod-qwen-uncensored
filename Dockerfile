FROM ghcr.io/ggml-org/llama.cpp:full-cuda13

# Override entrypoint
ENTRYPOINT []

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip3 install --break-system-packages --no-cache-dir runpod requests

# Copy handler
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
