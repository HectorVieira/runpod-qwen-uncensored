FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Override entrypoint
ENTRYPOINT []

# Install Ollama binary directly via python (curl/wget may be absent)
RUN python3 - <<'PYEOF'
import urllib.request, os
url = "https://ollama.com/download/ollama-linux-amd64"
print("Downloading Ollama...", flush=True)
urllib.request.urlretrieve(url, "/usr/bin/ollama")
os.chmod("/usr/bin/ollama", 0o755)
print("Done", flush=True)
PYEOF
RUN ollama --version
RUN pip install --no-cache-dir requests runpod

# Model will be loaded from network volume at runtime
ENV OLLAMA_HOST="0.0.0.0:11434"
ENV HF_HUB_OFFLINE="1"

# Copy handler
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
