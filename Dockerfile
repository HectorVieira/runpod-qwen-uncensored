FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Install Ollama manually
RUN curl -L -o /usr/local/bin/ollama https://ollama.com/download/ollama-linux-amd64 \
    && chmod +x /usr/local/bin/ollama

# Install Python deps
RUN python3.10 -m pip install --no-cache-dir runpod requests

# Copy files
COPY handler.py /handler.py
COPY Modelfile /Modelfile

CMD ["python3.10", "-u", "/handler.py"]
