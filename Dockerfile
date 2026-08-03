FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python deps
RUN pip install --no-cache-dir runpod requests

# Copy files
COPY handler.py /handler.py
COPY Modelfile /Modelfile

CMD ["python3", "-u", "/handler.py"]
