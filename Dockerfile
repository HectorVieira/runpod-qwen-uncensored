FROM ollama/ollama:latest

# Create model directory
RUN mkdir -p /models

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Copy handler
COPY handler.py /handler.py
COPY requirements.txt /requirements.txt
COPY Modelfile /Modelfile

# Install Python deps
RUN pip3 install --no-cache-dir -r /requirements.txt

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:11434/api/tags || exit 1

# -u for unbuffered output
CMD ["python3", "-u", "/handler.py"]
