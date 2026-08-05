FROM hungmol/llama.cpp:server-cuda

# Override entrypoint
ENTRYPOINT []

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Find and copy llama-server to /usr/local/bin
RUN find / -name "llama-server" -type f 2>/dev/null | head -5 \
    && cp $(find / -name "llama-server" -type f 2>/dev/null | head -1) /usr/local/bin/llama-server 2>/dev/null \
    && chmod +x /usr/local/bin/llama-server 2>/dev/null || echo "llama-server not found, will check at runtime"

# Install Python deps
RUN pip3 install --no-cache-dir runpod requests

# Copy handler
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
