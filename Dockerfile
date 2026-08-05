FROM hungmol/llama.cpp:server-cuda

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Install Python deps
RUN pip3 install --no-cache-dir runpod requests

# Copy handler
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
