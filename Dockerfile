FROM hungmol/llama.cpp:server-cuda

# Install Python deps
RUN pip install --no-cache-dir runpod requests

# Copy handler
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
