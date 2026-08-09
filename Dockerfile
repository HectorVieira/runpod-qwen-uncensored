FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Override entrypoint
ENTRYPOINT []

# Install vLLM and dependencies
RUN pip3 install --break-system-packages --no-cache-dir vllm requests runpod

# Download model on startup
ENV HF_TOKEN=""
ENV MODEL_NAME="symrex/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V6-dequantized-oQ8e-mtp"

# Copy handler
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
