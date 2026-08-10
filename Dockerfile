FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Override entrypoint
ENTRYPOINT []

# Install vLLM and dependencies
RUN pip install --no-cache-dir vllm requests runpod

# Model will be loaded from network volume
ENV VLLM_MODEL_PATH="/workspace/models/vllm-model/"
ENV HF_HUB_OFFLINE="1"

# Copy handler
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
