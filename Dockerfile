# syntax=docker/dockerfile:1
#
# RunPod Serverless worker: uncensored Qwen3.8-27B-OBLITERATED GGUF served by llama.cpp.
#
# Why the official llama.cpp CUDA image instead of Ollama/vLLM:
#   * llama-server mmap()s the GGUF straight off the network volume, so a cold
#     start never has to copy a 17-26 GB blob into ephemeral container storage.
#   * It exposes an OpenAI-compatible /v1/chat/completions with --jinja, which is
#     what tool calling needs.
#   * Using a prebuilt image keeps the GitHub -> RunPod autobuild to a couple of
#     minutes instead of a full CUDA compile.
#
# Pin the build explicitly: `latest` moves daily and would silently change the
# runtime under an autodeploying endpoint. Bump deliberately.
ARG LLAMA_CPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda-b11046
FROM ${LLAMA_CPP_IMAGE}

# The base image's ENTRYPOINT is /app/llama-server; RunPod must run our handler,
# which supervises llama-server as a child process instead.
ENTRYPOINT []
USER root

# The base image ships HEALTHCHECK curl -f http://localhost:8080/health, which is
# wrong for this container in two ways: llama-server is not listening at all
# until the model finishes downloading, and /health answers 503 (not 2xx) while
# the model loads. `curl -f` fails on both, so Docker marks the worker unhealthy
# after ~90s and RunPod recycles it -- an endless restart loop that never lets a
# 25 GB download finish. Readiness is reported through the handler's `status`
# command instead, so the container-level probe is simply disabled.
HEALTHCHECK NONE

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# The CUDA runtime base ships no Python; RunPod's SDK needs it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# runpod      -> serverless runtime
# aiohttp     -> async proxying to llama-server, including SSE passthrough
# requests    -> readiness probing of llama-server
# huggingface_hub[hf_transfer] -> fetches the GGUF once, on first boot
RUN pip3 install --no-cache-dir \
      runpod \
      aiohttp \
      requests \
      "huggingface_hub[hf_transfer]"

# Shared libs and llama-server both live in /app in the base image; ggml loads
# its backend .so files relative to the executable, so keep /app on the path.
ENV LD_LIBRARY_PATH=/app \
    LLAMA_SERVER_BIN=/app/llama-server

# ---------------------------------------------------------------------------
# Runtime configuration. Everything here can be overridden per-endpoint in the
# RunPod console without rebuilding the image.
# ---------------------------------------------------------------------------
# Serverless network volumes are mounted at /runpod-volume (NOT /workspace,
# which is the Pod convention).
ENV MODEL_DIR=/runpod-volume/models \
    MODEL_REPO=OBLITERATUS/Qwen3.8-27B-OBLITERATED \
    MODEL_FILE=Qwen3.8-27B-OBLITERATED-Q6_K.gguf \
    MODEL_ALIAS=qwen3.8-27b-obliterated \
    MODEL_AUTO_DOWNLOAD=1 \
    CTX_SIZE=131072 \
    GPU_LAYERS=99 \
    LOAD_MODE=mmap \
    REPEAT_PENALTY=1.15 \
    REASONING=off \
    LLAMA_HOST=127.0.0.1 \
    LLAMA_PORT=8080 \
    SERVER_START_TIMEOUT=900

COPY handler.py /handler.py
# The GGUF's own template has no tool support; this one does.
COPY chat_template_qwen38.jinja /app/chat_template_qwen38.jinja
ENV CHAT_TEMPLATE_FILE=/app/chat_template_qwen38.jinja

CMD ["python3", "-u", "/handler.py"]
