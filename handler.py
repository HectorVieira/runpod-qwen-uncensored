"""
RunPod Serverless Handler for Qwen3.6-35B-A3B Uncensored (Ollama)
Uses network volume at /runpod-volume for persistent model storage
"""
import runpod
import subprocess
import time
import json
import requests
import signal
import sys
import os
import threading

# Configuration
MODEL_NAME = "qwen3.6-35b-uncensored"
MODEL_PATH = "/runpod-volume/models/qwen3.6-35b-uncensored.gguf"
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
MODFILE_PATH = "/Modelfile"

server_process = None


def start_ollama():
    """Start Ollama server in background."""
    global server_process

    print("Starting Ollama server...", flush=True)

    server_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Read output in real-time
    def read_output():
        for line in server_process.stdout:
            print(f"[ollama] {line.strip()}", flush=True)

    t = threading.Thread(target=read_output, daemon=True)
    t.start()

    # Wait for Ollama to be ready
    print("Waiting for Ollama to be ready...", flush=True)
    for i in range(60):
        try:
            r = requests.get(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags", timeout=5)
            if r.status_code == 200:
                print(f"Ollama ready after {i+1}s", flush=True)
                return True
        except:
            pass
        time.sleep(1)

    print("Ollama failed to start within 60s", flush=True)
    return False


def download_model_if_needed():
    """Download model to volume if not present."""
    print(f"Checking model at {MODEL_PATH}...", flush=True)

    # Check if model already exists on volume
    if os.path.exists(MODEL_PATH):
        size_gb = os.path.getsize(MODEL_PATH) / (1024**3)
        print(f"Model already exists ({size_gb:.1f} GB)", flush=True)
        return True

    # Create models directory on volume
    os.makedirs("/runpod-volume/models", exist_ok=True)

    print("Model not found. Downloading APEX (23.9GB)...", flush=True)

    # Try APEX first (smaller, fits in 32GB)
    urls = [
        "https://huggingface.co/LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V6-GGUF/resolve/main/Hermes3.6-35B-A3B-Uncensored-Genesis-V6-APEX.gguf",
        "https://huggingface.co/LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V6-GGUF/resolve/main/Hermes3.6-35B-A3B-Uncensored-Genesis-V6-APEX-Compact.gguf",
    ]

    for url in urls:
        filename = url.split("/")[-1]
        print(f"Trying {filename}...", flush=True)

        try:
            result = subprocess.run(
                ["wget", "--tries=3", "--timeout=300", "-q", "-O", MODEL_PATH, url],
                capture_output=True,
                text=True,
                timeout=1800  # 30 min timeout
            )

            if result.returncode == 0 and os.path.exists(MODEL_PATH):
                size_gb = os.path.getsize(MODEL_PATH) / (1024**3)
                print(f"Downloaded {filename} ({size_gb:.1f} GB)", flush=True)
                return True
            else:
                print(f"Download failed: {result.stderr}", flush=True)
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)

        except Exception as e:
            print(f"Error: {e}", flush=True)
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)

    print("Failed to download model", flush=True)
    return False


def create_ollama_model():
    """Create model in Ollama from GGUF file."""
    print(f"Creating Ollama model {MODEL_NAME}...", flush=True)

    # Check if model already exists
    try:
        r = requests.get(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags", timeout=10)
        models = [m['name'] for m in r.json().get('models', [])]

        if MODEL_NAME in models or f"{MODEL_NAME}:latest" in models:
            print(f"Model {MODEL_NAME} already exists in Ollama", flush=True)
            return True
    except:
        pass

    # Read Modelfile
    with open(MODFILE_PATH, 'r') as f:
        modelfile_content = f.read()

    # Create model
    try:
        r = requests.post(
            f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/create",
            json={
                "name": MODEL_NAME,
                "modelfile": modelfile_content
            },
            timeout=600  # 10 minutes
        )

        if r.status_code == 200:
            print(f"Model {MODEL_NAME} created successfully", flush=True)
            return True
        else:
            print(f"Failed to create model: {r.text}", flush=True)
            return False

    except Exception as e:
        print(f"Error creating model: {e}", flush=True)
        return False


def handler(job):
    """RunPod serverless handler."""
    job_input = job.get("input", {})

    messages = job_input.get("messages", [])
    prompt = job_input.get("prompt", "")
    max_tokens = job_input.get("max_tokens", 1024)
    temperature = job_input.get("temperature", 0.7)
    top_p = job_input.get("top_p", 0.9)
    stream = job_input.get("stream", False)

    if not messages and prompt:
        messages = [{"role": "user", "content": prompt}]

    if not messages:
        return {"error": "No messages or prompt provided"}

    try:
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        }

        print(f"Request: messages={len(messages)}, max_tokens={max_tokens}", flush=True)

        r = requests.post(
            f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat",
            json=payload,
            timeout=300
        )

        if r.status_code != 200:
            print(f"Ollama error {r.status_code}: {r.text[:200]}", flush=True)
            return {"error": f"Ollama returned {r.status_code}: {r.text}"}

        result = r.json()

        # Convert to OpenAI format
        choice = result.get("message", {})
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {
                    "role": choice.get("role", "assistant"),
                    "content": choice.get("content", "")
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "completion_tokens": result.get("eval_count", 0),
                "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
            }
        }

    except Exception as e:
        print(f"Handler error: {e}", flush=True)
        return {"error": str(e)}


def cleanup(signum, frame):
    """Cleanup on shutdown."""
    global server_process
    if server_process:
        server_process.terminate()
        server_process.wait(timeout=10)
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    print("=== RunPod Ollama Handler Starting ===", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Model path: {MODEL_PATH}", flush=True)

    # Check volume exists
    if not os.path.exists("/runpod-volume"):
        print("ERROR: /runpod-volume not found! Attach network volume.", flush=True)
        sys.exit(1)

    # Create models directory
    os.makedirs("/runpod-volume/models", exist_ok=True)

    # Download model if needed
    if not download_model_if_needed():
        print("Failed to download model", flush=True)
        sys.exit(1)

    # Start Ollama
    if start_ollama():
        # Create Ollama model from GGUF
        if create_ollama_model():
            print("Model ready, entering RunPod handler loop...", flush=True)
            runpod.serverless.start({"handler": handler})
        else:
            print("Failed to create Ollama model", flush=True)
            sys.exit(1)
    else:
        print("Failed to start Ollama", flush=True)
        sys.exit(1)
