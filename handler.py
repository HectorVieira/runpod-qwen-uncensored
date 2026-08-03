"""
RunPod Serverless Handler - Qwen3.6-35B-A3B Uncensored (Ollama)
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

MODEL_NAME = "qwen3.6-35b-uncensored"
MODEL_DIR = "/runpod-volume/models"
MODEL_PATH = f"{MODEL_DIR}/qwen3.6-35b-uncensored.gguf"
OLLAMA_HOST = "0.0.0.0"
OLLAMA_PORT = 11434

server_process = None

MODEL_URLS = [
    "https://huggingface.co/LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V6-GGUF/resolve/main/Hermes3.6-35B-A3B-Uncensored-Genesis-V6-APEX.gguf",
    "https://huggingface.co/LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V6-GGUF/resolve/main/Hermes3.6-35B-A3B-Uncensored-Genesis-V6-APEX-Compact.gguf",
]


def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for url in MODEL_URLS:
        filename = url.split("/")[-1]
        print(f"Downloading {filename}...", flush=True)
        try:
            result = subprocess.run(
                ["wget", "--tries=3", "--timeout=300", "-q", "-O", MODEL_PATH, url],
                capture_output=True, text=True, timeout=3600
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
    return False


def handler(job):
    job_input = job.get("input", {})
    messages = job_input.get("messages", [])
    prompt = job_input.get("prompt", "")
    if not messages and prompt:
        messages = [{"role": "user", "content": prompt}]
    if not messages:
        return {"error": "No messages or prompt provided"}
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": job_input.get("max_tokens", 1024),
                "temperature": job_input.get("temperature", 0.7),
            }
        }
        r = requests.post(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat", json=payload, timeout=300)
        if r.status_code != 200:
            return {"error": f"Ollama {r.status_code}: {r.text[:200]}"}
        result = r.json()
        choice = result.get("message", {})
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [{"index": 0, "message": {"role": choice.get("role", "assistant"), "content": choice.get("content", "")}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": result.get("prompt_eval_count", 0), "completion_tokens": result.get("eval_count", 0), "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)}
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("=== Handler Starting ===", flush=True)

    if not os.path.exists("/runpod-volume"):
        print("ERROR: /runpod-volume not found!", flush=True)
        sys.exit(1)

    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        size_gb = os.path.getsize(MODEL_PATH) / (1024**3)
        print(f"Model exists: {size_gb:.1f} GB", flush=True)
    else:
        print("Model not found. Downloading...", flush=True)
        if not download_model():
            print("Failed to download model", flush=True)
            sys.exit(1)

    print("Starting Ollama...", flush=True)
    server_process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def read_output():
        for line in server_process.stdout:
            print(f"[ollama] {line.strip()}", flush=True)

    t = threading.Thread(target=read_output, daemon=True)
    t.start()

    print("Waiting for Ollama...", flush=True)
    for i in range(120):
        try:
            r = requests.get(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags", timeout=5)
            if r.status_code == 200:
                print(f"Ollama ready after {i+1}s", flush=True)
                break
        except:
            pass
        time.sleep(1)
    else:
        print("Ollama failed to start", flush=True)
        sys.exit(1)

    r = requests.get(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags", timeout=10)
    models = [m['name'] for m in r.json().get('models', [])]
    print(f"Existing models: {models}", flush=True)

    if MODEL_NAME not in models and f"{MODEL_NAME}:latest" not in models:
        print(f"Creating model {MODEL_NAME}...", flush=True)
        modelfile = f"FROM {MODEL_PATH}\nPARAMETER num_ctx 8192\nPARAMETER num_gpu 99"
        r = requests.post(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/create", json={"name": MODEL_NAME, "modelfile": modelfile}, timeout=600)
        if r.status_code == 200:
            print("Model created", flush=True)
        else:
            print(f"Failed: {r.text}", flush=True)
            sys.exit(1)

    print("Starting RunPod handler...", flush=True)
    runpod.serverless.start({"handler": handler})
