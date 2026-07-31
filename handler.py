"""
RunPod Serverless Handler - Minimal for debugging
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
OLLAMA_HOST = "0.0.0.0"
OLLAMA_PORT = 11434

server_process = None


def handler(job):
    """RunPod serverless handler."""
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

        r = requests.post(
            f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat",
            json=payload,
            timeout=300
        )

        if r.status_code != 200:
            return {"error": f"Ollama returned {r.status_code}: {r.text[:200]}"}

        result = r.json()
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
        return {"error": str(e)}


if __name__ == "__main__":
    print("=== Handler Starting ===", flush=True)

    # Check volume
    if os.path.exists("/runpod-volume"):
        print(f"Volume exists: {os.listdir('/runpod-volume')}", flush=True)
    else:
        print("WARNING: /runpod-volume not found!", flush=True)

    # Check model
    if os.path.exists(MODEL_PATH):
        size_gb = os.path.getsize(MODEL_PATH) / (1024**3)
        print(f"Model exists: {size_gb:.1f} GB", flush=True)
    else:
        print(f"Model not found at {MODEL_PATH}", flush=True)

    # Start Ollama
    print("Starting Ollama...", flush=True)
    server_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Read output
    def read_output():
        for line in server_process.stdout:
            print(f"[ollama] {line.strip()}", flush=True)

    t = threading.Thread(target=read_output, daemon=True)
    t.start()

    # Wait for Ollama
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

    # Check for existing model
    r = requests.get(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags", timeout=10)
    models = [m['name'] for m in r.json().get('models', [])]
    print(f"Existing models: {models}", flush=True)

    if MODEL_NAME not in models and f"{MODEL_NAME}:latest" not in models:
        print(f"Creating model {MODEL_NAME}...", flush=True)

        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file not found at {MODEL_PATH}", flush=True)
            print("Please download the model to the volume first.", flush=True)
            sys.exit(1)

        # Create from Modelfile
        modelfile = f"FROM {MODEL_PATH}\nPARAMETER num_ctx 8192\nPARAMETER num_gpu 99"

        r = requests.post(
            f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/create",
            json={"name": MODEL_NAME, "modelfile": modelfile},
            timeout=600
        )

        if r.status_code == 200:
            print(f"Model created successfully", flush=True)
        else:
            print(f"Failed to create model: {r.text}", flush=True)
            sys.exit(1)

    print("Starting RunPod handler...", flush=True)
    runpod.serverless.start({"handler": handler})
