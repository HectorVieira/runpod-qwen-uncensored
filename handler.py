"""
RunPod Serverless Handler - Qwen3.6-35B-A3B Uncensored (llama.cpp)
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
import shutil

MODEL_PATH = "/runpod-volume/models/qwen3.6-35b-uncensored.gguf"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080

server_process = None


def find_llama_server():
    """Search for llama-server binary"""
    # Common locations
    paths = [
        "/usr/local/bin/llama-server",
        "/usr/bin/llama-server",
        "/opt/llama.cpp/build/bin/llama-server",
        "/app/llama-server",
        "llama-server",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    
    # shutil.which
    path = shutil.which("llama-server")
    if path:
        return path
    
    # find command
    try:
        result = subprocess.run(
            ["find", "/", "-name", "llama-server", "-type", "f", "-executable"],
            capture_output=True, text=True, timeout=15
        )
        for line in result.stdout.strip().split("\n"):
            if line and os.path.isfile(line) and os.access(line, os.X_OK):
                return line
    except:
        pass
    
    return None


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
            "messages": messages,
            "max_tokens": job_input.get("max_tokens", 1024),
            "temperature": job_input.get("temperature", 0.7),
            "stream": False,
        }
        r = requests.post(f"http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions", json=payload, timeout=300)
        if r.status_code != 200:
            return {"error": f"llama-server {r.status_code}: {r.text[:200]}"}
        return r.json()
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("=== Handler Starting ===", flush=True)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}", flush=True)
        sys.exit(1)

    size_gb = os.path.getsize(MODEL_PATH) / (1024**3)
    print(f"Model: {size_gb:.1f} GB", flush=True)

    llama_path = find_llama_server()
    if not llama_path:
        print("ERROR: llama-server not found", flush=True)
        # List all files in common locations
        for d in ["/usr/local/bin", "/usr/bin", "/opt", "/app"]:
            if os.path.exists(d):
                print(f"  {d}: {os.listdir(d)[:10]}", flush=True)
        sys.exit(1)
    print(f"llama-server: {llama_path}", flush=True)

    cmd = [
        llama_path,
        "--model", MODEL_PATH,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--n-gpu-layers", "35",
        "--ctx-size", "4096",
        "--parallel", "1",
        "--cont-batching",
        "--flash-attn",
    ]

    print(f"Starting llama-server...", flush=True)
    server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def read_output():
        for line in server_process.stdout:
            print(f"[llama] {line.strip()}", flush=True)

    t = threading.Thread(target=read_output, daemon=True)
    t.start()

    print("Waiting for server...", flush=True)
    for i in range(180):
        try:
            r = requests.get(f"http://{SERVER_HOST}:{SERVER_PORT}/health", timeout=5)
            if r.status_code == 200:
                print(f"Server ready after {i+1}s", flush=True)
                break
        except:
            pass
        time.sleep(1)
    else:
        print("Server failed to start", flush=True)
        sys.exit(1)

    print("Starting RunPod handler...", flush=True)
    runpod.serverless.start({"handler": handler})
