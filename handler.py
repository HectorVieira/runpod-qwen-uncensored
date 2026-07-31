"""
RunPod Serverless Handler for Qwen3.6-35B-A3B Uncensored
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
MODEL_PATH = "/models/qwen3.6-35b-uncensored.gguf"
SERVER_PORT = 8080
SERVER_HOST = "0.0.0.0"

server_process = None


def start_server():
    """Start llama-server in background."""
    global server_process

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}", flush=True)
        for root, dirs, files in os.walk("/models"):
            for f in files:
                print(f"  {os.path.join(root, f)}", flush=True)
        return False

    cmd = [
        "llama-server",
        "--model", MODEL_PATH,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--n-gpu-layers", "-1",
        "--ctx-size", "8192",
        "--parallel", "2",
        "--cont-batching",
        "--flash-attn",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--jinja",
        "--metrics",
    ]

    print(f"Starting llama-server...", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)

    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Read output in real-time for debugging
    def read_output():
        for line in server_process.stdout:
            print(f"[llama] {line.strip()}", flush=True)

    t = threading.Thread(target=read_output, daemon=True)
    t.start()

    # Wait for server to be ready
    print("Waiting for server to be ready...", flush=True)
    for i in range(180):
        try:
            r = requests.get(f"http://{SERVER_HOST}:{SERVER_PORT}/health", timeout=5)
            if r.status_code == 200:
                print(f"Server ready after {i+1}s", flush=True)
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"Health check: {e}", flush=True)
        time.sleep(1)

    print("Server failed to start within 180s", flush=True)
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
            "model": "qwen3.6-35b-uncensored",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        print(f"Request: messages={len(messages)}, max_tokens={max_tokens}", flush=True)

        r = requests.post(
            f"http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions",
            json=payload,
            timeout=300
        )

        if r.status_code != 200:
            print(f"LLM error {r.status_code}: {r.text[:200]}", flush=True)
            return {"error": f"LLM returned {r.status_code}: {r.text}"}

        result = r.json()
        usage = result.get("usage", {})
        print(f"Response: {usage.get('completion_tokens', 0)} tokens", flush=True)

        return result

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

    print("=== RunPod Handler Starting ===", flush=True)
    print(f"Model: {MODEL_PATH}", flush=True)
    print(f"Port: {SERVER_PORT}", flush=True)

    if start_server():
        print("Server started, entering RunPod handler loop...", flush=True)
        runpod.serverless.start({"handler": handler})
    else:
        print("Failed to start server", flush=True)
        sys.exit(1)
