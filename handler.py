"""
RunPod Serverless Handler - vLLM with local model from network volume
"""
import runpod
import subprocess
import time
import json
import requests
import sys
import os
import threading

MODEL_PATH = "/workspace/models/vllm-model"
VLLM_HOST = "0.0.0.0"
VLLM_PORT = 8000


def start_vllm():
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--dtype", "auto",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.9",
        "--enforce-eager",
        "--trust-remote-code",
        "--tool-call-parser", "hermes",
        "--enable-auto-tool-choice",
    ]
    
    print(f"Starting vLLM...", flush=True)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    def read_output():
        for line in process.stdout:
            print(f"[vllm] {line.strip()}", flush=True)
    
    t = threading.Thread(target=read_output, daemon=True)
    t.start()
    
    print("Waiting for vLLM...", flush=True)
    for i in range(300):
        try:
            r = requests.get(f"http://{VLLM_HOST}:{VLLM_PORT}/health", timeout=5)
            if r.status_code == 200:
                print(f"vLLM ready after {i+1}s", flush=True)
                return True
        except:
            pass
        time.sleep(1)
    
    print("vLLM failed to start", flush=True)
    return False


def handler(job):
    job_input = job.get("input", {})
    
    openai_input = job_input.get("openai_input", {})
    if openai_input:
        messages = openai_input.get("messages", [])
        tools = openai_input.get("tools", [])
        max_tokens = openai_input.get("max_tokens", 4096)
        temperature = openai_input.get("temperature", 0.6)
        stream = openai_input.get("stream", False)
    else:
        messages = job_input.get("messages", [])
        tools = job_input.get("tools", [])
        prompt = job_input.get("prompt", "")
        if not messages and prompt:
            messages = [{"role": "user", "content": prompt}]
        max_tokens = job_input.get("max_tokens", 4096)
        temperature = job_input.get("temperature", 0.6)
        stream = job_input.get("stream", False)
    
    if not messages:
        return {"error": "No messages provided"}
    
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "stream": stream,
    }
    
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    
    try:
        r = requests.post(
            f"http://{VLLM_HOST}:{VLLM_PORT}/v1/chat/completions",
            json=payload,
            stream=stream,
            timeout=(30, 600)
        )
        
        if r.status_code != 200:
            return {"error": f"vLLM {r.status_code}: {r.text[:200]}"}
        
        if stream:
            for line in r.iter_lines():
                if line:
                    yield line.decode('utf-8') + '\n\n'
            return
        else:
            return r.json()
    
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("=== Handler Starting ===", flush=True)
    print(f"Model path: {MODEL_PATH}", flush=True)
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}", flush=True)
        sys.exit(1)
    
    # Check for safetensors files
    files = os.listdir(MODEL_PATH)
    safetensors = [f for f in files if f.endswith('.safetensors')]
    print(f"Found {len(safetensors)} safetensors files", flush=True)
    
    if not start_vllm():
        sys.exit(1)
    
    print("Starting RunPod handler...", flush=True)
    runpod.serverless.start({"handler": handler})
