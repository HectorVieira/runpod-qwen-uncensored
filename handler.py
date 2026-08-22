"""
RunPod Serverless Handler - Ollama serving local GGUF from network volume
"""
import runpod
import subprocess
import time
import json
import requests
import os
import threading

MODEL_PATH = "/workspace/models/qwen3.6-35b-uncensored.gguf"
MODEL_NAME = "qwen35b"
OLLAMA_HOST = "0.0.0.0"
OLLAMA_PORT = 11434

MODELFILE = f"FROM {MODEL_PATH}\nPARAMETER num_ctx 32768\nPARAMETER num_gpu 99\nPARAMETER temperature 0.7\nPARAMETER top_p 0.9\n"

def start_ollama():
    # inicia o ollama serve
    print("Starting Ollama serve...", flush=True)
    srv = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    def read_output():
        for line in srv.stdout:
            print(f"[ollama] {line.strip()}", flush=True)
    threading.Thread(target=read_output, daemon=True).start()
    time.sleep(5)
    # cria o modelo a partir do gguf
    print("Creating Ollama model from GGUF...", flush=True)
    try:
        subprocess.run(["ollama", "create", MODEL_NAME, "-f", "-"],
                       input=MODELFILE.encode(), timeout=600, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ollama create failed: {e}", flush=True)
        return False
    # espera saude
    print("Waiting for Ollama...", flush=True)
    for i in range(300):
        try:
            r = requests.get(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags", timeout=5)
            if r.status_code == 200:
                print(f"Ollama ready after {i+1}s", flush=True)
                return True
        except:
            pass
        time.sleep(1)
    print("Ollama failed to start", flush=True)
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
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    try:
        r = requests.post(
            f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/v1/chat/completions",
            json=payload, timeout=300
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if start_ollama():
        runpod.serverless.start({"handler": handler})
    else:
        print("FATAL: Ollama did not start", flush=True)
