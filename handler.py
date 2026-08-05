"""
Diagnostic handler
"""
import sys
import os
import subprocess

print("=== DIAGNOSTIC ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"CWD: {os.getcwd()}", flush=True)
print(f"PATH: {os.environ.get('PATH', 'N/A')}", flush=True)

# Check model
if os.path.exists("/runpod-volume/models/qwen3.6-35b-uncensored.gguf"):
    size = os.path.getsize("/runpod-volume/models/qwen3.6-35b-uncensored.gguf")
    print(f"Model: {size / 1024 / 1024 / 1024:.1f} GB", flush=True)
else:
    print("Model: NOT FOUND", flush=True)

# Check llama-server
import shutil
llama_path = shutil.which("llama-server")
print(f"llama-server: {llama_path}", flush=True)

if llama_path:
    result = subprocess.run([llama_path, "--help"], capture_output=True, text=True, timeout=5)
    print(f"llama-server help: {result.stdout[:200]}", flush=True)

# Check runpod
try:
    import runpod
    print(f"RunPod SDK: OK", flush=True)
except Exception as e:
    print(f"RunPod SDK ERROR: {e}", flush=True)

print("=== END ===", flush=True)

def handler(job):
    return {"status": "ok"}

if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
