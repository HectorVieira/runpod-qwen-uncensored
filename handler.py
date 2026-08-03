"""
Diagnostic handler - check Ollama and volume
"""
import sys
import os
import subprocess

print("=== DIAGNOSTIC ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"CWD: {os.getcwd()}", flush=True)

# Check volume
if os.path.exists("/runpod-volume"):
    print(f"Volume: {os.listdir('/runpod-volume')}", flush=True)
else:
    print("Volume: NOT FOUND", flush=True)

# Check Ollama
import shutil
ollama_path = shutil.which("ollama")
print(f"Ollama: {ollama_path}", flush=True)

if ollama_path:
    result = subprocess.run([ollama_path, "--version"], capture_output=True, text=True)
    print(f"Ollama version: {result.stdout.strip()}", flush=True)

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
