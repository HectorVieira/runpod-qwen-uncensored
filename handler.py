"""
Minimal diagnostic handler - just prints system info
"""
import sys
import os

print("=== DIAGNOSTIC ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"CWD: {os.getcwd()}", flush=True)
print(f"PATH: {os.environ.get('PATH', 'N/A')}", flush=True)

# Check volume
if os.path.exists("/runpod-volume"):
    print(f"Volume: {os.listdir('/runpod-volume')}", flush=True)
else:
    print("Volume: NOT FOUND", flush=True)

# Check Ollama
import shutil
ollama_path = shutil.which("ollama")
print(f"Ollama binary: {ollama_path}", flush=True)

# Check if we can import runpod
try:
    import runpod
    print(f"RunPod SDK: {runpod.__version__}", flush=True)
except Exception as e:
    print(f"RunPod SDK ERROR: {e}", flush=True)

print("=== END DIAGNOSTIC ===", flush=True)

# Simple handler
def handler(job):
    return {"status": "ok", "message": "Diagnostics passed"}

if __name__ == "__main__":
    print("Starting handler...", flush=True)
    import runpod
    runpod.serverless.start({"handler": handler})
