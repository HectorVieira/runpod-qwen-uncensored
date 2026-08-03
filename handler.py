"""
Download model to volume - run once then delete endpoint
"""
import subprocess
import os
import requests
import runpod

MODEL_DIR = "/runpod-volume/models"
MODEL_PATH = f"{MODEL_DIR}/qwen3.6-35b-uncensored.gguf"

MODEL_URLS = [
    "https://huggingface.co/LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V6-GGUF/resolve/main/Hermes3.6-35B-A3B-Uncensored-Genesis-V6-APEX.gguf",
    "https://huggingface.co/LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V6-GGUF/resolve/main/Hermes3.6-35B-A3B-Uncensored-Genesis-V6-APEX-Compact.gguf",
]


def handler(job):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        size_gb = os.path.getsize(MODEL_PATH) / (1024**3)
        return {"status": "already_exists", "size_gb": round(size_gb, 1)}

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
                print(f"Downloaded! ({size_gb:.1f} GB)", flush=True)
                return {"status": "downloaded", "file": filename, "size_gb": round(size_gb, 1)}
            else:
                print(f"Failed: {result.stderr}", flush=True)
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
        except Exception as e:
            print(f"Error: {e}", flush=True)
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)

    return {"status": "failed"}


if __name__ == "__main__":
    print("=== Download Handler ===", flush=True)
    runpod.serverless.start({"handler": handler})
