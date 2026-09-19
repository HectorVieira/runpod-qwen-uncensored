"""
RunPod Serverless handler: uncensored Qwen3.6-35B-A3B GGUF served by llama.cpp.

Architecture
------------
    RunPod endpoint -> handler.py -> llama-server -> /runpod-volume/models/*.gguf

llama-server is launched once per worker and mmap()s the GGUF directly from the
network volume. Nothing is copied into the container, so cold starts only pay
for loading the weights into VRAM rather than duplicating a 17-26 GB file.

Job inputs accepted (all optional except one of messages/prompt):

    {"input": {"messages": [...]}}                  # OpenAI-ish, short form
    {"input": {"prompt": "..."}}                    # single user turn
    {"input": {"openai_input": {"messages": [...]}}}  # RunPod OpenAI proxy form
    {"input": {"command": "status"}}                # readiness / download report
    {"input": {"command": "download"}}              # force model fetch, blocking

Extra sampling/tool fields inside `openai_input` or the top-level input
(temperature, top_p, max_tokens, tools, stop, ...) are forwarded to llama-server.
"""

import os
import shlex
import subprocess
import threading
import time

import requests
import runpod

# --------------------------------------------------------------------------- #
# Configuration -- every value is overridable from the endpoint's env vars.
# --------------------------------------------------------------------------- #
MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")
MODEL_REPO = os.environ.get(
    "MODEL_REPO",
    "LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V6-GGUF",
)
MODEL_FILE = os.environ.get(
    "MODEL_FILE",
    "Hermes3.6-35B-A3B-Uncensored-Genesis-Final-APEX.gguf",
)
# Optional multimodal projector; leave empty to serve text-only.
MMPROJ_FILE = os.environ.get("MMPROJ_FILE", "").strip()

MODEL_AUTO_DOWNLOAD = os.environ.get("MODEL_AUTO_DOWNLOAD", "1").strip() == "1"
CTX_SIZE = os.environ.get("CTX_SIZE", "32768").strip()
GPU_LAYERS = os.environ.get("GPU_LAYERS", "99").strip()
# Offload N MoE expert layers to CPU; needed when the GGUF nearly fills VRAM.
N_CPU_MOE = os.environ.get("N_CPU_MOE", "").strip()

LLAMA_HOST = os.environ.get("LLAMA_HOST", "127.0.0.1").strip()
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
SERVER_START_TIMEOUT = int(os.environ.get("SERVER_START_TIMEOUT", "900"))
SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", "/app/llama-server")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "qwen3.6-35b-uncensored")

# Optional knobs: empty means "do not pass the flag", so the pinned build's own
# default applies. llama-server validates the value and aborts loudly on a bad
# one, which surfaces in the worker logs.
#
# NOTE for future edits: at build b11046 the CLI parser ALWAYS consumes the next
# token as the value for these options -- there is no optional-value mechanism.
# A bare `--flash-attn` or `--reasoning` would swallow the following argument and
# fail with "unknown value". Never emit them without a value.
LOAD_MODE = os.environ.get("LOAD_MODE", "").strip()    # auto|none|mmap|mlock|mmap+mlock|dio
FLASH_ATTN = os.environ.get("FLASH_ATTN", "").strip()  # on|off|auto  (value MANDATORY)
REASONING = os.environ.get("REASONING", "").strip()    # on|off|auto
PARALLEL = os.environ.get("PARALLEL", "").strip()      # concurrent slots

# Escape hatch for flags this file does not model explicitly.
EXTRA_ARGS = shlex.split(os.environ.get("LLAMA_EXTRA_ARGS", ""))

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
MMPROJ_PATH = os.path.join(MODEL_DIR, MMPROJ_FILE) if MMPROJ_FILE else ""

BASE_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}"
# Fields we translate rather than pass through verbatim.
_CONTROL_KEYS = {"command", "messages", "prompt", "openai_input", "stream"}

# --------------------------------------------------------------------------- #
# Shared worker state
# --------------------------------------------------------------------------- #
_lock = threading.Lock()
_state = {
    "proc": None,
    "ready": False,
    "error": None,
    "downloading": False,
    "download_note": "",
}


def log(msg):
    print(f"[handler] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Model provisioning
# --------------------------------------------------------------------------- #
def _download_model():
    """Fetch the GGUF onto the network volume. Safe to call repeatedly.

    huggingface_hub resumes partial downloads, which matters because a worker can
    be reaped by the idle timeout in the middle of a multi-GB transfer.
    """
    from huggingface_hub import hf_hub_download

    try:
        import hf_transfer  # noqa: F401

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    except Exception:
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

    os.makedirs(MODEL_DIR, exist_ok=True)
    log(f"Downloading {MODEL_REPO}/{MODEL_FILE} -> {MODEL_DIR} (this can take a while)")
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=MODEL_DIR,
    )
    log(f"Download complete: {path}")
    return path


def _background_download():
    try:
        _download_model()
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller via status
        with _lock:
            _state["error"] = f"model download failed: {exc}"
        log(f"Model download failed: {exc}")
    finally:
        with _lock:
            _state["downloading"] = False


VOLUME_ROOT = "/runpod-volume"


def ensure_model():
    """Return True once the GGUF is on disk, kicking off a download if needed."""
    if os.path.isfile(MODEL_PATH):
        return True

    # Refuse to download into ephemeral container storage. If the network volume
    # is not attached, /runpod-volume does not exist and a 25 GB fetch would land
    # on the (small) container disk and vanish on the next cold start.
    if MODEL_DIR.startswith(VOLUME_ROOT) and not os.path.isdir(VOLUME_ROOT):
        with _lock:
            _state["error"] = (
                f"network volume is not mounted at {VOLUME_ROOT}. Attach a volume "
                f"to this endpoint, or point MODEL_DIR at a path that persists."
            )
        log(f"ERROR: {_state['error']}")
        return False

    if not MODEL_AUTO_DOWNLOAD:
        with _lock:
            _state["error"] = (
                f"model not found at {MODEL_PATH} and MODEL_AUTO_DOWNLOAD is off"
            )
        return False

    with _lock:
        already = _state["downloading"]
        if not already:
            _state["downloading"] = True

    if not already:
        log(f"Model missing at {MODEL_PATH}; starting background download")
        threading.Thread(target=_background_download, daemon=True).start()
    return False


# --------------------------------------------------------------------------- #
# llama-server lifecycle
# --------------------------------------------------------------------------- #
def build_command():
    cmd = [
        SERVER_BIN,
        "--model", MODEL_PATH,
        "--host", LLAMA_HOST,
        "--port", str(LLAMA_PORT),
        "--ctx-size", CTX_SIZE,
        "--n-gpu-layers", GPU_LAYERS,
        "--alias", MODEL_ALIAS,
        # Required for tool / function calling with the model's chat template.
        "--jinja",
    ]
    if N_CPU_MOE:
        cmd += ["--n-cpu-moe", N_CPU_MOE]
    if MMPROJ_PATH and os.path.isfile(MMPROJ_PATH):
        cmd += ["--mmproj", MMPROJ_PATH]
    if LOAD_MODE:
        cmd += ["--load-mode", LOAD_MODE]
    if FLASH_ATTN:
        cmd += ["--flash-attn", FLASH_ATTN]
    if REASONING:
        cmd += ["--reasoning", REASONING]
    if PARALLEL:
        cmd += ["--parallel", PARALLEL]
    cmd += EXTRA_ARGS
    return cmd


def _pump_output(proc):
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        print(f"[llama-server] {line.rstrip()}", flush=True)


def start_server():
    if not ensure_model():
        return False

    with _lock:
        if _state["ready"]:
            return True

    cmd = build_command()
    log("Starting llama-server: " + " ".join(shlex.quote(c) for c in cmd))

    env = dict(os.environ)
    env.setdefault("LD_LIBRARY_PATH", "/app")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=os.path.dirname(SERVER_BIN) or "/app",
        env=env,
    )
    with _lock:
        _state["proc"] = proc
    threading.Thread(target=_pump_output, args=(proc,), daemon=True).start()

    deadline = time.time() + SERVER_START_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            msg = f"llama-server exited early with code {proc.returncode}"
            log(msg)
            with _lock:
                _state["error"] = msg
            return False
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                log("llama-server is ready")
                with _lock:
                    _state["ready"] = True
                    _state["error"] = None
                return True
        except requests.RequestException:
            pass
        time.sleep(2)

    msg = f"llama-server not ready after {SERVER_START_TIMEOUT}s"
    log(msg)
    with _lock:
        _state["error"] = msg
    return False


def supervise():
    """Restart llama-server if it dies, so a long-lived worker self-heals."""
    while True:
        time.sleep(10)
        with _lock:
            proc = _state["proc"]
            ready = _state["ready"]
        if proc is None:
            continue
        if proc.poll() is not None:
            log(f"llama-server died (code {proc.returncode}); restarting")
            with _lock:
                _state["ready"] = False
            start_server()


# --------------------------------------------------------------------------- #
# Request handling
# --------------------------------------------------------------------------- #
def _status_payload():
    with _lock:
        ready = _state["ready"]
        error = _state["error"]
        downloading = _state["downloading"]

    payload = {
        "ready": ready,
        "model_path": MODEL_PATH,
        "model_present": os.path.isfile(MODEL_PATH),
        "downloading": downloading,
    }
    if error:
        payload["error"] = error
    if not ready and not error:
        payload["detail"] = (
            "Model is still downloading or loading into VRAM. Retry shortly."
            if downloading
            else "llama-server is still starting. Retry shortly."
        )
    return payload


def _normalise(job_input):
    """Accept both RunPod's OpenAI proxy shape and a plain messages list."""
    openai_input = job_input.get("openai_input") or {}
    source = openai_input or job_input

    messages = source.get("messages") or []
    if not messages:
        prompt = job_input.get("prompt") or source.get("prompt")
        if prompt:
            messages = [{"role": "user", "content": prompt}]

    payload = {k: v for k, v in source.items() if k not in _CONTROL_KEYS}
    payload["messages"] = messages
    payload.setdefault("model", MODEL_ALIAS)
    # RunPod returns a single JSON document, so never ask upstream for SSE.
    payload["stream"] = False
    return payload


def handler(job):
    job_input = job.get("input") or {}

    command = (job_input.get("command") or "").strip().lower()
    if command in {"status", "health"}:
        return _status_payload()
    if command == "props":
        # Surfaces the model's embedded chat template and its capabilities, which
        # is how you confirm tool calling is actually available before relying on it.
        try:
            r = requests.get(f"{BASE_URL}/props", timeout=30)
            return r.json()
        except requests.RequestException as exc:
            return {"error": f"llama-server /props failed: {exc}"}
    if command == "download":
        try:
            _download_model()
        except Exception as exc:  # noqa: BLE001
            with _lock:
                _state["error"] = f"model download failed: {exc}"
            return _status_payload()
        return _status_payload()

    payload = _normalise(job_input)
    if not payload["messages"]:
        return {"error": "No messages provided"}

    with _lock:
        ready = _state["ready"]
    if not ready:
        # Distinguish "still warming up" from "actually broken".
        status = _status_payload()
        status["error"] = status.get("error", "model_not_ready")
        return status

    try:
        r = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=600,
        )
    except requests.RequestException as exc:
        return {"error": f"llama-server request failed: {exc}"}

    if r.status_code != 200:
        return {
            "error": f"llama-server returned HTTP {r.status_code}",
            "detail": r.text[:2000],
        }
    return r.json()


if __name__ == "__main__":
    log(f"Model path: {MODEL_PATH}")
    if start_server():
        threading.Thread(target=supervise, daemon=True).start()
        runpod.serverless.start({"handler": handler})
    else:
        # Keep the worker alive so /run can report *why* it is unhealthy rather
        # than the endpoint dying with no diagnosable output.
        log(f"FATAL: {_state['error']}")
        runpod.serverless.start({"handler": handler})
