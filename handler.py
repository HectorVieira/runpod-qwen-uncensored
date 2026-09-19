"""
RunPod Serverless handler: uncensored Qwen3.8-27B-OBLITERATED GGUF served by llama.cpp.

Architecture
------------
    RunPod endpoint -> handler.py -> llama-server -> /runpod-volume/models/*.gguf

llama-server is launched once per worker and mmap()s the GGUF directly from the
network volume. Nothing is copied into the container, so cold starts only pay
for loading the weights into VRAM rather than duplicating a 25 GB file.

Two ways in
-----------
1. RunPod's OpenAI passthrough. The platform's OpenAI gateway turns
   `POST /openai/v1/chat/completions` into a job shaped like

       {"input": {"openai_route": "/v1/chat/completions",
                  "openai_input": {...the OpenAI body...}}}

   and a bare `GET /openai/v1/models` into `{"input": {"openai_route": "/v1/models"}}`.
   That gateway relays Server-Sent Events only, so a streaming request is proxied
   through byte-for-byte. This is why the handler is an async generator: the
   RunPod SDK only enters its streaming path when the handler *function* is a
   generator, so every result is yielded.

2. Plain jobs on /run and /runsync, using `messages`, `prompt`, or the diagnostic
   `command` values.

Job inputs accepted:

    {"input": {"openai_route": "/v1/chat/completions",
               "openai_input": {...}}}              # what the OpenAI gateway sends
    {"input": {"openai_route": "/v1/models"}}       # bare route -> GET
    {"input": {"messages": [...]}}                  # short form
    {"input": {"prompt": "..."}}                    # single user turn
    {"input": {"command": "status"}}                # readiness / download / disk report
    {"input": {"command": "props"}}                 # chat template + capabilities
    {"input": {"command": "download"}}              # force model fetch, blocking
    {"input": {"command": "cleanup"}}               # delete other *.gguf from the volume

Extra sampling/tool fields (temperature, top_p, max_tokens, tools, stop, ...)
are forwarded to llama-server.

WARNING for maintainers: because the handler is a generator, RunPod aggregates
yielded values, so /runsync returns `output` as a LIST of chunks rather than one
object. That is inherent to the streaming contract, not a choice made here.
"""

import json
import os
import shlex
import shutil
import subprocess
import threading
import time

import aiohttp
import requests
import runpod

# --------------------------------------------------------------------------- #
# Configuration -- every value is overridable from the endpoint's env vars.
# --------------------------------------------------------------------------- #
MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")
MODEL_REPO = os.environ.get(
    "MODEL_REPO",
    "OBLITERATUS/Qwen3.8-27B-OBLITERATED",
)
MODEL_FILE = os.environ.get(
    "MODEL_FILE",
    "Qwen3.8-27B-OBLITERATED-Q6_K.gguf",
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
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "qwen3.8-27b-obliterated")

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
# The OBLITERATED model card is explicit that this is mandatory: without it,
# greedy decoding degenerates into repeating imports/boilerplate, and agent
# harnesses get stuck in tool-call loops.
REPEAT_PENALTY = os.environ.get("REPEAT_PENALTY", "").strip()

# Escape hatch for flags this file does not model explicitly.
EXTRA_ARGS = shlex.split(os.environ.get("LLAMA_EXTRA_ARGS", ""))

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
MMPROJ_PATH = os.path.join(MODEL_DIR, MMPROJ_FILE) if MMPROJ_FILE else ""

BASE_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}"
VOLUME_ROOT = "/runpod-volume"

# Route the OpenAI gateway uses when it does not name one explicitly.
DEFAULT_CHAT_ROUTE = "/v1/chat/completions"
# Longest a single generation may take, in seconds.
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "600"))

# Fields we translate rather than forward verbatim.
_CONTROL_KEYS = {"command", "messages", "openai_route", "prompt", "openai_input"}

# --------------------------------------------------------------------------- #
# Shared worker state
# --------------------------------------------------------------------------- #
_lock = threading.Lock()
_state = {
    "proc": None,
    "ready": False,
    "error": None,
    "downloading": False,
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
    if REPEAT_PENALTY:
        cmd += ["--repeat-penalty", REPEAT_PENALTY]
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
    """Keep llama-server running, and start it once the model is available.

    This must run unconditionally. When the model is missing at boot,
    start_server() returns immediately without ever launching anything, so
    without this thread a worker would answer "not ready" forever even after a
    download completed -- the supervisor is the only thing that brings the
    server up in that case.
    """
    while True:
        time.sleep(10)
        with _lock:
            proc = _state["proc"]
            ready = _state["ready"]
            downloading = _state["downloading"]

        if ready or downloading:
            continue
        if proc is not None and proc.poll() is None:
            continue  # alive; start_server is already waiting on its health
        if not os.path.isfile(MODEL_PATH):
            continue  # nothing to serve yet

        if proc is None:
            log("llama-server was never started; starting it now")
        else:
            log(f"llama-server is gone (code {proc.returncode}); restarting")
        with _lock:
            _state["ready"] = False
        start_server()


# --------------------------------------------------------------------------- #
# Response helpers
# --------------------------------------------------------------------------- #
def _openai_error(message, err_type="worker_error"):
    """OpenAI-shaped error, which is what the platform gateway expects."""
    return {"error": {"message": str(message), "type": err_type, "code": None}}


def _disk_payload():
    """Space on the volume holding the models; two of these GGUFs do not fit."""
    target = MODEL_DIR if os.path.isdir(MODEL_DIR) else "/"
    try:
        u = shutil.disk_usage(target)
        return {
            "disk_total_gb": round(u.total / 1e9, 2),
            "disk_used_gb": round(u.used / 1e9, 2),
            "disk_free_gb": round(u.free / 1e9, 2),
        }
    except Exception as exc:  # noqa: BLE001
        return {"disk_error": str(exc)}


def _cleanup_other_models():
    """Delete every *.gguf in MODEL_DIR except the configured MODEL_FILE.

    Deliberately explicit and narrow: only a direct child of MODEL_DIR, only
    *.gguf, never the configured model. It exists because the network volume is
    smaller than two of these models, so swapping means removing the old one
    first. Reports what it freed rather than failing silently.
    """
    removed, freed = [], 0
    errors = []
    try:
        names = sorted(os.listdir(MODEL_DIR))
    except Exception as exc:  # noqa: BLE001
        return {"cleanup_error": f"cannot list {MODEL_DIR}: {exc}"}

    for name in names:
        path = os.path.join(MODEL_DIR, name)
        if not name.lower().endswith(".gguf") or name == MODEL_FILE:
            continue
        if not os.path.isfile(path):
            continue
        try:
            size = os.path.getsize(path)
            os.remove(path)
            removed.append(name)
            freed += size
            log(f"cleanup: removed {name} ({size / 1e9:.2f} GB)")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")

    payload = {
        "removed": removed,
        "freed_gb": round(freed / 1e9, 2),
        "kept": MODEL_FILE,
    }
    if errors:
        payload["cleanup_error"] = "; ".join(errors)
    payload.update(_disk_payload())
    return payload


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
    payload.update(_disk_payload())
    if error:
        payload["error"] = error
    if not ready and not error:
        payload["detail"] = (
            "Model is still downloading or loading into VRAM. Retry shortly."
            if downloading
            else "llama-server is still starting. Retry shortly."
        )
    return payload


def _resolve_route(job_input):
    """Map a job to (route, method, body) per RunPod's OpenAI passthrough.

    Returns (None, None, None) when the job is not an OpenAI passthrough, in
    which case the caller falls back to the messages/prompt shortcuts.
    """
    if job_input.get("openai_input"):
        route = job_input.get("openai_route") or DEFAULT_CHAT_ROUTE
        return route, "POST", dict(job_input["openai_input"])

    if job_input.get("openai_route"):
        # A bare route with no body is a read-only request, e.g. /v1/models.
        return job_input["openai_route"], "GET", None

    return None, None, None


def _shortcut_body(job_input):
    """Build a chat body from the plain messages/prompt short form."""
    messages = job_input.get("messages") or []
    if not messages:
        prompt = job_input.get("prompt")
        if prompt:
            messages = [{"role": "user", "content": prompt}]
    if not messages:
        return None

    body = {k: v for k, v in job_input.items() if k not in _CONTROL_KEYS}
    body["messages"] = messages
    body.setdefault("model", MODEL_ALIAS)
    # Short-form callers get one JSON document; the platform has no stream to
    # relay for them, so never ask upstream for SSE here.
    body["stream"] = False
    return body


async def _request_json(route, method, body):
    timeout = aiohttp.ClientTimeout(total=GENERATION_TIMEOUT)
    url = f"{BASE_URL}{route}"
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, url, json=body) as resp:
                raw = await resp.text()
                if resp.status >= 400:
                    return _openai_error(
                        f"llama-server HTTP {resp.status} on {method} {route}: {raw[:1000]}",
                        err_type="upstream_error",
                    )
                try:
                    return json.loads(raw)
                except ValueError:
                    # A route such as /props may legitimately return non-JSON.
                    return raw
    except aiohttp.ClientError as exc:
        return _openai_error(f"llama-server request failed: {exc}", "upstream_error")
    except Exception as exc:  # noqa: BLE001
        return _openai_error(f"unexpected error: {exc}")


async def _stream_route(route, body):
    """Yield llama-server's SSE bytes verbatim so the gateway can relay them."""
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=GENERATION_TIMEOUT)
    url = f"{BASE_URL}{route}"
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=body) as resp:
                if resp.status >= 400:
                    raw = await resp.text()
                    yield _openai_error(
                        f"llama-server HTTP {resp.status} on POST {route}: {raw[:1000]}",
                        err_type="upstream_error",
                    )
                    return
                async for chunk in resp.content.iter_any():
                    yield chunk.decode("utf-8", errors="replace")
    except aiohttp.ClientError as exc:
        yield _openai_error(f"llama-server stream failed: {exc}", "upstream_error")
    except Exception as exc:  # noqa: BLE001
        yield _openai_error(f"unexpected stream error: {exc}")


# --------------------------------------------------------------------------- #
# Handler
# --------------------------------------------------------------------------- #
async def handler(job):
    """Async generator: RunPod only streams when the handler is a generator."""
    job_input = job.get("input") or {}

    command = (job_input.get("command") or "").strip().lower()
    if command in {"status", "health"}:
        yield _status_payload()
        return
    if command == "props":
        yield await _request_json("/props", "GET", None)
        return
    if command == "cleanup":
        yield _cleanup_other_models()
        return
    if command == "download":
        try:
            _download_model()
            with _lock:
                _state["error"] = None
        except Exception as exc:  # noqa: BLE001
            with _lock:
                _state["error"] = f"model download failed: {exc}"
            log(f"Download failed: {exc}")
        yield _status_payload()
        return

    route, method, body = _resolve_route(job_input)
    if route is None:
        body = _shortcut_body(job_input)
        if body is None:
            yield _openai_error("No messages provided", "invalid_request_error")
            return
        route, method = DEFAULT_CHAT_ROUTE, "POST"

    with _lock:
        ready = _state["ready"]
    if not ready:
        status = _status_payload()
        status.setdefault("error", "model_not_ready")
        yield status
        return

    wants_stream = (
        method != "GET" and isinstance(body, dict) and body.get("stream") is True
    )

    if wants_stream:
        async for part in _stream_route(route, body):
            yield part
    else:
        yield await _request_json(route, method, body)


if __name__ == "__main__":
    log(f"Model path: {MODEL_PATH}")
    if start_server():
        log("Serving.")
    else:
        # Not fatal: the model may simply not be downloaded yet. The supervisor
        # below retries as soon as it appears, and `status` reports why.
        log(f"Initial start incomplete: {_state['error']}")

    # Always supervise, even after a failed initial start -- otherwise a model
    # fetched later would never be served by this worker.
    threading.Thread(target=supervise, daemon=True).start()

    # return_aggregate_stream keeps every yielded chunk in the final job result
    # instead of dropping them, so a non-streaming caller still receives the
    # full completion (as a list of chunks).
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
