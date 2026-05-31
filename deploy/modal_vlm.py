"""Serverless VLM endpoint on Modal — for demoing on a small local GPU (RTX 1650).

The 4GB RTX 1650 runs the cheap path (OCR + KIE) locally; only router-flagged HARD
cases call this serverless VLM. Modal gives **scale-to-zero** (you pay only the
seconds a hard case is actually inferring) and an OpenAI-compatible endpoint, so
the app plugs in with NO code change:

    modal deploy deploy/modal_vlm.py          # prints a URL like https://<you>--docai-vlm-serve.modal.run
    export DOCAI_VLM_MODE=api
    export VLM_API_BASE="https://<you>--docai-vlm-serve.modal.run/v1"
    export VLM_API_KEY="dummy"                # vLLM doesn't require one by default
    export VLM_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
    # now run the local pipeline; hard cases hit Modal, easy cases stay local.

Cold start: first hard case after idle waits ~30-60s while the container spins up
(mitigate with min_containers=1 if you need it always warm). See https://modal.com/docs .
"""
import modal

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"   # fits a 24GB A10G/L4; use Qwen2.5-VL-3B for L4-small

image = (
    modal.Image.debian_slim()
    .pip_install("vllm>=0.6.3", "transformers>=4.49", "qwen-vl-utils", "accelerate")
)
app = modal.App("docai-vlm")


@app.function(
    image=image,
    gpu="A10G",                 # 24GB; cheapest that holds the 7B VL model
    scaledown_window=300,       # scale to zero after 5 min idle (cost control)
    timeout=600,
    # min_containers=1,         # uncomment to keep one warm (no cold start, more $)
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    """Launch a vLLM OpenAI-compatible server (/v1/chat/completions, vision)."""
    import subprocess
    subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL, "--port", "8000",
        "--gpu-memory-utilization", "0.92", "--max-model-len", "8192",
    ])
