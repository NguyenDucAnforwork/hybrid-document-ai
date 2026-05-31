"""VLM fallback (Tier-2). Default disabled; `api`/`remote_gpu` swap-by-config.

We keep the full interface + guardrails to show we understand VLM-OCR failure
modes (repeated phrases, token ceiling, latency cascade) without spending the
disk/VRAM budget on a local 7B model. Donut local is a documented option.
"""
from __future__ import annotations
import os
import json
import re

PROMPT = (
    "You are a strict document information extraction system. Extract fields "
    "merchant_name, invoice_id, date, total_amount, payment_method from this "
    "receipt. Return ONLY valid JSON. Use null if not visible. Do not guess."
)


def _guardrail_ok(text: str) -> bool:
    if not text or len(text) > 4000:
        return False
    # repeated-phrase / whitespace explosion detection
    if re.search(r"(.{4,})\1{4,}", text):
        return False
    return True


_local = {"model": None, "processor": None}
LOCAL_MODEL = os.environ.get("DOCAI_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")


def _load_local():
    """Lazy-load a real VLM (Qwen2.5-VL) for hard-case extraction.

    Device from DOCAI_VLM_DEVICE (default cpu — this box's driver 12.2 is too old
    for the py3.13 CUDA wheels; in production this runs on a proper GPU via vLLM,
    config `vlm.mode=remote_gpu`).
    """
    if _local["model"] is None:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        dev = os.environ.get("DOCAI_VLM_DEVICE", "cpu")
        dtype = torch.bfloat16 if dev != "cpu" else torch.float32
        _local["model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            LOCAL_MODEL, torch_dtype=dtype).to(dev).eval()
        _local["processor"] = AutoProcessor.from_pretrained(LOCAL_MODEL)
    return _local["model"], _local["processor"]


def _vlm_local(image_bytes: bytes) -> dict | None:
    import io
    from PIL import Image
    model, processor = _load_local()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(model.device)
    gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)  # guardrail: low cap
    trimmed = gen[:, inputs.input_ids.shape[1]:]
    out = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    if not _guardrail_ok(out):
        return None
    try:
        return json.loads(re.search(r"\{.*\}", out, re.S).group())
    except Exception:
        return None


def vlm_extract(image_bytes: bytes) -> dict | None:
    """Returns extracted fields or None (disabled/failed -> human review)."""
    mode = os.environ.get("DOCAI_VLM_MODE", "disabled")
    if mode == "disabled":
        return None
    if mode == "local":          # real VLM on local GPU (H100) for hard cases
        try:
            return _vlm_local(image_bytes)
        except Exception:
            return None
    if mode == "api":
        base = os.environ.get("VLM_API_BASE")
        key = os.environ.get("VLM_API_KEY", "")
        if not base:
            return None
        try:
            import base64, httpx
            b64 = base64.b64encode(image_bytes).decode()
            r = httpx.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": os.environ.get("VLM_MODEL", "qwen2.5-vl"),
                    "max_tokens": 256,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ]}],
                }, timeout=30,
            )
            txt = r.json()["choices"][0]["message"]["content"]
            if not _guardrail_ok(txt):
                return None
            return json.loads(re.search(r"\{.*\}", txt, re.S).group())
        except Exception:
            return None
    return None
