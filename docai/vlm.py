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


def vlm_extract(image_bytes: bytes) -> dict | None:
    """Returns extracted fields or None (disabled/failed -> human review)."""
    mode = os.environ.get("DOCAI_VLM_MODE", "disabled")
    if mode == "disabled":
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
