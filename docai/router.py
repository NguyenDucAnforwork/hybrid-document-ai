"""Hybrid confidence router (Processing Layer).

Decides traditional_ocr vs vlm_fallback per document. 80-90% of traffic stays
on the cheap/fast traditional path; only low-confidence docs escalate to VLM.
"""
from __future__ import annotations
from .config import REQUIRED_FIELDS, MIN_FIELD_CONFIDENCE


def route_decision(fields: dict) -> tuple[bool, bool, list[str]]:
    """Return (needs_review, should_vlm, reasons)."""
    reasons = []
    for rf in REQUIRED_FIELDS:
        v = fields.get(rf, (None, 0.0))
        if v[0] is None:
            reasons.append(f"missing:{rf}")
    for f, (val, conf) in fields.items():
        if val is not None and conf < MIN_FIELD_CONFIDENCE:
            reasons.append(f"low_conf:{f}={conf}")
    needs_review = len(reasons) > 0
    return needs_review, needs_review, reasons
