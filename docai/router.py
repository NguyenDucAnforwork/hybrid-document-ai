"""Hybrid confidence router (Processing Layer).

Decides traditional_ocr vs vlm_fallback per document. 80-90% of traffic stays
on the cheap/fast traditional path; only low-confidence docs escalate to VLM.
"""
from __future__ import annotations
from .config import REQUIRED_FIELDS, MIN_FIELD_CONFIDENCE, FIELD_CONFIDENCE_THRESHOLDS


def route_decision(fields: dict, required: list[str] | None = None) -> tuple[bool, bool, list[str]]:
    """Return (needs_review, should_vlm, reasons). `required` is per doc-type.

    Each field is evaluated against its own calibrated threshold
    (FIELD_CONFIDENCE_THRESHOLDS) rather than a single global value.
    total_amount uses a higher bar (0.80) because its ECE ≈ 0.51 means the
    model is systematically overconfident on wrong extractions.
    """
    # Review is driven by the REQUIRED fields only. Optional fields being
    # low-confidence must not saturate the router — else every real-world doc
    # gets flagged and the router loses all discriminating power.
    reasons = []
    for rf in (required if required is not None else REQUIRED_FIELDS):
        val, conf = fields.get(rf, (None, 0.0))
        threshold = FIELD_CONFIDENCE_THRESHOLDS.get(rf, MIN_FIELD_CONFIDENCE)
        if val is None:
            reasons.append(f"missing:{rf}")
        elif conf < threshold:
            reasons.append(f"low_conf:{rf}={conf:.2f}")
    needs_review = len(reasons) > 0
    return needs_review, needs_review, reasons
