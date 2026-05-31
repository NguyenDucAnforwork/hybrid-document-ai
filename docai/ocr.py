"""OCR engine (Processing Layer): RapidOCR (PP-OCRv4/v5 ONNX) wrapper.

Interface lets us swap to Triton-served ONNX in production without code change.
"""
from __future__ import annotations
import numpy as np

_engine = None
OCR_VERSION = "rapidocr-onnxruntime/pp-ocrv4"


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def run_ocr(image_bgr: np.ndarray) -> list[dict]:
    """Return tokens: [{text, bbox:[x0,y0,x1,y1], conf}]."""
    result, _ = _get_engine()(image_bgr)
    tokens = []
    for box, text, conf in (result or []):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        tokens.append({
            "text": text,
            "bbox": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
            "conf": float(conf),
        })
    return tokens


def warmup():
    """Force model load + first inference (avoid cold-start in real requests)."""
    dummy = np.full((64, 256, 3), 255, np.uint8)
    try:
        run_ocr(dummy)
    except Exception:
        pass
