"""OCR engine (Processing Layer): RapidOCR (PP-OCRv4/v5 ONNX) wrapper.

Interface lets us swap to Triton-served ONNX in production without code change.
"""
from __future__ import annotations
import os
import numpy as np

_engine = None
OCR_VERSION = "rapidocr-onnxruntime/pp-ocrv4"


def _engine_kwargs() -> dict:
    """ONNX Runtime thread budget. Defaults intra=1, inter=1 to avoid CPU
    oversubscription when multiple OCR worker processes run in parallel
    (effective load = workers * intra_threads). Override per-deployment via
    DOCAI_OCR_INTRA_THREADS / DOCAI_OCR_INTER_THREADS."""
    intra = int(os.environ.get("DOCAI_OCR_INTRA_THREADS", "1"))
    inter = int(os.environ.get("DOCAI_OCR_INTER_THREADS", "1"))
    return {"intra_op_num_threads": intra, "inter_op_num_threads": inter}


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        try:
            _engine = RapidOCR(**_engine_kwargs())
        except TypeError:
            # Older RapidOCR signatures may not accept thread kwargs.
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


def run_ocr_batch(images: list[np.ndarray]) -> list[list[dict]]:
    """Sequential batch entry (in-process). True parallelism comes from
    docai.serving.ocr_pool.ProcessPoolOCR; this keeps a single-process fallback."""
    return [run_ocr(im) for im in images]


def warmup():
    """Force model load + first inference (avoid cold-start in real requests)."""
    dummy = np.full((64, 256, 3), 255, np.uint8)
    try:
        run_ocr(dummy)
    except Exception:
        pass
