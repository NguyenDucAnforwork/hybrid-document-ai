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


def _crop(image_bgr: np.ndarray, box) -> np.ndarray:
    xs = [p[0] for p in box]; ys = [p[1] for p in box]
    x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
    x1, y1 = int(max(xs)), int(max(ys))
    c = image_bgr[y0:y1, x0:x1]
    return c if c.size else image_bgr


def run_ocr(image_bgr: np.ndarray) -> list[dict]:
    """Return tokens: [{text, bbox:[x0,y0,x1,y1], conf}].

    Detector is always RapidOCR. The recognizer is swappable (WP-3): with
    DOCAI_OCR_RECOGNIZER=ppocr_vi_mcocr_ft the fine-tuned CRNN re-recognizes each
    detected box; the token schema is identical so KIE/router/pipeline are
    unchanged (enables clean on/off + downstream anti-regression)."""
    from . import config
    result, _ = _get_engine()(image_bgr)
    rows = result or []

    rec = None
    if config.OCR_RECOGNIZER != "rapidocr_default":
        from .ocr_recognizer import get_recognizer
        rec = get_recognizer()           # None if artifacts missing -> graceful fallback

    use_ft = rec is not None
    if use_ft and config.OCR_RECOGNIZER == "auto":
        # Task D: route by language — only use the VI recognizer on Vietnamese docs.
        use_ft = _diacritic_ratio(" ".join(t for _, t, _ in rows)) >= config.OCR_VI_DIACRITIC_MIN

    if use_ft and rows:
        crops = [_crop(image_bgr, box) for box, _, _ in rows]
        ft = rec.recognize(crops)
        rows = [(box, ft[i][0], ft[i][1]) for i, (box, _, _) in enumerate(rows)]

    tokens = []
    for box, text, conf in rows:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        tokens.append({
            "text": text,
            "bbox": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
            "conf": float(conf),
        })

    if config.LINE_REGROUP:               # Task B: split horizontal two-field merges
        from .line_grouping import regroup_tokens
        tokens = regroup_tokens(tokens)
    return tokens


_VN_DIAC = set("ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
               "ÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỹỴ")


def _diacritic_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c in _VN_DIAC for c in text) / len(text)


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
