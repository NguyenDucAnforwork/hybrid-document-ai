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


_last_stats = {"rerec": 0, "total_boxes": 0}     # WP-3 ablation: last run_ocr re-recognition count


def _aabb(box):
    xs = [p[0] for p in box]; ys = [p[1] for p in box]
    return [min(xs), min(ys), max(xs), max(ys)]


def _safe_crop(image_bgr, b):
    c = image_bgr[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
    return c if c.size else image_bgr


def _is_field_critical(text: str, aabb, H: int) -> bool:
    """A box worth the FT recognizer: top region (merchant/address), or date/money/
    anchor content (timestamp/total). Skips generic mid-receipt lines (Task B)."""
    from .kie import DATE_RE, MONEY_RE, ANCHORS
    cy = (aabb[1] + aabb[3]) / 2
    if cy < 0.30 * max(H, 1):
        return True
    if DATE_RE.search(text or "") or MONEY_RE.search(text or ""):
        return True
    low = (text or "").lower()
    for f in ("date", "total_amount"):
        if any(a and a in low for a in ANCHORS.get(f, [])):
            return True
    return False


def _poly(aabb):
    x0, y0, x1, y1 = aabb
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _projection_split_boxes(aabbs, image_bgr, ratio):
    """Expand tall boxes into per-row sub-boxes (WP-3 Task E). Order preserved."""
    from .line_grouping import projection_split_box
    import statistics
    heights = [b[3] - b[1] for b in aabbs] or [1]
    med = statistics.median(heights)
    out = []
    for b in aabbs:
        if (b[3] - b[1]) > ratio * med:
            subs = projection_split_box(image_bgr, b)
            if subs and len(subs) >= 2:
                out.extend(subs); continue
        out.append(b)
    return out


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
    if use_ft and config.OCR_RECOGNIZER == "auto" and rows:
        # Task D: route by language. NOTE: the default (Chinese-dict) recognizer cannot
        # emit Vietnamese diacritics, so we must probe with the FT recognizer itself on a
        # few boxes — measuring diacritics on default text would never detect Vietnamese.
        probe_boxes = [_aabb(b) for b, _, _ in rows[:8]]
        probe = rec.recognize([_safe_crop(image_bgr, b) for b in probe_boxes])
        use_ft = _diacritic_ratio(" ".join(t for t, _ in probe)) >= config.OCR_VI_DIACRITIC_MIN

    if use_ft and rows:
        default_rows = rows
        boxes = [_aabb(box) for box, _, _ in default_rows]
        H = image_bgr.shape[0]
        if config.OCR_FIELD_CRITICAL and not config.PROJECTION_SPLIT:
            crit = [i for i, b in enumerate(boxes)
                    if _is_field_critical(default_rows[i][1], b, H)]
            crops = [_safe_crop(image_bgr, boxes[i]) for i in crit]
            ft = rec.recognize(crops)
            rows = list(default_rows)                 # keep default text for non-critical
            for j, i in enumerate(crit):
                rows[i] = (default_rows[i][0], ft[j][0], ft[j][1])
            _last_stats.update(rerec=len(crit), total_boxes=len(default_rows))
        else:
            if config.PROJECTION_SPLIT and boxes:
                boxes = _projection_split_boxes(boxes, image_bgr, config.PROJECTION_SPLIT_RATIO)
            crops = [_safe_crop(image_bgr, b) for b in boxes]
            ft = rec.recognize(crops)
            rows = [(_poly(b), ft[i][0], ft[i][1]) for i, b in enumerate(boxes)]
            _last_stats.update(rerec=len(boxes), total_boxes=len(default_rows))
    else:
        _last_stats.update(rerec=0, total_boxes=len(rows))

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
