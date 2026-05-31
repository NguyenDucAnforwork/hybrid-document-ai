"""Single-document pipeline glue (Processing Layer orchestration of stages)."""
from __future__ import annotations
import time
import cv2
import numpy as np

from . import metrics
from .quality import check_quality
from .ocr import run_ocr, OCR_VERSION
from .kie import KIEModel
from .router import route_decision
from .vlm import vlm_extract
from .schemas import DocumentResult, FieldValue
from .registry import active_path
from .config import ALL_FIELDS

_kie: KIEModel | None = None


def get_kie() -> KIEModel:
    global _kie
    if _kie is None:
        p = active_path("kie")
        _kie = KIEModel.load(p) if p and p.exists() else KIEModel()
    return _kie


def _decode(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode image")
    return img


def process_document(doc_id: str, image_bytes: bytes) -> DocumentResult:
    t0 = time.perf_counter()
    img = _decode(image_bytes)

    q = check_quality(img)
    metrics.blur_observed.observe(q.blur_score)

    # Preprocess: upscale small scans so OCR has enough resolution (real docs).
    h, w = img.shape[:2]
    if min(h, w) < 720:
        scale = 720.0 / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    with metrics.stage_latency.labels("ocr").time():
        tokens = run_ocr(img)

    kie = get_kie()
    with metrics.stage_latency.labels("kie").time():
        extracted = kie.extract(tokens)

    needs_review, should_vlm, reasons = route_decision(extracted)
    route = "traditional_ocr"

    if should_vlm:
        vlm_fields = vlm_extract(image_bytes)
        if vlm_fields:
            route = "vlm_fallback"
            metrics.fallback_total.inc()
            from .kie import norm_field
            for f in ALL_FIELDS:
                raw = vlm_fields.get(f)
                if raw is not None:
                    nv = norm_field(f, str(raw))   # normalize VLM output to schema
                    if nv is not None:
                        extracted[f] = (nv, 0.80)
            needs_review, _, reasons = route_decision(extracted)

    fields = {}
    for f in ALL_FIELDS:
        val, conf = extracted.get(f, (None, 0.0))
        fields[f] = FieldValue(value=val, confidence=conf)
        if val is not None:
            metrics.field_confidence_observed.observe(conf)
            if conf < 0.75:
                metrics.low_confidence_total.inc()

    needs_review = needs_review or (not q.quality_pass)   # severe blur -> human
    if needs_review:
        metrics.human_review_total.inc()
    metrics.documents_processed_total.labels("needs_review" if needs_review else "success").inc()
    metrics.stage_latency.labels("total").observe(time.perf_counter() - t0)

    return DocumentResult(
        document_id=doc_id, route=route, fields=fields, quality=q,
        needs_human_review=needs_review,
        model_versions={"ocr": OCR_VERSION, "kie": kie.version},
    )
