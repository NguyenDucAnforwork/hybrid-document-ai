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
from . import doctypes
from .classifier import get_classifier
from .statement import extract_statement, reconcile

_kie: KIEModel | None = None


def get_kie() -> KIEModel:
    global _kie
    if _kie is None:
        p = active_path("kie")
        _kie = KIEModel.load(p) if p and p.exists() else KIEModel()
    return _kie


def _normalize_txns(txns) -> list[dict]:
    """Normalize VLM-returned transactions to our schema (ISO date, float amounts)."""
    from .kie import norm_date
    from .statement import signed_money
    out = []
    for t in (txns or []):
        if not isinstance(t, dict):
            continue
        out.append({
            "date": norm_date(str(t.get("date", ""))),
            "description": (str(t["description"]).lower() if t.get("description") else None),
            "amount": signed_money(str(t.get("amount", ""))),
            "balance": signed_money(str(t.get("balance", ""))),
        })
    return out


def _decode(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode image")
    return img


def _deskew(img: np.ndarray, angle: float) -> np.ndarray:
    """Correct image skew before OCR. angle from minAreaRect convention."""
    if abs(angle) < 0.5:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for c in text if '一' <= c <= '鿿' or '　' <= c <= '〿')
    return cjk / len(text)


def _filter_cjk_hallucination(extracted: dict, tokens: list[dict]) -> dict:
    """Null out field values that are CJK hallucinations from PP-OCRv4.

    PP-OCRv4 is primarily trained on Chinese text and can hallucinate CJK
    characters when the input image is rotated or low-resolution Latin text.
    If the overall OCR corpus is not a Chinese document (corpus CJK ratio < 0.3)
    but a specific field value is predominantly CJK, it's a hallucination.
    """
    corpus = " ".join(t.get("text", "") for t in tokens)
    if _cjk_ratio(corpus) > 0.3:
        return extracted  # looks like a real Chinese document — don't filter

    filtered = {}
    for field, payload in extracted.items():
        val, conf = payload
        if val is not None and isinstance(val, str) and _cjk_ratio(val) > 0.3:
            filtered[field] = (None, 0.0)  # hallucination → null, triggers human review
        else:
            filtered[field] = payload
    return filtered


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

    # Preprocess: correct skew before OCR — PP-OCRv4 hallucinates CJK on rotated Latin text.
    if q.is_rotated:
        img = _deskew(img, q.skew_angle)

    with metrics.stage_latency.labels("ocr").time():
        tokens = run_ocr(img)

    # Document-type router: receipt vs bank_statement (multi-document).
    H, W = img.shape[:2]
    clf = get_classifier()
    doc_type, dt_conf = clf.predict(tokens, W, H)
    dt = doctypes.get(doc_type)
    line_items = []
    kie_ver = "n/a"

    stmt_recon = None
    with metrics.stage_latency.labels("kie").time():
        if dt.name == "bank_statement":
            extracted, line_items = extract_statement(tokens)
            stmt_recon = reconcile(line_items, extracted.get("opening_balance", (None,))[0],
                                   extracted.get("closing_balance", (None,))[0])
            kie_ver = "statement-rules+table"
        elif dt.name == "payment_order":
            from .kv import kv_extract
            extracted = kv_extract(tokens, dt)
            kie_ver = "kv-rules"
        else:
            kie = get_kie()
            extracted = kie.extract(tokens)
            kie_ver = kie.version

    extracted = _filter_cjk_hallucination(extracted, tokens)
    needs_review, should_vlm, reasons = route_decision(extracted, dt.required)
    # Statement-specific trigger: if the parsed table fails balance reconciliation,
    # the rule parser got it wrong -> escalate to the VLM to re-read the table.
    if stmt_recon is not None and stmt_recon < 0.7:
        should_vlm = True
        needs_review = True          # unreliable table -> flag for human even if VLM off
        reasons.append(f"low_table_reconciliation:{stmt_recon}")
    route = "traditional_ocr"

    if should_vlm:
        vlm_fields = vlm_extract(image_bytes, prompt=dt.vlm_prompt)
        if vlm_fields:
            route = "vlm_fallback"
            metrics.fallback_total.inc()
            from .kie import norm_field
            for f in dt.fields:
                raw = vlm_fields.get(f)
                if raw is not None:
                    nv = norm_field(f, str(raw)) if dt.name == "receipt" else raw
                    if nv is not None:
                        extracted[f] = (nv, 0.80)
            if dt.name == "bank_statement" and vlm_fields.get("transactions"):
                line_items = _normalize_txns(vlm_fields["transactions"])
            needs_review, _, reasons = route_decision(extracted, dt.required)
            if stmt_recon is not None:   # re-check table after VLM
                r2 = reconcile(line_items, extracted.get("opening_balance", (None,))[0],
                               extracted.get("closing_balance", (None,))[0])
                needs_review = needs_review or (r2 < 0.7)

    fields = {}
    for f in dt.fields:
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
        document_id=doc_id, document_type=dt.name, route=route, fields=fields,
        line_items=line_items, quality=q, needs_human_review=needs_review,
        model_versions={"ocr": OCR_VERSION, "kie": kie_ver,
                        "doctype": f"{clf.version}({dt_conf:.2f})"},
    )
