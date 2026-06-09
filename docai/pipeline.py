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
from .layoutlmv3_onnx import get_layoutlmv3_onnx, onnx_mode

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
    if not image_bytes:
        raise ValueError("cannot decode image: empty bytes")
    arr = np.frombuffer(image_bytes, np.uint8)
    try:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except cv2.error:
        raise ValueError("cannot decode image: corrupted file")
    if img is None:
        raise ValueError("cannot decode image: unsupported format")
    return img


def _deskew(img: np.ndarray, angle: float) -> np.ndarray:
    """Correct image skew before OCR. angle from minAreaRect convention."""
    if abs(angle) < 0.5:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _extract_numeric_amounts(tokens: list[dict]) -> list[float]:
    """Extract plausible currency amounts from OCR tokens.

    Filters out years (1900-2100), sub-cent noise (< 0.1), and values that
    exceed the receipt sanity ceiling (> 50 000).
    """
    import re
    amounts = []
    for t in tokens:
        text = t.get("text", "").replace(",", "").replace(" ", "").strip()
        if not re.fullmatch(r"\d{1,6}(\.\d{1,2})?", text):
            continue
        val = float(text)
        if val < 0.1 or val > 50_000:
            continue
        if 1900.0 <= val <= 2100.0:  # looks like a year
            continue
        amounts.append(val)
    return amounts


def _cross_validate_total(extracted: dict, tokens: list[dict]) -> list[str]:
    """Flag when a clearly larger currency amount is visible in the OCR corpus.

    The grand total on a receipt is the largest monetary value. If the KIE
    picked a subtotal or tax line, the real total is still visible as a bigger
    number in the token stream — catch it here rather than relying on the
    miscalibrated confidence score (ECE ≈ 0.51).
    """
    total_val, _ = extracted.get("total_amount", (None, 0.0))
    if total_val is None:
        return []
    try:
        total = float(total_val)
    except (TypeError, ValueError):
        return []
    if total <= 0:
        return []

    amounts = _extract_numeric_amounts(tokens)
    if not amounts:
        return []

    max_amount = max(amounts)
    # 2.1x: conservative enough to avoid flagging correct receipts where the second
    # price tier is around half the total, but catches barcode/phone numbers that are
    # 2-3x the actual total (empirically validated on SROIE n=80).
    if max_amount >= total * 2.1 and max_amount != total:
        return [f"total_may_be_subtotal:got={total},max_seen={max_amount:.2f}"]
    return []


def _sanity_check(extracted: dict, doc_type: str,
                  tokens: list[dict] | None = None) -> list[str]:
    """Plausibility guardrails — catches KIE picking wrong tokens (phone number as total,
    transposed year as date). Returns list of failure reasons → triggers needs_review."""
    import re
    flags = []

    date_val = (extracted.get("date") or (None,))[0]
    if date_val:
        m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", str(date_val))
        if m:
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if not (2000 <= year <= 2035):
                flags.append(f"implausible_year:{year}")
            if not (1 <= month <= 12):
                flags.append(f"implausible_month:{month}")
            if not (1 <= day <= 31):
                flags.append(f"implausible_day:{day}")

    if doc_type == "receipt":
        total_val = (extracted.get("total_amount") or (None,))[0]
        if total_val is not None:
            try:
                amount = float(total_val)
                if amount <= 0:
                    flags.append(f"nonpositive_total:{amount}")
                elif amount < 0.5:
                    # Sub-50-cent amounts are almost certainly a decimal/line-item confusion
                    flags.append(f"implausibly_small_total:{amount}")
                elif amount > 50_000:
                    # POS receipts >50k are almost always a barcode/phone extracted as total
                    flags.append(f"implausible_total:{amount}")
            except (TypeError, ValueError):
                pass

        # Cross-validate: if a much larger amount is visible in the corpus,
        # we likely picked a subtotal. Deterministic; not affected by conf calibration.
        if tokens:
            flags.extend(_cross_validate_total(extracted, tokens))

    return flags


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
    try:
        img = _decode(image_bytes)
    except ValueError as e:
        from .schemas import QualityReport as _QR
        from .config import ALL_FIELDS
        dummy_q = _QR(blur_score=0, is_blurry=False, is_dark=False,
                                low_resolution=False, is_rotated=False,
                                quality_pass=False, issues=["decode_error"],
                                action="request_reupload")
        metrics.decode_error_total.inc()
        return DocumentResult(
            document_id=doc_id, document_type="unknown", route="error",
            fields={f: FieldValue() for f in ALL_FIELDS},
            quality=dummy_q, needs_human_review=True,
            model_versions={"ocr": "n/a", "kie": "n/a", "doctype": "n/a"},
            error=str(e),
        )

    q = check_quality(img)
    metrics.blur_observed.observe(q.blur_score)

    # Preprocess: cap oversized images to avoid multi-second latency spikes.
    # Resize down to max 3000px on the longest side before any other processing.
    h, w = img.shape[:2]
    if max(h, w) > 3000:
        scale = 3000.0 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # Preprocess: upscale small scans so OCR has enough resolution (real docs).
    if min(h, w) < 720:
        scale = 720.0 / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Preprocess: correct small-angle skew (5°–44°) before OCR.
    # Near-90° angles from minAreaRect are unreliable (landscape vs. portrait ambiguity)
    # and cause catastrophic rotation errors when applied. CJK filter handles hallucinations.
    if q.is_rotated and abs(q.skew_angle) < 45:
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
    lm_ver = "disabled"

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
            lm = get_layoutlmv3_onnx()
            mode = onnx_mode()
            if lm is not None and mode in {"merchant", "all"}:
                lm_pred = lm.predict(img, tokens)
                lm_ver = lm.version
                merchant_val = lm_pred.get("merchant_name")
                cur_val, cur_conf = extracted.get("merchant_name", (None, 0.0))
                if merchant_val and (cur_val is None or cur_conf < 0.75):
                    extracted["merchant_name"] = (merchant_val, max(cur_conf, 0.72))
                if mode == "all":
                    for field in ("date", "total_amount"):
                        val = lm_pred.get(field)
                        if val is not None:
                            cur_v, cur_c = extracted.get(field, (None, 0.0))
                            if cur_v is None or cur_c < 0.60:
                                extracted[field] = (val, max(cur_c, 0.65))

    extracted = _filter_cjk_hallucination(extracted, tokens)
    sanity_flags = _sanity_check(extracted, dt.name, tokens)
    needs_review, should_vlm, reasons = route_decision(extracted, dt.required)
    if sanity_flags:
        needs_review = True
        reasons.extend(sanity_flags)
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
        model_versions={"ocr": OCR_VERSION, "kie": kie_ver, "layoutlmv3": lm_ver,
                        "doctype": f"{clf.version}({dt_conf:.2f})"},
    )
