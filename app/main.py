"""Deployment Layer — FastAPI service (v1 API).

Endpoints:
  Legacy: /health  /metrics  /documents/extract  /batch_jobs
  v1:     POST /v1/documents                    → upload, return document_id
          POST /v1/extraction_jobs              → async job over uploaded docs
          GET  /v1/extraction_jobs/{job_id}     → status + summary
          GET  /v1/documents/{doc_id}/result    → per-doc result
          POST /v1/documents/{doc_id}/feedback  → human correction (training data loop)
"""
from __future__ import annotations
import json
import logging
import time
import uuid
from pathlib import Path
import sys

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from docai import storage, metrics, ocr            # noqa
from docai.pipeline import process_document, get_kie  # noqa
from docai.orchestrator import run_job              # noqa
from docai.batcher import DynamicBatcher            # noqa

logger = logging.getLogger("docai.api")
logging.basicConfig(level=logging.INFO,
                    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')

MAX_FILE_BYTES = 20 * 1024 * 1024   # 20MB
ALLOWED_MIME = {"image/jpeg", "image/png", "image/tiff", "application/pdf"}

app = FastAPI(title="Hybrid Document AI", version="1.0.0")
ocr_batcher: DynamicBatcher | None = None


# ── middleware: request_id + latency ──────────────────────────────────────────

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - t0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-Ms"] = f"{latency*1000:.1f}"
    metrics.request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=str(response.status_code),
    ).inc()
    metrics.request_latency.labels(endpoint=request.url.path).observe(latency)
    logger.info(f"request_id={request_id} path={request.url.path} "
                f"status={response.status_code} latency_ms={latency*1000:.1f}")
    return response


# ── startup ───────────────────────────────────────────────────────────────────

ocr_pool = None


@app.on_event("startup")
async def _startup():
    storage.init()
    ocr.warmup()
    get_kie()
    global ocr_batcher, ocr_pool
    # CPU OCR via a process pool (true multi-core), fronted by the dynamic batcher.
    from docai.serving.ocr_pool import ProcessPoolOCR
    ocr_pool = ProcessPoolOCR()
    ocr_pool.warmup()
    ocr_batcher = DynamicBatcher(batch_fn=ocr_pool.run_many)
    ocr_batcher.start()


@app.on_event("shutdown")
async def _shutdown():
    if ocr_pool is not None:
        ocr_pool.shutdown()


# ── shared helpers ────────────────────────────────────────────────────────────

def _validate_file(data: bytes, filename: str):
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_FILE_BYTES//1024//1024}MB)")
    ext = Path(filename or "").suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".pdf"}:
        raise HTTPException(415, f"Unsupported file type: {ext}")


def _emit_doc_metrics(res):
    metrics.document_type_total.labels(doc_type=res.document_type).inc()
    if res.needs_human_review:
        metrics.human_review_total.inc()
    if res.route == "vlm_fallback":
        metrics.fallback_total.inc()
    if hasattr(res, "quality") and res.quality:
        q = res.quality
        if hasattr(q, "blur_score"):
            metrics.blur_observed.observe(q.blur_score)
    for field, fdata in (res.fields or {}).items():
        if hasattr(fdata, "confidence"):
            metrics.field_confidence_observed.observe(fdata.confidence)
        if fdata.value is None:
            metrics.missing_required_field_total.labels(field=field).inc()
    # Business safety: high-confidence wrong output guard
    avg_conf = sum(
        (f.confidence for f in res.fields.values() if hasattr(f, "confidence")), 0
    ) / max(len(res.fields), 1)
    if avg_conf > 0.85 and res.needs_human_review:
        metrics.high_risk_auto_accept_total.inc()


# ── legacy endpoints (backward-compat) ───────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "kie_version": get_kie().version}


@app.get("/metrics")
async def get_metrics():
    body, ctype = metrics.render()
    return Response(body, media_type=ctype)


@app.post("/documents/extract")
async def extract(file: UploadFile = File(...)):
    data = await file.read()
    _validate_file(data, file.filename or "doc")
    try:
        res = process_document(file.filename or "doc", data)
    except Exception as e:
        metrics.documents_processed_total.labels(status="error").inc()
        raise HTTPException(400, str(e))
    _emit_doc_metrics(res)
    metrics.documents_processed_total.labels(status="ok").inc()
    return JSONResponse(res.model_dump())


@app.post("/batch_jobs")
async def create_batch(files: list[UploadFile] = File(...)):
    if not 1 <= len(files) <= 50:
        raise HTTPException(400, "1..50 files allowed")
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    storage.create_job(job_id, len(files))
    docs = []
    for f in files:
        raw = await f.read()
        _validate_file(raw, f.filename or "doc")
        doc_id = f"{job_id}_{uuid.uuid4().hex[:6]}"
        storage.add_doc(doc_id, job_id)
        docs.append((doc_id, raw))
    metrics.batch_jobs_total.inc()
    summary = await run_job(job_id, docs)
    job = storage.get_job(job_id)
    return {"job_id": job_id, "status": job["status"],
            "total_documents": len(files), "summary": summary.model_dump()}


@app.get("/batch_jobs/{job_id}")
async def get_batch(job_id: str):
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.get("/batch_jobs/{job_id}/results")
async def get_batch_results(job_id: str):
    if not storage.get_job(job_id):
        raise HTTPException(404, "job not found")
    return {"job_id": job_id, "results": storage.get_results(job_id)}


# ── v1 API ────────────────────────────────────────────────────────────────────

_v1_docs: dict = {}   # in-memory store (swap for Redis/Postgres in production)


@app.post("/v1/documents", status_code=202)
async def v1_upload_document(
    file: UploadFile = File(...),
    x_idempotency_key: str = Header(default=""),
):
    """Upload a document; returns document_id. Extraction is lazy (trigger via /v1/extraction_jobs)."""
    data = await file.read()
    _validate_file(data, file.filename or "doc")

    # Idempotency: same key → same document_id (prevent double-submit)
    if x_idempotency_key and x_idempotency_key in _v1_docs:
        existing = _v1_docs[x_idempotency_key]
        return {"document_id": existing["document_id"], "status": "already_uploaded"}

    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    _v1_docs[doc_id] = {
        "document_id": doc_id,
        "filename": file.filename,
        "size_bytes": len(data),
        "status": "uploaded",
        "result": None,
        "_raw": data,
    }
    if x_idempotency_key:
        _v1_docs[x_idempotency_key] = _v1_docs[doc_id]
    return {"document_id": doc_id, "status": "uploaded", "size_bytes": len(data)}


@app.post("/v1/extraction_jobs", status_code=202)
async def v1_create_extraction_job(body: dict):
    """Trigger extraction on a list of already-uploaded document_ids."""
    doc_ids: list[str] = body.get("document_ids", [])
    if not 1 <= len(doc_ids) <= 50:
        raise HTTPException(400, "1..50 document_ids required")
    missing = [d for d in doc_ids if d not in _v1_docs]
    if missing:
        raise HTTPException(404, f"Unknown document_ids: {missing}")

    job_id = f"ejob_{uuid.uuid4().hex[:10]}"
    docs = [(d, _v1_docs[d]["_raw"]) for d in doc_ids]

    storage.create_job(job_id, len(doc_ids))
    for d in doc_ids:
        storage.add_doc(d, job_id)
    metrics.batch_jobs_total.inc()
    summary = await run_job(job_id, docs)
    job = storage.get_job(job_id)
    return {"job_id": job_id, "status": job["status"],
            "document_count": len(doc_ids), "summary": summary.model_dump()}


@app.get("/v1/extraction_jobs/{job_id}")
async def v1_get_job(job_id: str):
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.get("/v1/documents/{doc_id}/result")
async def v1_get_result(doc_id: str):
    doc = _v1_docs.get(doc_id)
    if not doc:
        raise HTTPException(404, "document not found")
    results = storage.get_results(None)   # returns all; filter by doc_id
    doc_result = next((r for r in results if r.get("document_id") == doc_id), None)
    return {"document_id": doc_id, "status": doc.get("status"), "result": doc_result}


class FeedbackPayload(BaseModel):
    corrected_fields: dict           # e.g. {"total_amount": "12.50", "date": "2024-01-15"}
    correction_type: str = "field_value"   # field_value | wrong_doc_type | false_review_flag
    # Optional: original model extraction before correction. When provided,
    # stored alongside the correction so the training script can compute
    # token-level BIO labels for LayoutLMv3 without re-running inference.
    original_extraction: dict | None = None
    pipeline_flags: list[str] | None = None   # reasons[] from DocumentResult


@app.post("/v1/documents/{doc_id}/feedback", status_code=201)
async def v1_submit_feedback(doc_id: str, payload: FeedbackPayload):
    """Human correction → append to feedback log for active-learning retraining.

    Schema is LayoutLMv3-training-ready: a conversion script pairs this entry
    with the source image (identified by filename) and the OCR token stream to
    produce (words, bboxes, labels) training samples.
    """
    doc = _v1_docs.get(doc_id)
    if not doc:
        raise HTTPException(404, "document not found")

    feedback_dir = Path("docs/feedback")
    feedback_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "document_id": doc_id,
        "filename": doc.get("filename"),
        "original_extraction": payload.original_extraction,
        "corrected_fields": payload.corrected_fields,
        "correction_type": payload.correction_type,
        "pipeline_flags": payload.pipeline_flags,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    log_path = feedback_dir / "corrections.jsonl"
    with open(log_path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")

    metrics.human_feedback_total.labels(correction_type=payload.correction_type).inc()
    logger.info(f"feedback doc_id={doc_id} type={payload.correction_type} "
                f"fields={list(payload.corrected_fields.keys())}")
    return {"status": "accepted", "document_id": doc_id,
            "note": "correction logged for retraining pipeline"}
