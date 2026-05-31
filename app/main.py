"""Deployment Layer — FastAPI service.

Endpoints (PLAN.md §Module A): /health /metrics /documents/extract /batch_jobs.
Serving: OCR dynamic batcher + in-process async orchestrator (memory queue).
"""
from __future__ import annotations
import uuid
from pathlib import Path
import sys

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from docai import storage, metrics, ocr            # noqa
from docai.pipeline import process_document, get_kie  # noqa
from docai.orchestrator import run_job              # noqa
from docai.batcher import DynamicBatcher            # noqa

app = FastAPI(title="Hybrid Document AI", version="0.1.0")
ocr_batcher: DynamicBatcher | None = None


@app.on_event("startup")
async def _startup():
    storage.init()
    ocr.warmup()        # avoid cold-start on first real request
    get_kie()           # load active KIE model version
    global ocr_batcher
    ocr_batcher = DynamicBatcher(batch_fn=lambda imgs: [ocr.run_ocr(i) for i in imgs])
    ocr_batcher.start()


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
    try:
        res = process_document(file.filename or "doc", data)
    except Exception as e:
        raise HTTPException(400, str(e))
    return JSONResponse(res.model_dump())


@app.post("/batch_jobs")
async def create_batch(files: list[UploadFile] = File(...)):
    if not 1 <= len(files) <= 50:
        raise HTTPException(400, "1..50 files allowed")
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    storage.create_job(job_id, len(files))
    docs = []
    for f in files:
        doc_id = f"{job_id}_{uuid.uuid4().hex[:6]}"
        storage.add_doc(doc_id, job_id)
        docs.append((doc_id, await f.read()))
    metrics.batch_jobs_total.inc()
    summary = await run_job(job_id, docs)   # async; returns when done (in-process)
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
