"""Async batch orchestration: per-doc state machine, retry, dead-letter.

Memory queue backend (default; no docker/redis on this box). Same semantics as
Redis Streams consumer-group + retry counter + dead-letter — swap by config.
"""
from __future__ import annotations
import asyncio
from . import storage, metrics
from .pipeline import process_document
from .schemas import BatchSummary

MAX_RETRIES = 2
_dead_letter: list[str] = []


async def _process_one(doc_id: str, image_bytes: bytes):
    storage.set_doc(doc_id, state="processing")
    for attempt in range(MAX_RETRIES + 1):
        try:
            res = await asyncio.to_thread(process_document, doc_id, image_bytes)
            state = "needs_review" if res.needs_human_review else "completed"
            storage.set_doc(doc_id, state=state, result=res.model_dump())
            return res
        except Exception as e:  # partial failure isolation
            storage.set_doc(doc_id, inc_retry=True, error=str(e))
            if attempt < MAX_RETRIES:
                await asyncio.sleep(0.1 * (attempt + 1))
                continue
            storage.set_doc(doc_id, state="failed", error=str(e))
            _dead_letter.append(doc_id)
            metrics.documents_processed_total.labels("failed").inc()
            return None


async def run_job(job_id: str, docs: list[tuple[str, bytes]], concurrency: int = 4):
    storage.set_job(job_id, status="processing")
    sem = asyncio.Semaphore(concurrency)

    async def guarded(doc_id, data):
        async with sem:
            return await _process_one(doc_id, data)

    results = await asyncio.gather(*(guarded(d, b) for d, b in docs))

    summary = BatchSummary(total=len(docs))
    for r in results:
        if r is None:
            summary.failed += 1
        elif r.needs_human_review:
            summary.needs_review += 1
            summary.success += 1
        else:
            summary.success += 1
        if r is not None and r.route == "vlm_fallback":
            summary.vlm_fallback += 1

    if summary.failed == len(docs):
        status = "failed"
    elif summary.failed > 0:
        status = "partial_completed"
    else:
        status = "completed"
    storage.set_job(job_id, status=status, summary=summary.model_dump())
    return summary
