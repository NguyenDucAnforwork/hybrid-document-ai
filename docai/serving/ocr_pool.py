"""Serving Layer — CPU OCR via a PROCESS pool (not a thread pool).

Why a process pool: OCR is CPU-bound. The previous serving path ran OCR in a
single thread (`asyncio.to_thread` over a sequential list comprehension), so
concurrent requests serialized on one core. A ProcessPoolExecutor gives true
multi-core parallelism, sidestepping the GIL for the Python-side work.

Anti-oversubscription contract (see docs): with W workers each running ONNX
Runtime at T intra-op threads, effective load is W*T. Defaults are deliberately
conservative (W=2, T=1, inter_op=1) so 2 workers * 1 thread != core blow-up.
Always sweep (workers x intra_threads x concurrency) — tuning one alone misleads.

Sub-cost attribution: run_many_timed() returns serialize_ms / per-worker
worker_ms / pool_wait_ms so a throughput shortfall can be debugged (encode/decode
overhead vs. worker compute vs. queueing).
"""
from __future__ import annotations
import os
import time

import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor


def _worker_init(intra_threads: int) -> None:
    # Pin each worker's ONNX Runtime thread budget BEFORE the engine is built.
    os.environ["DOCAI_OCR_INTRA_THREADS"] = str(intra_threads)
    os.environ["DOCAI_OCR_INTER_THREADS"] = "1"


def _worker_run(buf: bytes):
    """Decode bytes -> run OCR. Returns (tokens, worker_ms). Engine cached per process."""
    from docai import ocr as _ocr
    t0 = time.perf_counter()
    arr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    tokens = _ocr.run_ocr(img)
    return tokens, (time.perf_counter() - t0) * 1000.0


class ProcessPoolOCR:
    def __init__(self, workers: int | None = None, intra_threads: int | None = None):
        self.workers = int(workers or os.environ.get("DOCAI_OCR_WORKERS", "2"))
        self.intra = int(intra_threads or os.environ.get("DOCAI_OCR_INTRA_THREADS", "1"))
        self._pool: ProcessPoolExecutor | None = None

    def start(self) -> None:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=_worker_init,
                initargs=(self.intra,),
            )

    def warmup(self) -> None:
        """Pay model-load cold start now, in every worker (avoid first-request lag)."""
        self.start()
        dummy = np.full((64, 256, 3), 255, np.uint8)
        _, buf = cv2.imencode(".jpg", dummy)
        list(self._pool.map(_worker_run, [buf.tobytes()] * self.workers))

    def run_many(self, images: list[np.ndarray]) -> list[list[dict]]:
        return self.run_many_timed(images)[0]

    def run_many_timed(self, images: list[np.ndarray]):
        """Returns (list_of_tokens, meta) with serialize/worker/wait breakdown."""
        self.start()
        from .. import profiling, metrics

        ts = time.perf_counter()
        bufs = []
        for im in images:
            ok, b = cv2.imencode(".jpg", im)
            bufs.append(b.tobytes())
        serialize_ms = (time.perf_counter() - ts) * 1000.0

        t1 = time.perf_counter()
        out = list(self._pool.map(_worker_run, bufs))
        wall_ms = (time.perf_counter() - t1) * 1000.0

        tokens = [o[0] for o in out]
        worker_ms = [round(o[1], 3) for o in out]
        # Wall minus the slowest worker ≈ queueing/dispatch/IPC overhead.
        pool_wait_ms = max(0.0, wall_ms - (max(worker_ms) if worker_ms else 0.0))

        meta = {
            "serialize_ms": round(serialize_ms, 3),
            "worker_ms": worker_ms,
            "pool_wall_ms": round(wall_ms, 3),
            "pool_wait_ms": round(pool_wait_ms, 3),
            "workers": self.workers,
            "intra_threads": self.intra,
        }
        profiling.record("ocr_serialize", serialize_ms)
        profiling.record("ocr_pool_wait", pool_wait_ms)
        try:
            metrics.ocr_batch_size.observe(len(images))
            metrics.ocr_pool_workers.set(self.workers)
        except Exception:
            pass
        return tokens, meta

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
