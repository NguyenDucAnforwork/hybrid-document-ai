"""Per-document stage profiler (Processing Layer).

Times each pipeline stage into Prometheus AND an in-band per-request `timings`
dict, so the profiler script can report per-stage p50/p95/p99 and the API can
return where time went. Cheap by design: a contextvar + perf_counter, no deps.

Sub-timings (e.g. ocr_serialize, ocr_pool_wait) are added with `record()` so the
process-pool path can attribute serialize/worker/wait cost separately.
"""
from __future__ import annotations
import time
import contextvars
from contextlib import contextmanager

from . import metrics

# Request-scoped accumulator. None outside a profiled call.
_timings: contextvars.ContextVar["dict | None"] = contextvars.ContextVar(
    "docai_timings", default=None)


def begin() -> dict:
    """Start a request-scoped timing collector and return the dict."""
    d: dict[str, float] = {}
    _timings.set(d)
    return d


def end() -> None:
    """Clear the request-scoped collector."""
    _timings.set(None)


def record(name: str, ms: float) -> None:
    """Add a (sub-)timing in milliseconds to the current collector, if any."""
    sink = _timings.get()
    if sink is not None:
        sink[name] = round(sink.get(name, 0.0) + ms, 3)


@contextmanager
def collect():
    """Context-managed collector (used by scripts/tests)."""
    tok = _timings.set({})
    try:
        yield _timings.get()
    finally:
        _timings.reset(tok)


@contextmanager
def stage(name: str):
    """Time a pipeline stage into both Prometheus and the in-band collector."""
    t0 = time.perf_counter()
    try:
        with metrics.stage_latency.labels(name).time():
            yield
    finally:
        record(name, (time.perf_counter() - t0) * 1000.0)
