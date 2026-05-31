"""Prometheus metrics + simple drift signals (MLOps monitoring)."""
from __future__ import annotations
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

documents_processed_total = Counter("documents_processed_total", "docs processed", ["status"])
batch_jobs_total = Counter("batch_jobs_total", "batch jobs created")
stage_latency = Histogram("stage_latency_seconds", "per-stage latency", ["stage"])
fallback_total = Counter("vlm_fallback_total", "docs routed to VLM fallback")
low_confidence_total = Counter("low_confidence_total", "docs with a low-confidence field")
human_review_total = Counter("human_review_total", "docs flagged for human review")
queue_size = Gauge("queue_size", "pending docs in queue", ["stage"])

# Drift signals: distributions of input quality + output confidence.
blur_observed = Histogram("input_blur_score", "input blur score distribution",
                          buckets=(20, 50, 100, 200, 400, 800))
field_confidence_observed = Histogram("field_confidence", "field confidence distribution",
                                      buckets=(0.3, 0.5, 0.6, 0.7, 0.75, 0.85, 0.95, 1.0))


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
