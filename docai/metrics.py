"""Prometheus metrics + simple drift signals (MLOps monitoring)."""
from __future__ import annotations
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# --- System ---
documents_processed_total = Counter("documents_processed_total", "docs processed", ["status"])
batch_jobs_total = Counter("batch_jobs_total", "batch jobs created")
stage_latency = Histogram("stage_latency_seconds", "per-stage latency", ["stage"],
                          buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 5.0, 10.0))
queue_size = Gauge("queue_size", "pending docs in queue", ["stage"])
ocr_batch_size = Histogram("ocr_batch_size", "images per OCR pool batch",
                           buckets=(1, 2, 4, 8, 16, 32))
ocr_pool_workers = Gauge("ocr_pool_workers", "configured OCR process-pool workers")
request_count = Counter("request_count_total", "total API requests", ["method", "endpoint", "status_code"])
request_latency = Histogram("request_latency_seconds", "API request latency", ["endpoint"],
                            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0))

# --- Model quality proxy ---
fallback_total = Counter("vlm_fallback_total", "docs routed to VLM fallback")
low_confidence_total = Counter("low_confidence_total", "docs with a low-confidence field")
human_review_total = Counter("human_review_total", "docs flagged for human review")
extraction_schema_valid_total = Counter("extraction_schema_valid_total", "valid schema extractions", ["valid"])
document_type_total = Counter("document_type_total", "doc type distribution", ["doc_type"])

# --- Input errors ---
decode_error_total = Counter("decode_error_total", "docs rejected at decode (empty/corrupt/unsupported)")

# --- Business safety (fintech-critical) ---
amount_reconciliation_fail_total = Counter(
    "amount_reconciliation_fail_total",
    "statements where running balance did not reconcile (potential extraction error)")
high_risk_auto_accept_total = Counter(
    "high_risk_auto_accept_total",
    "docs auto-accepted despite low confidence (should trigger review)")
missing_required_field_total = Counter(
    "missing_required_field_total",
    "extractions with a required field missing or null", ["field"])
human_feedback_total = Counter(
    "human_feedback_total",
    "corrections submitted via feedback endpoint", ["correction_type"])

# --- Drift signals ---
blur_observed = Histogram("input_blur_score", "input blur score distribution",
                          buckets=(20, 50, 100, 200, 400, 800))
field_confidence_observed = Histogram("field_confidence", "field confidence distribution",
                                      buckets=(0.3, 0.5, 0.6, 0.7, 0.75, 0.85, 0.95, 1.0))
ocr_line_count_observed = Histogram("ocr_line_count", "OCR detected line count per document",
                                    buckets=(2, 5, 10, 20, 35, 50, 80))


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
