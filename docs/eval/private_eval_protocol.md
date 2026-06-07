# Private Blind Evaluation Protocol

## Motivation
Synthetic data is fine for training augmentation and chaos/stress tests,
but **final performance claims must rest on real-world, independently annotated data**.
This private set uses documents the system has never seen, annotated by a human
reviewer following a strict schema — a standard in fintech document processing.

## Dataset spec (target: 120 documents)

| Category | Count | Source |
|---|---|---|
| Receipt / invoice (Vietnamese) | 30 | Photographed from real transactions (info redacted) |
| Payment order / ủy nhiệm chi | 30 | Public sample forms from bank websites, info anonymized |
| Bank statement (table) | 30 | Screenshots with account numbers / names masked |
| Degraded images (blur/skew/low-light) | 30 | Subsampled from above + degradation applied |

## Annotation schema

Each document annotated as `docs/eval/annotations/{doc_id}.json`:

```json
{
  "doc_id": "pb001",
  "document_type": "receipt",
  "annotator": "reviewer_A",
  "annotation_date": "2026-06-07",
  "quality_flags": ["low_resolution"],
  "fields": {
    "merchant_name": "Siêu Thị Co.opMart",
    "date": "2024-03-15",
    "total_amount": "125000",
    "payment_method": "CASH"
  },
  "notes": "receipt partially occluded at top-right corner"
}
```

For bank statements, also annotate `line_items`:
```json
{
  "line_items": [
    {"date": "2024-03-01", "description": "Tiền điện EVNHCM", "amount": "-150000", "balance": "2350000"},
    {"date": "2024-03-03", "description": "Nhận chuyển khoản", "amount": "+500000", "balance": "2850000"}
  ]
}
```

## Annotation guidelines

1. **Exact string match** for amounts: use the number as-is on the document (e.g., `"1.250.000"` not `"1250000"`)
2. **Date**: ISO-8601 format `YYYY-MM-DD` always
3. **Redaction**: mask PAN (card numbers), full account numbers > 6 digits with `[REDACTED]`
4. **Ambiguous fields**: if a field is genuinely unreadable, set to `null` — do NOT guess
5. **Two-pass rule**: annotator A labels; annotator B spot-checks 20% for agreement

## Metrics to report

| Metric | Why it matters for VNPAY |
|---|---|
| Field F1 / ANLS / CER per field | Standard extraction quality |
| `all_required_correct` rate | % docs where every critical field is correct |
| `human_review_rate` | System's self-awareness — what % it refused to auto-accept |
| `high_confidence_wrong` rate | Most dangerous failure: confident but wrong |
| `amount_reconciliation_fail` | For statements: balance sanity check failure rate |
| Latency p50 / p95 | Production SLA |

## Privacy and publication

- Do **not** commit raw images to the public repo
- Store at `$DOCAI_WORKSPACE/eval/private_blind/` (outside repo, on local machine)
- Commit only: this protocol, annotation schema, and the `*_summary.md` metrics file
- Before any demo: double-check no PAN / full account numbers remain visible
