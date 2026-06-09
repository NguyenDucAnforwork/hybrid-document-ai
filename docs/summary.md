# summary.md — Hybrid Document AI: main results

One-page result summary for the **Hybrid Document AI** project (Receipt/Invoice/Statement
OCR + KIE, 3-layer Processing·Serving·Deployment + full MLOps). Detailed rationale lives in
`docs/lessons-learned.md` (ADR-1…17); reproduction in `docs/reproduce.md`; raw logs in
`docs/logs/`.

## 1. Extraction quality — real SROIE (n=80, end-to-end real OCR)

| field | F1 | exact | ANLS | CER | note |
|---|---|---|---|---|---|
| date | 0.75 | 0.74 | **0.84** | 0.14 | strong |
| total_amount | 0.54 | 0.54 | **0.60** | 0.37 | moderate (OCR digit errors) |
| merchant_name | 0.05 | 0.00 | 0.02–0.71* | 0.92 | known limitation; *0.71 ANLS with LayoutLMv3 |

\* merchant_name F1 (exact-match) heavily under-counts: the model gets the name roughly right
(LayoutLMv3 ANLS 0.71) but exact string match punishes minor OCR/punctuation diffs.
Targeted fix specced in the `MerchantNameExtractor` work package (WP-2, not yet built).

**Go-live audit (n=30, strict):** date exact **80%**, total exact **37%**, **0** CJK hallucinations,
SILENT_WRONG 5/30 (catastrophic cases caught by sanity/cross-validate guards).

## 2. Multi-document (3 types) — HARD test data

| metric | easy | **HARD** |
|---|---|---|
| 3-way doc routing | 1.00 | **1.00** |
| stmt table row-F1 | 1.00 | 0.92 |
| stmt amount-acc (rules) | 1.00 | **0.46** |
| payment_order mean field | — | ~0.88 |

Key finding: rule-based table parsing does NOT generalize across bank layouts; a small VLM
(Qwen2.5-VL-3B) doesn't rescue it (0.33). **But the balance-reconciliation guard flags ~87%**
of hard statements as `needs_human_review` → wrong financial figures are not emitted silently.

## 3. Hybrid OCR + VLM on hard cases (router-gated)

Confidence router escalates only flagged docs to a real VLM (fired 3/12 blurred receipts),
improving every field: date ANLS 0.68→**0.93**, total 0.58→**0.66**, merchant 0.00→**0.17**.
Cost: 1.4s→17.8s on CPU → exactly why the VLM runs only on router-flagged docs.

## 4. Latency / performance pack (ADR-17) — **NEW**

Stage profiler + process-pool OCR + workers×intra_threads×concurrency sweep + smoke CI gate.
Numbers below: **synthetic n=30, small images, LayoutLMv3/VLM off, 48-core box** — *not* the
real-SROIE ~2s figure; they isolate where pipeline time goes.

| stage | warm p50 (ms) | warm p95 | warm p99 |
|---|---|---|---|
| **total** | **535** | 656 | 673 |
| **ocr** | **514** (~96%) | 641 | 662 |
| kie | 0.4 | 3.0 | 4.6 |
| quality | 7.2 | 15.7 | 23.1 |
| decode | 3.5 | 8.6 | 11.9 |
| preprocess | 1.4 | 4.2 | 4.2 |
| classify | 0.1 | 0.3 | 0.3 |

- **Cold start: 1485 ms** (OCR engine load 1460 ms) → all workers `warmup()` on deploy.
- **OCR is 96% of latency** → optimization target is OCR/serving, never KIE.
- **Thread/worker sweep (honest):** throughput plateaus ~70–140 docs/min; **adding workers does
  not help** on this workload — best `W=1, T=2`; `W=4` is *worse* (44–72). Oversubscription
  (large `W×T`) degrades, confirming ADR-16 with data. Process-pool gives the correct multi-core
  path for genuine concurrent traffic, not a free throughput win on single small images.
- **Latency baseline:** warm total p50 **532.5 ms** committed (`docs/logs/latency_baseline.json`);
  CI smoke gate fails on > +40% regression.

CV line this supports: *“profiled a hybrid OCR-KIE pipeline and optimized CPU-bound inference
serving under production latency constraints, with p50/p95/p99 benchmarks, monitoring and a CI
regression gate.”*

## 5. Robustness (real SROIE, n=30, severity 0.6) — highlights

Router **catches blur/motion-blur** (needs_review→1.0, refuses garbage) but **misses
rotate/perspective** (CER 1.2–1.4 yet confidence ~0.87, ECE ~0.51–0.55) → high-confidence wrong
output. Documented as the failure a single clean accuracy hides; fix = geometry-aware confidence.

## 6. MLOps lifecycle

Training-pipeline DAG (ingest→validate→train→eval-gate→register, lineage) · staged model
registry · eval-as-CI-gate · **latency smoke gate (ADR-17)** · Prometheus + drift + alert rules
(`monitoring/alerts.yaml`, incl. `OCRStageLatencyP95`) · chaos · DR runbook · runnable Docker
Compose (api+redis+minio+prometheus+grafana).

## 7. Honest limitations
- merchant_name exact-match weak (WP-2 planned: `MerchantNameExtractor` + field router).
- Statement table extraction at scale needs a table-structure model / larger VLM (guard catches it meanwhile).
- Latency numbers in §4 are synthetic small images; real SROIE scans run ~2s p50 (OCR-bound).
- Router blind to geometric warp (rotate/perspective).
