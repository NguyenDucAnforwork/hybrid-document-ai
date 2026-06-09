# summary.md — Hybrid Document AI: main results

One-page result summary for the **Hybrid Document AI** project (Receipt/Invoice/Statement
OCR + KIE, 3-layer Processing·Serving·Deployment + full MLOps). Rationale: `docs/lessons-learned.md`
(ADR-1…17); reproduction: `docs/reproduce.md`; raw logs: `docs/logs/`. Every number below cites
its source log.

## 1. Receipt KIE — real SROIE, end-to-end real OCR (n=80)
Source: `docs/logs/benchmark_20260607_0055.md`

- **macro-F1 0.442 · macro-ANLS 0.521 · all-required-correct 0.362 · latency p50 1082ms / p95 1672ms**

| field | F1 | exact | ANLS | CER | note |
|---|---|---|---|---|---|
| date | **0.7692** | 0.769 | 0.886 | 0.098 | strong — works for reconciliation |
| total_amount | **0.5063** | 0.506 | 0.578 | 0.412 | moderate (OCR digit errors) |
| merchant_name | **0.05** | 0.05 | 0.10 | 0.877 | known limitation ↓ (see §4) |

merchant_name F1 (exact-match) is the weakest field. Note the F1-vs-ANLS gap is partly a metric
artifact (exact-match punishes minor OCR/punctuation diffs) and partly real — addressed by the
LayoutLMv3 finding in §4 and the planned `MerchantNameExtractor` (WP-2).

## 2. Production safety — router + guardrail closure (the "banking" result)
Source: `docs/lessons-learned.md` ADR-15 (n=80) · README go-live audit (n=30 strict)

| metric (n=80) | before | after | 
|---|---|---|
| **SILENT_WRONG** (wrong required field, `needs_review=False`) | 13/80 (16.3%) | **3/80 (3.8%)** |
| **Router Recall** (flags doc when model is wrong) | 83.7% | **92.5%** |
| false_review (correct doc flagged anyway) | 12/80 (15%) | 25/80 (31.2%) |

**This is the core fintech trade-off:** silent-wrong cut 4× by **reviewing more** (false_review
15%→31.2%). In banking, a controlled refusal beats a silent wrong number. Driven by *deterministic
sanity checks*, not confidence tuning (total_amount ECE ≈ 0.51 → confidence untrustworthy).
Strict go-live audit (n=30): date exact **80%**, total exact **37%**, **0** CJK hallucinations, SILENT_WRONG 5/30.

## 3. KIE model comparison — why hybrid routing is the right call (SROIE n=80)
Source: `docs/logs/model_comparison_raw.json`

| field | metric | rule | logistic-KIE | LayoutLMv3 | winner |
|---|---|---|---|---|---|
| merchant_name | ANLS | 0.546 | 0.0997 | **0.7148** | LayoutLMv3 |
| date | F1 | 0.800 | **0.775** | 0.1875 | rule/logistic |
| total_amount | F1 | 0.000 | **0.4875** | 0.0375 | logistic |
| (inference) | latency | 1.5ms | 60.2ms | 41.4ms | — |

**Not "LayoutLMv3 beats all":** logistic-KIE wins date/total (OCR line-grouped tokens ≠ SROIE
box annotations → train/infer gap hurts LayoutLMv3 there); LayoutLMv3 wins merchant_name decisively
(multimodal layout). Production router = logistic for date/total, LayoutLMv3 for merchant.

## 4. Multi-document — 3 types, HARD statement set
Source: `docs/logs/multidoc_20260607_0105.md`

- **3-way routing accuracy: 1.0** (receipt / bank_statement / payment_order)
- **statement table (HARD): row-F1 0.926 · amount-acc 0.463 · description-acc 0.996**
- payment_order header fields ~0.9–1.0 (mean ~0.88)

**Honest:** amount-acc 0.463 is **not production-grade** for banking statement extraction — rule-based
table parsing does not generalize across layouts, and a small VLM (Qwen2.5-VL-3B) doesn't rescue it.
**But** the balance-reconciliation guard flags ~87% of hard statements `needs_human_review` → wrong
financial figures are not emitted silently. A table-structure model (Table-Transformer/LayoutLMv3)
or larger VLM is the real fix.

## 5. Hybrid OCR + VLM on hard cases (router-gated)
Source: `docs/logs/vlm_compare_20260531_1129.md`

VLM fired on **3/12** blurred receipts and improved every field:

| field | OCR F1 → OCR+VLM F1 | OCR ANLS → OCR+VLM ANLS |
|---|---|---|
| date | 0.6364 → **0.8333** | 0.675 → 0.925 |
| total_amount | 0.4167 → **0.50** | 0.575 → 0.661 |
| merchant_name | 0.00 → **0.1667** | 0.0 → 0.167 |

Cost: mean latency **1.4s → 17.8s (CPU)** — useful but expensive, exactly why the VLM runs only on
router-flagged docs (production uses GPU/vLLM `vlm.mode=remote_gpu`).

## 6. Latency / performance pack (ADR-17) — NEW
Source: `docs/logs/profile_20260609_0916.md`, `bench_threads_20260609_0918.md`, `latency_baseline.json`

Stage profiler + process-pool OCR + workers×intra_threads×concurrency sweep + smoke CI gate.
Numbers below: **synthetic n=30, small images, LayoutLMv3/VLM off, 48-core box** — they isolate
*where pipeline time goes* (real SROIE end-to-end is p50 **1082ms**, see §1).

| stage | warm p50 (ms) | warm p95 | warm p99 |
|---|---|---|---|
| **total** | **535** | 656 | 673 |
| **ocr** | **514 (~96%)** | 641 | 662 |
| kie | 0.4 | 3.0 | 4.6 |
| quality | 7.2 | 15.7 | 23.1 |
| decode | 3.5 | 8.6 | 11.9 |
| preprocess | 1.4 | 4.2 | 4.2 |
| classify | 0.1 | 0.3 | 0.3 |

- **Cold start 1485ms** (OCR engine load 1460ms) → `warmup()` every worker on deploy.
- **OCR = 96% of latency** → optimize OCR/serving, never KIE.
- **Thread/worker sweep (honest):** throughput plateaus ~70–140 docs/min; **adding workers does not
  help** on this workload — best `W=1,T=2`; `W=4` is worse (44–72). Oversubscription (large `W×T`)
  degrades, confirming ADR-16 with data. Process-pool gives the correct multi-core path for genuine
  concurrent traffic, not a free win on single small images.
- **Latency baseline** warm total p50 **532.5ms** committed; CI smoke gate fails on > +40% regression.

CV line: *“profiled a hybrid OCR-KIE pipeline and optimized CPU-bound inference serving under
production latency constraints, with p50/p95/p99 benchmarks, monitoring and a CI regression gate.”*

## 6b. OCR recognizer fine-tune — MC-OCR 2021 (WP-3) — NEW
Source: `docs/wp3-ocr-finetune-report.md`, `docs/logs/ocr_rec_eval_20260609_1112.md`

Fine-tuned a compact CRNN+CTC Vietnamese receipt recognizer (torch→ONNX), optional config-
switchable adapter. OCR-level, MC-OCR val n=1300:

| recognizer | CER | exact-line | WER | p50 ms/crop |
|---|---|---|---|---|
| default (RapidOCR, Chinese dict) | 0.3197 | 0.149 | 0.836 | 27.1 |
| **fine-tuned CRNN** | **0.0853** | **0.599** | **0.246** | **9.4** |

**CER ↓ 73.3% relative**, and faster. Train 227s, peak VRAM 1316 MB (≤1h/≤5GB ✓). Artifacts on HF
(`hybrid-docai-kie:ocr/vi_mcocr_crnn_ft`, dataset `hybrid-docai-mcocr-ocr`).

**Per-field, crop-level (leakage-free):** SELLER −74.8%, ADDRESS −82.4%, TIMESTAMP −67.3%,
**TOTAL_COST −64.6%** (0.27→0.095), diacritics −78.7%. The money field downstream `total` depends
on does improve strongly at crop level.

**Full-image pipeline (det+rec, n=80):** macro CER **0.337→0.265 (~21%)** — much smaller than crop
−73% (TIMESTAMP ≈ 0%). **Bottleneck shifts to the detector/line-grouping, not the recognizer**:
behind the real detector, mis-segmentation dominates and recognizer quality can't fix it. `needs_review`
rises 0.66→0.80 (SROIE-tuned KIE doesn't benefit from Vietnamese). Next lever = detector, not more rec FT.

**Honest caveats:** crop-level gain is partly because RapidOCR's default rec uses a Chinese dict and
can't emit Vietnamese diacritics ("in-language wins", not SOTA); full-image gold is train-only
(recognizer in-domain → optimistic); `mcocr_val_sample_df.csv` is a stub, not downstream gold.

**Detector error analysis** (`docs/wp3-detector-analysis.md`, `scripts/eval_detector_mcocr.py`):
det_field_recall **0.978** (detector doesn't miss fields) → don't swap detector / don't prioritize
deskew. Per-field cause: **TIMESTAMP = REC_ERROR** (crop-distribution gap, not detection),
**ADDRESS = OVERMERGE** (vertical multi-line merge caps the recognizer gain), TOTAL_COST mostly OK.
Fixes shipped flag-gated: **D** language routing (`DOCAI_OCR_RECOGNIZER=auto`, VN→CRNN / EN→default,
fixes needs_review↑ + SROIE-regression risk), **C** geometry-risk→needs_review (`skew≥8°`).

**Grouping fixes are a dead-end here; the recognizer crop-gap was the real bottleneck.** Task B
(horizontal anchor-split) and Task E (vertical projection row-split) both measured **null** on
ADDRESS — the failures are garbled recognizer output on *detector* crops, not merges. **Task F —
detector-style crop augmentation** (extract detector boxes matched to gold, short fine-tune with
crop/pad/blur jitter) — fixed **all four fields**: full-image macro CER **0.265 → 0.205**
(SELLER 0.179→0.111, ADDRESS 0.319→0.255, TIMESTAMP 0.454→0.376, TOTAL 0.152→0.108), clean-val CER
also improved 0.085→0.063. Now the config-default recognizer. The measurement bought the real lesson:
not the detector, not deskew, not grouping — **fine-tune the recognizer on detector-style crops.**

## 7. Robustness (real SROIE, n=30, severity 0.6)
Source: `docs/logs/robustness_*.md`

Router **catches blur/motion-blur** (needs_review→1.0, refuses garbage) but **misses
rotate/perspective** (CER 1.2–1.4 yet confidence ~0.87, ECE ~0.51–0.55) → high-confidence wrong
output. The failure a single clean accuracy hides; fix = geometry-aware confidence + deskew.

## 8. MLOps lifecycle
Training-pipeline DAG (ingest→validate→train→eval-gate→register, lineage) · staged model registry ·
eval-as-CI-gate · **latency smoke gate (ADR-17)** · Prometheus + drift + alert rules
(`monitoring/alerts.yaml`, incl. `OCRStageLatencyP95`) · chaos · DR runbook · runnable Docker Compose
(api+redis+minio+prometheus+grafana).

## 9. Honest limitations
- **merchant_name F1 0.05** (exact-match) — WP-2 planned: `MerchantNameExtractor` + field router (target F1 ≥ 0.25).
- **Statement amount-acc 0.463** — not production-grade; needs table-structure model / larger VLM. Guard catches it meanwhile.
- §6 latency numbers are synthetic small images; **real SROIE end-to-end p50 = 1082ms** (OCR-bound).
- Router blind to geometric warp (rotate/perspective).
