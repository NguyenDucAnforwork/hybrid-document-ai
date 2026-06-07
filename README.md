# Hybrid Document AI — Receipt/Invoice OCR + KIE (production-grade, MLOps)

> Portfolio for **AI Engineer @ VNPAY** (track *NLP–Computer Vision–MLOps*).
> A **multi-model document-understanding pipeline** on the 3-layer Banking/Enterprise
> architecture (**Processing · Serving · Deployment**) with a full **MLOps lifecycle**
> (training-pipeline DAG → eval-gate → staged model registry → monitoring → CI/CD →
> chaos/DR). Evaluated on **real SROIE scans** under **banking-realistic degradations**.

## 🔗 Links
- 🟢 **Live demo (HF Space):** https://huggingface.co/spaces/banhchungtuongot/hybrid-docai-demo
- 🐙 Code: https://github.com/NguyenDucAnforwork/hybrid-document-ai
- 🤗 Model (registry, staged + lineage): https://huggingface.co/banhchungtuongot/hybrid-docai-kie
- 🤗 Dataset: https://huggingface.co/datasets/banhchungtuongot/hybrid-docai-receipts

## Why this isn't "another OCR demo"
Most OCR demos die in production because they optimize the *model* and ignore data,
pipeline, serving and operations. Here the model is **1 of 3 layers**:

| Layer | Implemented |
|---|---|
| **Processing** (multi-model) | Quality → upscale → OCR (RapidOCR/PP-OCR ONNX) → **KIE**: layout line-grouping → feature vector → **calibrated scikit-learn field classifier** + **LayoutLMv3-base fine-tuned** (BIO token classification, SROIE) → **hybrid confidence router** → **real VLM fallback (Qwen2.5-VL) on hard cases** |
| **Serving** | **dynamic micro-batcher** (runnable) · stage queues · CPU/VRAM caps · v1 REST API (idempotency key, feedback loop, request_id) · Triton + vLLM + **KServe** (autoscale/scale-to-zero) artifacts |
| **Deployment + MLOps** | FastAPI · **runnable Docker Compose** (api + redis + minio + prometheus + grafana) · Prometheus + drift + **alert rules** · **staged model registry + lineage** · **training-pipeline DAG** (KFP) · **eval-as-CI-gate** · **chaos engineering** · DR runbook |

**KIE is a learned multi-model stage, not regex.** Candidate-gen (regex/keyword/layout-graph)
→ features → a **calibrated** sklearn classifier (trained on real SROIE + synthetic),
with a confidence router that sends uncertain docs to human review / VLM instead of
silently emitting wrong data.

## Multi-document (not receipt-only) — 3 types
A bank needs more than invoices. The pipeline is **document-type-agnostic** — only the
schema/anchors/prompt are per-type (`docai/doctypes.py`); quality/OCR/router/VLM/MLOps are
shared. A learned **document-type router** (`docai/classifier.py`) classifies first, then
dispatches:
- **receipt/invoice** → calibrated sklearn KIE (key-value).
- **bank statement** → header KIE **+ transaction TABLE parsing** (`docai/statement.py`:
  row clustering + column detection → `{date, description, amount, balance}`).
- **payment order / ủy nhiệm chi** → generic anchor KV extractor (`docai/kv.py`).

Adding a type = one registry entry (+ optional extractor); everything else is reused.

### Honest results on **HARD** test data (`docs/logs/multidoc_*.md`)
Statements use a **hard generator** (random column schema/order incl. separate Debit/Credit,
parenthesis & CR/DR negatives, VN/EN headers, footer distractor rows, x-jitter, multi-word
descriptions) — *not* the easy single-layout that gave a misleading 1.00.

| metric | easy (misleading) | **HARD** |
|---|---|---|
| 3-way doc routing | 1.00 | 1.00 |
| stmt table row-F1 | 1.00 | 0.92 |
| stmt **amount-acc** (rules) | 1.00 | **0.46** |
| stmt **amount-acc** (rules→VLM-3B fallback) | — | **0.33** (no gain) |
| stmt desc-acc (after left-edge col fix) | — | **0.99** |
| payment_order mean field | — | ~0.88 |

**Key findings (the whole point of harder data):**
1. **Rule-based table parsing does NOT generalize** across bank layouts — amount-acc 1.00→**0.46**.
2. **A small VLM (Qwen2.5-VL-3B) doesn't rescue it either** (0.33) — statement table extraction at
   scale genuinely needs a **table-structure model** (Table-Transformer / LayoutLMv3) or a larger VLM.
3. **But the system stays safe:** a **balance-reconciliation guard** (`statement.reconcile`: does
   running balance match parsed amounts?) flags **~87%** of hard statements as `needs_human_review`
   — so wrong financial figures are *not emitted silently*, even when extraction is unreliable.
   (Debugging the harder data also found & fixed real bugs: a missing-description render bug, CR/DR
   column-shift, and VN Nợ/Có header matching.)

Routing stays 1.00 because the three doc types differ strongly in global features (genuinely easy).

### Batch demo (5–10 documents)
The live Space has a **Batch tab**: drop 5–10 mixed documents → each is auto-routed
(receipt/statement/payment_order), extracted, and shown in a results table (type · route ·
review-flag · key field · #transactions) with a batch summary. Backed by the async
`/batch_jobs` API + orchestrator (per-doc state machine, retry, dead-letter) for production.

## Data & evaluation (the production-hard part)
- **Real data:** SROIE 2019 — **626 real scanned receipts** (Malaysian, thermal-printer, genuinely noisy). Token-level gold → `merchant_name / date / total_amount`.
- **Banking degradations** (`docai/augment.py`): dark, low-contrast, blur, motion-blur, rotate, perspective, low-res, JPEG, noise, tear/occlusion, fade — for a **robustness curve**, plus **Vietnamese** synthetic receipts (multilingual).
- **Strong metrics:** exact-match · F1 · **CER** · **ANLS** · **ECE (calibration)** — not a single clean accuracy.

### Results — clean real SROIE (n=80, end-to-end real OCR)
| field | F1 | exact | ANLS | CER | note |
|---|---|---|---|---|---|
| date | 0.75 | 0.74 | **0.84** | 0.14 | strong |
| total_amount | 0.54 | 0.54 | **0.60** | 0.37 | moderate (OCR digit errors) |
| merchant_name | 0.00 | 0.00 | 0.02 | 0.92 | **known limitation** ↓ |

Latency p50 ~2.0s (real scans, CPU). **merchant_name is honestly hard**: company names
span multiple lines and OCR fuses words ("ABC MART"→"ABCMART"); line-selection heuristics
fail → it needs a sequence-labeling model (LayoutLMv3). It is **optional** (not a required
field) and routes to human review. The financial fields that matter for reconciliation —
**date and total_amount — work** (ANLS 0.84 / 0.60).

### KIE model comparison (SROIE test n=80)

Fine-tuned **LayoutLMv3-base** (BIO token classification, `training/train_layoutlmv3.py`,
~4 min on RTX 3090) — compared against rule-based and logistic-KIE baselines:

| field | metric | rule | logistic-KIE | **LayoutLMv3** | winner |
|---|---|---|---|---|---|
| merchant_name | ANLS | 0.55 | 0.10 | **0.71** | LayoutLMv3 |
| date | F1 | 0.80 | **0.775** | 0.19 | logistic |
| total_amount | F1 | 0.00 | **0.49** | 0.04 | logistic |
| full pipeline | p50 latency | ~2s | ~2s | ~2–3s | — |

**Kết luận: hybrid routing là production call đúng.** Không phải "LayoutLMv3 beats all" —
mỗi model có điểm mạnh khác nhau:
- **merchant_name**: LayoutLMv3 thắng rõ (ANLS 0.10→0.71) nhờ multimodal layout context — tên công ty in to/nhiều dòng, chính xác là bài toán mà token-sequence + layout embedding giải được.
- **date / total_amount**: logistic-KIE vẫn tốt hơn vì OCR tokens từ RapidOCR (line-grouped, pixel bbox) khác với SROIE ground-truth box annotations (token-per-word, precise bbox) — train/infer gap khiến LayoutLMv3 suy giảm ở 2 field này.
- **Production router**: dùng logistic cho date/total (nhanh, chính xác), LayoutLMv3 cho merchant_name (multimodal layout). Generate bởi `scripts/compare_models.py --limit 80 --out docs/logs`.

### Robustness curve (real SROIE, n=30, severity 0.6) — `docs/logs/robustness_*.md`
| degradation | macro-F1 | ANLS | CER | needs_review | ECE |
|---|---|---|---|---|---|
| clean | 0.41 | 0.48 | 0.51 | 0.03 | 0.43 |
| dark | 0.43 | 0.50 | 0.50 | 0.03 | 0.41 |
| fade | 0.46 | 0.50 | 0.48 | 0.07 | 0.38 |
| jpeg | 0.39 | 0.48 | 0.55 | 0.10 | 0.45 |
| low_res | 0.32 | 0.44 | 0.54 | 0.03 | 0.53 |
| noise | 0.36 | 0.42 | 0.57 | 0.17 | 0.46 |
| tear | 0.36 | 0.42 | 0.57 | 0.07 | 0.47 |
| rotate | 0.36 | 0.44 | 1.18 | 0.03 | 0.51 |
| perspective | 0.33 | 0.41 | 1.39 | 0.03 | 0.54 |
| blur | 0.35 | 0.44 | 0.55 | **1.00** | 0.49 |
| motion_blur | **0.01** | 0.05 | 0.95 | **1.00** | 0.29 |
| mixed_hard | 0.28 | 0.39 | 1.25 | 0.43 | 0.55 |

### Hybrid OCR + **VLM** on hard cases (real Qwen2.5-VL) — `docs/logs/vlm_compare_*.md`
The confidence router escalates only flagged (hard) docs to a real VLM (Qwen2.5-VL-3B);
on 12 blurred SROIE receipts it fired on **3/12** and improved every field:
| field | OCR-only ANLS | **OCR+VLM ANLS** | OCR-only F1 | **OCR+VLM F1** |
|---|---|---|---|---|
| merchant_name | 0.00 | **0.17** | 0.00 | **0.17** |
| date | 0.68 | **0.93** | 0.64 | **0.83** |
| total_amount | 0.58 | **0.66** | 0.42 | **0.50** |

Cost: mean latency 1.4s → 17.8s (CPU; production uses GPU/vLLM `vlm.mode=remote_gpu`).
That cost is exactly why the VLM runs **only on router-flagged docs**, not the whole batch —
the textbook hybrid trade-off. Enable with `DOCAI_VLM_MODE=local` (`docai/vlm.py`).

**Demoing on a small GPU (e.g. RTX 1650 / 4GB)?** The VLM won't fit locally, but the
OpenAI-compatible `mode=api` lets the local box run OCR+KIE while hard cases hit a remote
VLM. **This is live now:** the public HF Space (CPU, OCR+KIE) escalates hard cases to a
**Modal serverless GPU** running Qwen2.5-VL (`deploy/modal_vlm.py`, scale-to-zero) — verified
end-to-end (a blurred receipt → `route=vlm_fallback` → VLM recovers the merchant the OCR path
misses). Same wiring works on an RTX 1650. Alternatives (managed Qwen API, local Ollama
`qwen2.5vl:3b`): **`docs/vlm-deployment.md`**. Note: first hard case after idle waits ~60-70s
(Modal cold start + model load).

**Two honest findings worth more than a single number:**
1. The router **catches blur/motion-blur** (OCR confidence collapses → `needs_review`=1.0, correctly refusing garbage) and reacts to `mixed_hard` (0.43). Dark/fade are tolerated (OCR robust).
2. The router **does NOT catch rotate/perspective**: CER explodes (1.2–1.4) yet confidence stays ~0.87 and `needs_review`≈0.03 → **high-confidence wrong output** (ECE ≈ 0.51–0.55). Root cause: confidence reflects *line-selection* + *OCR self-reported* certainty, and OCR stays (wrongly) confident on geometrically warped text. Fix: a deskew/perspective-correction preprocessing stage + geometry-aware confidence. This is exactly the kind of failure a single clean accuracy hides.

## MLOps lifecycle (maps the 4 lectures) — see `docs/mlops.md`
- **Training pipeline DAG** `mlops/pipeline.py`: ingest → validate → train → eval-gate → register, with **caching / retry / resume** and a **run manifest (lineage)**. Production target: Kubeflow `mlops/kfp_pipeline.py`.
- **Model registry**: stages `developing→staging→production→archived` + lineage (run/data/params/metrics). `transition()` for promote/rollback.
- **Serving**: dynamic batcher (runnable) + KServe `deploy/kserve.yaml` (scale-to-zero/out) + Triton/vLLM.
- **Monitoring**: `/metrics` + drift signals + `monitoring/alerts.yaml`. **Chaos**: `mlops/chaos.py`. **DR**: `docs/runbook-dr.md`.

## Demo (chạy ngay — 3 paths)

### Path 1 — API trực tiếp (~30s setup, không cần Docker)

```bash
# Terminal 1 — khởi động API
cd /workspace/hybrid-document-ai
export DOCAI_WORKSPACE=/workspace/docai-ws
export DOCAI_VLM_MODE=disabled
/opt/miniforge3/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```bash
# Terminal 2 — test sau khi thấy "Application startup complete"

# Health check
curl http://localhost:8000/health | python3 -m json.tool

# Extract 1 ảnh receipt
curl -s -X POST http://localhost:8000/documents/extract \
  -F "file=@$DOCAI_WORKSPACE/data/sroie/test/images/000.jpg" | python3 -m json.tool

# v1 API — upload → poll result
DOC_ID=$(curl -s -X POST http://localhost:8000/v1/documents \
  -F "file=@$DOCAI_WORKSPACE/data/sroie/test/images/001.jpg" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['doc_id'])")
curl -s http://localhost:8000/v1/documents/$DOC_ID/result | python3 -m json.tool

# Batch (5 ảnh cùng lúc)
curl -s -X POST http://localhost:8000/batch_jobs \
  -F "files=@$DOCAI_WORKSPACE/data/sroie/test/images/000.jpg" \
  -F "files=@$DOCAI_WORKSPACE/data/sroie/test/images/001.jpg" \
  -F "files=@$DOCAI_WORKSPACE/data/sroie/test/images/002.jpg" \
  -F "files=@$DOCAI_WORKSPACE/data/sroie/test/images/003.jpg" \
  -F "files=@$DOCAI_WORKSPACE/data/sroie/test/images/004.jpg" | python3 -m json.tool

# Prometheus metrics
curl -s http://localhost:8000/metrics | grep -E "^docai_"
```

### Path 2 — Docker Compose (full stack: Redis + MinIO + Prometheus + Grafana)

```bash
cd /workspace/hybrid-document-ai
export DOCAI_WORKSPACE=/workspace/docai-ws
docker compose -f deploy/docker-compose.yml up --build
```

Sau khi up: API `http://localhost:8000` · Prometheus `http://localhost:9090` ·
Grafana `http://localhost:3000` (admin/admin) · MinIO console `http://localhost:9001` (minio/minio123).

### Path 3 — Scripts eval & benchmark

```bash
export DOCAI_WORKSPACE=/workspace/docai-ws
export DOCAI_VLM_MODE=disabled

# So sánh 3 model (rule vs logistic vs LayoutLMv3)
/opt/miniforge3/bin/python scripts/compare_models.py \
  --test-data $DOCAI_WORKSPACE/data/sroie/test/labels.json \
  --test-img-dir $DOCAI_WORKSPACE/data/sroie/test/images \
  --layoutlmv3-dir $DOCAI_WORKSPACE/models/layoutlmv3/model

# ONNX benchmark (PyTorch CUDA vs FP32-CPU vs INT8-CPU)
/opt/miniforge3/bin/python training/export_onnx.py \
  --model-dir $DOCAI_WORKSPACE/models/layoutlmv3/model \
  --out-dir $DOCAI_WORKSPACE/models/layoutlmv3

# Load test throughput (cần API đang chạy ở Path 1)
/opt/miniforge3/bin/python scripts/load_test.py \
  --url http://localhost:8000 \
  --img-dir $DOCAI_WORKSPACE/data/sroie/test/images
```

## Reproduce từ đầu (training pipeline)

```bash
# 0. Môi trường — dùng conda (torch CUDA wheels ~11GB, cần /data)
/opt/miniforge3/bin/conda create -n docai python=3.11 -y
conda activate docai
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "transformers==4.49.0" "accelerate>=0.26.0"  # pin version, tránh float8 incompatibility

export DOCAI_WORKSPACE=/workspace/docai-ws
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | head -1 | cut -d= -f2)

# 1. Data SROIE thật
python scripts/prepare_sroie.py \
    --src $DOCAI_WORKSPACE/sroie_src/data \
    --out $DOCAI_WORKSPACE/data/sroie

# 2. Train KIE (sklearn logistic + MLOps DAG)
python -m mlops.pipeline --version v4

# 2b. Fine-tune LayoutLMv3 (~4 min RTX 3090)
python training/train_layoutlmv3.py --epochs 5 --batch 4

# 3. So sánh 3 model (rule / logistic / LayoutLMv3)
python scripts/compare_models.py --limit 80 --out docs/logs

# 4. Benchmark + robustness curve
python scripts/run_benchmark.py --data $DOCAI_WORKSPACE/data/sroie/test --f1-threshold 0.2
python scripts/eval_robustness.py --data $DOCAI_WORKSPACE/data/sroie/test --limit 30

# 5. Load test (batch 1/5/10 docs, 3 rounds)
python scripts/load_test.py \
    --url http://localhost:8000 \
    --img-dir $DOCAI_WORKSPACE/data/sroie/test/images

# 6. Chaos + CI
python -m mlops.chaos
```

Full walkthrough: **`test.ipynb`** · reproduce step-by-step: `docs/reproduce.md`.

## API v1
| Endpoint | Mô tả |
|---|---|
| `POST /v1/documents` | Upload ảnh + idempotency key; trả `document_id` |
| `POST /v1/extraction_jobs` | Tạo extraction job từ `document_id` đã upload |
| `POST /v1/documents/{id}/feedback` | Human correction → training data loop |
| `GET /metrics` | Prometheus metrics (system, model-quality, business-safety, drift) |
| `POST /batch_jobs` | Async batch (5–10 docs, per-doc state machine) |

Response có `X-Latency-Ms` header + `request_id` trong JSON. Mọi request được log JSON structured.

## Constraints honored
≤ 15GB disk (artifacts on `/data` hoặc `/workspace`; `/home` không để artifacts nặng) ·
≤ 4GB VRAM (VLM off-box: api/remote; LayoutLMv3 inference 41ms/doc GPU) ·
conda env `docai` Python 3.11 (torch CUDA wheels ~11GB thực tế, vượt ước tính ban đầu) ·
RTX 3090 cho LayoutLMv3 fine-tune (~4 phút) ·
no k8s on dev box (KServe/KFP/Triton là swap-by-config artifacts; live serving on HF Space).
