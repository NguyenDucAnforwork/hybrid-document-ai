# Hybrid Document AI — Receipt/Invoice OCR + KIE (production-grade)

> Portfolio for **AI Engineer @ VNPAY** (track *NLP–Computer Vision–MLOps*).
> A **multi-model document-understanding pipeline** built around the 3-layer
> Banking/Enterprise architecture: **Processing · Serving · Deployment**, with a
> full **MLOps lifecycle** (training pipeline → eval-gate → model registry →
> monitoring → CI/CD). Designed for **inference/deployment first**, not training.

## Why this isn't "another OCR demo"
Most OCR demos die in production because they optimize the *model* and ignore
data, pipeline, serving and operations. Here the model is **1 of 3 layers**:

| Layer | What's implemented |
|---|---|
| **Processing** (multi-model) | Quality → Layout → Text-Detect → OCR (RapidOCR/PP-OCR ONNX) → **KIE 2-tier** → hybrid router → VLM fallback |
| **Serving** | **Dynamic micro-batcher** (runnable) · stage queues · CPU/VRAM caps · Triton model-repo + vLLM (artifacts) |
| **Deployment + MLOps** | FastAPI · Docker/Compose · Prometheus + drift · **model registry/versioning** · **eval-as-CI-gate** · K8s HPA |

**KIE is not just regex.** Tier-1 = candidate-generation (regex/keyword/layout-graph)
→ feature vector → **scikit-learn field classifier** (calibrated confidence,
*trained, versioned, eval-gated*). Tier-2 = **VLM OCR-free** (Donut/Qwen-VL) for
low-confidence "hard cases" via a confidence router — the hybrid architecture.

## Results (measured, end-to-end real OCR on 40 receipts)
| field | F1 | exact-match |
|---|---|---|
| date | 1.00 | 1.00 |
| total_amount | 1.00 | 1.00 |
| invoice_id | 0.95 | 0.95 |
| payment_method | 0.93 | 0.93 |
| merchant_name | 0.45 | 0.45 |
| **macro-F1** | **0.865** | — |

Latency p50 **508 ms** / p95 **658 ms** (CPU). `merchant_name` is the hardest field
(real OCR reads store names inconsistently) — exactly the "real-world data is
messy" problem; low-confidence docs are routed to human review / VLM, not guessed.

## Quickstart
```bash
pip install -r requirements.txt
export DOCAI_WORKSPACE=/data/nvidia-ai-workspace          # heavy artifacts off /home
python -c "from docai.synth import generate; generate('$DOCAI_WORKSPACE/data/r',120,42)"
python training/train_kie.py --data $DOCAI_WORKSPACE/data/r --version v1   # train + register
uvicorn app.main:app --port 8000
curl -F file=@<receipt>.png localhost:8000/documents/extract
python scripts/run_benchmark.py --data $DOCAI_WORKSPACE/data/r --f1-threshold 0.6  # eval-gate
```
Full step-by-step + manual rerun: **`test.ipynb`** and **`docs/reproduce.md`**.

## API
`POST /documents/extract` · `POST /batch_jobs` · `GET /batch_jobs/{id}[/results]`
· `GET /health` · `GET /metrics`

## Artifacts on HuggingFace
- Model: `banhchungtuongot/hybrid-docai-kie` (KIE classifier + registry)
- Dataset: `banhchungtuongot/hybrid-docai-receipts` (synthetic labeled receipts)

## Layout
`docai/` core pipeline · `app/` FastAPI · `training/` KIE training · `scripts/`
benchmark · `serving/triton/` Triton configs · `deploy/` Docker/Compose/HPA ·
`docs/` PLAN + lessons-learned + debug-workflows + reproduce + logs.

## Constraints honored
≤ 2h build · ≤ 15GB disk (heavy artifacts on `/data`, `/home` was full) · ≤ 4GB
VRAM (VLM kept off-box: `disabled`/`api`/`remote_gpu`). See `docs/lessons-learned.md`.
