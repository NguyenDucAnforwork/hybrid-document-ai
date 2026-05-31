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
| **Processing** (multi-model) | Quality → upscale → OCR (RapidOCR/PP-OCR ONNX) → **KIE**: layout line-grouping → feature vector → **calibrated scikit-learn field classifier** → **confidence router** → **real VLM fallback (Qwen2.5-VL) on hard cases** |
| **Serving** | **dynamic micro-batcher** (runnable) · stage queues · CPU/VRAM caps · Triton + vLLM + **KServe** (autoscale/scale-to-zero) artifacts |
| **Deployment + MLOps** | FastAPI · Docker/Compose · Prometheus + drift + **alert rules** · **staged model registry + lineage** · **training-pipeline DAG** (KFP) · **eval-as-CI-gate** · **chaos engineering** · DR runbook |

**KIE is a learned multi-model stage, not regex.** Candidate-gen (regex/keyword/layout-graph)
→ features → a **calibrated** sklearn classifier (trained on real SROIE + synthetic),
with a confidence router that sends uncertain docs to human review / VLM instead of
silently emitting wrong data.

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

**Two honest findings worth more than a single number:**
1. The router **catches blur/motion-blur** (OCR confidence collapses → `needs_review`=1.0, correctly refusing garbage) and reacts to `mixed_hard` (0.43). Dark/fade are tolerated (OCR robust).
2. The router **does NOT catch rotate/perspective**: CER explodes (1.2–1.4) yet confidence stays ~0.87 and `needs_review`≈0.03 → **high-confidence wrong output** (ECE ≈ 0.51–0.55). Root cause: confidence reflects *line-selection* + *OCR self-reported* certainty, and OCR stays (wrongly) confident on geometrically warped text. Fix: a deskew/perspective-correction preprocessing stage + geometry-aware confidence. This is exactly the kind of failure a single clean accuracy hides.

## MLOps lifecycle (maps the 4 lectures) — see `docs/mlops.md`
- **Training pipeline DAG** `mlops/pipeline.py`: ingest → validate → train → eval-gate → register, with **caching / retry / resume** and a **run manifest (lineage)**. Production target: Kubeflow `mlops/kfp_pipeline.py`.
- **Model registry**: stages `developing→staging→production→archived` + lineage (run/data/params/metrics). `transition()` for promote/rollback.
- **Serving**: dynamic batcher (runnable) + KServe `deploy/kserve.yaml` (scale-to-zero/out) + Triton/vLLM.
- **Monitoring**: `/metrics` + drift signals + `monitoring/alerts.yaml`. **Chaos**: `mlops/chaos.py`. **DR**: `docs/runbook-dr.md`.

## Quickstart
```bash
pip install -r requirements.txt
export DOCAI_WORKSPACE=/data/nvidia-ai-workspace          # heavy artifacts off /home
python scripts/prepare_sroie.py                            # real SROIE -> tokens+gold
python -m mlops.pipeline --version v4                      # DAG: validate->train->eval-gate->register
python scripts/run_benchmark.py --data $DOCAI_WORKSPACE/data/sroie/test --f1-threshold 0.2
python scripts/eval_robustness.py --data $DOCAI_WORKSPACE/data/sroie/test --limit 30  # robustness curve
uvicorn app.main:app --port 8000                           # API
python -m mlops.chaos                                      # resilience
```
Full walkthrough: **`test.ipynb`** · reproduce: `docs/reproduce.md`.

## Constraints honored
≤ 15GB disk (artifacts on `/data`; `/home` was full) · ≤ 4GB VRAM (VLM off-box: api/remote) ·
no k8s on dev box (KServe/KFP/Triton are swap-by-config artifacts; live serving on HF Space).
