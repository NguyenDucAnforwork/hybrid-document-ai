# PLAN.md — Hybrid Document AI (OCR + KIE) — Production-grade, VNPAY-aligned

> Hồ sơ ứng tuyển **AI Engineer @ VNPAY**, track **"NLP – Computer Vision – MLOps"**.
> JD yêu cầu rõ: *CV/OCR · MLOps: training pipeline, model serving, monitoring, CI/CD for ML · production deployment*.
> Thiết kế theo **3-layer chuẩn Banking/Enterprise** (Processing · Serving · Deployment) + **hybrid OCR↔VLM**.
> Ngân sách cứng: **≤ 2 giờ thực thi**, **≤ 5 GB disk**, **≤ 4 GB VRAM** (mô phỏng GPU triển khai ngân hàng; máy thật có H100 nhưng tự siết để chứng minh tối ưu tài nguyên).

---

## 0. Vì sao bản plan này khác "demo OCR thường gặp"

Bài viết tham chiếu nói đúng: đa số demo OCR chết khi lên production vì chỉ lo *kết quả model*, bỏ quên **dữ liệu, pipeline, serving, vận hành**. Plan này cố tình đảo trọng tâm: **model chỉ là 1 trong 3 layer**. Ba thứ chứng minh năng lực AI-Engineer-cho-bank:

1. **Processing = multi-model pipeline**, KHÔNG phải "1 model OCR duy nhất": Layout → Text Detection → Recognition → **KIE có thành phần học được** → router → VLM reasoning cho case khó.
2. **Serving Layer thật**: dynamic batching, request scheduling, GPU/CPU resource cap, Triton model-repo + vLLM (artifact + 1 phần chạy được).
3. **MLOps lifecycle khép kín**: training pipeline (KIE model) → eval-as-CI-gate → **model registry/versioning** → serving theo version → **monitoring + drift** → CI/CD.

> Phản hồi đã tiếp thu: KIE **không** chỉ rule+regex. KIE là **2-tier**: (Tier-1) candidate-generation rồi **field classifier scikit-learn** (trainable, calibrated, versioned); (Tier-2) **VLM OCR-free** cho hard cases. Đây là "hybrid architecture" mà bài viết khuyến nghị, áp ngay ở tầng KIE.

---

## 1. Kiến trúc 3-layer (ánh xạ trực tiếp JD + bài viết)

```
┌────────────────────────── DEPLOYMENT LAYER ──────────────────────────┐
│ FastAPI REST · Docker/Compose · Prometheus+Grafana · K8s HPA(artifact)│
│ Model Versioning/Registry · CI/CD (lint→test→eval-gate→build)         │
└───────────────────────────────┬───────────────────────────────────────┘
                                 │
┌────────────────────────── SERVING LAYER ─────────────────────────────┐
│ Request Scheduler → Dynamic Micro-Batcher (OCR ONNX)                  │
│ Resource mgmt (CPU pool, VRAM cap 4GB) · Triton model-repo(config.pbtxt)│
│ VLM serving: vLLM OpenAI-compatible (remote_gpu/api)                  │
└───────────────────────────────┬───────────────────────────────────────┘
                                 │
┌────────────────────────── PROCESSING LAYER (multi-model) ────────────┐
│ Quality/Preproc → Layout Detect → Text Detect → Recognition(OCR)      │
│   → KIE 2-tier:  [candidate-gen] → [sklearn field classifier]         │
│   → Confidence Router ── low-conf ──► VLM OCR-free (Donut) / Qwen-VL   │
│   → Validate/Normalize (Pydantic) → DocumentResult JSON               │
└───────────────────────────────────────────────────────────────────────┘
        ▲ async batch orchestration: per-doc state machine, retry, dead-letter
```

**Hybrid router (trọng tâm):** mỗi document đi đường **traditional OCR pipeline** (nhanh, rẻ, kiểm soát được — 80–90% lưu lượng). Chỉ doc **low-confidence / missing required / layout lạ** mới rớt xuống **VLM path** (đắt, chậm, mạnh). Router quyết định bằng confidence — đo được, log được, tinh chỉnh được. Đây là điểm khác biệt so với "đập VLM cho mọi ảnh".

---

## 2. Processing Layer — multi-model pipeline (chi tiết kỹ thuật)

| Stage | Model/Kỹ thuật | Đầu ra | Ghi chú budget |
|---|---|---|---|
| Quality | OpenCV: Laplacian-var(blur), brightness, resolution, skew(minAreaRect) | `QualityReport` | CPU, ~ms |
| Layout Detect | Heuristic layout-graph (line/column clustering) — interface sẵn cho PP-Structure/`layout` model | regions (header/body/total-zone) | giữ nhẹ; model là pluggable |
| Text Detect | **RapidOCR DB** (PP-OCRv4/v5 det, ONNX) | text boxes | ONNX CPU |
| Recognition | **RapidOCR CRNN** (rec, ONNX) | `[{text,bbox,conf}]` | ONNX CPU |
| **KIE Tier-1** | **candidate-gen (regex+keyword+layout-graph) → feature vector → `scikit-learn` field classifier** | field→value + **calibrated confidence** | model train được, ~30MB |
| **KIE Tier-2** | **Donut** OCR-free (receipt-finetuned) hoặc **Qwen-VL** (api/vLLM) | field JSON cho hard cases | optional local ≤4GB / remote |
| Confidence Router | luật ngưỡng + xác suất classifier | route ∈ {traditional, vlm} | quyết định fallback |
| Postprocess | Pydantic v2 validate + normalize (date→ISO, money→int VND) | `DocumentResult` | — |

### KIE Tier-1 — vì sao không chỉ regex (điểm user yêu cầu)
1. **Candidate generation** sinh ứng viên cho mỗi trường (regex date/money/tax, keyword anchor đa ngữ VN/EN, layout-graph: vị trí tương đối với keyword, cột phải/dưới).
2. **Feature vector / candidate** (~10–15 đặc trưng): ocr_conf, khoảng cách tới keyword anchor (chuẩn hoá), vị trí y tương đối, là-số-lớn-nhất-gần-đáy, regex-match-type, độ dài token, font-height tương đối…
3. **Field classifier `scikit-learn`** (Logistic Regression / GradientBoosting) train trên SROIE để chọn candidate đúng cho từng trường + **xác suất calibrated** làm confidence. → *học được, versioned, eval-gate*. Đây chính là "training pipeline" trong JD, gọn nhẹ đủ chạy CPU vài phút.
4. **Confidence ensemble:** `field_confidence = w·P(classifier) + (1−w)·(0.4·ocr + 0.3·pattern + 0.2·layout + 0.1·proximity)`. Trọng số `w` lưu trong model registry.

> Đường nâng cấp (documented, không bắt buộc chạy): thay Tier-1 bằng **LayoutLMv3 token-classification** (ONNX) — cùng interface `KIEModel`, swap qua registry. Giải thích đánh đổi accuracy/latency/disk trong lessons-learned.

---

## 3. Serving Layer — thứ phân biệt "demo" với "production"

| Thành phần | Thực thi | Trạng thái |
|---|---|---|
| **Dynamic micro-batching** | Batcher gom request OCR trong cửa sổ `max_delay_ms` (vd 20ms) tới `max_batch=8`, infer 1 lần qua ONNX | **CHẠY ĐƯỢC** — artifact serving thật |
| **Request scheduling** | hàng đợi theo stage (`quality/ocr/kie/vlm`) để VLM nặng không nghẽn OCR nhẹ | **CHẠY ĐƯỢC** (queue backend) |
| **Resource management** | OCR: CPU thread pool cố định; VLM: `VRAM_CAP=4GB` (env `CUDA`/`gpu_mem_fraction`), `vlm_concurrency=1` | **CHẠY ĐƯỢC** (semaphore) |
| **Triton Inference Server** | `serving/triton/model_repository/{det,rec}/config.pbtxt` với `dynamic_batching{}` — ONNX drop-in | **ARTIFACT** (không chạy: tốn disk/cần container) |
| **vLLM cho VLM** | OpenAI-compatible endpoint; dùng ở `vlm.mode=remote_gpu` | **ARTIFACT + client chạy được** |

> Thông điệp: hiểu **request scheduling + dynamic batching + GPU resource mgmt** (đúng từ khoá bài viết), thể hiện bằng **một batcher chạy thật** + config Triton/vLLM chuẩn production. Trên máy này VLM/Triton để dạng artifact vì disk 5GB; swap-in bằng config.

---

## 4. Deployment Layer + MLOps lifecycle (JD nhấn mạnh)

```
data(SROIE) ──> training pipeline(KIE) ──> eval ──> [eval-gate: F1≥thr?]
                                                       │pass
                                            model registry (version+metrics+date)
                                                       │
serving load "active version" ──> FastAPI ──> Prometheus/Grafana + drift monitor
                                                       │
                              CI/CD: lint → unit test → eval-gate → build image
```

| MLOps mục JD | Thực thi cụ thể |
|---|---|
| **Training pipeline** | `training/train_kie.py`: SROIE → features → fit sklearn → save `models/kie/<version>/` + `metrics.json`. Reproducible (seed cố định). |
| **Model serving** | Serving Layer mục 3; serving đọc model theo **active version** trong registry. |
| **Model versioning/registry** | `models/registry.yaml`: `{model, version, path, metrics, created, active}`. Output JSON nhúng `model_versions` để truy vết. |
| **Monitoring** | Prometheus `/metrics` (latency p50/p95, fallback_rate, low_confidence_rate, human_review_rate, queue_size, failed_total) + **drift signals** (phân phối blur/brightness input, phân phối confidence, OOD rate) + Grafana dashboard JSON. |
| **CI/CD for ML** | `.github/workflows/ci.yml`: ruff + pytest + **eval-gate** (chạy benchmark, fail nếu field-F1 < threshold) + docker build. |
| **Production deployment** | Dockerfile + docker-compose (api+redis+minio+triton target) + K8s `hpa.yaml` (autoscale theo queue depth) — artifact. |

**Anti-fraud nod (JD có "anti-fraud vision"):** quality+confidence+drift làm tín hiệu phát hiện **document bất thường/giả mạo** (blur bất thường, layout lệch template, confidence sụp) → flag `suspicious` cho review. Ghi rõ là hướng mở rộng.

---

## 5. Tech stack chốt (đã cân disk/VRAM)

| Layer | Chọn | Lý do/budget |
|---|---|---|
| API | FastAPI + uvicorn | async, /metrics, /health |
| Validation/Schema | Pydantic v2 | output contract |
| OCR | **RapidOCR (onnxruntime)** = PP-OCRv4/v5 ONNX | ~vài trăm MB, CPU |
| Img/quality | opencv-python-headless, numpy, Pillow | nhẹ |
| **KIE model** | **scikit-learn** (LogReg/GBDT) | ~30MB, train CPU vài phút, JD liệt kê sklearn |
| VLM (hard case) | Donut (transformers) *optional local* / Qwen-VL qua vLLM *remote/api* | Donut-base ~0.7GB chạy ≤4GB; mặc định `disabled`→`api` |
| Queue/state | `MemoryQueue`(default) / Redis Streams(prod) | không docker/redis trên máy |
| Object store | filesystem(default) / MinIO(prod) | disk |
| Metadata | SQLite | nhẹ, chuẩn |
| Serving | dynamic batcher (tự viết) + Triton/vLLM config | chạy 1 phần + artifact |
| Metrics | prometheus-client | /metrics |
| Eval | Levenshtein(CER) + field exact-match/F1 + calibration | scripts |
| CI | ruff + pytest + eval-gate | GitHub Actions |

**Disk guard:** core (fastapi+uvicorn+pydantic+rapidocr-onnxruntime+opencv-headless+numpy+pillow+scikit-learn+prometheus-client+pytest+httpx+python-Levenshtein) ước tính **~2–2.5GB**. Donut/torch **chỉ cài nếu** `du -sh` còn ≥1.5GB dư; nếu không → VLM dùng `api`/`remote`. `pip install --no-cache-dir`, kiểm `du -sh` sau mỗi nhóm.

---

## 6. Data & checkpoint

- **SROIE 2019** (đã chọn): dùng **cả train lẫn eval cho KIE classifier** (không chỉ eval như bản cũ). Subset: ~150–200 ảnh train, ~50 ảnh test (`download_sroie.py --train 200 --test 50`). 4 trường gold: company/date/address/total → map sang `merchant_name/date/total_amount` + bổ sung `invoice_id/payment_method` bằng candidate-gen.
- **Ảnh VN synthetic** (Pillow, 10 ảnh, keyword "TỔNG CỘNG/THÀNH TIỀN/Số HĐ") — kiểm KIE đa ngữ, không dùng dữ liệu cá nhân.
- **Model artifacts:** RapidOCR ONNX tự tải (<100MB, cache đếm vào disk); KIE sklearn tự train (~vài MB); Donut tải **chỉ khi** chọn local.
- Không train OCR/VLM. Chỉ **train KIE classifier nhẹ** (đúng tinh thần inference-first + 1 training pipeline minh hoạ MLOps).

---

## 7. Repo structure (mục tiêu)

```
hybrid-document-ai/
  app/main.py  app/schemas.py
  app/routes/{documents,batch_jobs,health}.py
  processing/
    quality_check.py preprocessing.py layout.py
    ocr_engine.py            # RapidOCR wrapper + OCREngine interface
    kie/
      candidate_gen.py       # regex+keyword+layout-graph -> candidates+features
      features.py            # feature vector
      classifier.py          # sklearn KIEModel (load active version)
      confidence.py          # ensemble + calibration
    vlm/
      base.py                # VLMClient interface
      donut_client.py        # optional local OCR-free
      vllm_client.py         # OpenAI-compatible remote
      prompt.py guardrails.py# JSON-only, max_tokens, timeout, repeat-detect
    router.py                # hybrid confidence router
    postprocess.py orchestrator.py
  serving/
    batcher.py               # dynamic micro-batching (RUNNABLE)
    scheduler.py resource.py # stage queues + VRAM/CPU caps
    triton/model_repository/{det,rec}/config.pbtxt   # artifact
  storage/{object_store,metadata_store,queue_backend}.py
  monitoring/{metrics,logging_config,drift}.py
  mlops/
    registry.py  models/registry.yaml
    grafana_dashboard.json
  training/
    train_kie.py             # SROIE -> features -> fit -> eval -> register
    prepare_sroie.py
  configs/{app,receipt_schema,vlm,batch,serving}.yaml
  scripts/{download_sroie,make_vi_synthetic,run_benchmark,summarize_benchmark}.py
  tests/test_{quality,kie,confidence,router,api,batch,batcher}.py
  .github/workflows/ci.yml
  deploy/{Dockerfile,docker-compose.yml,hpa.yaml}
  docs/{PLAN,lessons-learned,debug-workflows,reproduce}.md  docs/logs/
  requirements.txt  README.md
```

---

## 8. Logic cốt lõi (chốt số để code không lệch)

**Quality:** `blur=var(Laplacian)`; is_blurry `<100`; is_dark `mean<50`; low_res `w<720 or h<720`; rotated `|skew|>5°`.
**Router:** `needs_review = missing_required OR any(field_conf<0.75) OR conflict>0`; `vlm_fallback = needs_review AND vlm.mode!=disabled`.
**Output JSON:** đúng schema mô tả gốc + thêm `model_versions:{ocr,kie}` và `route` để truy vết MLOps.
**VLM guardrails:** `max_new_tokens` thấp, JSON-only, `timeout`, retry≤2, detect repeated-phrase/whitespace → invalid thì `needs_human_review` (theo cảnh báo failure-mode VLM-OCR).

---

## 9. Kế hoạch theo phase (timeboxed 2 giờ, mỗi phase 1 checkpoint chạy được)

| Phase | Thời lượng | Việc | Done khi… |
|---|---|---|---|
| **P0 Setup** | 0:00–0:12 | venv, install core, `du -sh`, scaffold, `/health` | `/health` 200, disk<3GB |
| **P1 OCR + Serving batcher** | 0:12–0:35 | `ocr_engine` (RapidOCR), `download_sroie`, **dynamic batcher**, warm-up | OCR 1 ảnh ra box+text+conf; batcher gom ≥2 req thành 1 batch (test) |
| **P2 KIE multi-model** | 0:35–1:05 | candidate_gen+features, **train_kie.py** (sklearn), classifier+confidence ensemble, postprocess | `POST /documents/extract` ra JSON đúng schema có confidence từ classifier; `models/kie/v1` + metrics.json |
| **P3 Router + VLM + batch/state** | 1:05–1:30 | hybrid router, VLMClient stub/api, metadata(SQLite), queue(memory), orchestrator(retry/dead-letter), batch routes | `POST /batch_jobs` 20 ảnh → summary; doc low-conf gắn route=vlm/needs_review |
| **P4 MLOps: metrics+eval-gate+registry+monitoring** | 1:30–1:52 | metrics+drift, registry.py, `run_benchmark` (CER+field-F1+calibration), eval-gate, grafana json, ci.yml | bảng benchmark vào docs/logs; `/metrics` có drift+fallback; eval-gate pass/fail in ra |
| **P5 Deploy artifacts + docs + test** | 1:52–2:00 | Dockerfile/compose/hpa/triton config, điền docs, smoke tests | repo chạy lại theo reproduce.md; artifacts tồn tại |

**Thứ tự hy sinh nếu trượt giờ:** Donut local → Grafana json → hpa/triton config → ảnh VN synthetic → một phần test. **Không bao giờ hy sinh:** OCR→KIE(classifier)→JSON, hybrid router, batch state machine, eval-gate, model registry/versioning, /metrics, docs.

---

## 10. Done criteria (nghiệm thu — production-grade)

**Processing (multi-model):**
- [ ] OCR (RapidOCR det+rec) ra `[{text,bbox,conf}]`.
- [ ] KIE Tier-1 dùng **sklearn classifier đã train** (không chỉ regex) → field + confidence calibrated.
- [ ] Hybrid router gắn `route` + kích hoạt VLM path cho low-confidence (stub/api/Donut).
- [ ] Output JSON đúng schema + `model_versions` + `route`.

**Serving:**
- [ ] **Dynamic batcher chạy được** (test chứng minh gom batch); resource cap (CPU pool, vlm_concurrency=1).
- [ ] Triton `config.pbtxt` + vLLM client artifact tồn tại.

**Deployment + MLOps:**
- [ ] `POST /batch_jobs` async, state machine per-doc, retry≤2, dead-letter, partial_completed.
- [ ] **Model registry/versioning** hoạt động (serving load active version; output truy vết được).
- [ ] **Eval-as-gate**: benchmark in CER + field exact-match + F1 + calibration; gate fail khi F1<thr; kết quả lưu `docs/logs/`.
- [ ] `/metrics` Prometheus + ≥1 drift signal; `/health`.
- [ ] `ci.yml`, Dockerfile, docker-compose, hpa.yaml tồn tại (artifact production).
- [ ] Docs đầy đủ (PLAN, lessons-learned+ADR, debug-workflows, reproduce) + reproduce chạy lại được.

**Báo cáo so sánh (README):** Setting A (OCR+rule-only) vs B (OCR+sklearn-KIE) vs C (+VLM fallback) — F1/latency/fallback_rate/cost.

**Out of scope (nói rõ):** train OCR/VLM from scratch; chạy Triton/MinIO/Redis thật trên máy này (có interface+config swap); finetune VLM; vector/semantic search.

---

## 11. Rủi ro & giảm thiểu

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Disk 5GB tràn (onnxruntime+opencv+torch) | Cao | core không-torch trước; Donut/torch chỉ nếu dư ≥1.5GB; `--no-cache-dir`; `du -sh` từng bước; VLM fallback về api |
| RapidOCR/onnx lỗi trên Py3.13 | TB | pin onnxruntime cp313; fallback easyocr sau interface; cuối cùng venv 3.11 |
| KIE classifier ít data → overfit | TB | features đơn giản + regularization + calibration; report cả rule-only baseline (Setting A) để thấy giá trị gia tăng |
| 2h không đủ | Cao | phase timeboxed + thứ tự hy sinh; mỗi phase checkpoint chạy được |
| VLM/Triton không chạy thật bị hỏi | Chắc chắn | trình bày là artifact production + swap-by-config; batcher/serving cap chạy thật để chứng minh hiểu serving |

---

## 12. Lệnh chạy (rút gọn — chi tiết ở reproduce.md)

```bash
python -m venv .venv && . .venv/bin/activate && pip install --no-cache-dir -r requirements.txt
python scripts/download_sroie.py --train 200 --test 50
python training/train_kie.py            # -> models/kie/v1 + metrics.json + register
uvicorn app.main:app --port 8000
curl -F file=@data/sroie/test/X.jpg localhost:8000/documents/extract | jq
python scripts/run_benchmark.py --data data/sroie/test --out docs/logs/   # eval-gate
curl localhost:8000/metrics ; pytest -q
```

---

## 13. Theo dõi tiến độ (cập nhật khi thực thi)

| Phase | Trạng thái | Ghi chú |
|---|---|---|
| P0 Setup | ✅ | venv 743MB trên /data (no torch); /home đầy nên artifacts off-box |
| P1 OCR + Batcher | ✅ | RapidOCR ONNX; dynamic batcher test pass (max_seen>=2) |
| P2 KIE multi-model | ✅ | sklearn classifier + layout line-grouping; kie v1→**v2** (macro_test 0.992) |
| P3 Router + Batch | ✅ | confidence router + batch state machine + retry/dead-letter |
| P4 MLOps | ✅ | eval-gate PASS **macro-F1 0.98** (v1 0.865→v2 0.98); registry/versioning; /metrics+drift |
| P5 Deploy + docs | ✅ | Dockerfile/compose/hpa/triton/ci; HF+GitHub pushed; test.ipynb |

**Artifacts:** GitHub https://github.com/NguyenDucAnforwork/hybrid-document-ai · HF model `banhchungtuongot/hybrid-docai-kie` · HF dataset `banhchungtuongot/hybrid-docai-receipts`
