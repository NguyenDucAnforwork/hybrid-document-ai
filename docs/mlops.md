# MLOps — vòng đời ML khép kín (ánh xạ 4 bài giảng)

> Triết lý (MLOps Intro): *ML production là bài toán hệ thống, không phải notebook/model.*
> "ML code" chỉ là một ô nhỏ; xung quanh là data verification, training pipeline, serving,
> monitoring, metadata, testing. Dưới đây là cái đã **chạy thật** trong repo và **target production**.

## 1. Training pipeline (MLOps - Training Pipeline → Kubeflow)
| Nguyên tắc bài giảng | Repo (chạy thật) | Production target |
|---|---|---|
| DAG nhiều bước, không chạy thủ công | `mlops/pipeline.py`: ingest → validate → train → evaluate → register | `mlops/kfp_pipeline.py` (Kubeflow, mỗi step 1 container) |
| Caching / retry / resume | per-step marker `*.json`, skip nếu đã done, retry n lần | KFP caching + `set_retry` |
| Traceability / reproducibility | `manifest.json` mỗi run: inputs/outputs/metrics/dur + **lineage** (datasets, seed, params) | KFP run + artifact lineage UI |
| Data verification trước train | `mlops/data_validation.py` (schema/coverage/sanity), **gate** | cùng component, gate |
| Linh hoạt phần cứng/ngôn ngữ | step chạy subprocess/độc lập | container riêng mỗi step |

Chạy: `python -m mlops.pipeline --run-id demo --version v_pipe` → DAG có gate eval, chỉ promote `production` nếu pass.

## 2. Model Serving (MLOps - Model Serving → KServe)
| Bài giảng | Repo | Target |
|---|---|---|
| Chuẩn hoá serving đa framework | interface `OCREngine`/`VLMClient`; ONNX drop-in | KServe `deploy/kserve.yaml` + Triton runtime |
| Autoscaling (scale-to-zero / scale-out) | — | KServe `minReplicas: 0`, `maxReplicas: 10`, target concurrency |
| Dynamic batching | **`docai/batcher.py` chạy thật** (gom request OCR theo cửa sổ) | Triton `dynamic_batching{}` |
| Cold-start (mặt trái scale-to-zero) | warm-up OCR lúc `startup` | doc runbook (giữ minReplicas≥1 cho tier nóng) |
| **Live demo** | **HuggingFace Space (Gradio)** — URL công khai chạy pipeline thật | KServe trên k8s |

## 3. Model Registry & lineage (MLOps - More and beyond)
- `docai/registry.py`: version + **stage** `developing→staging→production→archived` + **lineage** (run nào, data nào, metric nào sinh ra model).
- `transition()` promote/rollback (prod cũ tự `archived`). Output JSON nhúng `model_versions` → audit "model version nào tạo ra kết quả nào".
- Quản lý metadata trước, UI tập trung sau — đúng thứ tự bài giảng.

## 4. Monitoring / Chaos / Disaster Recovery (More and beyond)
- **Monitoring:** `/metrics` (latency p50/p95, fallback_rate, human_review_rate, queue_size) + **drift signals** (phân phối `input_blur_score`, `field_confidence`). Alert rules: `monitoring/alerts.yaml` (drift, fallback spike, p95, confidence drop). Tracing/GPU-util qua OpenTelemetry + DCGM (target).
- **Chaos engineering:** `mlops/chaos.py` bơm input OOD (noise/blank/corrupt/tiny) → assert hệ thống **không crash**, route `needs_human_review` thay vì phát data sai. Chạy: `python -m mlops.chaos`.
- **Disaster recovery:** xem `docs/runbook-dr.md`.
- **GPU sharing / cost:** scale-to-zero cho tier VLM nhàn rỗi; time-slicing/MIG cho nhiều model nhỏ trên 1 GPU (rủi ro memory/capacity — ghi trong runbook).

## Còn thiếu so với production đầy đủ (trung thực)
Chưa chạy thật: KServe/Triton/vLLM trên k8s, KFP cluster, OTel collector, Grafana, GPU MIG. Đều có artifact + swap-by-config; lý do: budget 15GB disk/4GB VRAM/không k8s trên máy dev. Demo serving thật chạy trên HF Space.
