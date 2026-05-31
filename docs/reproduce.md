# reproduce.md — Chạy lại từ máy sạch

Mục tiêu: người khác / CI dựng lại pipeline, **train KIE**, serve, và benchmark mà không cần hỏi thêm.

## 0. Môi trường đã kiểm chứng
- Linux, Python 3.13 (venv). Không cần GPU/docker/redis/MinIO cho core MVP.
- Disk: ~2.5–3 GB (core deps + RapidOCR ONNX + SROIE subset + KIE model). Torch/Donut chỉ cài nếu chọn VLM local.

## 1. Cài đặt
```bash
cd hybrid-document-ai
python3 -m venv .venv && source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt
du -sh .venv   # phải < 3GB; nếu vượt, xem debug-workflows "disk đầy"
```

## 2. Dữ liệu
```bash
python scripts/download_sroie.py --train 200 --test 50   # data/sroie/{train,test}
python scripts/make_vi_synthetic.py --n 10               # (optional) receipt tiếng Việt
```

## 3. Train KIE model (MLOps training pipeline)
```bash
python training/train_kie.py --data data/sroie/train --out models/kie/v1 --seed 42
# -> models/kie/v1/{model.joblib, metrics.json}; tự đăng ký vào mlops/models/registry.yaml (active)
```
Inference-first: chỉ train **KIE classifier nhẹ** (sklearn). KHÔNG train OCR/VLM.

## 4. Chạy service
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
curl -s localhost:8000/health
```
Serving Layer: dynamic batcher + queue backend `memory` chạy in-process. Đổi sang Redis/MinIO/Triton qua `configs/app.yaml` + `deploy/docker-compose.yml`.

## 5. Smoke test
```bash
curl -s -F file=@data/sroie/test/<anh>.jpg localhost:8000/documents/extract | jq
# Kiểm output có: fields{value,confidence}, route, model_versions{ocr,kie}, needs_human_review
JOB=$(curl -s -F 'files=@a.jpg' -F 'files=@b.jpg' localhost:8000/batch_jobs | jq -r .job_id)
curl -s localhost:8000/batch_jobs/$JOB | jq            # state + summary
curl -s localhost:8000/batch_jobs/$JOB/results | jq
curl -s localhost:8000/metrics | grep -E 'fallback_rate|documents_processed|low_confidence'
```

## 6. Benchmark + eval-gate (MLOps)
```bash
python scripts/run_benchmark.py --data data/sroie/test --out docs/logs/ --f1-threshold 0.70
echo "exit=$?"   # !=0 nếu field-F1 < threshold (gate fail) — đây là cổng CI/CD
python scripts/summarize_benchmark.py docs/logs/benchmark_raw.json
```
Sinh `docs/logs/benchmark_<ngày>.md`: CER, field exact-match, F1, calibration, latency p50/p95, fallback_rate, needs_review_rate. So Setting A (rule-only) vs B (sklearn-KIE) vs C (+VLM).

## 7. Test + CI
```bash
pytest -q
ruff check .            # local mirror của .github/workflows/ci.yml
```
CI gate (`.github/workflows/ci.yml`): ruff → pytest → eval-gate → docker build.

## 8. Chuyển production (swap-by-config, không sửa code)
`configs/app.yaml`:
```yaml
queue:   {backend: redis, url: "redis://redis:6379"}
storage: {backend: minio, endpoint: "minio:9000", bucket: documents}
serving: {ocr_via: triton, triton_url: "triton:8001"}
vlm:     {mode: remote_gpu, base_url: "http://vllm:8000/v1", min_confidence: 0.75}
```
```bash
cd deploy && docker compose up      # api + redis + minio + triton + vllm (target)
kubectl apply -f deploy/hpa.yaml    # autoscale theo queue depth
```

## 9. Phiên bản pin
- `requirements.txt` pin exact version.
- Model versions ghi trong `mlops/models/registry.yaml` + nhúng trong mỗi output JSON (`model_versions`).
