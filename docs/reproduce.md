# reproduce.md — Chạy lại từ máy sạch (bản cập nhật, khớp code thật)

Mục tiêu: dựng lại pipeline — ingest data thật (SROIE) → train KIE → serve → benchmark
→ robustness → VLM hard-case (Modal) — không cần hỏi thêm. Mọi lệnh dưới đây khớp script
hiện có (`scripts/`, `training/`, `mlops/`).

## 0. Môi trường
- Linux, Python 3.11. **Heavy artifacts để trên `/workspace` hoặc `/data`** (conda env/data/models/cache) vì `/home` máy gốc đầy.
- **Bắt buộc:** set `DOCAI_WORKSPACE` và `HF_TOKEN` TRƯỚC khi import `docai`.
- **Lưu ý disk:** conda env + torch CUDA wheels thực tế ~12GB (vượt ước tính ban đầu ~2GB). Cần ít nhất 15GB trống trên partition chứa conda env.

```bash
# Tạo conda env (dùng miniforge, không dùng venv vì torch CUDA cần conda solver)
export DOCAI_WORKSPACE=/workspace/docai-ws            # đổi sang chỗ có >=15GB
/opt/miniforge3/bin/conda create -n docai python=3.11 -y
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate docai

# Core dependencies
pip install -r requirements.txt                        # core (no torch) ~2GB

# Torch + transformers cho LayoutLMv3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "transformers==4.49.0"                    # PHẢI pin: 5.x incompatible với torch 2.6
pip install "accelerate>=0.26.0"                      # bắt buộc khi dùng HuggingFace Trainer

# HuggingFace token (cần để tải LayoutLMv3 weights)
export HF_TOKEN=$(grep HF_TOKEN_geminipro /workspace/.env | cut -d= -f2)
# Hoặc set thủ công: export HF_TOKEN=hf_xxxx

# (chỉ khi chạy VLM LOCAL) thêm: pip install qwen-vl-utils
```

## 1. Dữ liệu
```bash
# 1a. SROIE thật (626 receipt scan) — clone mirror rồi ingest sang tokens+gold
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/zzzDavid/ICDAR-2019-SROIE \
    $DOCAI_WORKSPACE/sroie_src
# Chỉ định --src và --out rõ ràng (không hardcode /data/nvidia-ai-workspace)
python scripts/prepare_sroie.py \
    --src $DOCAI_WORKSPACE/sroie_src/data \
    --out $DOCAI_WORKSPACE/data/sroie \
    --n-test 80                                        # -> $DOCAI_WORKSPACE/data/sroie/{train,test}
# 1b. Synthetic receipts (EN + tiếng Việt, 5 field, multilingual)
python -c "import os;from docai.synth import generate;generate(os.environ['DOCAI_WORKSPACE']+'/data/receipts',120,42)"
# 1c. Multi-document synthetic: statements (easy train + HARD train/test) + payment orders
python - <<'PYG'
import os
from docai.synth import generate_statements as gs, generate_payment_orders as gp
WS=os.environ['DOCAI_WORKSPACE']
gs(f"{WS}/data/statements",100,7); gs(f"{WS}/data/statements_hard",60,13,hard=True)
gs(f"{WS}/data/statements_test_hard",40,99,hard=True)
gp(f"{WS}/data/payment_orders",80,11); gp(f"{WS}/data/payment_orders_test",30,23)
PYG
```
The HARD statement generator randomizes column schema/order (incl. Debit/Credit), negative
format (`-x`, `(x)`, `CR/DR`), VN/EN headers, footer distractors, x-jitter — to avoid easy-data.

### Multi-document: train the 3-way doc-type router + eval
```bash
python training/train_doctype.py \
  --receipts $DOCAI_WORKSPACE/data/sroie/train $DOCAI_WORKSPACE/data/receipts \
  --statements $DOCAI_WORKSPACE/data/statements $DOCAI_WORKSPACE/data/statements_hard \
  --payment-orders $DOCAI_WORKSPACE/data/payment_orders --version v3
# eval_multidoc.py cần chỉ định rõ từng tập test (không đọc env hardcoded)
python scripts/eval_multidoc.py \
  --statements $DOCAI_WORKSPACE/data/statements_test_hard \
  --receipts $DOCAI_WORKSPACE/data/sroie/test \
  --payment-orders $DOCAI_WORKSPACE/data/payment_orders_test \
  --limit 30                                         # HARD statements -> docs/logs/multidoc_*.md
```
Auto-routes: receipt → sklearn KIE; bank_statement → header KIE + **table parsing**
(`docai/statement.py`); payment_order → anchor KV (`docai/kv.py`). Output JSON has
`document_type` + `line_items`. On HARD data, statement amount/description accuracy drops to
0.35/0.55 (rule-based table parsing limit — see README "Key finding").

## 2. Train KIE — qua MLOps DAG (khuyến nghị) hoặc trực tiếp
```bash
# DAG: ingest -> validate -> train -> eval-gate -> register (cache/retry/resume + manifest lineage)
python -m mlops.pipeline --run-id repro --version v4
# hoặc trực tiếp (train kết hợp SROIE thật + synthetic; sklearn calibrated):
python training/train_kie.py --data $DOCAI_WORKSPACE/data/sroie/train $DOCAI_WORKSPACE/data/receipts --version v4 --seed 42
```
Sinh `$DOCAI_WORKSPACE/models/kie/v4/{model.joblib,metrics.json}` + đăng ký vào `$DOCAI_WORKSPACE/models/registry.yaml`
(stage `staging`→`production`, kèm lineage). Inference-first: chỉ train KIE classifier nhẹ; KHÔNG train OCR/VLM.

### 2b. Fine-tune LayoutLMv3 (BIO token classification cho merchant_name)
```bash
# Cần torch + transformers đã cài ở bước 0; RTX 3090 ~4 phút (CPU ~30+ phút)
python training/train_layoutlmv3.py --epochs 5 --batch 4
# -> $DOCAI_WORKSPACE/models/layoutlmv3/{config.json, pytorch_model.bin, metrics.json}
# Val F1 ~0.91 (box-file tokens); test ANLS merchant ~0.71 (OCR tokens, xem ADR-9 về train/infer gap)
```
**Lưu ý train/infer gap:** SROIE training dùng ground-truth box annotations (token-per-word,
precise bbox); inference dùng RapidOCR (line-grouped tokens, pixel bbox). Bbox phải được
normalize theo W/H ảnh thật (không clip cứng 1000). Xem `docs/debug-workflows.md` để biết chi tiết.

## 3. Chạy service + smoke test
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
curl -s localhost:8000/health

IMG=$DOCAI_WORKSPACE/data/sroie/test/images/$(ls $DOCAI_WORKSPACE/data/sroie/test/images | head -1)

# v1 API endpoints
# Upload + idempotency key
curl -s -X POST -F "file=@$IMG" -H "Idempotency-Key: test-001" \
    localhost:8000/v1/documents | jq
# Tạo extraction job
curl -s -X POST -H "Content-Type: application/json" \
    -d '{"document_id": "<id từ bước trên>"}' \
    localhost:8000/v1/extraction_jobs | jq
# Human feedback / correction (tạo training data loop)
curl -s -X POST -H "Content-Type: application/json" \
    -d '{"merchant_name": "ABC MART", "total_amount": "12.50"}' \
    localhost:8000/v1/documents/<id>/feedback | jq

# Legacy endpoint (vẫn hoạt động)
curl -s -F "file=@$IMG" localhost:8000/documents/extract | jq
curl -s -F "files=@$IMG" localhost:8000/batch_jobs | jq

# Kiểm tra metrics (4 nhóm: system, model-quality, business-safety, drift)
curl -s localhost:8000/metrics | grep -E 'documents_processed|vlm_fallback|amount_reconciliation_fail|human_feedback|document_type_total'

# Response có X-Latency-Ms header + request_id trong body
curl -v -F "file=@$IMG" localhost:8000/v1/documents 2>&1 | grep -E 'X-Latency|request_id'
```

## 4. Đánh giá (data thật + metric mạnh)
```bash
# 4a. Benchmark SROIE clean: F1 + exact + ANLS + CER + eval-gate (SROIE macro ~0.43 -> dùng thr thấp)
python scripts/run_benchmark.py --data $DOCAI_WORKSPACE/data/sroie/test --out docs/logs --f1-threshold 0.2
echo "gate exit=$?"     # !=0 nếu macro-F1 < threshold (cổng CI)

# 4b. So sánh 3 model (rule / logistic-KIE / LayoutLMv3) trên SROIE test
python scripts/compare_models.py --limit 80 --out docs/logs
# -> docs/logs/model_comparison_*.md (merchant ANLS / date F1 / total F1 per model)

# 4c. Robustness curve theo từng degradation (tối/mờ/nghiêng/nhiễu/rách/...)
python scripts/eval_robustness.py --data $DOCAI_WORKSPACE/data/sroie/test --limit 30 --severity 0.6
# -> docs/logs/robustness_*.md  (CER/ANLS/ECE/needs_review per degradation)

# 4d. Load test (batch 1/5/10 docs, 3 rounds — bottleneck analysis)
# Chạy sau khi đã start uvicorn (bước 3)
python scripts/load_test.py \
    --url http://localhost:8000 \
    --img-dir $DOCAI_WORKSPACE/data/sroie/test/images
# Kết quả tham khảo: batch=1 p50=2.71s 22docs/min, batch=5 p50=12.1s 25docs/min, batch=10 p50=26s 23docs/min
# Throughput plateau ~22-25 docs/min bất kể batch size -> bottleneck là CPU OCR (rapidocr single-thread)
```

## 5. VLM hard-case fallback (3 cách — xem `docs/vlm-deployment.md`)
```bash
# (A) Modal serverless GPU — KHUYẾN NGHỊ cho máy GPU nhỏ (RTX 1650). Endpoint đang LIVE.
modal token set --token-id <ak-...> --token-secret <as-...>
modal deploy deploy/modal_vlm.py            # -> https://<you>--docai-vlm-vlm-serve.modal.run
export DOCAI_VLM_MODE=api VLM_API_KEY=dummy
export VLM_API_BASE="https://<you>--docai-vlm-vlm-serve.modal.run/v1"
export VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
# (B) Local GPU/CPU:  export DOCAI_VLM_MODE=local DOCAI_VLM_DEVICE=cpu   (cần cài torch+transformers)
# (C) Tắt:            export DOCAI_VLM_MODE=disabled   (default; ca khó -> human review)

# So sánh OCR-only (B) vs OCR+VLM (C) trên ca khó (tạo hard case bằng --degrade):
python scripts/compare_vlm.py --data $DOCAI_WORKSPACE/data/sroie/test --limit 12 --degrade blur --severity 0.4
# -> docs/logs/vlm_compare_*.md
```

## 6. Tests + Chaos + CI
```bash
pytest -q                                   # unit: quality/kie/batcher/metrics/chaos
python -m mlops.chaos                        # OOD/blank/corrupt/tiny -> graceful (needs_review, no crash)
python -m mlops.data_validation $DOCAI_WORKSPACE/data/sroie/train/labels.json
```
CI (`.github/workflows/ci.yml`): ruff → pytest → train → eval-gate.

## 7. Production (swap-by-config, không sửa code) — artifact + local stack
### Local stack (runnable ngay):
```bash
# Stack: api + redis (healthcheck) + minio (healthcheck) + prometheus + grafana (auto-provisioned datasource)
cd deploy && docker compose up
# Grafana: http://localhost:3000 (admin/admin) — datasource Prometheus đã tự provision
# Prometheus: http://localhost:9090
# MinIO: http://localhost:9001 (minioadmin/minioadmin)
# KServe / Triton / vLLM KHÔNG có trong stack này (cloud-only targets)
```

### Cloud config (swap-by-config):
- `configs/app.yaml`: `queue.backend=redis`, `storage.backend=minio`, `serving.ocr_via=triton`, `vlm.mode=remote_gpu`.
- `deploy/`: `kserve.yaml` (autoscale/scale-to-zero), `hpa.yaml`, `modal_vlm.py` (đang chạy thật).
- `mlops/kfp_pipeline.py` (Kubeflow), `monitoring/alerts.yaml`. Vận hành/DR: `docs/runbook-dr.md`.

## 8. Artifacts đã publish (backup)
- **Code:** github.com/NguyenDucAnforwork/hybrid-document-ai
- **Model + registry:** huggingface.co/banhchungtuongot/hybrid-docai-kie (`kie/v1,v2,v4` + `registry.yaml`)
- **Dataset:** huggingface.co/datasets/banhchungtuongot/hybrid-docai-receipts (120 synthetic + labels)
- **Live demo:** huggingface.co/spaces/banhchungtuongot/hybrid-docai-demo (gọi Modal VLM cho hard case)
- **Reports:** `docs/logs/*.md` (benchmark, robustness, vlm_compare) — commit trong repo.
- SROIE thật KHÔNG commit (tải lại bằng bước 1a; là dataset public).

## 9. Pin phiên bản
`requirements.txt` (core). LayoutLMv3 + VLM local: `transformers==4.49.0` + torch/torchvision khớp nhau.
**Quan trọng:** transformers 5.x incompatible với torch 2.6 (float8_e8m0fnu type chưa tồn tại) —
luôn pin `transformers==4.49.0` khi cài torch specific version. Xem `docs/debug-workflows.md`.
Model version + lineage trong `registry.yaml` và nhúng `model_versions` trong mỗi output JSON.

## 10. Lưu ý disk
- conda env `docai` (không có torch): ~1.5GB
- torch CUDA (cu121 wheels): ~8GB
- torchvision: ~500MB
- transformers + accelerate + model weights (LayoutLMv3-base): ~2GB
- **Tổng thực tế: ~12GB** (ước tính ban đầu ~2GB là sai — không tính CUDA wheels)
- SROIE raw scans: ~1GB thêm
- Cần ít nhất **15GB trống** trên partition chứa conda env trước khi bắt đầu.
