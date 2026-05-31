# reproduce.md — Chạy lại từ máy sạch (bản cập nhật, khớp code thật)

Mục tiêu: dựng lại pipeline — ingest data thật (SROIE) → train KIE → serve → benchmark
→ robustness → VLM hard-case (Modal) — không cần hỏi thêm. Mọi lệnh dưới đây khớp script
hiện có (`scripts/`, `training/`, `mlops/`).

## 0. Môi trường
- Linux, Python 3.11–3.13. **Heavy artifacts để trên `/data`** (venv/data/models/cache) vì `/home` máy gốc đầy.
- **Bắt buộc:** set `DOCAI_WORKSPACE` TRƯỚC khi import `docai` (config đọc env lúc import).
```bash
export DOCAI_WORKSPACE=/data/nvidia-ai-workspace      # đổi sang chỗ có >=10GB nếu máy khác
python3 -m venv $DOCAI_WORKSPACE/venv && source $DOCAI_WORKSPACE/venv/bin/activate
pip install -r requirements.txt                       # core (no torch) ~2GB
# (chỉ khi chạy VLM LOCAL) thêm: pip install "transformers==4.49.0" torch torchvision accelerate qwen-vl-utils
```

## 1. Dữ liệu
```bash
# 1a. SROIE thật (626 receipt scan) — clone mirror rồi ingest sang tokens+gold
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/zzzDavid/ICDAR-2019-SROIE \
    $DOCAI_WORKSPACE/sroie_src
python scripts/prepare_sroie.py --n-test 80           # -> $WS/data/sroie/{train,test}
# 1b. Synthetic (EN + tiếng Việt, 5 field, multilingual)
python -c "import os;from docai.synth import generate;generate(os.environ['DOCAI_WORKSPACE']+'/data/receipts',120,42)"
```

## 2. Train KIE — qua MLOps DAG (khuyến nghị) hoặc trực tiếp
```bash
# DAG: ingest -> validate -> train -> eval-gate -> register (cache/retry/resume + manifest lineage)
python -m mlops.pipeline --run-id repro --version v4
# hoặc trực tiếp (train kết hợp SROIE thật + synthetic; sklearn calibrated):
python training/train_kie.py --data $DOCAI_WORKSPACE/data/sroie/train $DOCAI_WORKSPACE/data/receipts --version v4 --seed 42
```
Sinh `$WS/models/kie/v4/{model.joblib,metrics.json}` + đăng ký vào `$WS/models/registry.yaml`
(stage `staging`→`production`, kèm lineage). Inference-first: chỉ train KIE classifier nhẹ; KHÔNG train OCR/VLM.

## 3. Chạy service + smoke test
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
curl -s localhost:8000/health
IMG=$DOCAI_WORKSPACE/data/sroie/test/images/$(ls $DOCAI_WORKSPACE/data/sroie/test/images | head -1)
curl -s -F file=@$IMG localhost:8000/documents/extract | jq   # fields{value,confidence}, route, model_versions, needs_human_review
curl -s -F "files=@$IMG" localhost:8000/batch_jobs | jq        # async batch + summary
curl -s localhost:8000/metrics | grep -E 'documents_processed|vlm_fallback|human_review'
```

## 4. Đánh giá (data thật + metric mạnh)
```bash
# 4a. Benchmark SROIE clean: F1 + exact + ANLS + CER + eval-gate (SROIE macro ~0.43 -> dùng thr thấp)
python scripts/run_benchmark.py --data $DOCAI_WORKSPACE/data/sroie/test --out docs/logs --f1-threshold 0.2
echo "gate exit=$?"     # !=0 nếu macro-F1 < threshold (cổng CI)
# 4b. Robustness curve theo từng degradation (tối/mờ/nghiêng/nhiễu/rách/...)
python scripts/eval_robustness.py --data $DOCAI_WORKSPACE/data/sroie/test --limit 30 --severity 0.6
# -> docs/logs/robustness_*.md  (CER/ANLS/ECE/needs_review per degradation)
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

## 7. Production (swap-by-config, không sửa code) — artifact
- `configs/app.yaml`: `queue.backend=redis`, `storage.backend=minio`, `serving.ocr_via=triton`, `vlm.mode=remote_gpu`.
- `deploy/`: `docker-compose.yml` (redis+minio+triton+vllm+prometheus), `kserve.yaml` (autoscale/scale-to-zero), `hpa.yaml`, `modal_vlm.py` (đang chạy thật).
- `mlops/kfp_pipeline.py` (Kubeflow), `monitoring/alerts.yaml`. Vận hành/DR: `docs/runbook-dr.md`.

## 8. Artifacts đã publish (backup)
- **Code:** github.com/NguyenDucAnforwork/hybrid-document-ai
- **Model + registry:** huggingface.co/banhchungtuongot/hybrid-docai-kie (`kie/v1,v2,v4` + `registry.yaml`)
- **Dataset:** huggingface.co/datasets/banhchungtuongot/hybrid-docai-receipts (120 synthetic + labels)
- **Live demo:** huggingface.co/spaces/banhchungtuongot/hybrid-docai-demo (gọi Modal VLM cho hard case)
- **Reports:** `docs/logs/*.md` (benchmark, robustness, vlm_compare) — commit trong repo.
- SROIE thật KHÔNG commit (tải lại bằng bước 1a; là dataset public).

## 9. Pin phiên bản
`requirements.txt` (core). VLM local: `transformers==4.49.0` + torch/torchvision khớp nhau
(driver cũ → dùng CPU; xem `docs/lessons-learned.md`). Model version + lineage trong `registry.yaml`
và nhúng `model_versions` trong mỗi output JSON.
