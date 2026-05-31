# debug-workflows.md — Quy trình xử lý sự cố

Ghi **triệu chứng → chẩn đoán → sửa**. Cập nhật liên tục khi thực thi.

## Debug pipeline theo stage (nguyên tắc)
Khi 1 doc sai/lỗi, **cô lập theo stage**, đừng đoán. Mỗi stage emit 1 dòng log structured `{doc_id, stage, status, dur_ms, model_version}`. Dò lỗi = grep theo `doc_id`.
1. `quality` → in QualityReport (blur/brightness/res/skew).
2. `ocr` → dump raw `[{text,bbox,conf}]` ra `docs/logs/ocr_dump_<doc>.json`.
3. `kie.candidate_gen` → log candidate + feature vector mỗi field.
4. `kie.classifier` → log P(field|candidate) + candidate được chọn.
5. `confidence` → log 4 thành phần ensemble + xác suất classifier.
6. `router` → route (traditional/vlm) + lý do (missing/low-conf/conflict).
7. `vlm` → raw output + guardrail verdict (json valid? repeated? timeout?).
8. `postprocess` → field nào fail Pydantic.
9. `orchestrator` → state transition + retry count + lý do dead-letter.

---

## Sổ tay lỗi (điền khi gặp)
### [TEMPLATE]
- **Triệu chứng / Chẩn đoán / Nguyên nhân gốc / Cách sửa / Phòng ngừa**

---

## Pre-mortem (lỗi dự kiến — xác nhận khi chạy)

### Disk đầy khi `pip install`
- **Triệu chứng:** `No space left on device` (máy này free dao động 3–5GB, đang bị thứ khác ăn).
- **Chẩn đoán:** `df -h /home`, `du -sh .venv ~/.cache/pip`.
- **Sửa:** `opencv-python-headless`; `pip install --no-cache-dir`; xoá `~/.cache/pip`; **KHÔNG cài torch/Donut** → VLM về `api`/`remote`; bỏ FUNSD.

### RapidOCR/onnxruntime không cài trên Py3.13
- **Chẩn đoán:** `pip install rapidocr-onnxruntime -v`.
- **Sửa:** pin onnxruntime có cp313 wheel → fallback `easyocr` sau interface `OCREngine` → cuối cùng venv Python 3.11.

### Model OCR cold-start chậm/treo request đầu
- **Sửa:** warm-up lúc FastAPI `startup` (OCR 1 ảnh dummy); ghi thời gian warm-up vào metrics. Đây cũng là lý do cần Serving Layer tách khỏi request.

### Dynamic batcher: deadlock / không gom được batch
- **Triệu chứng:** request treo tới timeout, hoặc batch luôn size=1.
- **Chẩn đoán:** log `{batch_size, wait_ms}` mỗi flush.
- **Sửa:** đảm bảo flush khi `len>=max_batch` HOẶC `elapsed>=max_delay_ms` (đúng 1 trong 2); future/event resolve đúng request; có timeout per-request để không treo vô hạn.

### KIE classifier: F1 thấp / lỗi shape feature
- **Triệu chứng:** F1≈0 dù OCR đọc đúng; hoặc `ValueError: feature shape mismatch`.
- **Chẩn đoán:** so feature vector lúc train vs lúc infer (cùng thứ tự/độ dài?); kiểm chuẩn hoá value (số tiền so theo int `235000`, không theo chuỗi `235,000`) đồng nhất giữa pred/gold/eval.
- **Sửa:** đóng băng feature order trong `features.py`; cùng hàm normalize ở postprocess + eval; nếu thiếu data → thêm regularization/giảm feature; luôn report baseline rule-only để biết classifier có thực sự cải thiện.

### Train/eval không reproduce (số khác mỗi lần)
- **Sửa:** seed cố định (`--seed 42`), pin version sklearn, ghi metrics.json + registry. Nếu vẫn lệch → kiểm thứ tự đọc file dataset (sort path).

### Dataset SROIE 404
- **Sửa:** thử HF `darentang/sroie` → GitHub `zzzDavid/ICDAR-2019-SROIE` → ảnh VN synthetic làm tập tối thiểu. Ghi nguồn thật vào `docs/logs/`.

### Batch kẹt `processing`
- **Chẩn đoán:** đếm doc theo state trong SQLite; xem dead-letter; worker còn sống?
- **Sửa:** enforce `per_image_timeout_sec`; quá retry → `failed`+dead-letter; job có ≥1 success + có fail → `partial_completed` (không kẹt).

### VLM output hỏng (failure-mode kinh điển)
- **Triệu chứng:** whitespace lặp / phrase lặp / chạm token ceiling / timeout.
- **Sửa:** guardrails: `max_new_tokens` thấp, JSON-only + parse strict, detect repeated n-gram, timeout + retry≤1; invalid → `needs_human_review` (không để cascade latency).

### `/metrics` rỗng / `Duplicated timeseries`
- **Sửa:** một registry global; `.inc()` trong orchestrator; định nghĩa metric 1 lần ở module load, không tạo lại mỗi request.

### eval-gate fail trong CI nhưng local pass
- **Chẩn đoán:** so dataset/seed/threshold giữa CI và local; CI có tải đúng subset không.
- **Sửa:** pin `--f1-threshold`, commit subset nhỏ cố định cho CI (hoặc cache), log F1 thực vào artifact CI.
