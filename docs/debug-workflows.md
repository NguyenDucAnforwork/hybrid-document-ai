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

## WP-3 OCR recognizer fine-tune — sự cố thực tế

### WP3-1: CRNN collapse về all-blank (val CER = 1.0)
- **Triệu chứng:** eval đầu tiên ft CER=1.0, exact=0.0, pred rỗng toàn bộ (default CER=0.34) → rel -197%.
- **Chẩn đoán:** CER=1.0 + pred="" ⇒ CTC decode ra toàn blank (idx 0). Train chỉ 3 epoch (~123 step), lr 1e-4 (giá trị cho *fine-tune pretrained*, không phải from-scratch).
- **Nguyên nhân gốc:** CRNN from-scratch chưa thoát "all-blank regime" của CTC; lr thấp + quá ít step (mới dùng <1 phút trong budget 1h).
- **Cách sửa:** lr 1e-3, 60 epoch → val CER 0.0854, exact 0.60; vẫn 227s, peak VRAM 1316MB.
- **Phòng ngừa:** CTC from-scratch đừng mượn HP của fine-tune; sanity-check pred KHÔNG rỗng sau epoch 1; tận dụng hết budget.

### WP3-2: ONNX export crash "Module onnx is not installed"
- **Triệu chứng:** train xong, best.pt lưu OK, `torch.onnx.export` ném `OnnxExporterError`.
- **Nguyên nhân:** `pip install torch` không kéo theo package `onnx` (exporter cần).
- **Cách sửa:** `pip install onnx`; re-export từ best.pt (không train lại).
- **Phòng ngừa:** cài `onnx` cùng torch; export là bước riêng đọc checkpoint, không nhồi cuối train loop.

### WP3-3: per-field CER ra rỗng (SELLER/ADDRESS… thiếu dòng)
- **Triệu chứng:** report chỉ có ALL/diacritics/digit_heavy; nhãn field n=0.
- **Nguyên nhân gốc:** `_crop_label_map` key bằng đường dẫn tuyệt đối nhưng lookup bằng basename (`Path(p).name`) → không khớp.
- **Cách sửa:** key map theo `Path(cp).name` → SELLER -74.8%, ADDRESS -82.4%, TOTAL_COST -64.6%.
- **Phòng ngừa:** chuẩn hóa key (basename) cả ghi lẫn đọc; assert tổng n field ≈ n_val.

### WP3-4: gain crop-level (-73%) ≫ full-image (~-21%) → bottleneck là detector
- **Triệu chứng:** crop-level CER -73% nhưng full-image macro chỉ -21%, TIMESTAMP ~0%.
- **Chẩn đoán:** dump failure full-image → detector gộp nhầm vùng + sai reading order (`Co. optTo. H B. Mi`), có dòng pred rỗng (detector miss).
- **Nguyên nhân gốc:** recognizer tốt không cứu input bị segment sai; bottleneck dịch sang detector/crop/line-grouping.
- **Cách (WP sau):** cải thiện detector/line-grouping, KHÔNG fine-tune recognizer thêm (ADR-19).
- **Phòng ngừa:** luôn đo full-image (det+rec) trước khi claim downstream; crop-level chỉ là chặn trên.

### WP3-5: Task B (anchor split) null result — sai trục lỗi (ngang vs dọc)
- **Triệu chứng:** thêm horizontal anchor-split (`DOCAI_LINE_REGROUP=1`) nhưng overmerge_rate 0.07→0.07 (không đổi).
- **Chẩn đoán:** taxonomy theo field cho thấy OVERMERGE tập trung ở **ADDRESS (24/32)** = merge **dọc** nhiều dòng địa chỉ; ca merge **ngang** (`Ngày…Tổng tiền`) hiếm (TIMESTAMP 4, TOTAL 0).
- **Nguyên nhân gốc:** fix đúng cơ chế (split ngang theo anchor, unit pass) nhưng **sai trục**: over-merge ở đây là dọc.
- **Cách (next):** in-box horizontal-projection row split cho box cao nhiều dòng (kỹ thuật khác). Giữ anchor-split (đúng cho ca ngang, flag-gated).
- **Phòng ngừa:** luôn đo before/after; đừng assume một fix "hợp lý" sẽ giảm metric khi chưa biết trục lỗi (đo taxonomy theo field trước).

### WP3-6: TIMESTAMP full-image kém dù detector recall=1.0
- **Triệu chứng:** TIMESTAMP full-image CER ~0.45 (gần như không cải thiện) dù crop-level 0.098.
- **Chẩn đoán:** taxonomy TIMESTAMP = REC_ERROR 26 (coverage 1.0, overmerge 4) → box đúng nhưng recognizer đọc sai.
- **Nguyên nhân gốc:** crop của detector ≠ phân phối crop train (line time/date `10:44:08-15/08/2020`); recognizer chưa robust với crop detector-style.
- **Cách (next):** train/augment recognizer bằng crop sinh từ detector thật (không chỉ crop gold); hoặc normalize crop trước recognize. → **đã làm ở WP3-8.**

### WP3-7: Task E (projection row-split) cũng null cho ADDRESS
- **Triệu chứng:** thêm `DOCAI_PROJECTION_SPLIT=1` (tách box cao theo valley) nhưng ADDRESS full-image CER 0.319→0.316 (null), overmerge 0.07→0.07, recall 0.978→0.963 (over-split nhẹ).
- **Chẩn đoán:** xem failure → pred là **rác recognizer** trên crop detector (`188 Hau Giang…`→`P a Ta, xự Q H…`), không phải nhiều dòng dính nhau.
- **Nguyên nhân gốc:** ADDRESS loss KHÔNG phải merge dọc mà là crop-distribution gap (giống TIMESTAMP). Hai fix grouping (B, E) đều sai hướng.
- **Cách sửa:** → Task F (WP3-8). Giữ projection split flag-gated, default OFF.
- **Phòng ngừa:** khi 2 fix cùng họ (grouping) đều null, dừng đào sâu họ đó — đổi giả thuyết (sang recognizer/crop).

### WP3-8: Task F (detector-style crop augmentation) — fix đúng, mọi field tốt lên
- **Triệu chứng (mục tiêu):** đóng crop-distribution gap cho TIMESTAMP/ADDRESS.
- **Cách:** trích 3862 detector-style crops (box match gold) + short fine-tune từ v1 (`--init-from`, mix 2×, `--augment` pad/crop jitter + blur + contrast).
- **Kết quả:** full-image macro CER 0.265→0.205; SELLER 0.179→0.111, ADDRESS 0.319→0.255, TIMESTAMP 0.454→0.376, TOTAL 0.152→0.108; clean-val 0.085→0.063. Model F thành default.
- **Phòng ngừa/đo:** TIMESTAMP suýt trượt ≤0.35 (0.376) → time/date strings vẫn khó nhất; latency full-image đo trên máy rảnh (FT re-recognize nặng hơn default, số p50 nhiễu theo tải máy).

### WP3-9: `auto` routing không bao giờ phát hiện tiếng Việt (signal vòng tròn)
- **Triệu chứng:** ablation cho `auto` macro CER = 0.34 (= default), mean #rerec = 0 → auto KHÔNG route doc MC-OCR sang FT.
- **Chẩn đoán:** `auto` tính tỉ lệ dấu tiếng Việt trên **output của recognizer default** (PP-OCR dict tiếng Trung). Default KHÔNG thể sinh dấu tiếng Việt → ratio ≈ 0 cho cả doc tiếng Việt → luôn ở dưới threshold.
- **Nguyên nhân gốc:** dùng chính cái cần phát hiện (khả năng đọc tiếng Việt) làm tín hiệu, nhưng tín hiệu lấy từ model KHÔNG đọc được tiếng Việt → vòng tròn.
- **Cách sửa:** probe bằng **FT recognizer** trên ~8 box rồi đo dấu trên output FT (FT mới sinh được dấu). VI→ratio cao→FT; EN→ratio thấp→default.
- **Phòng ngừa:** tín hiệu routing phải độc lập với thứ đang route; ablation 4 nhánh là cách lộ ra (auto≡default là cờ đỏ).

### WP3-10: config trỏ Task F nhưng vẫn load model v1
- **Triệu chứng:** ablation `ft_all` macro 0.265 (số của v1) thay vì 0.205 (Task F), dù config default đã đổi sang `vi_mcocr_crnn_ft_taskf`.
- **Nguyên nhân gốc:** `ocr_recognizer.load()` **hardcode** `os.environ.get("DOCAI_OCR_REC_MODEL", MODELS_DIR/"...vi_mcocr_crnn_ft/...")` → bỏ qua `config.OCR_REC_MODEL` đã cập nhật.
- **Cách sửa:** `load()` đọc `config.OCR_REC_MODEL/OCR_REC_DICT` (đã resolve env + default Task F).
- **Phòng ngừa:** một nguồn sự thật cho path (config), KHÔNG lặp default ở nhiều chỗ. Verify bằng `r.session._model_path` sau load.

### WP3-11: ablation chạy serial quá chậm / latency nhiễu khi chạy concurrent
- **Triệu chứng:** ablation 4 config × 80 ảnh "rất chậm" dù GPU gần rảnh.
- **Chẩn đoán:** OCR chạy **CPU** (onnxruntime CPU + RapidOCR), GPU không giúp; script còn chạy `process_document` (OCR lần 2) chỉ để lấy needs_review → 2× OCR mỗi ảnh × 4 config serial.
- **Cách sửa:** bỏ pass `process_document` (thêm `--with-review` mặc định off); thêm `--only` để chạy 4 config song song (48 core). **Nhưng:** chạy concurrent thổi p50 lên ~4× do contention CPU → **latency tuyệt đối phải đo SERIAL**; CER thì concurrent vẫn đúng.
- **Phòng ngừa:** tách "đo đúng" (CER, concurrent OK) khỏi "đo nhanh" (latency, phải serial/quiet box).

---

## Pre-mortem (lỗi dự kiến — xác nhận khi chạy)

### Disk đầy khi `pip install`
- **Triệu chứng:** `No space left on device` (máy này free dao động 3–5GB, đang bị thứ khác ăn).
- **Chẩn đoán:** `df -h /home`, `du -sh .venv ~/.cache/pip`.
- **Sửa:** `opencv-python-headless`; `pip install --no-cache-dir`; xoá `~/.cache/pip`; **KHÔNG cài torch/Donut** → VLM về `api`/`remote`; bỏ FUNSD.

### ĐÃ GẶP: `merchant_name` F1 thấp (0.45) — OCR tách token tiêu đề
- **Triệu chứng:** GOLD=`abc mart` PRED=`abc`. Per-field eval chỉ merchant tệ.
- **Chẩn đoán:** in `[t['text'] for t in run_ocr(img)]` → thấy `['ABC ','MART',...]` (OCR tách 2 token), trong khi training dùng token cả-dòng.
- **Sửa:** thêm `group_lines()` (gộp token cùng hàng y) vào `candidates()` → train/infer đồng nhất. **0.45→1.0**.
- **Phòng ngừa:** luôn so token thật (infer) với token train; mismatch candidate = nghi ngờ đầu tiên.

### ĐÃ GẶP: `total_amount` chọn nhầm số nhỏ — money regex bắt cả ngày/ID
- **Triệu chứng:** PRED=12000 (item) thay vì 120000 (TONG CONG). Xảy ra sau khi thêm line-grouping.
- **Chẩn đoán:** in candidate values cho field total → thấy dòng `Date: 01/12/2025` cho ra "money"=1122025 (regex `\d[\d.,]{2,}` nuốt cả ngày) → thành `max_money` giả → feature `is_max_number` dạy ngược.
- **Sửa:** `MONEY_RE = \d{1,3}(?:[.,]\d{3})+` (bắt buộc dấu phân cách nghìn). `total_amount`→1.0.
- **Phòng ngừa:** regex trích xuất phải **loại trừ lẫn nhau** giữa các field (date/ID/money); test trên token thật, không chỉ chuỗi đẹp.

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

---

## LayoutLMv3 / transformers — lỗi đã gặp

### ĐÃ GẶP: transformers 5.x incompatible với torch 2.6 (float8_e8m0fnu)
- **Triệu chứng:** `ModuleNotFoundError` hoặc `AttributeError` khi import `AutoProcessor` từ transformers; stack trace đề cập `torch.float8_e8m0fnu`.
- **Chẩn đoán:** transformers ≥5.0 dùng `torch.float8_e8m0fnu` type — type này chưa tồn tại trong torch 2.6. Kiểm tra: `python -c "import torch; print(torch.__version__)"` + `pip show transformers`.
- **Sửa:** `pip install "transformers==4.49.0"` (downgrade về phiên bản tương thích với torch 2.6).
- **Phòng ngừa:** khi install torch specific version (đặc biệt torch 2.x), LUÔN pin transformers version tương thích ngay cùng lúc. Không để pip resolve transformers tự động — nó sẽ lấy latest 5.x và break. Rule: `pip install torch==2.6.x transformers==4.49.0 torchvision` trong một lệnh.

### ĐÃ GẶP: LayoutLMv3 train/infer gap — OCR tokens ≠ box-file tokens
- **Triệu chứng:** validation F1 trên SROIE box-file tokens đạt ~0.91, nhưng test F1 với OCR tokens (RapidOCR) thấp hơn nhiều (date/total F1 < 0.2, chỉ merchant_name còn tốt).
- **Chẩn đoán:** SROIE training dùng ground-truth box annotations — mỗi token là một từ riêng biệt, bbox chính xác pixel từ annotation file. RapidOCR trả về line-grouped tokens (nhiều từ gộp thành 1 token) với pixel bbox từ detector. Hai loại input khác nhau về (1) độ granularity token và (2) bbox coordinate space.
- **Sửa:**
  1. Normalize bbox theo W/H ảnh thật: `x_norm = int(x / img_w * 1000)` — không clip cứng 1000 khi ảnh không vuông.
  2. Apply `norm_field()` cho output LayoutLMv3 để comparable với gold (cùng normalization function dùng trong eval logistic).
  3. Khi đánh giá: luôn chạy inference qua OCR pipeline thật, không chỉ qua box-file loader.
- **Phòng ngừa:** luôn có một test set "OCR-tokenized" song song với "box-tokenized". Nếu 2 số khác nhiều → có train/infer mismatch. Kiểm tra bằng cách in tokens thật từ cả 2 source và so sánh.

### ĐÃ GẶP: norm_money() nhận float thay vì str
- **Triệu chứng:** `TypeError: expected string or bytes-like object, got float` trong DataLoader worker khi train LayoutLMv3; traceback trỏ vào `norm_money()` hoặc `re.sub()`.
- **Nguyên nhân:** SROIE gold labels có `total_amount` là float (ví dụ `10.4` từ JSON parse) thay vì string. `norm_money()` gọi `re.sub()` trên input — regex không nhận float.
- **Sửa:** trong data loader, convert gold value trước khi normalize: `total_val = str(total_raw) if total_raw is not None else ""`. Hoặc thêm type guard đầu hàm: `def norm_money(v): if not isinstance(v, str): v = str(v)`.
- **Phòng ngừa:** validate gold schema types trước khi feed vào model. Thêm unit test: `norm_money(10.4)` phải không raise. SROIE JSON có thể có mixed types (float/str) tùy field.

### ĐÃ GẶP: accelerate không được cài khi dùng HuggingFace Trainer
- **Triệu chứng:** `ImportError: Using the Trainer with PyTorch requires accelerate>=0.26.0: Please run pip install accelerate`.
- **Nguyên nhân:** `pip install transformers` không kéo `accelerate` theo mặc định dù Trainer phụ thuộc vào nó.
- **Sửa:** `pip install "accelerate>=0.26.0"` hoặc dùng extra: `pip install "transformers[torch]"` (kéo accelerate + torch đúng version).
- **Phòng ngừa:** khi setup môi trường cho LayoutLMv3/bất kỳ HF Trainer workflow nào, luôn install `accelerate` ngay từ đầu cùng transformers.

### Tối ưu LayoutLMv3 inference (kế hoạch — chưa implement)
- **Tình trạng hiện tại:** inference qua HuggingFace Trainer/pipeline, ~41ms/doc GPU (RTX 3090), ~800ms/doc CPU.
- **Bước 1 — ONNX export:**
  ```bash
  optimum-cli export onnx --model microsoft/layoutlmv3-base \
      --task token-classification \
      $DOCAI_WORKSPACE/models/layoutlmv3_onnx/
  ```
- **Bước 2 — INT8 quantization (CPU inference):**
  ```bash
  optimum-cli onnxruntime quantize \
      --onnx_model $DOCAI_WORKSPACE/models/layoutlmv3_onnx/ \
      --output $DOCAI_WORKSPACE/models/layoutlmv3_onnx_int8/ \
      --avx512
  ```
- **Kỳ vọng:** CPU latency ~800ms → ~200–300ms sau INT8 quantization. GPU latency đã đủ nhanh (41ms không phải bottleneck — OCR mới là bottleneck thật ở ~1.5–2s).
- **Lưu ý khi export:** LayoutLMv3 cần image + bbox + input_ids — ONNX export phải khai báo đủ dynamic axes cho cả 3 inputs.

### ĐÃ GẶP: zero-shot Table Transformer nuốt dòng đầu vào header row
- **Triệu chứng:** debug JSON của statement parser cho thấy ô header chứa cả anchor **và** giá trị transaction đầu tiên, ví dụ `Ngay 02/01/2024` hoặc `9,900,000 CR 62,094,716 So du`. Khi đó semantic mapping vẫn “có đủ cột”, nhưng amount/balance của các dòng sau bị lệch nặng.
- **Chẩn đoán:** `microsoft/table-transformer-structure-recognition` detect đúng khung bảng/row/column ở mức hình học, nhưng trên statement synthetic nó đôi lúc gộp row header với row transaction đầu tiên. Đây là lỗi structure tốt-bề-ngoài nhưng sai cell assignment.
- **Sửa:** thêm `header_contaminated` guard trong `docai/statement.py`; nếu header row của zero-shot parser đã chứa date/amount thật thì mode `hybrid` fallback về rule parser. Đồng thời lưu overlay + JSON vào `DOCAI_STATEMENT_DEBUG_DIR` để nhìn rõ row/column/cell assignment.
- **Phòng ngừa:** zero-shot table structure chỉ nên là một tín hiệu phụ/ablation path. Muốn thắng parser rules hiện tại phải fine-tune hoặc domain-adapt trên statement bank thật, không chỉ drop-in model pretrained.

### ĐÃ GẶP: PP-OCRv4 hallucinate chữ Hán trên ảnh rotate/low-res (CJK hallucination)
- **Triệu chứng:** `merchant_name` trả về Unicode Hán tự (e.g. `"我物出门，#不处或更"`) khi OCR ảnh receipt Latin/Malay bị rotate hoặc low-resolution. Phát hiện khi test live trên `000.jpg` (SROIE). JSON response có `\uXXXX` escape sequences decode ra chữ Trung Quốc.
- **Chẩn đoán:**
  1. Kiểm tra `quality.issues` trong response — nếu có `"rotated_image"` + `"low_resolution"` → nghi ngờ hallucination.
  2. Kiểm tra `skew_angle` trong response — nếu `abs(skew_angle) > 45` thì ảnh gần như lật 90°.
  3. Confirm bằng cách decode field value: `python3 -c "print('field_value_here')"` — nếu ra chữ Hán mà context là tiếng Anh → hallucination.
  4. Root cause: PP-OCRv4 train chủ yếu trên Chinese corpus → khi text Latin bị distort geometric → classify sai thành Chinese stroke patterns.
- **Sửa (đã implement trong `docai/pipeline.py`):**
  1. **`_deskew(img, angle)`**: Dùng `skew_angle` từ `QualityReport`, áp dụng `cv2.getRotationMatrix2D + warpAffine` trước `run_ocr()`. OCR nhận ảnh thẳng → không hallucinate.
  2. **`_filter_cjk_hallucination(extracted, tokens)`**: Post-KIE guard — nếu corpus tokens không phải Chinese doc (tổng CJK ratio < 30%) mà field value có > 30% CJK chars → set `(None, 0.0)`, trigger human review.
  3. **`QualityReport.skew_angle`**: Expose góc từ `check_quality()` (trước chỉ tính nội bộ không return).
- **Phòng ngừa:**
  - Luôn kiểm tra `quality.issues` và `skew_angle` khi field value trông bất thường.
  - Test với ảnh rotate khi setup OCR pipeline mới. Bất kỳ PP-OCR-based engine nào đều có risk này.
  - Khi thêm ngôn ngữ mới (Japanese, Korean): điều chỉnh CJK filter threshold hoặc dùng language detection (langdetect) thay vì heuristic ratio.
- **Lưu ý giới hạn của fix:** deskew chỉ hiệu quả khi `skew_angle` estimate chính xác. Ảnh bị vừa blur vừa rotate nặng → angle estimate sai → vẫn cần CJK filter làm lớp backup. Ảnh blur + rotate vẫn có `needs_human_review=True` qua quality gate.

### ĐÃ GẶP: Deskew regression — xoay ảnh sai 89° gây silent wrong output (CRITICAL)
- **Triệu chứng:** Sau khi thêm `_deskew()`, total_amount trả về 9744 thay vì 112, 519537 thay vì 2.5, date "0240-24-03" thay vì "2018-03-14" — và `needs_human_review=False`. Silent wrong trên 9/30 ảnh test, từ 0% lên 100%.
- **Root cause:** `cv2.minAreaRect` trên toàn bộ foreground pixels của SROIE trả về angle ~-89° cho hầu hết ảnh (không phải vì ảnh bị xoay, mà vì bounding box pixel fill có hình dạng vertical). `_deskew(img, -89°)` → xoay ảnh sai 89° → OCR đọc sai hoàn toàn → KIE extract wrong values với confidence cao.
- **Sửa:** Condition deskew chỉ với `abs(skew_angle) < 45` — chỉ correct genuine small-angle skew. Near-90° từ minAreaRect là ambiguous (portrait vs landscape), không auto-correct.
- **Bài học:** Không bao giờ apply transformation heuristic mà không verify trước trên sample. `minAreaRect` của tất cả foreground pixels ≠ text skew angle. Đây là silent correctness bug tệ hơn crash: system tự tin cao (conf > 0.9) nhưng output sai.

### ĐÃ GẶP: Silent wrong output — KIE chọn sub-total/barcode thay vì grand total
- **Triệu chứng:** `needs_human_review=False` nhưng `total_amount` sai: 9744 (phone number), 8.68 (tax line), 31.03 (subtotal vs 32.7). Date đôi khi trả về "2015-07-03" thay vì "2019-01-12".
- **Chẩn đoán:** Rule-based KIE candidate selection (`_rule_score`) chọn số lớn nhất trên receipt làm total, hoặc số gần keyword nhất làm date. Khi receipt có nhiều số (phone, invoice ID, subtotal) → sai candidate được chọn với confidence cao.
- **Sửa đã implement:** `_sanity_check()` trong pipeline.py bắt được cases catastrophic:
  - `total > 50,000` → implausible_total (bắt được barcode 519537)
  - `total ≤ 0` → nonpositive_total
  - `year < 2000 hoặc > 2035` → implausible_year (bắt được 1815, 0240)
  - `month > 12 hoặc day > 31` → implausible date components
- **Remaining limitation (5/30 cases — chấp nhận được):** "Plausible wrong" values (sub-total, tax line, nearby number) không bắt được bằng range check. Root fix: LayoutLMv3 hybrid routing — sequence labeling với layout context xác định đúng "TOTAL" token thay vì dùng regex candidate.
