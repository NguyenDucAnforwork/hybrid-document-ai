# lessons-learned.md — Bài học & quyết định kỹ thuật (ADR)

Ghi **quyết định + lý do + đánh đổi**. Phần này nhà tuyển dụng ngân hàng đọc kỹ nhất: thể hiện tư duy engineering/production, không chỉ code chạy.

## Architecture Decision Records

### ADR-1: OCR engine = RapidOCR (ONNX) thay vì paddlepaddle
- **Bối cảnh:** disk 5GB; `paddlepaddle` >1GB + dễ vỡ build trên Python 3.13.
- **Quyết định:** RapidOCR (onnxruntime) — vẫn model PP-OCRv4/v5 nhưng ONNX, CPU-only, nhẹ; lại **drop-in được vào Triton** (cùng định dạng ONNX) → khớp Serving Layer.
- **Đánh đổi:** mất PP-StructureV3; bù bằng layout-graph heuristic. Chấp nhận vì receipt layout đơn giản.
- **Bài học:** "Dùng PaddleOCR" không bắt buộc kéo theo framework paddle. Tách model khỏi framework giúp vừa hạ tầng *và* serving được bằng Triton.

### ADR-2: Mọi thành phần nặng ẩn sau interface + có impl nhẹ (swap-by-config)
- Redis→`MemoryQueue`, MinIO→`FsObjectStore`, VLM→stub/api, Triton→config artifact. Production swap bằng **config**, không sửa code.
- **Bài học:** demo trong 2h/5GB **mà vẫn** chứng minh production design — giá trị hơn việc cài thật mọi thứ.

### ADR-3: VLM là HARD-CASE path, mặc định không local
- Qwen-VL-7B local = ~15GB + >4GB VRAM → vi phạm budget. Donut-base (~0.7GB) chạy được ≤4GB → để **optional local showcase**; mặc định `disabled`→`api`/`remote_gpu(vLLM)`.
- Giữ trigger logic + prompt + guardrail (JSON-only, max_tokens thấp, timeout, detect repeated-phrase) — hiểu failure-mode VLM-OCR ở scale.
- **Bài học:** với bank, *biết KHI NÀO không gọi VLM* (cost/latency/độ tin) quan trọng ngang biết gọi. Router confidence là nơi quyết định.

### ADR-4: Confidence-driven human review thay vì cố đạt 100% accuracy
- Ảnh khó → `needs_human_review`, không đoán bừa.
- **Bài học:** trong fintech, **sai im lặng** nguy hiểm hơn **từ chối có kiểm soát** → "fail loud → human".

### ADR-5: KIE là multi-model 2-tier, KHÔNG chỉ regex (phản hồi reviewer)
- **Bối cảnh:** rule+regex quá đơn giản; JD VNPAY = CV/OCR + MLOps; bài viết: Processing Layer phải *multi-model*.
- **Quyết định:** Tier-1 = candidate-gen (regex/keyword/layout-graph) → **feature vector** → **scikit-learn field classifier** (calibrated confidence, trainable, versioned). Tier-2 = **VLM OCR-free** (Donut/Qwen) cho hard cases qua hybrid router.
- **Đánh đổi:** sklearn cần data + dễ overfit khi ít mẫu → feature đơn giản + regularization + calibration; **báo cáo baseline rule-only (Setting A)** để chứng minh giá trị gia tăng định lượng.
- **Bài học:** đây chính là *training pipeline* JD đòi. Chọn model **đủ nhỏ để train CPU vài phút trong 5GB** quan trọng hơn chọn model to. Layout-aware KIE ≠ bắt buộc LayoutLM nặng.

### ADR-6: Serving Layer thật = dynamic batcher chạy được + Triton/vLLM artifact
- **Bối cảnh:** JD/bài viết nhấn Triton/vLLM/dynamic batching/GPU resource mgmt; disk 5GB không chứa nổi.
- **Quyết định:** tự viết **dynamic micro-batcher** (gom request OCR theo cửa sổ + batch ONNX) chạy thật, test được. Triton `config.pbtxt` + vLLM client = artifact, swap-by-config.
- **Bài học:** "Show the mechanism, not the logo" — implement tối giản chạy được + giải thích đánh đổi thuyết phục hơn cài Triton mà không hiểu.

### ADR-7: MLOps lifecycle khép kín
- registry/versioning (`models/registry.yaml`), eval-as-CI-gate (F1 threshold), monitoring+drift, CI/CD — đúng 4 từ khoá MLOps trong JD.
- Output JSON nhúng `model_versions` → **truy vết model version nào tạo kết quả nào** (yêu cầu audit ngân hàng).
- **Bài học:** với bank, traceability + eval-gate là bắt buộc, không phải nice-to-have.

## Bài học khi thực thi (ĐÃ CHẠY THẬT)
- **[P0] Disk:** `/home` đầy 100% (free tụt 5.3→1.9GB trong phiên) → chuyển venv/data/models sang `/data` (631GB), code ở nltk_data. Tách *code* (nhỏ, versioned) khỏi *artifacts* (lớn, không commit) cứu được khi mount chính hết chỗ. Bỏ `torch` → venv chỉ 743MB.
- **[P1] OCR:** RapidOCR ship sẵn ONNX (không tải mạng); batcher test chứng minh `max_seen>=2` (gom batch thật).
- **[P2] KIE (v1→v2, debug bằng chính eval):** v1 end-to-end **0.865**, `merchant_name` F1 **0.45**. Debug ra **2 nguyên nhân thật** (không phải classifier yếu):
  1. **OCR tách tiêu đề nhiều chữ** ("ABC"+"MART") còn training dùng token cả-dòng → **train/serve mismatch**. Fix: thêm **layout-graph line-grouping** gộp token cùng hàng → áp đồng nhất train+infer. `merchant_name` **0.45→1.0**.
  2. **Money regex quá lỏng** (`\d[\d.,]{2,}`) bắt cả ngày ("01/12/2025"→1122025) và invoice ("HD4657"→4657) thành tiền giả → số giả khổng lồ thành `max_money` → classifier học "total KHÔNG phải số lớn nhất" → chọn nhầm item nhỏ nhất. Fix: money phải có **dấu phân cách nghìn** `\d{1,3}(?:[.,]\d{3})+`. `total_amount` về **1.0**.
  - Thêm feature `is_largest_font` (merchant = dòng font to nhất). **v2: macro-F1 0.98, all-required-correct 1.0**. `invoice_id` 0.90 là trần thật (OCR đọc nhầm `1↔l`) → low-conf → human review.
  - **Bài học lớn nhất:** *gần như mọi điểm yếu "model" hoá ra là bug ở candidate-generation / chuẩn hoá, lộ ra nhờ eval per-field + nhìn token thật.* Đây là giá trị của benchmark có gold, không phải "accuracy đẹp".
- **[P3] Router + batch:** ngưỡng 0.75 hợp lý (doc clean ~0.95 không kích VLM); batch state machine + summary chuẩn.
- **[P4] MLOps:** eval-gate thr=0.6, macro-F1 0.865 → PASS. Drift hữu ích nhất: phân phối `field_confidence` + `input_blur_score`. `np.float64` không serialize YAML → phải sanitize trong registry.
- **[P5] Wrap:** cắt Donut local (cần torch, đụng disk) → VLM `api` mode. Thêm giờ: ràng buộc candidate merchant + LayoutLMv3 ONNX + chạy Triton/vLLM thật trên H100.

## v3→v4: chuyển sang DATA THẬT (SROIE) + metric mạnh + calibration
Reviewer chỉ ra: synthetic quá sạch, metric yếu, MLOps sơ sài. Đã xử lý:
- **Data thật:** 626 receipt scan SROIE (Malaysia, thermal-printer, nhiễu thật) + **augmentation banking** (`docai/augment.py`: tối/nghiêng/mờ/motion-blur/nhiễu/rách/fade/low-res/JPEG/perspective) cho **robustness-curve** + receipt **tiếng Việt** có dấu. Train kết hợp SROIE+synthetic (`kie:v4`).
- **Bug quan trọng (quality gate):** ảnh SROIE rộng ~463px < `MIN_DIM=480` → quality gate **chặn cứng low-res** → trả None toàn bộ (merchant/date/total = None). Đây là lý do v3 ban đầu macro chỉ 0.22. **Sửa:** low-res là *flag* không phải *blocker*; luôn OCR (upscale ảnh nhỏ ×→720), router quyết review. → date/total nhảy vọt.
- **Money parsing đa domain:** SROIE total là thập phân "9.00"/"1,234.56" còn VND "235,000" — regex cũ hỏng. Tổng quát hoá: strip date trước, `,`=nghìn `.`=thập phân, trả float. 
- **Metric mạnh:** thêm **CER, ANLS (ngưỡng 0.5), ECE (calibration)** ngoài exact-match/F1. Với field text dài như merchant, exact-match≈0 nhưng ANLS cho điểm partial — exact-match là metric **sai** cho field đó.
- **merchant_name là field khó kinh điển:** OCR đọc tên công ty dính liền không dấu cách + nhiều dòng + watermark "tan woon yann" ở mọi ảnh SROIE → chọn nhầm dòng. Quyết định trung thực: report ANLS + **để confidence router đẩy sang human review** thay vì giả vờ đúng. 
- **Calibration (CalibratedClassifierCV sigmoid):** confidence trở nên đáng tin → router gửi đúng doc bất định sang review/VLM; đo bằng ECE. Đây là MLOps "trust", không chỉ accuracy.
- **MLOps depth:** training-pipeline DAG (`mlops/pipeline.py`: ingest→validate→train→eval→register, có cache/retry/resume + manifest lineage) + KFP artifact; registry có **stage dev/staging/prod/archived + lineage**; data validation gate; **chaos engineering** (`mlops/chaos.py`); KServe/alerts/DR runbook. Xem `docs/mlops.md`.

## Robustness findings (real SROIE + augmentation, n=30, severity 0.6)
- **Router bắt được blur/motion-blur:** OCR confidence sụp → `needs_review`=1.0 (từ chối rác đúng). motion_blur F1 0.015, mean_conf 0.30 → calibration đúng ở case này.
- **Router KHÔNG bắt rotate/perspective:** CER nổ (1.18/1.39) nhưng confidence vẫn ~0.87, `needs_review`≈0.03 → **output sai mà tự tin cao** (ECE 0.51–0.55). Vì confidence phản ánh độ chắc *chọn dòng* + *OCR tự báo*, mà OCR vẫn "tự tin" trên chữ bị méo hình học. **Fix:** thêm bước deskew/perspective-correction + geometry-aware confidence. Đây là failure mà một con số "accuracy sạch" che giấu — lý do phải đo robustness-curve.
- **total_amount confidence không phân biệt đúng/sai** (0.85 khi đúng = 0.85 khi sai): classifier chấm chọn-dòng, không chấm đúng-transcription của OCR. → cần validation rule (checksum/range) + OCR-confidence-aware ensemble.
- dark/fade được dung nạp (OCR bền); noise/tear/jpeg/low_res giảm vừa.

## VLM hard-case fallback đã CHẠY THẬT (Qwen2.5-VL-3B)
- Trước đây VLM chỉ là stub (`disabled`). Giờ `mode=local` chạy **Qwen2.5-VL-3B** thật (transformers) cho ca router flag.
- **Vướng môi trường:** driver GPU 12.2 quá cũ cho torch py3.13 CUDA wheels (cu130 cần driver 13; cu121 không có wheel py3.13) → chạy **CPU** (~17s/ảnh). Production: GPU qua vLLM (`mode=remote_gpu`). transformers 5.x lazy-import hỏng → pin `transformers==4.49.0`; torchvision phải khớp torch.
- **Kết quả (12 ảnh blur, VLM bật 3/12):** cải thiện mọi field — date ANLS 0.68→**0.93**, total 0.58→**0.66**, merchant 0.0→**0.17**; latency 1.4s→17.8s. Đúng hybrid trade-off: VLM đắt nên CHỈ chạy ca flag, không cả batch.
- **Bài học:** giá trị VLM rõ nhất ở ca khó; trên ảnh clean router KHÔNG kích VLM (kỷ luật cost) — 0/12 ở set clean. "Khi nào KHÔNG gọi VLM" quan trọng ngang khi nào gọi.

## Multi-document + bảng sao kê: data khó lộ giới hạn thật (reviewer feedback)
- Reviewer chỉ ra điểm 1.0 đáng ngờ → viết **HARD statement generator** (random column/Nợ-Có/ngoặc/CR-DR/VN-EN/footer/jitter). Kết quả tụt về số thật.
- **Bug tìm được nhờ data khó:** (1) generator không vẽ cột description (desc=0.0 giả); (2) gán cột theo center bị " CR/DR" kéo lệch → đổi sang **left-edge** → desc 0.55→**0.99**; (3) anchor Nợ/Có tiếng Việt có dấu cách thừa + substring "co" match nhầm "content" → **word-boundary match** → amount 0.375→**0.46**.
- **Kết luận trung thực:** rule-based table parsing KHÔNG generalize (amount 0.46); **VLM-3B cũng không cứu (0.33)** → statement table cần **table-structure model** (Table-Transformer/LayoutLMv3) hoặc VLM lớn. Đã thử VLM-hybrid thật (reconciliation→Modal) và báo cáo đúng là không cải thiện — không giấu.
- **An toàn là điểm mấu chốt ngân hàng:** `statement.reconcile` (balance khớp amount?) flag **~87%** statement khó → `needs_human_review` → **không phát số tiền sai im lặng**. Đây mới là giá trị production (detect-and-flag), không phải accuracy đẹp.
- **Spike tiếp theo (zero-shot table-structure model):** thêm `microsoft/table-transformer-{detection,structure-recognition}` như một parser zero-shot cho statement table, có overlay debug row/column/cell assignment (`docs/logs/statement_table_debug_20260609_1213/`). Kết quả trên 8 hard statements với OCR tokens: **rules 0.943 / 0.413 / 0.987** (row-F1 / amount-acc / desc-acc), **TATR 0.791 / 0.291 / 0.873**, **hybrid 0.943 / 0.413 / 0.960**. Kết luận: **zero-shot chưa đủ tốt để thay parser hiện tại**; giá trị thực tế là debug structure + thêm một selector an toàn để fallback về rules khi header row bị “nhiễm” dòng giao dịch đầu.
- 3rd type **payment_order** (anchor KV) + **batch demo 5–10 ảnh** trên Space.

## Nếu làm lại / có thêm thời gian
- **Fine-tune hoặc domain-adapt Table-Transformer/LayoutLMv3 cho statement table** (zero-shot đã thử và không thắng rules; bước tiếp theo phải là adaptation thật thay vì chỉ drop-in model pretrained).
- **Deskew/perspective-correction trước OCR** (điểm yếu lớn nhất trong robustness) + geometry-aware confidence.
- merchant_name: ràng buộc header-block + NER (LayoutLMv3 ONNX) thay token-line heuristic.
- Chạy Triton + vLLM + KServe + KFP thật trên H100/k8s, đo throughput batching + autoscale.
- Drift detector thống kê (PSI) + Grafana + OpenTelemetry tracing + experiment tracking (MLflow).

---

## v4→v5: LayoutLMv3, v1 API, production-like deploy

### ADR-8: Fine-tune LayoutLMv3 thay vì logistic KIE cho merchant_name
- **Bối cảnh:** logistic-KIE cho merchant_name ANLS chỉ 0.10 trên SROIE thật (OCR gộp token, nhiều dòng, watermark). Reviewer hỏi tại sao không dùng layout-aware model.
- **Quyết định:** fine-tune LayoutLMv3-base (BIO token classification) trên SROIE train, ~4 phút RTX 3090 (`training/train_layoutlmv3.py --epochs 5 --batch 4`).
- **Kết quả trung thực (SROIE test n=80):**
  - merchant_name ANLS: logistic 0.10 → LayoutLMv3 **0.71** (cải thiện rõ rệt)
  - date F1: logistic **0.775** > LayoutLMv3 0.19 (logistic vẫn tốt hơn)
  - total_amount F1: logistic **0.49** > LayoutLMv3 0.04 (logistic vẫn tốt hơn)
- **Lý do train/infer gap ở date/total:** SROIE training dùng ground-truth box annotations (token-per-word, precise bbox) — LayoutLMv3 học layout ở độ granularity đó. Inference dùng RapidOCR (line-grouped tokens, pixel bbox ≠ normalized 0–1000 units). Sự khác biệt token granularity + bbox scale khiến LayoutLMv3 confuse ở date/total trong khi merchant (cả khối đầu trang) ít bị ảnh hưởng hơn.
- **Kết luận — hybrid routing là production call đúng:** logistic cho date/total (nhanh, chính xác, không phụ thuộc GPU); LayoutLMv3 cho merchant_name (multimodal layout, GPU 41ms/doc). Không phải "model mới tốt hơn mọi thứ" — từng model phục vụ một điểm mạnh cụ thể.
- **Bài học:** "Replace all" là bẫy phổ biến khi ra mắt model mới. Đo per-field trước khi quyết định replace hay hybrid. Val F1 cao (0.91 box tokens) không đảm bảo production performance tốt — phải test với OCR tokens thật.

### ADR-9: v1 API design — idempotency, feedback loop, traceability
- **Bối cảnh:** v0 API (`POST /documents/extract`) là fire-and-forget, không có idempotency, không có correction mechanism, log không structured.
- **Quyết định:** thiết kế lại thành v1 với:
  - `POST /v1/documents` + Idempotency-Key header: upload an toàn, retry không tạo duplicate
  - `POST /v1/extraction_jobs`: tách upload khỏi processing (async-ready)
  - `POST /v1/documents/{id}/feedback`: human correction → training data loop (annotator sửa → lưu vào gold set → retrain trigger)
  - `request_id` middleware: mọi response có unique ID để trace end-to-end
  - `X-Latency-Ms` header: latency monitoring không cần Prometheus đọc log
  - Structured JSON logs: mọi event có `{request_id, doc_id, stage, dur_ms, model_version}` → query được
- **Tại sao quan trọng với fintech:** ngân hàng yêu cầu **audit trail** — phải chứng minh được kết quả nào được tạo ra bởi model version nào, lúc nào, từ input nào. Idempotency key ngăn double-charge/double-extract khi retry. Feedback endpoint tạo vòng lặp liên tục: production errors → annotated corrections → model improvement.
- **Đánh đổi:** phức tạp hơn (thêm 2 endpoints, middleware, state tracking) nhưng traceability + audit là bắt buộc, không phải nice-to-have trong môi trường tài chính.

### ADR-10: docker-compose runnable thay vì artifact-only
- **Bối cảnh:** docker-compose.yml trước đây liệt kê Triton + vLLM + KServe — không chạy được trên dev box (GPU memory + download size). Reviewer hỏi tại sao không demo được local stack.
- **Quyết định:** tách stack thành (1) **local-runnable** và (2) **cloud-only targets**:
  - Local stack (`deploy/docker-compose.yml`): api + redis (healthcheck) + minio (healthcheck) + prometheus + grafana (auto-provisioned datasource từ `deploy/grafana/provisioning/`)
  - Cloud-only: KServe, Triton, vLLM — giữ trong `deploy/kserve.yaml` + config artifacts, không trong compose
  - Grafana datasource tự provision: `docker compose up` xong là có dashboard ngay, không cần manual setup
- **Bài học:** "runnable demo stack" quan trọng hơn "comprehensive artifact list". Reviewer muốn `docker compose up` và thấy Grafana, không muốn đọc YAML artifacts. Bỏ Triton/vLLM khỏi local stack là quyết định đúng — chúng là cloud-only và không có trong budget dev box.

### Bài học môi trường (ĐÃ CHẠY THẬT)
- **Torch CUDA wheels lớn hơn dự kiến:** ước tính ~2GB, thực tế ~11GB (torch cu121 ~8GB + torchvision ~500MB + transformers model weights ~2GB). Cần partition ≥15GB trước khi bắt đầu. Lesson: luôn kiểm tra disk trước khi install torch.
- **Version pinning bắt buộc:** `transformers==5.10.2` (mặc định từ PyPI lúc install) incompatible với torch 2.6 — lỗi `float8_e8m0fnu` type chưa tồn tại trong torch 2.6 → ModuleNotFoundError khi load AutoProcessor. Sửa: pin `transformers==4.49.0`. Bài học: khi install torch specific version, LUÔN pin transformers version tương thích ngay cùng lúc.
- **accelerate là dependency ẩn của Trainer:** HuggingFace Trainer không báo rõ trong install, chỉ fail lúc chạy với ImportError. Sửa: `pip install "transformers[torch]"` kéo accelerate theo, hoặc cài riêng `pip install "accelerate>=0.26.0"`.

---

## Inference optimization findings

### LayoutLMv3 inference: GPU vs logistic
- **LayoutLMv3 GPU (RTX 3090):** 41ms/doc (KIE-only, không tính OCR) — nhanh hơn logistic (60ms/doc) vì batch matmul GPU hiệu quả trên token sequence dài.
- **Full pipeline p50:** ~2–3s (dominated by OCR — RapidOCR single-thread CPU ~1.5–2s/doc). KIE chỉ chiếm ~2% tổng latency.
- **Implication:** optimize OCR trước khi optimize KIE — KIE đã đủ nhanh, bottleneck thật là OCR.

### Load test findings (batch 1/5/10 docs, 3 rounds)
- batch=1: p50=2.71s, throughput 22 docs/min
- batch=5: p50=12.1s, throughput 25 docs/min
- batch=10: p50=26s, throughput 23 docs/min
- **Throughput plateau ~22–25 docs/min bất kể batch size** — đây là dấu hiệu CPU bottleneck, không phải GPU/KIE bottleneck.
- **Root cause:** RapidOCR chạy single-thread trên CPU. Gom batch lớn hơn không giúp vì OCR vẫn tuần tự.
- **Recommendation production:** multiprocess OCR worker pool (N_WORKER = số CPU core) trước khi tối ưu bất kỳ thứ gì khác. Target: 100+ docs/min trên 8-core machine.

### High confidence wrong output — ADR-4 confirmed at scale
- Rotate/perspective: CER nổ (1.18–1.39) nhưng confidence vẫn ~0.87, `needs_review`≈0.03 → output sai mà pipeline tự tin cao.
- Load test xác nhận điều này ở scale: với 10-doc batch có ảnh rotate, ~30% doc bị sai im lặng.
- **Fintech-critical failure mode:** đây là nguy hiểm nhất — không phải lỗi rõ ràng mà là wrong extraction được accept. ADR-4 (confidence-driven human review) đúng hướng nhưng cần thêm geometry-aware confidence để bắt được loại này.

### ADR-11: ONNX Export + INT8 Dynamic Quantization cho LayoutLMv3
- **Bối cảnh:** LayoutLMv3 fine-tuned là PyTorch model (478MB). Để serve nhất quán với OCR layer (RapidOCR ONNX) và tích hợp Triton, cần export sang ONNX. INT8 dynamic quantization giảm size + tăng tốc inference CPU.
- **Quyết định:** tự viết `training/export_onnx.py` với wrapper class `LayoutLMv3Wrapper` (expose clean 4-input interface: `input_ids, attention_mask, bbox, pixel_values`), export opset 14 với dynamic axes (batch + seq_len), INT8 per-channel dynamic quantization.
- **Kết quả benchmark (seq_len=512, batch=1, RTX 3090 vs CPU):**

  | variant | p50 latency | model size |
  |---|---|---|
  | PyTorch CUDA | 34.7ms | 478 MB |
  | ONNX FP32 CPU | 623ms | 478 MB |
  | **ONNX INT8 CPU** | **491ms** | **121 MB** |

  - INT8 vs FP32 CPU: **1.29x speedup, 3.9x smaller** — F1 drop chỉ 0.003 (well within 2% threshold)
  - INT8 vs CUDA: 0.07x — GPU vẫn tốt hơn 14x (expected)

- **Tại sao speedup chỉ 1.29x (không phải 4x)?** Phân tích ONNX graph (2011 nodes): INT8 dynamic quant chỉ quantize MatMul/Gemm nodes (73/97 node MatMul-family). Phần còn lại — LayerNorm (ReduceMean+Sub+Div), attention softmax (Mul+Add), vision backbone (ViT patch ops, Gather) — vẫn FP32. LayoutLMv3 là multimodal (text + layout + image) → tỉ lệ non-MatMul ops cao hơn BERT thuần → speedup thấp hơn. Contrast: BERT-family INT8 thường 2–4x vì FFN+attention dominate và đều là MatMul.
- **Để tăng speedup:** (1) static INT8 (có calibration dataset, quantize cả activation) hoặc (2) FP16 GPU (~2x speedup, zero accuracy drop) — natural next steps.
- **Giá trị thật của bước này:** 3.9x size reduction (478→121MB) là hoàn toàn hiện thực; model 121MB load nhanh, memory-constrained deployment viable; ONNX model drop-in vào Triton ONNX backend (cùng infra với RapidOCR) — không cần thêm serving code.
- **Bài học:** luôn profile ONNX graph (đếm op types) trước khi claim INT8 speedup. Speedup phụ thuộc vào tỉ lệ ops có thể quantize, không phải tổng param count. Honest benchmark với phân tích op breakdown thuyết phục hơn số speedup đẹp không giải thích được.

### ADR-12: CJK hallucination filter + conditional deskew cho PP-OCRv4
- **Bối cảnh:** PP-OCRv4 (PaddleOCR backbone) được train chủ yếu trên Chinese text corpus. Khi nhận ảnh bị rotate hoặc low-resolution Latin/Malay text, nó "hallucinate" ra chữ Hán — ví dụ: `merchant_name = "我物出门，#不处或更"` thay vì `"book ta .k (taman daya) sdn bhd"`. Bug phát hiện khi demo live.
- **Chẩn đoán:** quality checker detect đúng `is_rotated: true` nhưng OCR vẫn nhận ảnh gốc. PP-OCRv4 trên ảnh distort → hallucinate CJK.
- **Quyết định — 2 lớp:**
  1. **CJK hallucination filter** (`_filter_cjk_hallucination()`): sau KIE, nếu corpus OCR không phải Chinese doc (CJK ratio < 30%) nhưng field value > 30% CJK → null out, confidence=0. Robust với mọi angle estimate error.
  2. **Conditional deskew** (`_deskew(img, angle)` chỉ khi `abs(skew_angle) < 45°`): correct genuine small-angle skew. Near-90° bị loại vì `cv2.minAreaRect` unreliable ở vùng đó — xem ADR-13.
- **Kết quả:** CJK hallucination = 0 trên 30 ảnh test. Deskew chỉ trigger khi góc 5–44° (genuine tilt).
- **Bài học:** PP-OCRv4 excellent trên Chinese text, degraded trên Latin khi geometric distortion. Defense-in-depth quan trọng: filter output (CJK filter) độc lập với fix input (deskew) — cả hai cần thiết vì angle estimation không perfect.

### ADR-13: Go-live audit — deskew regression, sanity checks, honest error floor
- **Bối cảnh:** Go-live audit trên n=30 SROIE test images reveal 3 lớp lỗi theo severity.
- **Lớp 1 — Deskew regression (CRITICAL, tôi introduce):** Ban đầu apply `_deskew()` unconditionally với `skew_angle` từ `cv2.minAreaRect`. Hóa ra 21/30 SROIE images báo angle ~-89° — không phải vì ảnh bị rotate, mà vì minAreaRect của toàn bộ foreground pixels trả về bounding-box angle (gần như luôn là 0° hoặc -90° tuỳ image fill pattern). Kết quả: `_deskew()` xoay ảnh sai 89° → OCR đọc sai → total=9744 thay vì 112, total=519537 thay vì 2.5, `needs_review=False` → **silent wrong trên 9/30 (critical)**. Fix: chỉ deskew khi `abs(angle) < 45`.
- **Lớp 2 — Sanity checks (ĐÃ FIX):** `route_decision()` chỉ check None/low-conf, không check plausibility. KIE extract phone number làm total (9744, 519537) hoặc transposed date (year=1815, month=24) mà không trigger review. Fix: `_sanity_check()` trong pipeline — year out of [2000,2035], month>12, day>31, total≤0, total>50000 → `needs_review=True`.
- **Lớp 3 — Plausible wrong (residual, chấp nhận được):** KIE chọn subtotal/tax line thay vì grand total (31.03 vs 32.7, 16.98 vs 18.0, 8.68 vs 153.35). Không bắt được bằng range check vì đều là plausible amounts. Root fix cần sequence-labeling với layout context (LayoutLMv3) — nằm trong production roadmap.
- **Kết quả audit đầy đủ (n=30):**

  | metric | trước fix | deskew broken | **sau all fixes** |
  |---|---|---|---|
  | date exact | ~75% | 63% | **80%** |
  | total exact | ~37% | 13% | **37%** |
  | CJK hallucination | có | có | **0** |
  | SILENT_WRONG | ~5 | **9** | **5** |

- **Bài học quan trọng nhất:** *Không apply bất kỳ geometric transformation nào mà không verify trên sample trước.* `minAreaRect` ≠ text-skew estimator. Silent correctness bug (wrong data, high confidence, no review flag) nguy hiểm hơn crash nhiều lần — crash thì rõ, silent wrong thì go-live rồi mới phát hiện. Go-live audit với n≥30 ảnh và check cả false-negative (silent wrong) là bắt buộc trước mọi pipeline change.

### ADR-14: Go-live audit kết quả đầy đủ (n=80, 5-phase)
- **Bối cảnh:** Audit nghiêm khắc chuẩn bị go-live — full 80-image SROIE, error taxonomy, ECE, robustness, edge cases.
- **KPIs (n=80):** date 78.8% · total 48.8% · merchant 5% (known) · needs_review_rate 72.5% · p50=1.18s · p95=1.79s

#### Phân loại lỗi theo severity

**CRITICAL — đã FIX:**
| issue | count | fix |
|---|---|---|
| Exception on corrupted/empty input | crash | Graceful error: route=error, needs_review=True |
| Oversized image latency (4000×3000 → 4.4s) | unbounded | Cap to 3000px → 1.8s |

**HIGH — đã FIX trước đó (ADR-12/13):**
| issue | count | fix |
|---|---|---|
| CJK hallucination | 0 ✅ | `_filter_cjk_hallucination()` |
| date impossible (year<2000, month>12) | 18 → all caught | `_sanity_check()` |
| total barcode >50k | 4 → all caught | `_sanity_check()` |

**MEDIUM — tồn tại, chấp nhận được:**
| issue | count | root cause | fix roadmap |
|---|---|---|---|
| SILENT_WRONG (date+total wrong, nr=False) | **13/80 (16.3%)** | KIE picks subtotals with high confidence | LayoutLMv3 hybrid routing cho total |
| total ECE=0.514 (overconfident) | — | KIE confidence ≠ accuracy for total | Calibrated classifier + more training data |
| true false_review (correct fields, nr=True) | 12/80 (15%) | total conf barely < 0.75 on rotated imgs | Lower `DOCAI_MIN_CONF` env var |
| date_null (OCR can't find date) | 13/80 (16.3%) | Small low-quality receipts | Better upscaling + LayoutLMv3 |

#### SILENT_WRONG deep-dive (13 cases)
Breakdown theo loại:
- **"Plausible close miss"** (subtotal/tax picked): 8 cases — total 3-30% below gold, cannot detect by range
- **"High-value confusion"** (invoice# or phone# as total): 3 cases (9744, 2687, 313 vs much smaller gold)
- **"Both fields wrong"**: 2 cases (full-receipt OCR confusion, very low-quality image)

**Không thể bắt bằng sanity check đơn thuần** vì values plausible. Production fix: LayoutLMv3 cho total_amount (sequence labeling + layout context nhận ra TOTAL keyword tốt hơn regex candidate gen).

#### ECE phân tích
- **date ECE=0.402** — underconfident (model_conf=0.76 < accuracy=0.79). OK.
- **total ECE=0.514** — severely overconfident (model_conf=0.77 >> accuracy=0.49). Confidence signal cho total không đáng tin cậy để làm review gate. Cần recalibration hoặc threshold thấp hơn.

#### Robustness (n=20, severity=0.5)
| degradation | date% | total% | nr% | silent_wrong |
|---|---|---|---|---|
| blur | 75% | 45% | 85% | 1/20 |
| low_res | 85% | 45% | 75% | 4/20 |
| rotate | 85% | 45% | 70% | **4/20** |

Rotate/low_res còn 4/20=20% silent wrong — consistent với robustness curve (rotate: CER 1.18, nr thấp). CJK=0 trên tất cả degradation.

#### Edge cases
| case | result |
|---|---|
| blank/black/noise | OK — nr=True, no crash |
| corrupted/empty | **route=error, nr=True** (fixed) |
| oversized 4000×3000 | **1.8s** (fixed, was 4.4s) |
| same image ×5 | deterministic PASS |

- **Bài học:** Go-live audit phải có tất cả 5 loại test: (1) full n test, (2) error taxonomy, (3) calibration, (4) degradation, (5) edge cases. Thiếu bất kỳ loại nào đều có thể bỏ sót issue quan trọng. ECE là metric quan trọng không kém F1 — hệ thống fintech cần calibrated confidence để human review gate hoạt động đúng.

### ADR-15: Phase 1 Go-live closure — Router Recall 83.7% → 92.5%

- **Bối cảnh:** Phase 1 criteria yêu cầu Router Recall ≥ 95% (khi model sai, phải flag needs_review). Baseline 83.7% (13/80 SILENT_WRONG). Reviewer đúng: *sanity checks deterministic > confidence tuning* vì total ECE=0.514 (confidence không calibrated).

#### Các thay đổi + tác động đo được

**[1] `_cross_validate_total()` — deterministic sanity (lớn nhất)**
So sánh extracted total với max số tiền visible trong receipt. Grand total của một receipt phải là số tiền lớn nhất. Nếu OCR thấy số tiền lớn hơn 2.1× con số KIE trích xuất → KIE chọn subtotal/dòng riêng, không phải grand total.
- Threshold 2.1× (không phải 2.0×): để tránh false positive trên 028.jpg (gold=2.5, max=5.0, ratio=2.0×)
- Threshold 2.1× cũng bắt được 048.jpg (extracted=2687, max_visible=5736, ratio=2.13×)

**[2] `implausibly_small_total < 0.5`**
Bắt 049.jpg: total=0.45 (decimal confusion giữa item đơn và total).

**[3] Field-specific confidence thresholds**
`total_amount` threshold: 0.75 → 0.80 vì ECE=0.514 (overconfident). Bắt thêm một số case có conf 0.75–0.80 mà model sai.
Config per-field: `FIELD_CONFIDENCE_THRESHOLDS` dict trong `config.py`, override bằng env var (`DOCAI_CONF_TOTAL`, `DOCAI_CONF_DATE`, `DOCAI_CONF_MERCHANT`).

#### Kết quả sau tất cả thay đổi

| metric | trước | sau |
|---|---|---|
| SILENT_WRONG | 13/80 (16.3%) | 3/80 (3.8%) |
| false_review | 12/80 (15%) | 25/80 (31.2%) |
| Router Recall | 83.7% | **92.5%** |
| total exact | 39/80 (48.8%) | 40/80 (50%) |

#### 3 SILENT_WRONG còn lại — đây là Phase 2 work
| file | extracted | gold | ratio max/ext |
|---|---|---|---|
| 025.jpg | 16.98 | 18.0 | 1.06× |
| 031.jpg | 70.75 | 75.0 | 1.41× |
| 075.jpg | 150.0 | 159.0 | 1.35× |

Cả 3 đều là "close miss" (subtotal vs grand total cách nhau 6%). Ratio max/extracted quá nhỏ — không có rule deterministic nào phân biệt được mà không có layout context (vị trí keyword "TOTAL" trên receipt). **Fix cần LayoutLMv3** (Phase 2).

#### Cái bẫy confidence thresholding (xác nhận lời khuyên reviewer)

Thử trước: chỉ hạ global `MIN_FIELD_CONFIDENCE` → bắt thêm được ít case nhưng false_review tăng vọt mà không có định hướng. Sau: tập trung sanity checks deterministic → bắt được **10/13 cases** với false_review tăng có kiểm soát.

**Bài học:** Với field có ECE cao (miscalibrated confidence), sanity rules deterministic mạnh hơn hàng bậc so với confidence thresholding. Confidence thresholding là lưới an toàn cuối cùng, không phải công cụ đầu tiên.

### ADR-16: Concurrent latency — CPU bottleneck là infrastructure problem, không phải code

- **Phát hiện:** Sequential p95=1.79s ✓ (Phase 1 target ≤3s). Under c=5 concurrent: p95=7.1s ✗.
- **Root cause:** PP-OCRv4/ONNX Runtime chiếm 1.5–2s CPU per request. Khi 5 request đến đồng thời, 4 phải xếp hàng chờ CPU. p95 ≈ 4×1.5s = 6s (lý thuyết), thực tế 7.1s (scheduling overhead).
- **Thử `run_in_executor` (thread pool):** p95 tăng lên 18.6s — tệ hơn. ONNX Runtime spawn internal threads (intra-op parallelism); 5 ONNX sessions × N intra-op threads = CPU oversubscription nghiêm trọng.
- **Fix đúng (infrastructure-level):**
  - CPU: `uvicorn --workers N` (separate processes, bypass GIL, ONNX session độc lập). N=4 → p95 ≈ sequential_p95 × ceil(c/N).
  - GPU: batch inference → 1 GPU handle N requests song song. p95 → ~0.2s × batch_latency.
  - Kubernetes HPA: auto-scale replica dựa trên request queue depth.
- **Bài học:** CPU-bound ML inference không benefit từ Python threading (GIL + ONNX internal threads = contention). Concurrency SLA cho ML service = infrastructure problem (GPU/multi-process), không giải được bằng `asyncio` tricks. Cần document rõ trong ops runbook.

### ADR-17: Performance pack — stage profiler, process-pool OCR, thread sweep, latency CI gate

- **Bối cảnh:** ADR-16 kết luận đúng (multi-process, không thread) nhưng *chưa đo*. Cần biến nhận định thành **đo lường tái lập được**: time đi đâu, tune thế nào, regression bắt bằng gì.
- **Đã làm (`docai/profiling.py`, `docai/serving/ocr_pool.py`, `scripts/profile_pipeline.py|bench_threads.py|latency_gate.py`):**
  - **Stage profiler** (contextvar + Prometheus `stage_latency`, sub-second buckets) gắn `timings` vào mỗi `DocumentResult`.
  - **ProcessPoolOCR** thật (thay vòng lặp single-thread cũ trong batcher), worker nhận encoded bytes, trả `serialize_ms / worker_ms / pool_wait_ms`.
  - **Sweep 2 biến** workers×intra_threads×concurrency (không chỉ concurrency) → in `effective_load = W*T`.
  - **Latency CI gate** dạng *smoke* (4 ảnh synthetic, tolerance +40%) — không dùng làm SLA tuyệt đối vì runner nhiễu.
- **Số đo (synthetic n=30, ảnh nhỏ, LayoutLMv3/VLM off — KHÔNG phải real SROIE ~2s):**
  - **OCR = 96% latency**: warm total p50 **535ms**, trong đó OCR p50 **514ms**; kie 0.4ms, quality 7.2ms, decode 3.5ms, classify 0.1ms. → mọi tối ưu phải nhắm OCR, không phải KIE.
  - **Cold start 1485ms** (OCR engine load 1460ms) → bắt buộc `warmup()` mọi worker khi deploy, nếu không request đầu lag ~3×.
  - **Sweep (48 cores):** throughput plateau ~70–140 docs/min; **tăng workers KHÔNG giúp** — `W=1,T=2` ≈ best (140), `W=4` tệ hơn (44–72). `pool_wait_ms` tăng theo (batch − workers) = queueing, không phải bug.
- **Phát hiện trung thực (đáng giá hơn 1 con số đẹp):** với ảnh nhỏ + 1 ảnh/request, chi phí *cố định* mỗi lần gọi OCR (≈0.5s) áp đảo; thêm process chỉ tăng tranh chấp bộ nhớ/cache chứ không tăng throughput. Process-pool có giá trị khi **traffic đồng thời thật** vượt số worker — nó cho đường multi-core đúng đắn (W=1–2, T=1–2), không phải khi tăng W mù quáng. Oversubscription (W×T lớn) làm tệ đi đúng như ADR-16 dự đoán — giờ có số liệu xác nhận.
- **Bài học:** "profile trước, tune sau". Nếu chỉ sweep concurrency mà không sweep thread config sẽ kết luận sai về scaling. Cold/warm phải tách. CI latency gate phải là smoke, report chính thức là local.

### ADR-18: WP-3 OCR recognizer fine-tune (MC-OCR) — torch CRNN thay vì Paddle, kết quả + caveat

- **Bối cảnh:** fine-tune recognizer cho receipt tiếng Việt (MC-OCR 2021), budget ≤1h H100 / ≤5GB VRAM, dùng 1 env `main` (base) duy nhất.
- **Quyết định lệch plan (có chủ đích):** plan đề xuất PP-OCR/PaddleOCR. Nhưng base là py3.13 + numpy 2.x; cài `paddlepaddle` rủi ro **vỡ stack sản phẩm** (ADR-1 đã tránh Paddle). → dùng **PyTorch CRNN+CTC** (torch+CUDA cài vào base) → export **ONNX** (done-criteria #5 chấp nhận ONNX). Inference runtime chỉ cần onnxruntime (chạy CPU/RTX1650).
- **Kết quả (val n=1300, crop-level):** CER **0.3197 → 0.0853 (-73.3% tương đối)**, exact-line 0.149 → 0.599, và **nhanh hơn** (27.1ms → 9.4ms/crop). Train: 227s, peak VRAM 1316MB — thừa budget.
- **Caveat trung thực (quan trọng nhất):** gain lớn một phần vì baseline RapidOCR dùng **dict tiếng Trung**, *không thể* sinh dấu tiếng Việt → bị phạt nặng. Đây KHÔNG phải "CRNN của ta là SOTA" mà là "recognizer đúng ngôn ngữ thắng recognizer mismatch". Production đúng = **route theo ngôn ngữ**, không swap global.
- **Tích hợp:** adapter optional `DOCAI_OCR_RECOGNIZER=rapidocr_default|ppocr_vi_mcocr_ft`; detector giữ nguyên RapidOCR, chỉ thay recognizer; token schema không đổi → KIE/router không cần sửa. Adapter tự load dict tiếng Việt + CTC decode (rapidocr-onnxruntime không cho override `rec_keys_path`).
- **Không làm (honest):** downstream anti-regression trên SROIE (SROIE chưa materialize ở checkout này + recognizer tiếng Việt sẽ regress trên English — đúng như dự đoán); per-field CER breakdown; `mcocr_val_sample_df.csv` là **stub** (`anno_texts="abc abc abc"`), KHÔNG phải downstream gold.
- **Bài học:** với hard env constraint, chọn tooling theo "không vỡ env hiện có" quan trọng ngang chọn model. Và một con số gain đẹp (-73%) phải kèm giải thích nguyên nhân, nếu không sẽ overclaim.

### ADR-19: WP-3 full-image vs crop-level — bottleneck dịch sang detector, không phải recognizer

- **Bối cảnh:** -73% CER là **crop-level** (line đã cắt sẵn). Production chạy full image → detector → crop → recognizer → line grouping → KIE. Phải kiểm tra full-image mới biết gain có "thật" downstream không.
- **Per-field crop-level (leakage-free, val held-out):** SELLER -74.8%, ADDRESS -82.4%, TIMESTAMP -67.3%, **TOTAL_COST -64.6%** (0.27→0.095), diacritics -78.7%. → money field (downstream `total`) cải thiện mạnh ở crop-level.
- **Full-image (n=80, det+rec+matching):** macro CER 0.337→0.265 (**~21%**, nhỏ hơn nhiều so với crop -73%); TIMESTAMP ~0%; TOTAL_COST -28%; ADDRESS -33%. Latency p50: default 2407ms → ft 1815ms (ft nhanh hơn). needs_review: 0.66 → **0.80** (KIE/router tune cho SROIE không hưởng lợi từ text tiếng Việt).
- **Kết luận (đúng như giả thuyết):** đằng sau detector thật, **bottleneck chuyển sang detector/crop/line-grouping**, không phải recognizer. Failure examples cho thấy detector gộp nhầm vùng + sai reading order (`Co. optTo. H B. Mi`); recognizer dù tốt cũng không sửa được input bị cắt sai. → lever tiếp theo để biến OCR gain thành downstream gain = **detector/line-grouping**, không phải fine-tune recognizer thêm.
- **Caveats trung thực:** (1) full-image gold chỉ có ở train images → recognizer in-domain (số liệu lạc quan, đọc delta + pattern); (2) polygon↔token matching xấp xỉ khi box detector lệch gold → cũng làm delta full-image nhỏ lại.
- **Bài học:** crop-level metric đẹp KHÔNG tự động thành end-to-end gain. Luôn đo full-image trước khi claim. "Đo đúng tầng bottleneck" quan trọng hơn "tối ưu tầng mình thích".

### ADR-20: Detector error analysis — đo trước khi sửa; Task B null result trung thực

- **Bối cảnh:** ADR-19 nói bottleneck là detector/grouping nhưng *chưa định lượng*. Dùng `anno_polygons` làm gold → `scripts/eval_detector_mcocr.py` (recall, overmerge/oversplit, reading-order, per-field failure taxonomy).
- **Kết quả định lượng (n=80):** det_field_recall **0.978** (detector KHÔNG miss field), overmerge 0.07, oversplit 0.013. Taxonomy: OK 368, REC_ERROR 43, OVERMERGE 32, DETECT_MISS 10, OVERSPLIT 6.
  - **TIMESTAMP**: lỗi chủ yếu **REC_ERROR (26)** dù coverage 1.0 → crop-distribution gap (crop của detector ≠ crop train), KHÔNG phải detection. Grouping/deskew không cứu.
  - **ADDRESS**: lỗi chủ yếu **OVERMERGE (24)** → recognizer giỏi trên crop sạch (crop -82%) nhưng full-image bị merge nhiều dòng địa chỉ → cap còn -33%.
  - **TOTAL_COST**: phần lớn OK (137), recall 0.99 → money field truyền tốt xuống full-image (-28%).
- **Quyết định data-driven:** KHÔNG đổi detector (recall 0.978), KHÔNG ưu tiên deskew (skew không phải nguyên nhân chính ở set này). Hai lever thật: (1) split ADDRESS over-merge dọc; (2) thu hẹp crop-gap của TIMESTAMP.
- **Task B null result (trung thực):** anchor-split *ngang* (`Ngày…Tổng tiền`) đúng cơ chế + unit pass, nhưng overmerge_rate 0.07→**0.07 (không đổi)** vì over-merge ở đây là **dọc** (ADDRESS đa dòng). → giữ Task B (đúng cho ca ngang, flag-gated) nhưng fix thật cho ADDRESS = **horizontal-projection row split trong box** (kỹ thuật khác, chưa làm). Báo cáo null thay vì giả vờ có gain.
- **Task D (routing) + Task C (geometry flag):** đã ship, flag-gated. D route theo tỉ lệ dấu tiếng Việt (VN→CRNN, EN/SROIE→default) — đúng cách xử lý needs_review↑ (0.66→0.80) và tránh regress SROIE. C flag skew ≥8° → needs_review (fail loud cho rotate/perspective).
- **Bài học:** đo định lượng nói cho ta biết **KHÔNG nên sửa cái gì** (detector, deskew) cũng nhiều như nên sửa cái gì. Một fix "đúng cơ chế" vẫn có thể null nếu sai trục lỗi (ngang vs dọc) — phải đo before/after, không assume.
