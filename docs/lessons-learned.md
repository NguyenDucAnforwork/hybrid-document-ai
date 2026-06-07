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
- 3rd type **payment_order** (anchor KV) + **batch demo 5–10 ảnh** trên Space.

## Nếu làm lại / có thêm thời gian
- **Table-Transformer/LayoutLMv3 cho statement table** (thay rule parser — đẩy amount-acc lên thật).
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

### ADR-12: Deskew preprocessing + CJK hallucination filter cho PP-OCRv4
- **Bối cảnh:** PP-OCRv4 (PaddleOCR backbone) được train chủ yếu trên Chinese text corpus. Khi nhận ảnh bị rotate hoặc low-resolution Latin/Malay text, nó "hallucinate" ra chữ Hán — ví dụ: `merchant_name = "我物出门，#不处或更"` thay vì `"book ta .k (taman daya) sdn bhd"`. Bug phát hiện khi demo live: `curl /documents/extract` với `000.jpg` trả về Unicode Hán tự mặc dù ảnh là receipt Malaysia.
- **Chẩn đoán:** quality checker đã detect đúng `is_rotated: true` + `skew_angle: -88.99°` nhưng OCR vẫn nhận ảnh gốc (chưa deskew). PP-OCRv4 OCR trên ảnh lật 90° → hallucinate CJK.
- **Quyết định — 2 lớp fix:**
  1. **Deskew trước OCR** (`pipeline.py: _deskew()`): dùng `skew_angle` từ quality report, áp dụng `cv2.getRotationMatrix2D + warpAffine` trước khi call `run_ocr()`. Fix root cause — OCR nhận ảnh thẳng.
  2. **CJK hallucination filter** (`pipeline.py: _filter_cjk_hallucination()`): sau KIE, nếu corpus OCR tokens không phải Chinese doc (CJK ratio < 30%) nhưng field value có >30% CJK chars → null out field + confidence=0. Defense-in-depth cho trường hợp deskew không đủ (ảnh blur + rotated, angle estimate không chính xác).
- **Kết quả sau fix:**
  - `000.jpg`: `merchant_name` không còn CJK — OCR đọc được text thật sau deskew; `date=2018-12-25`, `total_amount=9.0` đúng.
  - `needs_human_review` thay đổi từ true→false vì required fields (date/total) giờ correct.
- **Đánh đổi:**
  - `_deskew` thêm ~5-10ms overhead cho ảnh rotate. Chấp nhận vì chỉ trigger khi `is_rotated=True` (minority path).
  - Góc estimate từ `cv2.minAreaRect` đôi khi sai 90° (như image 000.jpg: -88.99° thay vì +1°). `warpAffine` vẫn xử lý đúng vì rotation matrix tương đương.
  - CJK filter không áp dụng khi doc thật là Chinese (corpus ratio > 30%) — correct behavior.
- **Bài học:** OCR engine bias của training data ảnh hưởng trực tiếp đến production reliability. PP-OCRv4 excellent trên Chinese text, degraded trên Latin khi có geometric distortion. **Deskew là preprocessing tối thiểu cần có** trước mọi OCR engine, không chỉ PP-OCRv4. Layer defense-in-depth (filter output, không chỉ fix input) quan trọng vì angle estimation không hoàn hảo 100%.
