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

## Nếu làm lại / có thêm thời gian
- Thay Tier-1 sklearn bằng LayoutLMv3 ONNX (so F1/latency/disk).
- Chạy Triton + vLLM thật trên H100, đo dynamic-batching throughput thực.
- Thêm drift detector thống kê (PSI) + alerting; experiment tracking (MLflow).
