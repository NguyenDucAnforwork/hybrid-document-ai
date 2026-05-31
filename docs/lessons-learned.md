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

## Nếu làm lại / có thêm thời gian
- **Deskew/perspective-correction trước OCR** (điểm yếu lớn nhất trong robustness) + geometry-aware confidence.
- merchant_name: ràng buộc header-block + NER (LayoutLMv3 ONNX) thay token-line heuristic.
- Chạy Triton + vLLM + KServe + KFP thật trên H100/k8s, đo throughput batching + autoscale.
- Drift detector thống kê (PSI) + Grafana + OpenTelemetry tracing + experiment tracking (MLflow).
