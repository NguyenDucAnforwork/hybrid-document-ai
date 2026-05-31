# Runbook — Operations & Disaster Recovery

Vận hành hệ thống Document AI trong production (ngân hàng). Map "MLOps - More and beyond".

## Model rollback (sự cố model mới kém)
1. `registry.transition("kie", "<good_version>", "production")` → prod cũ tự `archived`, serving load lại version production.
2. Output JSON nhúng `model_versions` → xác định version gây lỗi từ log.
3. Eval-gate trong CI/pipeline lẽ ra đã chặn; nếu lọt → hạ threshold drift alert.

## Disaster recovery: on-prem → cloud failover
- **Trigger:** inference on-prem (GPU ngân hàng) chết / quá tải.
- **Plan:** image Docker giống hệt (`deploy/Dockerfile`) đẩy lên cloud (KServe/Cloud Run); DNS/ingress trỏ sang. Model kéo từ HF registry (`banhchungtuongot/hybrid-docai-kie`) → không phụ thuộc local disk.
- **VLM:** đổi `vlm.mode` `remote_gpu`→`api` (provider managed) để không cần GPU cloud ngay.
- **RTO mục tiêu:** < 30 phút (image + model artifact đã sẵn ở registry).

## Cold-start (mặt trái scale-to-zero)
- Tier OCR/KIE (rẻ): giữ `minReplicas: 1` để tránh cold-start request đầu.
- Tier VLM (đắt): cho `scale-to-zero`; chấp nhận cold-start vì chỉ hard-case.
- Warm-up OCR ở `startup` đã có trong app.

## Khi alert kêu
| Alert | Hành động |
|---|---|
| `HighHumanReviewRate` | kiểm phân phối input (drift?); xem sample needs_review; cân nhắc retrain. |
| `VLMFallbackSpike` | OCR/KIE tụt chất lượng hoặc input đổi domain → kiểm OCR; giới hạn cost VLM. |
| `HighLatencyP95` | scale-out (HPA/KServe); kiểm dynamic batching; kiểm GPU contention. |
| `InputBlurDrift` / `ConfidenceDrop` | domain shift → trigger data collection + retrain pipeline. |

## Chaos drill (định kỳ)
`python -m mlops.chaos` — xác nhận OOD/blank/corrupt/tiny → `needs_human_review`, không crash, không phát data sai. Đưa vào CI để chống regression độ bền.

## Backup & ret- ention
- Model artifacts: HF registry (versioned). Metadata: SQLite/Postgres backup hằng ngày.
- Artifacts pipeline run (`manifest.json`) giữ để tái tạo/điều tra.
