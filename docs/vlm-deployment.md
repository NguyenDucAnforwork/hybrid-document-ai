# Deploy VLM khi demo trên GPU nhỏ (vd RTX 1650 / 4GB)

**Vấn đề:** Qwen2.5-VL-3B (bf16) cần ~7GB VRAM → RTX 1650 (4GB) chạy trực tiếp không nổi.
**Chìa khoá:** kiến trúc tách OCR/KIE (rẻ, local) khỏi VLM (đắt, ca khó). VLM gọi qua
endpoint **OpenAI-compatible** → đổi nơi chạy chỉ bằng config (`docai/vlm.py` mode `api`),
KHÔNG sửa code. Vì VLM chỉ chạy trên ca router-flag (vài %), latency/cost của nó không
ảnh hưởng phần lớn lưu lượng.

## Ma trận lựa chọn

| Hướng | VRAM local | Chi phí | Độ trễ | Khi nào dùng |
|---|---|---|---|---|
| **A. Modal serverless GPU** ⭐ | 0 (VLM ở cloud) | pay-per-second, **scale-to-zero** | cold-start ~30–60s, sau đó nhanh | Demo thực dụng nhất: bật khi cần, không giữ GPU |
| **B. Managed VLM API** | 0 | pay-per-token | thấp | Nhanh nhất để có demo: DashScope (Qwen), Gemini, OpenAI Vision, OpenRouter, Together/Fireworks |
| **C. Local VLM lượng tử hoá** | ~3GB | miễn phí | trung bình (RTX 1650 chậm) | Muốn chạy hẳn trên máy, offline |
| **D. Không VLM** (`disabled`) | 0 | 0 | — | Chỉ OCR+KIE + human review; vẫn là sản phẩm hợp lệ |

Tất cả A/B/C đều cắm vào `mode=api` (cùng `VLM_API_BASE`), trừ D.

## A. Modal (khuyến nghị) — `deploy/modal_vlm.py` ✅ ĐÃ DEPLOY & VERIFY
Endpoint đang chạy: `https://nguyenducanforwork--docai-vlm-vlm-serve.modal.run` (Qwen2.5-VL-3B trên L4,
serve `/v1/chat/completions` qua transformers + FastAPI, scale-to-zero). HF Space đang trỏ vào đây.
```bash
pip install modal && modal token set --token-id <ak-...> --token-secret <as-...>   # 1 lần
modal deploy deploy/modal_vlm.py                # -> URL https://<you>--docai-vlm-vlm-serve.modal.run
export DOCAI_VLM_MODE=api
export VLM_API_BASE="https://<you>--docai-vlm-vlm-serve.modal.run/v1"
export VLM_API_KEY=dummy
export VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
uvicorn app.main:app --port 8000                # local OCR+KIE; hard case -> Modal
```
- **scale-to-zero:** chỉ tốn tiền giây GPU thực sự infer (ca khó). Idle 5 phút → tắt.
- **cold-start ~60–70s** cho ca đầu sau idle (spin container + load model từ Volume cache); `min_containers=1` nếu cần luôn ấm.
- Model cache trong `modal.Volume` (không tải lại 7GB mỗi cold start). GPU L4 cho 3B; A10G cho 7B.
- Đã verify end-to-end: ảnh blur → HF Space (CPU OCR+KIE) → `route=vlm_fallback` → Modal trả merchant đúng.

## B. Managed API (nhanh nhất, không cần GPU)
Trỏ `VLM_API_BASE` tới endpoint OpenAI-compatible của nhà cung cấp:
- **DashScope (Qwen-VL official):** `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`, `VLM_MODEL=qwen-vl-max`
- **OpenRouter:** `https://openrouter.ai/api/v1`, `VLM_MODEL=qwen/qwen2.5-vl-7b-instruct`
- **Gemini / OpenAI Vision:** dùng SDK riêng (cần một adapter nhỏ; mặc định code đang theo chuẩn OpenAI chat+image_url).
Set `VLM_API_KEY=<key thật>`. Đây là cách dựng demo nhanh nhất.

## C. Local lượng tử hoá trên RTX 1650 (~3GB)
**Ollama (dễ nhất):**
```bash
ollama pull qwen2.5vl:3b            # GGUF ~3GB, hoặc 'minicpm-v', 'llava'
export DOCAI_VLM_MODE=api
export VLM_API_BASE="http://localhost:11434/v1"
export VLM_MODEL="qwen2.5vl:3b"
```
Ollama expose OpenAI-compatible + hỗ trợ ảnh → cắm thẳng `mode=api`, chạy hẳn trên RTX 1650.
**Lựa chọn nhẹ hơn nữa nếu vẫn thiếu VRAM:** SmolVLM-500M, Donut (OCR-free ~0.7GB), moondream2 —
hoặc Qwen2.5-VL-3B **4-bit** (bitsandbytes/AWQ) qua `mode=local` + `DOCAI_VLM_DEVICE=cuda`.

## D. Không VLM (`DOCAI_VLM_MODE=disabled`)
Pipeline vẫn chạy: OCR+KIE + confidence router → ca khó gắn `needs_human_review`. Đây là default
an toàn; date/total vẫn tốt (ANLS 0.84/0.60 trên SROIE). VLM là tăng cường, không bắt buộc.

## Tóm tắt khuyến nghị cho RTX 1650
1. **Demo có VLM, ít công:** Modal (`deploy/modal_vlm.py`) hoặc managed API (DashScope/OpenRouter).
2. **Chạy hẳn offline trên máy:** Ollama + `qwen2.5vl:3b` (mode=api → localhost:11434).
3. **Không cần VLM:** `mode=disabled` — vẫn demo được pipeline + human-review.
