# docs/logs/ — Nhật ký train/eval/benchmark & run

Project inference-first nhưng CÓ 1 training pipeline (KIE classifier). Log gồm:

| File | Nội dung | Sinh bởi |
|---|---|---|
| `setup_<ngày>.log` | output pip install, `du -sh`, version deps, disk | P0 |
| `train_kie_<ngày>.log` + `models/kie/<ver>/metrics.json` | feature importance, F1 train/val, calibration, seed | `training/train_kie.py` |
| `ocr_dump_<doc>.json` | raw OCR để debug KIE | pipeline `--debug` |
| `benchmark_raw.json` | pred vs gold + latency từng doc | `run_benchmark.py` |
| `benchmark_<ngày>.md` | bảng tổng hợp (xem dưới) | `summarize_benchmark.py` |
| `run_<ngày>.log` | structured log mỗi stage (1 dòng JSON) | logging_config |

## Bảng benchmark mẫu (điền số thật)
```
                       Setting A        Setting B          Setting C
                       (rule-only)      (sklearn-KIE)      (+VLM fallback)
field-F1 (macro)          -                -                  -
all-required-correct      -                -                  -
overall CER               -                -                  -
latency p50 / p95 (ms)    - / -            - / -              - / -
fallback_rate             0%               0%                 -%
needs_review_rate         -                -                  -
docs/min                  -                -                  -
```
→ Mục tiêu chứng minh: **B > A** (classifier cải thiện so với rule), và **C > B trên hard cases** (VLM cứu case khó) đổi lại latency/cost cao hơn.

## Eval-gate
`run_benchmark.py --f1-threshold 0.70` trả exit code !=0 nếu macro field-F1 < threshold → CI fail. Ghi F1 thực + verdict vào `benchmark_<ngày>.md`.

## Quy ước
- Mỗi benchmark ghi kèm: model versions (registry), version RapidOCR/sklearn/onnxruntime, số ảnh, nguồn dataset, seed.
- `.gitignore`: ảnh dataset + model binary; chỉ commit log số liệu + metrics.json.
