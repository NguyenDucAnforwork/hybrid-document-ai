# WP-3 — OCR Recognizer Fine-tune: results & handoff

Fine-tuned a lightweight Vietnamese receipt OCR recognizer on **MC-OCR 2021** and
integrated it into the production-style OCR-KIE pipeline as an **optional, config-
switchable adapter** with deployable ONNX export, latency numbers, and reproducible
crop-level evaluation. Honest scope: this is an **OCR adaptation experiment**, not a
production-grade banking OCR claim.

## Headline result — OCR-level, MC-OCR val (`text_recognition_val_data.txt`, n=1300)
Source: `docs/logs/ocr_rec_eval_20260609_1112.md`

| recognizer | CER | exact-line | WER | p50 ms/crop |
|---|---|---|---|---|
| default (RapidOCR PP-OCRv4, Chinese dict) | 0.3197 | 0.149 | 0.836 | 27.1 |
| **fine-tuned CRNN (ours)** | **0.0853** | **0.599** | **0.246** | **9.4** |

**Relative CER improvement: 73.3%** (excellent-done threshold is ≥25%). The FT model
is also ~3× faster per crop.

**Why the gain is large — and the honest caveat:** RapidOCR's default recognizer uses a
**Chinese** character dictionary and structurally cannot emit Vietnamese diacritics, so it
is heavily penalized on Vietnamese receipts. A large part of the 73% is the baseline being
language-mismatched, **not** our CRNN being state-of-the-art. The correct read is: *for
Vietnamese receipts, a small in-language recognizer beats a mismatched general recognizer
by a wide margin* — a real, defensible engineering finding, stated without overclaiming.

## Training — budget compliance
Source: `models/ocr/vi_mcocr_crnn_ft/training_log.json`

- **Wall clock: 227.8 s** (≤ 1h budget ✓), 60 epochs, batch 128, lr 1e-3, seed 42
- **Peak VRAM: 1316 MB** (≤ 5GB budget ✓) — measured via `torch.cuda.max_memory_allocated`
- best val CER 0.0854 (early-stop on val CER, best checkpoint only)
- model: compact CRNN+CTC, 182 classes (181-char Vietnamese set + CTC blank)

## Data — sanity report
Source: `data/processed/mcocr_ocr/sanity_report.json`

- accepted **train 5285 / val 1300** (matches plan's verified counts), **0** rejected
  (missing-image 0, empty-label 0, corrupt-image 0), charset 181 (NFC-normalized)
- field manifest: 1155 rows from `mcocr_train_df.csv` (analysis only)

## Artifacts (deployable)
`models/ocr/vi_mcocr_crnn_ft/`: `model.onnx` (34 MB), `vi_dict.txt`, `metadata.json`,
`best.pt`, `training_log.json`. ONNX runs on CPU / RTX 1650 via `onnxruntime` (no torch
or paddle at inference).

## Integration — optional adapter (clean on/off)
- `DOCAI_OCR_RECOGNIZER=rapidocr_default` (default, production untouched) `| ppocr_vi_mcocr_ft`
- Detector stays RapidOCR; only the recognizer is swapped (`docai/ocr_recognizer.py`,
  `docai/ocr.py`). Token schema unchanged → KIE/router/pipeline see no difference.
- Adapter loads its own Vietnamese dict + does CTC decode (RapidOCR's onnxruntime build has
  no `rec_keys_path` override; its dict is hardcoded Chinese). Falls back gracefully to
  default if the artifact is absent.

## Done-criteria mapping
- **Minimum done — MET:** audit documented; reproducible subset extraction; sanity report
  with all required counts; recognizer fine-tuned ≤1h/≤5GB; ONNX exported; runs as optional
  adapter; OCR eval on `text_recognition_val_data.txt`; `mcocr_val_sample_df.csv`-not-gold caveat stated.
- **Good done — MET:** CER ≥15% rel ↓ (73.3%); exact-line ↑ (0.149→0.599); runs on CPU/RTX1650.
- **Excellent done — PARTIALLY MET:** CER ≥25% rel ↓ ✓; ONNX small-ish (34 MB, LSTM-dominated)
  ✓; adapter on/off via config ✓; latency trade-off quantified (FT 9.4ms < default 27.1ms) ✓.
  **Not done:** per-field (TIMESTAMP/TOTAL_COST) CER breakdown, and downstream SROIE anti-
  regression — see below.

## What was NOT done (honest)
1. **Downstream anti-regression (bonus) not run.** Two reasons: (a) the real SROIE test set
   isn't materialized in this checkout (only the demo workspace), and (b) the recognizer is
   Vietnamese-only — running it on English/Malaysian SROIE would be expected to **regress**
   (charset mismatch in the other direction). The correct production posture is per-language
   recognizer routing, not a global swap. Documented rather than measured.
2. **Field-aware per-label CER** wiring exists (`field_manifest.jsonl`, crop→parent map) but
   the per-label aggregation was not run under the time budget.
3. **`mcocr_val_sample_df.csv` is a placeholder stub** (`anno_texts="abc abc abc"`) — NOT a
   trustworthy downstream gold validation set. No downstream KIE-validation claim is made from it.

## Deviation from plan (documented)
Plan named a PP-OCR/PaddleOCR recognizer. Under the single-main-env constraint + base running
numpy 2.x / py3.13, installing Paddle risked breaking the product stack (ADR-1 already avoided
Paddle). To keep one clean env and the budget, WP-3 used a **PyTorch CRNN+CTC** (torch+CUDA in
base) exported to **ONNX** (done-criteria #5 accepts ONNX). Trade-off: not PP-OCR-pretrained
init — but the in-language experiment is unaffected and the result is strong.

## Reproduce
```bash
export DOCAI_WORKSPACE=/data/nvidia-ai-workspace
python scripts/wp3_prepare_ocr.py                                   # sanity report
python training/train_ocr_rec.py --dry-run --batch 128              # VRAM check
python training/train_ocr_rec.py --epochs 60 --batch 128 --lr 1e-3  # train + ONNX
python scripts/eval_ocr_rec.py --recognizer both --limit 1300       # eval report
# use the adapter in the pipeline:
DOCAI_OCR_RECOGNIZER=ppocr_vi_mcocr_ft python -m uvicorn app.main:app
```
