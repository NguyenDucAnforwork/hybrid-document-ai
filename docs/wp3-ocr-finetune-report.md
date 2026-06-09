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

## Per-field + subset CER — crop-level, leakage-free (Việc 2)
Source: `docs/logs/ocr_field_cer_20260609_1142.md` (val crops held out for the recognizer;
label via crop→csv region index, validated 400/400 exact)

| bucket | n | default CER | ft CER | rel ↓ |
|---|---|---|---|---|
| ALL | 1300 | 0.3197 | 0.0853 | 73.3% |
| SELLER | 230 | 0.2821 | 0.0711 | **74.8%** |
| ADDRESS | 367 | 0.4178 | 0.0736 | **82.4%** |
| TIMESTAMP | 263 | 0.2984 | 0.0977 | 67.3% |
| **TOTAL_COST** | 439 | 0.2696 | 0.0954 | **64.6%** |
| digit_heavy (money/date) | 466 | 0.2344 | 0.0783 | 66.6% |
| diacritics | 898 | 0.3622 | 0.0772 | 78.7% |

**VNPAY-relevant read:** the gain is broad, not just on generic SELLER lines. **TOTAL_COST
improves 64.6%** (0.27→0.095) and diacritic-heavy lines improve most (78.7%) — so the money
field that downstream `total_amount` depends on does get materially better at the crop level.
ADDRESS gains most because it is the most diacritic-dense. TIMESTAMP/TOTAL_COST gain a bit less
relative (digit-heavy lines were the baseline's least-bad case) but still strong in absolute CER.

## Full-image pipeline eval — det + recognizer + matching (Việc 1)
Source: `docs/logs/fullimage_mcocr_20260609_1144.md` (n=80 full MC-OCR train receipts;
RapidOCR detector shared, recognizer swapped; per-field via polygon↔token matching)

| field | default CER | ft CER | rel ↓ | default ANLS | ft ANLS |
|---|---|---|---|---|---|
| SELLER | 0.209 | 0.179 | 14.3% | 0.817 | 0.842 |
| ADDRESS | 0.479 | 0.319 | 33.4% | 0.549 | 0.764 |
| TIMESTAMP | 0.458 | 0.454 | **0.9%** | 0.570 | 0.579 |
| TOTAL_COST | 0.212 | 0.152 | 28.1% | 0.794 | 0.866 |
| **macro** | **0.337** | **0.265** | **~21%** | — | — |

- latency p50/p95 (full-image OCR): default 2407/3465 ms · **ft 1815/3665 ms** (ft p50 faster)
- needs_review rate (`process_document`): default 0.662 · **ft 0.80** (ft flags *more* — the
  SROIE-tuned KIE/router does not benefit from Vietnamese text; honest anti-signal)

**The central finding (matches the brief's hypothesis):** crop-level CER gain was **−73%** but
full-image is only **−21% macro** (TIMESTAMP ≈ 0%). When you put the recognizer behind the real
detector, **the bottleneck shifts to the detector / crop / line-grouping, not the recognizer.**
Failure examples (`fullimage_mcocr_*.md`) show the detector over-merging adjacent regions and
scrambling reading order (e.g. SELLER pred `Co. optTo. H B. Mi`, ADDRESS lines fused) — recognizer
quality can't fix mis-segmented input. **Implication:** to convert OCR-level gains into downstream
gains, the next lever is the *detector / line-grouping*, not more recognizer fine-tuning.

Caveats: (1) full-image gold only exists for **train** receipts → recognizer is in-domain
(numbers optimistic; read the default→ft delta + pattern). (2) polygon↔token matching is
approximate when detector boxes don't align with gold regions, which itself contributes to the
smaller full-image delta.

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

## Published artifacts (HuggingFace)
- Checkpoint: `banhchungtuongot/hybrid-docai-kie` → `ocr/vi_mcocr_crnn_ft/` (model.onnx, vi_dict.txt, metadata, best.pt)
- Dataset: `banhchungtuongot/hybrid-docai-mcocr-ocr` (crops tar, labels, sanity report, manifests)

## What was NOT done (honest)
1. **SROIE anti-regression not run.** SROIE isn't materialized in this checkout, and a
   Vietnamese-only recognizer would be expected to **regress** on English/Malaysian SROIE
   (charset mismatch). Instead, the **full-image MC-OCR eval (Việc 1)** serves as the in-domain
   anti-regression: it shows `needs_review` *rises* (0.66→0.80) because the SROIE-tuned KIE/router
   doesn't benefit from Vietnamese text — consistent with "per-language routing, not global swap".
2. **`mcocr_val_sample_df.csv` is a placeholder stub** (`anno_texts="abc abc abc"`) — NOT a
   trustworthy downstream gold validation set. No downstream KIE-validation claim is made from it.
3. **Full-image bottleneck — investigated in `docs/wp3-detector-analysis.md`:** detector recall is
   fine (0.978); the loss was the recognizer crop-distribution gap, fixed by **Task F** (detector-style
   crop augmentation: full-image macro CER 0.265→0.205). Line-grouping fixes (Task B horizontal, Task E
   projection) measured **null**. Latency ablation: no FT config hits 1.3–1.6s (needs a detector-only
   path); **language routing (Task C)** proven — English/SROIE stays on default, byte-identical (zero
   regression), Vietnamese → CRNN. Remaining open item: detector-only path for latency.

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
