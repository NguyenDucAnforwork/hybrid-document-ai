# WP-3 — OCR Recognizer Fine-tune: detailed implementation spec

Implementation-level spec (files / functions / code) for the plan in
`docs/ocr-finetune-implementation-plan.md`. Scope unchanged: **fine-tune only a
lightweight PP-OCR mobile recognizer on MC-OCR 2021, export ONNX, expose as an
optional adapter, report OCR-level metrics honestly.** ≤1h H100, ≤5GB VRAM.

## Done Criteria (from `docs/ocr-finetune-implementation-plan.md`)

**Minimum done:** (1) Kaggle package structure audited/documented; (2) reproducible subset
extraction for `text_recognition_train_data.txt`, `text_recognition_val_data.txt`,
`text_recognition_mcocr_data/...`; (3) data sanity report with accepted-train, accepted-val,
missing-image, empty-label, corrupt-image counts; (4) lightweight recognizer fine-tuned from
pretrained init within **≤1h H100** and **≤5GB peak VRAM**; (5) ≥1 deployable artifact (Paddle
inference model **or ONNX**); (6) recognizer runs as optional adapter in pipeline or isolated
smoke test; (7) OCR-level eval report on `text_recognition_val_data.txt`; (8) handoff states
`mcocr_val_sample_df.csv` is **not** trustworthy downstream gold. *A reproducible, resource-
compliant, honestly-documented negative result still counts as minimum done.*

**Good done:** OCR val CER **≥15% relative ↓**; exact-line acc improves measurably; exported model
runs on RTX 1650 or CPU; downstream receipt anti-regression shows no key-field collapse and
**SILENT_WRONG does not increase materially**; end-to-end latency increase acceptable; team can
explain why OCR gains do/don't translate downstream.

**Excellent done:** OCR val CER **≥25% relative ↓**; timestamp/amount-like lines improve clearly;
ONNX small enough for local demo; downstream benchmarks stable/improved with no added silent-
failure risk; adapter switches on/off cleanly via config; final report quantifies **both** latency
and business-value trade-offs.

> **Implementation note (deviation, documented):** the plan names a PP-OCR/PaddleOCR-style
> recognizer. Under the *single main conda env* constraint + base running numpy 2.x / py3.13,
> installing `paddlepaddle` risks breaking the product stack (ADR-1 already avoided Paddle). To
> honor one clean env and the ≤1h/≤5GB budget, WP-3 fine-tunes a **compact PyTorch CRNN+CTC
> recognizer** (torch+CUDA already in base) and exports **ONNX** (done-criteria #5 accepts ONNX).
> Trade-off: not PP-OCR-pretrained init. But the experiment is still sharp — RapidOCR's default rec
> uses a **Chinese** dict and cannot represent Vietnamese diacritics, so a Vietnamese-trained CRNN
> is expected to win on MC-OCR val CER. Reported honestly either way.

## 0. Codebase-grounded findings that shape this spec (read first)

1. **Paddle is intentionally NOT in base (ADR-1).** Base/runtime uses
   `rapidocr-onnxruntime` (ONNX only). → Training is done in a **separate, throwaway
   conda env on `/data`** (PaddleOCR + paddlepaddle-gpu). The runtime base only ever
   sees the exported **ONNX + char dict** — no paddle dependency added to the product.
2. **`rapidocr-onnxruntime` (installed) accepts `rec_model_path`/`det_model_path`
   kwargs but has NO `rec_keys_path`** — its char dictionary is hardcoded to the
   Chinese `ch_ppocr_v3_rec` module (verified: `Rec` config keys = `use_cuda,
   module_name, class_name, model_path, rec_img_shape, rec_batch_num`). A Vietnamese
   recognizer uses a different charset, so **swapping `rec_model_path` alone would
   mis-decode.** → We ship our own tiny **CTC-decode adapter** (`docai/ocr_recognizer.py`)
   that owns the Vietnamese dict, rather than relying on RapidOCR's rec post-process.
3. **The core deliverable (OCR-level CER on `text_recognition_val_data.txt`) is
   crop-level — it needs NO detector.** Val lines are already cropped. So the eval
   path is just `rec(crop) → CER`, fully decoupled from pipeline detection. Full-
   pipeline integration (det boxes → our rec) is the *optional* harder layer.
4. Reuse existing infra: `eval/metrics.py:cer`, the `docs/logs/*.md` report
   convention, `training/export_onnx.py` style, `.env` (`KAGGLE_USERNAME/KEY`, `HF_TOKEN`),
   `docai/config.py` env-switch pattern, `docai/profiling.py` for latency.

## 1. Environments

**Training env (separate, on /data — not base):**
```bash
/home/nvidia-lab/miniconda3/bin/conda create -p /data/envs/ppocr-ft python=3.10 -y
# inside it: paddlepaddle-gpu (CUDA 12), paddleocr>=2.7, paddle2onnx, pynvml, opencv, lmdb
```
Rationale: keeps base clean; PP-OCR training tooling lives only here. Documented as
swap-by-artifact (same philosophy as ADR-1/ADR-6).

**Runtime/eval env:** existing base conda (already has `onnxruntime`, `cv2`, `numpy`,
`rapidocr-onnxruntime`). Adapter + eval add **no new runtime deps**.

## 2. New / changed files

| File | Type | Purpose |
|---|---|---|
| `scripts/wp3_download_mcocr.py` | new | Kaggle download of the minimal subset only |
| `scripts/wp3_prepare_ocr.py` | new | Materialize `data/processed/mcocr_ocr/` + sanity report |
| `training/ocr/build_charset.py` | new | Build Vietnamese `vi_dict.txt` from transcriptions |
| `training/ocr/ppocrv5_vi_mcocr_rec.yml` | new | PaddleOCR rec fine-tune config |
| `training/train_ocr_rec.py` | new | Budget-guarded wrapper around PaddleOCR `tools/train.py` |
| `training/export_ocr_onnx.py` | new | Paddle inference export → ONNX + `metadata.json` |
| `docai/ocr_recognizer.py` | new | `FineTunedRecognizer` ONNX + CTC decode (Vietnamese dict) |
| `docai/ocr.py` | edit | `DOCAI_OCR_RECOGNIZER` switch; det-boxes→adapter path |
| `docai/config.py` | edit | recognizer env vars |
| `scripts/eval_ocr_rec.py` | new | Crop-level CER/exact/WER eval (default vs FT) + field-aware |
| `docs/logs/ocr_rec_eval_*.md` | artifact | eval report |
| `models/ocr/ppocr_vi_mcocr_ft/` | artifact | `inference.*`, `model.onnx`, `metadata.json` |

---

## 3. Data pipeline

### 3.1 `scripts/wp3_download_mcocr.py`
Downloads only the WP-3 subset (not the 61k-entry full ZIP).
```python
# reads KAGGLE_USERNAME/KAGGLE_KEY from .env (do not print them)
DATASET = "domixi1989/vietnamese-receipts-mc-ocr-2021"
SUBSET = [                         # --full adds train_images/val_images/kie_data
    "text_recognition_train_data.txt",
    "text_recognition_val_data.txt",
    "text_recognition_mcocr_data/text_recognition_mcocr_data/",
    "mcocr_train_df.csv",          # optional field-aware analysis
]
def main():
    _load_env_kaggle()             # set os.environ from .env, then import kaggle
    import kaggle
    api = kaggle.KaggleApi(); api.authenticate()
    out = Path(os.environ.get("DOCAI_WORKSPACE",".")) / "data/raw/mcocr"
    for f in SUBSET:               # api.dataset_download_file(..., path=out); unzip
        ...
    # NOTE: mcocr_train_df.csv may arrive as a zip-disguised .csv when fetched
    # alone (per plan §2.2) -> _maybe_unzip_csv() handles both cases.
```
Functions: `_load_env_kaggle()`, `_maybe_unzip_csv(path)`, `main()`.
Target dir: `$DOCAI_WORKSPACE/data/raw/mcocr/`.

### 3.2 `scripts/wp3_prepare_ocr.py`
Materializes the processed dataset + the required sanity counts.
```text
$DOCAI_WORKSPACE/data/processed/mcocr_ocr/
  crops/                # validated crop images (or symlinks to raw)
  train.txt val.txt     # crop_path<TAB>transcription (Kaggle split preserved)
  manifest.jsonl        # {crop, text, split, h, w, n_chars}
  rejected.jsonl        # {crop, reason}
  sanity_report.json
```
```python
CROP_DIR = RAW/"text_recognition_mcocr_data/text_recognition_mcocr_data"
def load_label_file(p) -> list[tuple[str,str]]:
    # split on FIRST tab only; transcription may contain spaces
def validate(fname, text) -> str | None:        # returns reject reason or None
    # 2 tab-fields; crop exists; text.strip() non-empty; PIL opens; utf-8 normalize (NFC)
def build(split_file, split_name, accepted, rejected): ...
def main():
    # writes train.txt/val.txt/manifest/rejected + sanity_report.json with:
    #   accepted_train, accepted_val, missing_image, empty_label, corrupt_image
```
Expected (from plan): ~5,285 train / ~1,300 val lines, 6,585 crop images.
**Sanity report is a "minimum done" gate.** Unicode normalized to **NFC** (Vietnamese
diacritics) — recorded as a decision because it affects CER and the charset.

### 3.3 `training/ocr/build_charset.py`
```python
def build_charset(train_txt, val_txt, out="training/ocr/vi_dict.txt"):
    chars = sorted({c for _, t in load_label_file_all() for c in t})  # NFC
    # one char per line = PaddleOCR character_dict_path format
```
**Honest note in spec:** PP-OCRv5 `latin` head won't cover all Vietnamese diacritic
codepoints; building the dict from data means the rec head's final classifier is
(partially) re-initialized → 3 epochs is tight. Documented as a known risk; the
fallback is `PP-OCRv4_mobile_rec` (plan §5.3).

---

## 4. Training (separate env)

### 4.1 `training/ocr/ppocrv5_vi_mcocr_rec.yml`
PaddleOCR rec config, pretrained init, budget-tuned:
```yaml
Global:
  pretrained_model: <latin_PP-OCRv5_mobile_rec pretrained>   # init only
  character_dict_path: training/ocr/vi_dict.txt
  use_space_char: true
  epoch_num: 3
  eval_batch_step: [0, 200]
  save_model_dir: $WS/models/ocr/_train/ppocr_vi_mcocr_ft
  use_amp: true                      # mixed precision (VRAM)
  save_epoch_step: 1
Optimizer: {lr: {name: Cosine, learning_rate: 0.0001, warmup_epoch: 1}, regularizer: {factor: 1.0e-5}}
Train: {dataset: {label_file_list: [.../train.txt], data_dir: .../crops}, loader: {batch_size_per_card: 128, num_workers: 8}}
Eval:  {dataset: {label_file_list: [.../val.txt],  data_dir: .../crops}, loader: {batch_size_per_card: 128}}
Metric: {name: RecMetric, main_indicator: acc}   # also logs norm_edit_dis (1-CER)
```
seed 42, early-stop on val CER (best-only checkpoint).

### 4.2 `training/train_ocr_rec.py` — budget guard wrapper
Wraps PaddleOCR's `tools/train.py` (subprocess) and **enforces the ≤1h/≤5GB
envelope** the plan mandates.
```python
def gpu_mem_poller(stop, peak):           # pynvml, 1s poll -> peak["mb"]
def run_with_budget(cmd, wall_limit_s=3300, vram_limit_mb=5000):
    # spawn training subprocess; thread polls VRAM; if peak>limit -> log + SIGTERM
    # if elapsed>wall_limit -> SIGTERM, keep best checkpoint, exit "stopped_early"
def dry_run():                            # ~10 steps, report throughput + peak VRAM
def main():
    # 1) dry_run -> if peak>5GB: print "lower batch_size to 64/32" and exit 2
    # 2) full run_with_budget(); write training_log.json (wall_s, peak_vram_mb, best_cer)
```
Batch-size policy (plan §5.4): start 128 → 64 → 32 based on `dry_run()` peak VRAM.
Writes `models/ocr/ppocr_vi_mcocr_ft/training_log.json` with VRAM/wall evidence.

### 4.3 `training/export_ocr_onnx.py`
```python
def export():
    # 1) paddle: tools/export_model.py -> inference model (inference.pdmodel/.pdiparams)
    # 2) paddle2onnx -> model.onnx
    # 3) write metadata.json: {base_model, dict_path, rec_img_shape:[3,48,320],
    #      val_cer, val_exact, train_lineage(commit, data counts, seed), onnx_opset}
# output: $WS/models/ocr/ppocr_vi_mcocr_ft/{inference.*, model.onnx, metadata.json}
```
Mirrors `training/export_onnx.py` structure (argparse, timestamped log). ONNX size
reported (excellent-done wants it small for local demo).

---

## 5. Runtime adapter (base env, no paddle)

### 5.1 `docai/ocr_recognizer.py` (new)
Owns the Vietnamese dict + CTC decode so we don't depend on RapidOCR's hardcoded
Chinese post-process.
```python
class FineTunedRecognizer:
    def __init__(self, onnx_path, dict_path, img_shape=(3,48,320), batch=6):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.charset = ["blank"] + Path(dict_path).read_text(encoding="utf-8").splitlines()
        ...
    @classmethod
    def load(cls):                       # reads DOCAI_OCR_REC_MODEL / DOCAI_OCR_REC_DICT
        ...
    def _preprocess(self, crop_bgr):     # resize keep-ratio to H=48, pad to W=320, normalize
        ...
    def _ctc_decode(self, logits) -> tuple[str, float]:
        # argmax over time; collapse repeats; drop blank(0); conf = mean max-softmax
        ...
    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        # batched ONNX run -> list of (text, conf)
```
This is the ONLY new runtime code; pure onnxruntime + numpy.

### 5.2 `docai/config.py` (edit)
```python
OCR_RECOGNIZER = os.environ.get("DOCAI_OCR_RECOGNIZER", "rapidocr_default")  # |ppocr_vi_mcocr_ft
OCR_REC_MODEL  = os.environ.get("DOCAI_OCR_REC_MODEL",  str(MODELS_DIR/"ocr/ppocr_vi_mcocr_ft/model.onnx"))
OCR_REC_DICT   = os.environ.get("DOCAI_OCR_REC_DICT",   str(MODELS_DIR/"ocr/ppocr_vi_mcocr_ft/vi_dict.txt"))
```

### 5.3 `docai/ocr.py` (edit) — keep detector, swap recognizer
```python
def run_ocr(image_bgr):
    if config.OCR_RECOGNIZER == "rapidocr_default":
        ... # unchanged
    else:
        boxes = _get_engine().text_det(image_bgr)            # RapidOCR detector only
        crops = [_crop_perspective(image_bgr, b) for b in boxes]
        rec = _get_ft_recognizer()                            # cached FineTunedRecognizer
        texts = rec.recognize(crops)
        return [{"text": t, "bbox": _xyxy(b), "conf": c} for b,(t,c) in zip(boxes, crops_txt)]
```
Token schema (`text/bbox/conf`) is **unchanged**, so KIE/router/pipeline see no diff —
this is what makes it a clean on/off adapter and enables downstream anti-regression.
**Risk flagged:** reaching into RapidOCR's detector is internal API; if unstable, the
adapter degrades to **eval-only** mode (§6) and pipeline integration is marked optional
(the plan's "minimum done" is satisfied by eval-only).

---

## 6. Evaluation

### 6.1 `scripts/eval_ocr_rec.py` (core deliverable)
Crop-level, no detector. Compares default vs fine-tuned on `text_recognition_val_data.txt`.
```python
from eval.metrics import cer                      # reuse
def wer(pred, gold): ...                          # word-level, add here
def run(recognizer, val_lines, crop_dir): -> per-line (pred, cer, exact, wer, ms)
def main():
    # --recognizer default|ft ; --field-aware uses mcocr_train_df.csv crop->label map
    # report: macro CER, exact-line acc, WER, p50 latency/crop;
    #         relative CER improvement (good>=15%, excellent>=25%);
    #         field-aware CER by SELLER/ADDRESS/TIMESTAMP/TOTAL_COST
    # -> docs/logs/ocr_rec_eval_<ts>.md  (+ raw json)
```
Field-aware mapping (plan §5.6): crop `mcocr_public_<id>_<k>.jpg` → parent `<id>` →
`mcocr_train_df.csv` row → `anno_labels` (k-th region). Spec notes alignment is
best-effort and only for *analysis*, never a gold claim.

### 6.2 Downstream anti-regression (bonus)
Reuse `scripts/run_benchmark.py` with the env toggled — no new code:
```bash
DOCAI_OCR_RECOGNIZER=rapidocr_default python scripts/run_benchmark.py --data $WS/data/sroie/test ...
DOCAI_OCR_RECOGNIZER=ppocr_vi_mcocr_ft python scripts/run_benchmark.py --data $WS/data/sroie/test ...
```
Compare macro-F1, all-required-correct, and **SILENT_WRONG must not increase
materially** (good-done criterion). Honest caveat: MC-OCR is Vietnamese; SROIE is
Malaysian/English — gains may not transfer (and that's a documented, expected outcome).

---

## 7. H100 budget execution (maps plan §5.5)
| min | step | command |
|---|---|---|
| 0–10 | download + prepare + sanity | `wp3_download_mcocr.py`, `wp3_prepare_ocr.py`, `build_charset.py` |
| 10–15 | dry-run VRAM/throughput | `train_ocr_rec.py --dry-run` |
| 15–45 | fine-tune (budget-guarded) | `train_ocr_rec.py` |
| 45–55 | export | `export_ocr_onnx.py` |
| 55–60 | smoke eval + package | `eval_ocr_rec.py --recognizer ft` |
Overrun policy: SIGTERM, keep best checkpoint, document stop condition (a reproducible
negative result still counts as "minimum done").

## 8. Done-criteria mapping
- **Minimum:** §3 sanity report + §4 budget-compliant train + §4.3 ONNX + §6.1 eval + the
  `mcocr_val_sample_df.csv`-not-gold caveat (printed in eval report header).
- **Good:** val CER ≥15% relative ↓, runs on CPU/RTX1650, §6.2 no SILENT_WRONG increase.
- **Excellent:** CER ≥25% ↓, timestamp/amount lines improve, small ONNX, clean config on/off, latency + business trade-off quantified.

## 9. Honest caveats baked into deliverables
1. `mcocr_val_sample_df.csv` (`anno_texts="abc abc abc"`) is a **stub, not gold** — stated in eval report header and handoff.
2. RapidOCR can't take a custom rec dict via kwargs → we ship our own CTC adapter; full-pipeline swap uses RapidOCR's internal detector (flagged as internal API; eval-only fallback exists).
3. Paddle training is a **separate env**; the product gains no paddle dependency.
4. Vietnamese charset re-init + 3-epoch cap may limit gains; `PP-OCRv4_mobile_rec` fallback documented.
5. No production-readiness or downstream-KIE-validation claim (scope boundary from plan §3.3 / §7 NO-GO).

## 10. Deliverables checklist (plan §6)
audit note · `wp3_download_mcocr.py` · `wp3_prepare_ocr.py` + `sanity_report.json` ·
`ppocrv5_vi_mcocr_rec.yml` · `training_log.json` (VRAM/wall) · `models/ocr/ppocr_vi_mcocr_ft/`
(ONNX + metadata) · `docs/logs/ocr_rec_eval_*.md` · caveat note.
