# WP-3 Audited Implementation Plan

## MC-OCR OCR Fine-tune Under Hard Constraints

This document supersedes the earlier partial audit and rewrites the implementation
plan in `docs/wp3_mcocr_ocr_finetune_spec.md` based on the dataset structure that
 was actually verified from the Kaggle package.

Hard constraints for the implementation team:

- Total H100 budget: **<= 1 hour**
- Peak GPU memory: **<= 5 GB VRAM**
- Fine-tune corpus: **MC-OCR 2021 only**
- Keep the current production detector / pipeline story intact
- This WP-3 scope is **recognizer fine-tuning**, not full OCR stack retraining

---

## Task Description

The objective of this work package is to run a **small, production-oriented OCR
adaptation experiment** for the Hybrid Document AI project.

The goal is **not** to build a new OCR engine from scratch or to claim
production-ready banking OCR. The goal is to fine-tune only the **text
recognition module** of a lightweight PP-OCR/PaddleOCR-style mobile recognizer
on **MC-OCR 2021 Vietnamese receipt data**, then evaluate whether that adapted
recognizer improves OCR quality enough to justify integration cost, export
effort, and latency trade-offs.

The final deliverable should preserve the project's current production story:

- keep the current detector / OCR pipeline structure intact
- fine-tune only a lightweight recognizer
- export deployable artifacts such as Paddle inference model and ONNX
- integrate the recognizer as an **optional OCR adapter**
- report OCR-level gains honestly
- optionally measure downstream KIE anti-regression on existing internal sets
- preserve latency profiling, guardrails, confidence routing, and reproducible
  evaluation

Expected claim:

> Fine-tuned a lightweight Vietnamese receipt OCR recognizer on MC-OCR 2021 and
> integrated it into a production-style OCR-KIE pipeline with deployable export,
> latency profiling, and reproducible OCR-level evaluation.

Expected non-claim:

> Trained a production-grade banking OCR system.

Task boundaries:

- in scope:
  - MC-OCR subset audit
  - recognizer fine-tuning
  - export
  - optional adapter integration
  - OCR-level evaluation
  - downstream anti-regression as a bonus layer
- out of scope:
  - detector retraining
  - full OCR stack redesign
  - LayoutLMv3 retraining
  - VLM training
  - production-readiness claim

---

## Done Criteria

### Minimum done

The task is minimally complete when:

1. The actual Kaggle package structure is audited and documented correctly.
2. A reproducible subset-extraction path exists for:
   - `text_recognition_train_data.txt`
   - `text_recognition_val_data.txt`
   - `text_recognition_mcocr_data/text_recognition_mcocr_data/`
3. A data sanity report is generated with:
   - accepted train count
   - accepted val count
   - missing-image count
   - empty-label count
   - corrupt-image count
4. A lightweight recognizer is fine-tuned from pretrained weights within:
   - **<= 1 hour H100 wall clock**
   - **<= 5 GB peak VRAM**
5. At least one deployable inference artifact is exported:
   - Paddle inference model
   - or ONNX
6. The recognizer can run as an optional adapter in the local pipeline or in an
   isolated inference smoke test.
7. An OCR-level evaluation report exists on `text_recognition_val_data.txt`.
8. The handoff explicitly states that `mcocr_val_sample_df.csv` is **not** a
   trustworthy downstream gold validation set.

Important note:

- A negative experimental result can still count as "minimum done" if the run is
  reproducible, resource-compliant, and honestly documented.

### Good done

The task is good enough for README / CV / handoff if:

- OCR val CER improves by **>= 15% relative**
- exact line accuracy improves measurably on `text_recognition_val_data.txt`
- the exported model runs locally on:
  - RTX 1650
  - or CPU fallback
- downstream receipt anti-regression on existing internal benchmarks shows:
  - no obvious collapse on key fields
  - `SILENT_WRONG` does not increase materially
- end-to-end latency increase remains acceptable for demo usage
- the team can explain clearly why OCR improvement does or does not translate
  into downstream gains

### Excellent done

The task is excellent if:

- OCR val CER improves by **>= 25% relative**
- timestamp / amount-like lines improve clearly, not just generic lines
- the ONNX artifact is small enough for practical local demo deployment
- downstream receipt benchmarks show:
  - stable or improved key-field extraction
  - no increase in silent failure risk
- the optional OCR adapter can be switched on/off cleanly through config
- the final report includes both:
  - quantified latency trade-off
  - quantified business-value trade-off

---

## 1. Audit Summary

### 1.1 Credentials

Verified locally in `.env`:

- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
- `HF_TOKEN`

Do not copy raw secret values into any handoff document.

### 1.2 What was audited

The Kaggle dataset `domixi1989/vietnamese-receipts-mc-ocr-2021` was checked in
two ways:

1. Direct Kaggle file downloads for the named top-level files
2. ZIP manifest inspection of the full dataset package

The earlier "image-only mirror" conclusion was wrong because the first API-based
listing only exposed a partial slice of the dataset. The ZIP manifest is the
source of truth for this revised spec.

### 1.3 Verified package structure

The dataset ZIP contains **61,340** entries.

Top-level structure observed from the ZIP manifest:

- `data0_or_180/`: **35,987** entries
- `text_recognition_mcocr_data/`: **6,585** entries
- `rotation_corrector/`: **3,466** entries
- `rotation_corrector_kie/`: **3,465** entries
- `dataset/`: **2,311** entries
- `preprocessor/`: **2,311** entries
- `text_detector/`: **2,311** entries
- `kie_data/`: **2,307** entries
- `train_images/`: **1,155** entries
- `data0.7/`: **1,044** entries
- `val_images/`: **391** entries
- top-level files:
  - `mcocr_train_df.csv`
  - `mcocr_val_sample_df.csv`
  - `text_recognition_train_data.txt`
  - `text_recognition_val_data.txt`
  - `pre_dict.pkl`
  - `post_dict.pkl`
  - `results.csv`

---

## 2. Exact Data That Matters For WP-3

## 2.1 Primary OCR fine-tune data

### Crop images

Recognition crop images exist at:

```text
text_recognition_mcocr_data/text_recognition_mcocr_data/*.jpg
```

Verified count:

- **6,585** crop images

Sample paths:

- `text_recognition_mcocr_data/text_recognition_mcocr_data/mcocr_public_145013aagqw_0.jpg`
- `text_recognition_mcocr_data/text_recognition_mcocr_data/mcocr_public_145013aagqw_1.jpg`

### OCR train labels

Top-level file:

```text
text_recognition_train_data.txt
```

Verified line count:

- **5,285**

Verified format:

```text
crop_filename<TAB>transcription
```

Example:

```text
mcocr_public_145013snoxg_0.jpg<TAB>THE COFFEE HOUSE
```

### OCR validation labels

Top-level file:

```text
text_recognition_val_data.txt
```

Verified line count:

- **1,300**

Verified format:

```text
crop_filename<TAB>transcription
```

Important nuance:

- The OCR validation file also references `mcocr_public_*` crop names
- Therefore the OCR recognizer split is not the same thing as `val_images/`

### Mapping rule

For OCR fine-tune, the expected mapping is:

```text
text_recognition_train_data.txt / text_recognition_val_data.txt
  -> crop filename such as mcocr_public_xxxxx_7.jpg
  -> image file inside text_recognition_mcocr_data/text_recognition_mcocr_data/
```

This means:

- OCR recognizer fine-tuning is **GO**
- We have both crop images and transcriptions

---

## 2.2 Downstream KIE / field-labeled train data

### Train CSV

Top-level file:

```text
mcocr_train_df.csv
```

Important packaging note:

- When downloaded alone via Kaggle CLI, this file may arrive as a ZIP payload
  disguised with `.csv`
- Inside the full dataset ZIP, it is a normal CSV

Verified schema:

```text
img_id, anno_polygons, anno_texts, anno_labels, anno_num, anno_image_quality
```

Verified row count:

- **1,155**

Meaning:

- `img_id`: receipt image file name
- `anno_polygons`: list-like polygon annotations per labeled region
- `anno_texts`: `|||`-joined field texts
- `anno_labels`: `|||`-joined labels aligned with `anno_texts`
- `anno_num`: number of labeled regions
- `anno_image_quality`: image quality score

Verified label set:

- `SELLER`
- `ADDRESS`
- `TIMESTAMP`
- `TOTAL_COST`
- one rare noisy label: `TOTAL_TOTAL_COST`

Observed label frequency from the audited CSV:

- `TOTAL_COST`: **2,114**
- `ADDRESS`: **1,952**
- `TIMESTAMP`: **1,347**
- `SELLER`: **1,171**

This file is usable for:

- field-aware OCR error analysis
- silver crop filtering by field type
- downstream anti-regression studies on receipt fields

### Relation between OCR crop data and field-labeled train CSV

From the audit:

- OCR text files map to **1,153** unique parent receipt images
- Intersection with `mcocr_train_df.csv` image IDs: **1,153**
- `mcocr_train_df.csv` images missing from OCR text lists: **2**

So the OCR crop supervision and the field-labeled train receipts are almost fully
aligned, but not perfectly identical.

---

## 2.3 Validation caveat for downstream / KIE work

Top-level file:

```text
mcocr_val_sample_df.csv
```

Verified schema:

```text
img_id, anno_image_quality, anno_texts
```

Verified row count:

- **391**

Observed sampled values:

- `anno_texts` appears as placeholder text such as `abc abc abc`

Therefore:

- this file should **not** be treated as a real field-level gold validation set
- it is usable as a sample list / metadata stub only
- any official-looking downstream validation claim based only on this file would
  be weak

This is the most important caveat in the handoff.

---

## 2.4 Additional useful assets in the package

These are not required for the first OCR recognizer fine-tune, but they are real
and may help later:

- `train_images/train_images/*.jpg`: **1,155** full receipt images
- `val_images/val_images/*.jpg`: **391** full receipt images
- `kie_data/kie_data/boxes_and_transcripts/*.tsv`: **2,307** entries
- `text_detector/text_detector/txt/*.txt`: **2,311** entries
- `dataset/text_detector/txt/*.txt`: **2,311** entries

Sample observed formats:

### KIE TSV

Example line shape:

```text
id,x1,y1,x2,y2,x3,y3,x4,y4,text,label
```

Observed example:

```text
4,252,1218,412,1215,412,1239,252,1242,EL-02 0202/80/LO,TIMESTAMP
```

### Text detector TXT

Observed format:

```text
x1,y1,x2,y2,x3,y3,x4,y4,
```

This means the package also contains detector / KIE side artifacts, but WP-3
does not need to train those parts under the current constraint set.

---

## 3. Decision After Audit

### 3.1 What is now clearly feasible

The following is a **GO**:

- supervised OCR recognizer fine-tuning on MC-OCR crop images
- OCR-level validation using the provided `text_recognition_val_data.txt`
- field-aware post-hoc analysis using `mcocr_train_df.csv`

### 3.2 What is not strong enough yet

The following is **not** strong enough for a serious claim:

- downstream field-level validation based only on `mcocr_val_sample_df.csv`

### 3.3 Tech lead decision

Under banking-style scrutiny, the correct scope is:

> Fine-tune the OCR recognizer only, report OCR-level metrics honestly, and do
> not overclaim downstream KIE validation from the placeholder val CSV.

---

## 4. Recommended WP-3 Scope

WP-3 should focus on:

1. Prepare recognition crops and label lists from MC-OCR
2. Fine-tune a lightweight recognizer only
3. Export the recognizer for inference
4. Validate OCR quality on the provided OCR val split
5. Optionally run downstream anti-regression on existing internal receipt sets

WP-3 should explicitly avoid:

- detector retraining
- LayoutLMv3 retraining
- VLM training
- full end-to-end OCR stack redesign
- training anything that violates the 1 hour / 5 GB constraints

---

## 5. Detailed Implementation Plan

## 5.1 Minimal subset to extract

The implementation team does **not** need to extract the full Kaggle package.

Minimum required subset:

```text
text_recognition_train_data.txt
text_recognition_val_data.txt
text_recognition_mcocr_data/text_recognition_mcocr_data/
```

Recommended extra subset for field-aware analysis:

```text
mcocr_train_df.csv
train_images/train_images/
val_images/val_images/
kie_data/kie_data/boxes_and_transcripts/
```

---

## 5.2 Data preparation steps

### Step A: OCR dataset materialization

Build a local processed OCR dataset:

```text
data/processed/mcocr_ocr/
  crops/
  train.txt
  val.txt
  manifest.jsonl
  rejected.jsonl
```

Rules:

- `train.txt` and `val.txt` should remain in `crop_path<TAB>transcription` format
- paths must point to extracted crop images
- preserve the Kaggle-provided train/val split

### Step B: Sanity checks

Hard checks before training:

- every line has exactly 2 TAB-separated fields
- crop image exists
- transcription is non-empty after strip
- image opens successfully
- invalid UTF-8 is normalized or rejected

Required outputs:

- accepted train count
- accepted val count
- missing-image count
- empty-label count
- corrupt-image count

### Step C: Optional field-aware manifest

Parse `mcocr_train_df.csv` into a second manifest:

```text
img_id
polygon
text
label
image_quality
```

Use this manifest only for:

- field-specific OCR analysis
- building optional field-specific subsets:
  - `SELLER`
  - `ADDRESS`
  - `TIMESTAMP`
  - `TOTAL_COST`

Do not mix this optional manifest into the main OCR train/val split unless the
mapping logic is explicitly implemented and validated.

---

## 5.3 Model choice under <= 5 GB VRAM

### Primary model

Use a lightweight recognizer such as:

```text
latin_PP-OCRv5_mobile_rec
```

Reason:

- fits the resource envelope
- easier ONNX export story
- much lower implementation risk than a large recognizer
- enough capacity for a targeted Vietnamese receipt adaptation run

### Fallback

If v5 tooling is unstable, fallback to:

```text
PP-OCRv4_mobile_rec
```

### Explicit non-goals

Do not train:

- detector
- classifier
- LayoutLMv3
- Qwen / VLM
- Donut / end-to-end document model

---

## 5.4 Training recipe

### Runtime policy

- pretrained initialization only
- mixed precision on
- max epochs: **3**
- early stop on validation CER
- save best checkpoint only
- stop immediately if wall clock threatens the 1 hour budget

### Suggested starting hyperparameters

- input shape: model default mobile recognition shape
- learning rate: `1e-4`
- warmup ratio: `0.05`
- weight decay: `1e-5`
- seed: `42`

### Batch-size policy for VRAM cap

Start with:

- batch size `128`

If peak VRAM exceeds 5 GB:

- back off to `64`
- else `32`

No full run should start before a short dry-run confirms the memory ceiling.

---

## 5.5 H100 budget plan

Keep the whole activity inside this envelope:

- 0-10 min: subset extraction + sanity report
- 10-15 min: dry-run for throughput and VRAM
- 15-45 min: actual fine-tune
- 45-55 min: export checkpoint / ONNX
- 55-60 min: smoke evaluation and artifact packaging

If any stage overruns:

- stop early
- keep best checkpoint so far
- document the stop condition honestly

---

## 5.6 Evaluation plan

### OCR-level evaluation

Primary evaluation should use:

```text
text_recognition_val_data.txt
```

Report:

- CER
- exact line accuracy
- optionally WER
- latency per crop batch

### Field-aware analysis

Use `mcocr_train_df.csv` only for analysis or internal holdout construction.

Possible reports:

- CER by `SELLER`
- CER by `ADDRESS`
- CER by `TIMESTAMP`
- CER by `TOTAL_COST`

### Downstream anti-regression

If the team wants a practical business check, run the new recognizer through the
existing receipt pipeline and compare against the current system on:

- SROIE
- current local receipt benchmarks already in this repo

But this is a bonus layer, not the core success criterion of WP-3.

---

## 5.7 Export and integration

Expected output artifact:

```text
models/ocr/ppocr_vi_mcocr_ft/
  inference.*
  model.onnx
  metadata.json
```

Integration style:

- keep the current detector unchanged
- expose the fine-tuned recognizer behind a config switch

Suggested config shape:

```text
DOCAI_OCR_RECOGNIZER=rapidocr_default|ppocr_vi_mcocr_ft
```

---

## 6. Required Handoff Deliverables

The team receiving this spec should produce:

1. A reproducible audit script or notebook
2. A subset extraction script
3. A data sanity report
4. A recognizer training config
5. A short training log with VRAM / wall-clock evidence
6. Exported inference artifact
7. OCR-level eval report on `text_recognition_val_data.txt`
8. A caveat note that `mcocr_val_sample_df.csv` is not a trustworthy downstream
   gold set

---

## 7. Go / No-Go Summary

### GO

Proceed with WP-3 if the scope is:

- OCR recognizer fine-tune only
- OCR validation on `text_recognition_val_data.txt`
- resource cap strictly enforced

### Conditional GO

Proceed carefully on field-aware analysis if:

- `mcocr_train_df.csv` parsing is implemented correctly
- polygon/text/label alignment is verified

### NO-GO

Do **not** claim:

- official downstream field-level validation quality from `mcocr_val_sample_df.csv`
- end-to-end production readiness from this fine-tune alone
- detector improvement, since detector is out of scope here

---

## 8. Final Recommendation

The correct handoff message is:

> MC-OCR 2021 on Kaggle does contain enough data to fine-tune a lightweight OCR
> recognizer under the 1 hour H100 / 5 GB VRAM constraint. However, the package's
> downstream validation CSV is only a weak sample stub, so the implementation
> team should report OCR-level gains honestly and avoid overstating downstream KIE
> validation.
