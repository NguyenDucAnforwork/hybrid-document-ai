# WP-3 follow-up — detector + line-grouping error analysis (MC-OCR)

Measurement-first investigation of *why* the crop-level recognizer gain (−73% CER)
shrinks to ~21% full-image. Goal: quantify whether to fix with a **rule**, **deskew**,
or **a different detector** — instead of guessing.

Scripts: `scripts/eval_detector_mcocr.py` (Step 2 metrics + Task A taxonomy).
Gold = `anno_polygons` field regions in `mcocr_train_df.csv` (SELLER/ADDRESS/TIMESTAMP/
TOTAL_COST). n=80 full train receipts. Recognizer = fine-tuned CRNN.

## Step 2 — detector metrics (`docs/logs/detector_mcocr_20260609_1202.md`)

| metric | value | read |
|---|---|---|
| **det_field_recall** | **0.978** | detector finds nearly all field regions — **not** a detection-miss problem |
| overmerge_rate | 0.070 | one box spans ≥2 fields (32/459) |
| oversplit_rate | 0.013 | negligible |
| reading_order_error (img) | 0.823 | *likely overstated* — shared-box ties + gold-order assumptions; treat as soft |
| field coverage | SELLER .99 / ADDRESS .94 / TIMESTAMP 1.0 / TOTAL_COST .99 | all high |

Precision vs *all* lines is **not computable** (gold = field regions only, not every
text line); `boxes/img≈36`, `unmatched_box_rate=0.84` are unlabeled lines, **not errors**.
`KIE_SELECT_ERROR` not assessable — MC-OCR labels don't map to the SROIE KIE schema
(documented, not faked).

## Task A — failure taxonomy (dominant cause per gold region)

| field | coverage | DETECT_MISS | OVERMERGE | OVERSPLIT | REC_ERROR | OK |
|---|---|---|---|---|---|---|
| SELLER | 0.99 | 1 | 4 | 0 | 5 | 76 |
| ADDRESS | 0.94 | 8 | **24** | 1 | 3 | 97 |
| TIMESTAMP | 1.00 | 0 | 4 | 3 | **26** | 58 |
| TOTAL_COST | 0.99 | 1 | 0 | 2 | 9 | 137 |
| **total** | — | 10 | 32 | 6 | 43 | 368 |

### The answers asked for
- **TIMESTAMP errors are mainly REC_ERROR (26), not detection** (coverage 1.0). The
  recognizer errs on the *detector's* crop even though training-crop CER was 0.098 — a
  **crop-distribution gap** (detector boxes ≠ the clean training crops; time/date strings
  like `10:44:08-15/08/2020` are the hardest). → not fixable by grouping or deskew.
- **TOTAL_COST is mostly OK (137), recall 0.99** → the money field translates well; this is
  why full-image TOTAL_COST CER still improved (−28%). Good news for downstream `total`.
- **ADDRESS improvement is grouping-bound, not recognizer-bound.** Recognizer is excellent on
  clean address crops (crop-level −82%), but full-image is capped by **OVERMERGE (24)** +
  DETECT_MISS (8): multi-line addresses get merged into one box. So ADDRESS gains come from the
  recognizer *where grouping is clean*, and are lost where the detector over-merges.

## Decision (what the data says to fix)
- **Do NOT change the detector** (recall 0.978) and **do NOT prioritize deskew** (skew wasn't
  the dominant cause on this set).
- Two real levers: **(1) split ADDRESS vertical over-merges**; **(2) close the TIMESTAMP
  crop-distribution gap** (crop normalization / augment recognizer with detector-style crops).

## Tasks B / C / D — implemented, flag-gated, measured

### Task B — line-grouping / reading-order split (`docai/line_grouping.py`, `DOCAI_LINE_REGROUP=1`)
Splits a box containing ≥2 different field anchors (e.g. `Ngày: … Tổng tiền: …`) at the second
anchor with proportional bbox-x. Unit-verified correct.
**Measured effect on MC-OCR: overmerge_rate 0.070 → 0.070 (no change).** Honest null result:
the over-merge here is **vertical** (ADDRESS multi-line, 24/32), which a *horizontal* anchor
split cannot touch; the horizontal `Ngày…Tổng` merges are rare on this set. The fix is correct
for its case but doesn't move this metric. **Real next step:** in-box **horizontal-projection
row splitting** for tall multi-line boxes (different technique) — scoped, not yet built.

### Task D — language/doc recognizer routing (`DOCAI_OCR_RECOGNIZER=auto`)
Routes per document: Vietnamese-diacritic ratio of the default OCR text ≥ threshold → use the
fine-tuned VI recognizer; otherwise keep RapidOCR default (English/SROIE). Unit-verified: VN
text → FT, English text → default. Directly addresses the measured `needs_review` rise
(0.66→0.80) and prevents SROIE regression from a global swap — **per-language routing, not a
global recognizer**.

### Task C — geometry-risk flag (`docai/pipeline.py`, `DOCAI_GEOMETRY_RISK_ANGLE`)
When residual skew ≥ 8° (beyond the small-angle deskew the pipeline already applies),
`needs_human_review=True` with reason `geometry_risk:skew=…`. Targets the rotate/perspective
high-confidence-wrong failure mode (robustness ADR, ECE≈0.5) — fail loud instead of emitting
warped-text output. (Perspective_score proxy not yet added; skew used.)

## Bottom line
The full-image gap is **not detection** (recall 0.978). It is **ADDRESS vertical over-merge**
(needs projection row-splitting — Task B's horizontal split is the wrong axis here, measured null)
and **TIMESTAMP crop-distribution REC_ERROR** (needs crop-matched recognizer data). Task D
(routing) + Task C (geometry flag) are correct, cheap safety/quality fixes shipped now. This is
the measurement that tells us where NOT to spend effort (detector swap, deskew) as much as where to.
