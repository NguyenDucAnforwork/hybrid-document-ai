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

## Task E — in-box projection row-splitting (`docai/line_grouping.py`, `DOCAI_PROJECTION_SPLIT=1`)
Splits tall boxes (height > 1.8× median) into per-row sub-boxes via horizontal-projection
valleys, re-recognizes each. **Measured: ADDRESS full-image CER 0.319 → 0.316 (null);
overmerge_rate 0.07 → 0.07; det_field_recall 0.978 → 0.963 (slight over-split).** Another honest
null: the ADDRESS failures are **garbled recognizer output on detector crops**
(`188 Hau Giang…` → `P a Ta, xự Q H…`), not multi-line merges projection can separate. Kept
flag-gated, **default off**. Two grouping fixes (B horizontal, E vertical) both null → strong
evidence the ADDRESS loss is the **crop-distribution gap**, same root cause as TIMESTAMP.

## Task F — detector-style crop augmentation (`scripts/wp3_extract_det_crops.py` + `train_ocr_rec.py --augment`)
Root-cause fix: the recognizer was trained on clean gold crops but tested on the **detector's**
crops. Extracted **3862 detector-style crops** (all fields, boxes matched to gold), short
fine-tune from v1 (`--init-from`, mix det-crops 2×, `--augment`: pad/crop jitter + blur + contrast),
148 s, peak VRAM 1353 MB. **This is the fix that worked — for ALL fields, including the ones
grouping couldn't touch:**

| field | default | v1 ft | **Task F ft** | rel ↓ vs v1 |
|---|---|---|---|---|
| SELLER | 0.209 | 0.179 | **0.111** | −38% |
| ADDRESS | 0.479 | 0.319 | **0.255** | −20% (hits Task E's own target) |
| TIMESTAMP | 0.458 | 0.454 | **0.376** | −17% (missed ≤0.35 bar, but clear gain) |
| TOTAL_COST | 0.212 | 0.152 | **0.108** | −29% |
| **macro** | 0.337 | 0.265 | **0.205** | **−23%** |

ANLS up across the board (ADDRESS 0.55→0.80, TOTAL 0.79→0.90). Clean-crop val CER also *improved*
(0.085→0.063 — more data + robustness). Done-criteria: TIMESTAMP ≤0.35 **narrowly missed (0.376)**;
TOTAL/ADDRESS **did not regress — they improved strongly**. Task F model is now the config default
(`models/ocr/vi_mcocr_crnn_ft_taskf`). Latency note: full-image p50 is noisy run-to-run (machine
under shared load); the FT path re-recognizes crops so it is heavier than default — measure on a
quiet box before quoting.

## Task B (revisited) — latency ablation (`scripts/eval_latency_ablation.py`, clean sequential n=40)

| config | macro CER | p50 | p95 | mean #re-rec | TOTAL_COST CER |
|---|---|---|---|---|---|
| default (RapidOCR) | 0.38 | **1.09s** | 1.56s | 0 | 0.225 |
| ft_all | 0.24 | 2.88s | 3.46s | 41 | **0.101** |
| ft_critical | 0.26 | 2.51s | 2.95s | 25 | 0.148 |
| auto | 0.24 | 2.74s | — | 34 | 0.098 |

**No FT config reaches the 1.3–1.6s target.** The cost is structural: the pipeline runs **full
RapidOCR det+rec (~1.1s) and THEN re-crops + FT-recognizes** — the FT model itself is cheap
(batched), but RapidOCR's wasted rec + per-crop work adds ~1.8s. `ft_critical` (re-recognize only
top/date/money/anchor boxes) trims 2.88→2.51s but **regresses TOTAL_COST 0.101→0.148** (some total
lines not flagged critical). **Real latency fix = a detector-only path** (RapidOCR det, skip its
rec) — internal-API work, the next WP. Note: measure latency **serially** — concurrent ablation
runs inflate p50 ~4× from CPU contention (CER is concurrency-safe).

## Task C (revisited) — language-routing anti-regression (`scripts/eval_routing.py`)
Threshold tuned **0.015 → 0.06** (0.015 false-routed 30% of English to FT — the VI-CRNN hallucinates
diacritics on English crops). At 0.06, on a mixed set (MC-OCR VI + real SROIE EN):

- **Vietnamese → CRNN: 88%** · **English (SROIE) → default: 100%** · routing accuracy **0.942**
- **60/60 default-routed English docs are token-identical to default → zero regression on English.**

Answers the key question — *does the Vietnamese model break English receipts?* **No:** `auto` keeps
them on RapidOCR default with byte-identical output. Two bugs fixed en route (see debug WP3-9/10):
`auto` was reading diacritics off the Chinese-default output (never detected VI), and the loader
ignored the config's Task F model.

## Bottom line
The full-image gap is **not detection** (recall 0.978), **not deskew**, and **not line-grouping**
(Tasks B & E both measured null on ADDRESS). It is the **crop-distribution gap**: the recognizer
hadn't seen detector-style crops. **Task F (detector-style crop augmentation) fixed all four fields**
(macro 0.265→0.205), where two grouping attempts failed. The lesson the measurement bought: don't
swap the detector, don't deskew, don't keep tuning grouping — fine-tune the recognizer on
detector-style crops. Task D (language routing) + Task C (geometry flag) remain shipped as
safety/quality fixes.
