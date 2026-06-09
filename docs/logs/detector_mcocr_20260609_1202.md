# Detector + line-grouping error analysis (MC-OCR full-image) 20260609_1202

- n_imgs=80, gold field regions=459
- **det_field_recall=0.978**  overmerge_rate=0.07  oversplit_rate=0.013  reading_order_error(img)=0.823
- boxes/img≈36.3, unmatched_box_rate=0.842 (unlabeled lines, not errors)

### Failure taxonomy by field (dominant cause per gold region)
| field | coverage | DETECT_MISS | OVERMERGE | OVERSPLIT | REC_ERROR | OK |
|---|---|---|---|---|---|---|
| SELLER | 0.988 | 1 | 4 | 0 | 5 | 76 |
| ADDRESS | 0.94 | 8 | 24 | 1 | 3 | 97 |
| TIMESTAMP | 1.0 | 0 | 4 | 3 | 26 | 58 |
| TOTAL_COST | 0.993 | 1 | 0 | 2 | 9 | 137 |

### Overall taxonomy
| cause | count |
|---|---|
| DETECT_MISS | 10 |
| OVERMERGE | 32 |
| OVERSPLIT | 6 |
| REC_ERROR | 43 |
| OK | 368 |