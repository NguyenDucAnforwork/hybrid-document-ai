# Detector + line-grouping error analysis (MC-OCR full-image) 20260609_1317

- n_imgs=80, gold field regions=459
- **det_field_recall=0.963**  overmerge_rate=0.07  oversplit_rate=0.013  reading_order_error(img)=0.823
- boxes/img≈39.4, unmatched_box_rate=0.857 (unlabeled lines, not errors)

### Failure taxonomy by field (dominant cause per gold region)
| field | coverage | DETECT_MISS | OVERMERGE | OVERSPLIT | REC_ERROR | OK |
|---|---|---|---|---|---|---|
| SELLER | 0.965 | 3 | 4 | 0 | 3 | 76 |
| ADDRESS | 0.925 | 10 | 24 | 1 | 1 | 97 |
| TIMESTAMP | 0.989 | 1 | 4 | 3 | 25 | 58 |
| TOTAL_COST | 0.98 | 3 | 0 | 2 | 7 | 137 |

### Overall taxonomy
| cause | count |
|---|---|
| DETECT_MISS | 17 |
| OVERMERGE | 32 |
| OVERSPLIT | 6 |
| REC_ERROR | 36 |
| OK | 368 |