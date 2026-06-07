# Model Comparison (SROIE test, n=80) 20260607_0415

## F1 per field

| field | rule-only | logistic-KIE (prod) | LayoutLMv3-base |
|---|---|---|---|
| merchant_name | 0.1875 | 0.0375 | 0.2375 |
| date | 0.8 | 0.775 | 0.1875 |
| total_amount | 0.0 | 0.4875 | 0.0375 |

## ANLS per field

| field | rule-only | logistic-KIE | LayoutLMv3-base |
|---|---|---|---|
| merchant_name | 0.5463 | 0.0997 | 0.7148 |
| date | 0.8813 | 0.87 | 0.1875 |
| total_amount | 0.0 | 0.5396 | 0.0654 |

## Latency (KIE inference only, ms/doc)

| model | mean latency |
|---|---|
| rule-only | 1.5ms |
| logistic-KIE | 60.2ms |
| LayoutLMv3-base | 41.4ms |

> Note: latency above is KIE inference only (excludes OCR which is shared across models).
> Full pipeline (OCR + KIE) p50 ≈ 1-2s; see load_test logs for end-to-end numbers.