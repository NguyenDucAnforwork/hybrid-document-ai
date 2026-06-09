# Latency ablation (MC-OCR full-image, n=80) 20260609_1558

| config | macro CER | SELLER | ADDRESS | TIMESTAMP | TOTAL_COST | p50 ms | p95 ms | mean #rerec | needs_review |
|---|---|---|---|---|---|---|---|---|---|
| default | 0.3373 | 0.209 | 0.4786 | 0.4578 | 0.2118 | 1047.6 | 1523.0 | 0.0 | 1.0 |
| ft_all | 0.2653 | 0.1792 | 0.3186 | 0.4536 | 0.1523 | 1684.4 | 2131.4 | 36.3 | 1.0 |
| ft_critical | 0.273 | 0.1632 | 0.3322 | 0.4481 | 0.1764 | 1506.0 | 1944.1 | 21.2 | 1.0 |
| auto | 0.3373 | 0.209 | 0.4786 | 0.4578 | 0.2118 | 975.7 | 1335.3 | 0.0 | 1.0 |

Note: latency p50/p95 are noisy under shared-machine load; the FT path runs full RapidOCR (det+rec) THEN re-recognizes crops, so det+rec dominates. `auto` on this all-Vietnamese set behaves like ft_all (it routes VN→FT); its latency win is on English docs (Task C). `ft_critical` re-recognizes only field-critical boxes.