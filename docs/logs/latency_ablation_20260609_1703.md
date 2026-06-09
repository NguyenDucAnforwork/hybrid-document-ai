# Latency ablation (MC-OCR full-image, n=40) 20260609_1703

| config | macro CER | SELLER | ADDRESS | TIMESTAMP | TOTAL_COST | p50 ms | p95 ms | mean #rerec | needs_review |
|---|---|---|---|---|---|---|---|---|---|
| default | 0.3831 | 0.2504 | 0.5263 | 0.5658 | 0.2247 | 1091.3 | 1556.1 | 0.0 | 0.0 |
| ft_all | 0.2402 | 0.1434 | 0.3071 | 0.4668 | 0.1011 | 2883.2 | 3456.1 | 41.3 | 0.0 |
| ft_critical | 0.2595 | 0.1463 | 0.3095 | 0.4803 | 0.1481 | 2513.1 | 2949.7 | 25.1 | 0.0 |
| auto | 0.2417 | 0.1502 | 0.3123 | 0.4662 | 0.0975 | 2735.6 | 3310.4 | 34.1 | 0.0 |

Note: latency p50/p95 are noisy under shared-machine load; the FT path runs full RapidOCR (det+rec) THEN re-recognizes crops, so det+rec dominates. `auto` on this all-Vietnamese set behaves like ft_all (it routes VN→FT); its latency win is on English docs (Task C). `ft_critical` re-recognizes only field-critical boxes.