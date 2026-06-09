# Full-image MC-OCR pipeline eval 20260609_1144

- n_docs=80  (RapidOCR detector shared; recognizer swapped)
- **CAVEAT:** recognizer trained on crops from these train receipts → ft field numbers are optimistic (in-domain). Read the default→ft delta + per-field pattern.

| field | default CER | ft CER | rel ↓ | default ANLS | ft ANLS |
|---|---|---|---|---|---|
| SELLER | 0.209 | 0.1792 | 14.3% | 0.8171 | 0.8419 |
| ADDRESS | 0.4786 | 0.3186 | 33.4% | 0.5494 | 0.7644 |
| TIMESTAMP | 0.4578 | 0.4536 | 0.9% | 0.5698 | 0.5785 |
| TOTAL_COST | 0.2118 | 0.1523 | 28.1% | 0.7943 | 0.8656 |

- macro field CER: default 0.3373 → ft 0.2653
- latency p50/p95 (full-image OCR): default 2406.9/3464.7ms · ft 1815.2/3664.5ms
- needs_review rate (process_document): default 0.662 · ft 0.8

### Failure examples (ft, CER>0.5)
- `TIMESTAMP` cer=1.0 gold=`Ngày : 11/08/2020 08:06` pred=``
- `SELLER` cer=1.3 gold=`co.op mart` pred=`Co. optTo. H B. Mi`
- `SELLER` cer=1.89 gold=`Co.opMart HAU GIANG` pred=`P. audan o. d, K SA. T  Co. optTo. H B. Mi`
- `ADDRESS` cer=2.6 gold=`188 Hau Giang, P.6, Q.6, TpHCM` pred=`P a Ta, xự Q H - pAìh ha hmr HaConcu Ba D3u Hiy ờng, Tưxò, xi ưa c3 ch P. audan o. d, K SA. T `
- `ADDRESS` cer=2.94 gold=`Dat hang qua DT: 028.39.600.913` pred=`P a Ta, xự Q H - pAìh ha hmr HaConcu Batd: 12 Puờn tu ự on gpn Ha cac Ba D3u Hiy ờng, Tưxò, xi ưa c3 ch`
- `TIMESTAMP` cer=0.81 gold=`Ngày: 21/05/2020` pred=`Ngày: 24/08/20g0 p Viế THA`