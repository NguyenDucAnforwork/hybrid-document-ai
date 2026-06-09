# Full-image MC-OCR pipeline eval 20260609_1350

- n_docs=80  (RapidOCR detector shared; recognizer swapped)
- **CAVEAT:** recognizer trained on crops from these train receipts → ft field numbers are optimistic (in-domain). Read the default→ft delta + per-field pattern.

| field | default CER | ft CER | rel ↓ | default ANLS | ft ANLS |
|---|---|---|---|---|---|
| SELLER | 0.209 | 0.111 | 46.9% | 0.8171 | 0.9153 |
| ADDRESS | 0.4786 | 0.2552 | 46.7% | 0.5494 | 0.8015 |
| TIMESTAMP | 0.4578 | 0.3764 | 17.8% | 0.5698 | 0.6344 |
| TOTAL_COST | 0.2118 | 0.1084 | 48.8% | 0.7943 | 0.9026 |

- macro field CER: default 0.3373 → ft 0.2045
- latency p50/p95 (full-image OCR): default 937.1/1278.9ms · ft 2606.3/3103.8ms
- needs_review rate (process_document): default 1.0 · ft 1.0

### Failure examples (ft, CER>0.5)
- `TIMESTAMP` cer=1.0 gold=`Ngày : 11/08/2020 08:06` pred=``
- `SELLER` cer=1.2 gold=`co.op mart` pred=`Co.opMart HU GIANG`
- `SELLER` cer=0.84 gold=`Co.opMart HAU GIANG` pred=`Thời gi hu T 4 Co.opMart HU GIANG`
- `ADDRESS` cer=1.63 gold=`188 Hau Giang, P.6, Q.6, TpHCM` pred=`Da hang Gia Pa,- 2.230.10 .1 18 au Giang, P.,Q.QHo3 Thời gi hu T 4`
- `ADDRESS` cer=1.94 gold=`Dat hang qua DT: 028.39.600.913` pred=`Da hang Gia Pa,- 2.230.10 .1 Pa a P Hày 62 .2 H2 2  Hu 18 au Giang, P.,Q.QHo3`
- `TIMESTAMP` cer=0.56 gold=`Ngày: 21/05/2020` pred=`: 2/5/:2`