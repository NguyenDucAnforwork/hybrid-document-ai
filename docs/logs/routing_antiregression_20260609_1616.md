# Language-routing anti-regression 20260609_1616

- threshold (VI-diacritic ratio) = 0.015
- **VI (MC-OCR) routed to FT: 95%**
- **EN (SROIE) routed to default: 70%**
- EN identity check: 42/42 default-routed docs are token-identical to default → zero regression on English
- **routing accuracy = 0.825**

Conclusion: the Vietnamese CRNN does NOT touch English receipts — `auto` keeps them on RapidOCR default with byte-identical output, so SROIE metrics are unchanged. Vietnamese docs get the FT recognizer. Per-language routing, not a global swap.