---
title: Hybrid Document AI (OCR + KIE)
emoji: 🧾
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# Hybrid Document AI — Receipt OCR + KIE

Live demo of a production-grade document-understanding pipeline (VNPAY AI Engineer
portfolio). Quality → RapidOCR (PP-OCR) → calibrated scikit-learn KIE classifier →
confidence router (low-confidence → human review). Trained on real **SROIE** +
synthetic Vietnamese receipts.

- Code: https://github.com/NguyenDucAnforwork/hybrid-document-ai
- Model: https://huggingface.co/banhchungtuongot/hybrid-docai-kie
- Dataset: https://huggingface.co/datasets/banhchungtuongot/hybrid-docai-receipts
