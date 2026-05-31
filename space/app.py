"""Hybrid Document AI — live demo (HuggingFace Space, Gradio).

Runs the REAL pipeline: quality -> RapidOCR (PP-OCR) -> KIE (calibrated sklearn
classifier, downloaded from the HF model registry) -> confidence router.
"""
import os
os.environ.setdefault("DOCAI_WORKSPACE", "/tmp/docai-workspace")

import cv2
import numpy as np
import gradio as gr
from huggingface_hub import hf_hub_download

from docai import pipeline
from docai.kie import KIEModel

# Load the trained KIE model (v4, calibrated) from the HF model registry.
_model_file = hf_hub_download("banhchungtuongot/hybrid-docai-kie", "kie/v4/model.joblib")
pipeline._kie = KIEModel.load(_model_file)


def infer(image):
    if image is None:
        return {"error": "upload a receipt image"}, "no image"
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr)
    res = pipeline.process_document("upload", enc.tobytes())
    d = res.model_dump()
    rows = "\n".join(
        f"| {k} | {v['value']} | {v['confidence']} |" for k, v in d["fields"].items())
    summary = (
        f"### Route: `{d['route']}`  ·  needs_human_review: **{d['needs_human_review']}**\n\n"
        f"| field | value | confidence |\n|---|---|---|\n{rows}\n\n"
        f"**Quality:** blur={d['quality']['blur_score']} "
        f"issues={d['quality']['issues']}  ·  KIE `{d['model_versions'].get('kie')}`")
    return d, summary


DESC = (
    "# 🧾 Hybrid Document AI — Receipt OCR + KIE\n"
    "Production-grade pipeline (VNPAY AI Engineer portfolio). Upload a receipt: "
    "**RapidOCR (PP-OCR)** reads text, a **calibrated scikit-learn KIE classifier** "
    "extracts fields, and a **confidence router** flags low-confidence docs for human "
    "review (it won't silently emit wrong data). Trained on real **SROIE** receipts + "
    "synthetic VN data.\n\n"
    "Code: github.com/NguyenDucAnforwork/hybrid-document-ai")

examples = [[f"examples/{f}"] for f in sorted(os.listdir("examples"))] \
    if os.path.isdir("examples") else None

demo = gr.Interface(
    fn=infer, inputs=gr.Image(type="numpy", label="Receipt"),
    outputs=[gr.JSON(label="Structured result"), gr.Markdown(label="Summary")],
    title="Hybrid Document AI — OCR + KIE", description=DESC, examples=examples,
    allow_flagging="never")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
