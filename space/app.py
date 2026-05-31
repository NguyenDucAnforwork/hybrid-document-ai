"""Hybrid Document AI — live demo (HuggingFace Space, Gradio 5).

Runs the REAL pipeline: quality -> RapidOCR (PP-OCR) -> KIE (calibrated sklearn
classifier from the HF model registry) -> confidence router. (VLM hard-case path
runs on GPU; on this free CPU Space the traditional path is served.)
"""
import os
import json
os.environ.setdefault("DOCAI_WORKSPACE", "/tmp/docai-workspace")

import cv2
import numpy as np
import gradio as gr
from huggingface_hub import hf_hub_download

# Fix gradio_client schema bug ("argument of type 'bool' is not iterable") that
# 500s /gradio_api/info -> "No API found". Boolean JSON-schema nodes are valid
# (additionalProperties: true/false) but the parser doesn't guard for them.
import gradio_client.utils as _gcu
_orig_get_type = _gcu.get_type
_orig_js = _gcu._json_schema_to_python_type
def _safe_get_type(schema):
    return "Any" if isinstance(schema, bool) else _orig_get_type(schema)
def _safe_js(schema, defs=None):
    return "Any" if isinstance(schema, bool) else _orig_js(schema, defs)
_gcu.get_type = _safe_get_type
_gcu._json_schema_to_python_type = _safe_js

from docai import pipeline
from docai import classifier as _clf
from docai.kie import KIEModel

# Load trained models from the HF registry (Space has no /data workspace).
_model_file = hf_hub_download("banhchungtuongot/hybrid-docai-kie", "kie/v4/model.joblib")
pipeline._kie = KIEModel.load(_model_file)
_dt_file = hf_hub_download("banhchungtuongot/hybrid-docai-kie", "doctype/v2/model.joblib")
_clf._loaded = _clf.DocTypeClassifier.load(_dt_file)   # multi-document router


def infer(image):
    if image is None:
        return "{}", "Upload a receipt image first."
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, enc = cv2.imencode(".jpg", bgr)
    res = pipeline.process_document("upload", enc.tobytes())
    d = res.model_dump()
    rows = "\n".join(f"| {k} | {v['value']} | {v['confidence']} |"
                     for k, v in d["fields"].items())
    tx = d.get("line_items") or []
    tx_note = f"\n\n**Transactions parsed:** {len(tx)} rows" if tx else ""
    summary = (
        f"### Document type: `{d['document_type']}` · Route: `{d['route']}` · "
        f"needs_human_review: **{d['needs_human_review']}**\n\n"
        f"| field | value | confidence |\n|---|---|---|\n{rows}{tx_note}\n\n"
        f"**Quality:** blur={d['quality']['blur_score']} issues={d['quality']['issues']} "
        f"· models `{d['model_versions']}`")
    return json.dumps(d, ensure_ascii=False, indent=2), summary


DESC = (
    "# 🧾 Hybrid Document AI — Receipt OCR + KIE\n"
    "Production-grade pipeline (VNPAY AI Engineer portfolio). **RapidOCR (PP-OCR)** reads "
    "text → a **calibrated scikit-learn KIE classifier** extracts fields → a **confidence "
    "router** flags low-confidence docs for human review (won't silently emit wrong data). "
    "Trained on real **SROIE** receipts + synthetic VN data.\n\n"
    "date & total_amount work well; merchant_name is a known-hard field. "
    "Code: github.com/NguyenDucAnforwork/hybrid-document-ai")

with gr.Blocks(title="Hybrid Document AI — OCR + KIE") as demo:
    gr.Markdown(DESC)
    with gr.Row():
        inp = gr.Image(type="numpy", label="Receipt")
        with gr.Column():
            out_md = gr.Markdown(label="Summary")
            out_json = gr.Code(label="Structured result (JSON)", language="json")
    btn = gr.Button("Extract", variant="primary")
    btn.click(infer, inputs=inp, outputs=[out_json, out_md], api_name="predict")
    ex_dir = "examples"
    if os.path.isdir(ex_dir):
        gr.Examples([[f"{ex_dir}/{f}"] for f in sorted(os.listdir(ex_dir))], inputs=inp)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
