"""Hybrid Document AI — live demo (HuggingFace Space, Gradio 5).

Multi-document: routes receipt / bank_statement / payment_order, then extracts.
Two tabs: single document, and BATCH (5–10 images) for a production-style demo.
VLM hard-case fallback runs on Modal (serverless GPU) via mode=api.
"""
import os
import json
from pathlib import Path
import sys

os.environ.setdefault("DOCAI_WORKSPACE", "/tmp/docai-workspace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import gradio as gr
from huggingface_hub import hf_hub_download

# Fix gradio_client schema bug (bool 'additionalProperties') that 500s /info.
import gradio_client.utils as _gcu
_o1, _o2 = _gcu.get_type, _gcu._json_schema_to_python_type
_gcu.get_type = lambda s: "Any" if isinstance(s, bool) else _o1(s)
_gcu._json_schema_to_python_type = lambda s, d=None: "Any" if isinstance(s, bool) else _o2(s, d)

from docai import pipeline
from docai import classifier as _clf
from docai.kie import KIEModel

HF_REPO = os.environ.get("DOCAI_HF_MODEL_REPO", "banhchungtuongot/hybrid-docai-kie")
WORKSPACE = Path(os.environ["DOCAI_WORKSPACE"])


def _model_path(local_rel: str, hf_rel: str) -> str:
    local = WORKSPACE / local_rel
    if local.exists():
        return str(local)
    return hf_hub_download(HF_REPO, hf_rel)


# Prefer local demo artifacts when present; otherwise fall back to HF.
pipeline._kie = KIEModel.load(_model_path("models/kie/v4/model.joblib", "models/kie/v4/model.joblib"))
_clf._loaded = _clf.DocTypeClassifier.load(
    _model_path("models/doctype/v3/model.joblib", "models/doctype/v3/model.joblib"))

_KEYFIELD = {"receipt": "total_amount", "bank_statement": "closing_balance",
             "payment_order": "amount"}


def _run(image_bgr, name="doc"):
    ok, enc = cv2.imencode(".jpg", image_bgr)
    return pipeline.process_document(name, enc.tobytes()).model_dump()


def infer(image):
    if image is None:
        return "{}", "Upload a document image first."
    d = _run(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    rows = "\n".join(f"| {k} | {v['value']} | {v['confidence']} |" for k, v in d["fields"].items())
    tx = d.get("line_items") or []
    tx_note = f"\n\n**Transactions parsed:** {len(tx)} rows" if tx else ""
    summary = (f"### Document type: `{d['document_type']}` · Route: `{d['route']}` · "
               f"needs_human_review: **{d['needs_human_review']}**\n\n"
               f"| field | value | confidence |\n|---|---|---|\n{rows}{tx_note}\n\n"
               f"**Quality:** blur={d['quality']['blur_score']} · models `{d['model_versions']}`")
    return json.dumps(d, ensure_ascii=False, indent=2), summary


def batch_infer(files):
    """Process 5–10 documents at once (production-style batch demo)."""
    if not files:
        return [], {"error": "upload 1–10 images"}
    rows, counts = [], {}
    n_review = n_vlm = 0
    for fp in files[:10]:
        img = cv2.imread(fp)
        if img is None:
            continue
        d = _run(img, os.path.basename(fp))
        t = d["document_type"]; counts[t] = counts.get(t, 0) + 1
        n_review += int(d["needs_human_review"]); n_vlm += int(d["route"] == "vlm_fallback")
        kf = _KEYFIELD.get(t)
        kv = d["fields"].get(kf, {}).get("value") if kf else None
        rows.append([os.path.basename(fp), t, d["route"],
                     "✋" if d["needs_human_review"] else "✓",
                     f"{kf}={kv}", len(d.get("line_items") or [])])
    summary = {"total": len(rows), "by_type": counts,
               "needs_human_review": n_review, "vlm_fallback": n_vlm}
    return rows, summary


DESC = (
    "# 🧾 Hybrid Document AI — multi-document OCR + KIE (VNPAY portfolio)\n"
    "Auto-routes **receipt · bank statement · payment order (ủy nhiệm chi)** → extracts fields "
    "(+ transaction table for statements). Low-confidence / unreconciled docs escalate to a "
    "**VLM on Modal**. Code: github.com/NguyenDucAnforwork/hybrid-document-ai")
EX = Path(os.environ.get("DOCAI_EXAMPLES_DIR", WORKSPACE / "examples"))
ex = [[str(p)] for p in sorted(EX.iterdir()) if p.is_file()] if EX.is_dir() else None

with gr.Blocks(title="Hybrid Document AI — OCR + KIE") as demo:
    gr.Markdown(DESC)
    with gr.Tab("Single document"):
        with gr.Row():
            inp = gr.Image(type="numpy", label="Document")
            with gr.Column():
                out_md = gr.Markdown()
                out_json = gr.Code(label="Structured result (JSON)", language="json")
        gr.Button("Extract", variant="primary").click(infer, inp, [out_json, out_md], api_name="predict")
        if ex:
            gr.Examples(ex, inputs=inp)
    with gr.Tab("Batch (5–10 documents)"):
        bfiles = gr.File(file_count="multiple", file_types=["image"], label="Drop 5–10 images")
        btn = gr.Button("Process batch", variant="primary")
        btable = gr.Dataframe(headers=["file", "doc_type", "route", "review", "key_field", "#tx"],
                              label="Per-document results", wrap=True)
        bsum = gr.JSON(label="Batch summary")
        btn.click(batch_infer, bfiles, [btable, bsum], api_name="batch")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7861")),
        share=True,
    )
