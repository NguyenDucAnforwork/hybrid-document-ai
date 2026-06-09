"""Central config + paths. Heavy artifacts live on /data to spare /home."""
from __future__ import annotations
import os
from pathlib import Path

# Code repo root (this lives under /home/nvidia-lab/nltk_data as requested)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Heavy-artifact workspace. Dev: /data (631GB; /home full). Elsewhere (e.g. HF
# Space): fall back to a writable tmp dir so import never fails on mkdir.
def _default_workspace() -> str:
    env = os.environ.get("DOCAI_WORKSPACE")
    if env:
        return env
    if os.path.isdir("/data") and os.access("/data", os.W_OK):
        return "/data/nvidia-ai-workspace"
    return "/tmp/docai-workspace"


WORKSPACE = Path(_default_workspace())
DATA_DIR = Path(os.environ.get("DOCAI_DATA", WORKSPACE / "data"))
MODELS_DIR = Path(os.environ.get("DOCAI_MODELS", WORKSPACE / "models"))
ARTIFACTS_DIR = Path(os.environ.get("DOCAI_ARTIFACTS", WORKSPACE / "artifacts"))
META_DB = Path(os.environ.get("DOCAI_DB", WORKSPACE / "meta.db"))
LOGS_DIR = REPO_ROOT / "docs" / "logs"

for _p in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR):
    try:
        _p.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# Required fields for a PAYMENT document = what reconciliation needs (date+amount).
# merchant_name is valuable but optional: real OCR reads store names unreliably,
# so making it required would flag ~every doc and the router would lose its
# discriminating power. This is a product decision, documented in lessons-learned.
REQUIRED_FIELDS = ["date", "total_amount"]
ALL_FIELDS = ["merchant_name", "date", "total_amount", "invoice_id", "payment_method"]

# Confidence router thresholds.
# Global fallback; field-specific values override where model calibration differs.
MIN_FIELD_CONFIDENCE = float(os.environ.get("DOCAI_MIN_CONF", "0.75"))

# Per-field thresholds: total_amount uses a higher bar because its ECE ≈ 0.51
# (severely overconfident). date is well-calibrated (ECE ≈ 0.40); merchant is
# optional/noisy so the bar is lower to avoid saturating the router.
# Override individually via env vars without changing code.
FIELD_CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "total_amount": float(os.environ.get("DOCAI_CONF_TOTAL", "0.80")),
    "date":         float(os.environ.get("DOCAI_CONF_DATE",  "0.75")),
    "merchant_name": float(os.environ.get("DOCAI_CONF_MERCHANT", "0.65")),
}

# Quality thresholds.
BLUR_THRESHOLD = 100.0
DARK_THRESHOLD = 50.0
MIN_DIM = 480  # synthetic images are small; keep modest

# OCR recognizer selection (WP-3). Default keeps the production RapidOCR path
# untouched; 'ppocr_vi_mcocr_ft' swaps in the fine-tuned CRNN recognizer adapter
# (detector still RapidOCR). Switchable per-deployment with no code change.
#   rapidocr_default | ppocr_vi_mcocr_ft (force) | auto (route by language, WP-3 Task D)
OCR_RECOGNIZER = os.environ.get("DOCAI_OCR_RECOGNIZER", "rapidocr_default")
# Default -> the Task F detector-augmented model (full-image macro CER 0.265->0.205,
# strictly better than the clean-only v1); override via env for the v1 model.
OCR_REC_MODEL = os.environ.get("DOCAI_OCR_REC_MODEL", str(MODELS_DIR / "ocr/vi_mcocr_crnn_ft_taskf/model.onnx"))
OCR_REC_DICT = os.environ.get("DOCAI_OCR_REC_DICT", str(MODELS_DIR / "ocr/vi_mcocr_crnn_ft_taskf/vi_dict.txt"))
# WP-3 Task D: min Vietnamese-diacritic ratio of default OCR text to route a doc to
# the fine-tuned VI recognizer in 'auto' mode (English/SROIE stays on default).
OCR_VI_DIACRITIC_MIN = float(os.environ.get("DOCAI_OCR_VI_DIACRITIC_MIN", "0.06"))
# WP-3 Task B: split detector boxes that merge two fields horizontally.
LINE_REGROUP = os.environ.get("DOCAI_LINE_REGROUP", "0") == "1"
# WP-3 Task E: split tall over-merged boxes into rows via horizontal projection
# (FT/auto recognizer path only). Triggers when box height > ratio * median height.
PROJECTION_SPLIT = os.environ.get("DOCAI_PROJECTION_SPLIT", "0") == "1"
PROJECTION_SPLIT_RATIO = float(os.environ.get("DOCAI_PROJECTION_SPLIT_RATIO", "1.8"))
# WP-3 Task B (latency): only re-recognize field-critical boxes with the FT model
# (top region / date / money / field anchors); keep default text elsewhere.
OCR_FIELD_CRITICAL = os.environ.get("DOCAI_OCR_FIELD_CRITICAL", "0") == "1"
# WP-3 Task C: flag geometric risk (skew) -> needs_review.
GEOMETRY_RISK_ANGLE = float(os.environ.get("DOCAI_GEOMETRY_RISK_ANGLE", "8.0"))

# HuggingFace dataset/model repos (namespace resolved at runtime).
HF_USER = os.environ.get("HF_USER", "banhchungtuongot")
HF_MODEL_REPO = f"{HF_USER}/hybrid-docai-kie"
HF_DATASET_REPO = f"{HF_USER}/hybrid-docai-receipts"
