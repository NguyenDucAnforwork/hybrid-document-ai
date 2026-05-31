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
MIN_FIELD_CONFIDENCE = float(os.environ.get("DOCAI_MIN_CONF", "0.75"))

# Quality thresholds.
BLUR_THRESHOLD = 100.0
DARK_THRESHOLD = 50.0
MIN_DIM = 480  # synthetic images are small; keep modest

# HuggingFace dataset/model repos (namespace resolved at runtime).
HF_USER = os.environ.get("HF_USER", "banhchungtuongot")
HF_MODEL_REPO = f"{HF_USER}/hybrid-docai-kie"
HF_DATASET_REPO = f"{HF_USER}/hybrid-docai-receipts"
