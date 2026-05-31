"""Central config + paths. Heavy artifacts live on /data to spare /home."""
from __future__ import annotations
import os
from pathlib import Path

# Code repo root (this lives under /home/nvidia-lab/nltk_data as requested)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Heavy-artifact workspace on /data (631GB free) — /home is full.
WORKSPACE = Path(os.environ.get("DOCAI_WORKSPACE", "/data/nvidia-ai-workspace"))
DATA_DIR = Path(os.environ.get("DOCAI_DATA", WORKSPACE / "data"))
MODELS_DIR = Path(os.environ.get("DOCAI_MODELS", WORKSPACE / "models"))
ARTIFACTS_DIR = Path(os.environ.get("DOCAI_ARTIFACTS", WORKSPACE / "artifacts"))
META_DB = Path(os.environ.get("DOCAI_DB", WORKSPACE / "meta.db"))
LOGS_DIR = REPO_ROOT / "docs" / "logs"

for _p in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# Required fields for a receipt; missing any -> needs_human_review.
REQUIRED_FIELDS = ["merchant_name", "date", "total_amount"]
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
