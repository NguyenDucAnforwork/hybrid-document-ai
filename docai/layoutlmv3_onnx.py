"""Optional LayoutLMv3 ONNX inference for receipt KIE refinement."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .config import WORKSPACE
from .kie import norm_field

LABEL2ID = {
    "O": 0,
    "B-MERCHANT": 1,
    "I-MERCHANT": 2,
    "B-DATE": 3,
    "I-DATE": 4,
    "B-TOTAL": 5,
    "I-TOTAL": 6,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
ENTITY_MAP = {
    "MERCHANT": "merchant_name",
    "DATE": "date",
    "TOTAL": "total_amount",
}

_predictor: LayoutLMv3ONNXPredictor | None = None
_load_error: str | None = None


class LayoutLMv3ONNXPredictor:
    def __init__(self, model_dir: Path, onnx_path: Path, processor, session):
        self.model_dir = model_dir
        self.onnx_path = onnx_path
        self.processor = processor
        self.session = session
        self.version = f"layoutlmv3-onnx:{onnx_path.name}"

    @classmethod
    def load(cls, model_dir: str | Path | None = None,
             onnx_path: str | Path | None = None) -> "LayoutLMv3ONNXPredictor":
        from transformers import AutoProcessor
        import onnxruntime as ort

        model_dir = Path(model_dir or os.environ.get(
            "DOCAI_LAYOUTLMV3_DIR",
            WORKSPACE / "models" / "layoutlmv3" / "model",
        ))
        onnx_path = Path(onnx_path or os.environ.get(
            "DOCAI_LAYOUTLMV3_ONNX_PATH",
            WORKSPACE / "models" / "layoutlmv3" / "model_fp32.onnx",
        ))
        if not model_dir.exists():
            raise FileNotFoundError(f"missing LayoutLMv3 processor dir: {model_dir}")
        if not onnx_path.exists():
            raise FileNotFoundError(f"missing LayoutLMv3 ONNX file: {onnx_path}")

        processor = AutoProcessor.from_pretrained(str(model_dir), apply_ocr=False)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        return cls(model_dir=model_dir, onnx_path=onnx_path, processor=processor, session=session)

    def _normalize_boxes(self, tokens: list[dict], width: int, height: int) -> list[list[int]]:
        width = max(int(width), 1)
        height = max(int(height), 1)
        boxes = []
        for t in tokens:
            x0, y0, x1, y1 = t.get("bbox", [0, 0, 1, 1])
            boxes.append([
                max(0, min(1000, int(x0 / width * 1000))),
                max(0, min(1000, int(y0 / height * 1000))),
                max(0, min(1000, int(x1 / width * 1000))),
                max(0, min(1000, int(y1 / height * 1000))),
            ])
        return boxes

    def predict(self, image_bgr: np.ndarray, tokens: list[dict]) -> dict[str, str | float | None]:
        if image_bgr is None or not tokens:
            return {}

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        words = [t.get("text", "") or "" for t in tokens]
        boxes = self._normalize_boxes(tokens, width, height)

        encoding = self.processor(
            image_rgb,
            words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="np",
        )
        feed = {
            "input_ids": encoding["input_ids"].astype(np.int64),
            "attention_mask": encoding["attention_mask"].astype(np.int64),
            "bbox": encoding["bbox"].astype(np.int64),
            "pixel_values": encoding["pixel_values"].astype(np.float32),
        }
        logits = self.session.run(["logits"], feed)[0]
        preds = logits.argmax(axis=-1)[0].tolist()

        entity_tokens: dict[str, list[str]] = {f: [] for f in ENTITY_MAP.values()}
        word_ids = encoding.word_ids(0)
        prev_word = None
        for idx, wid in enumerate(word_ids):
            if wid is None or wid == prev_word:
                continue
            prev_word = wid
            label = ID2LABEL.get(preds[idx], "O")
            for ent, field in ENTITY_MAP.items():
                if ent in label and wid < len(words):
                    entity_tokens[field].append(words[wid])

        raw = {f: " ".join(v) if v else None for f, v in entity_tokens.items()}
        return {f: norm_field(f, v) if v is not None else None for f, v in raw.items()}


def onnx_mode() -> str:
    return os.environ.get("DOCAI_LAYOUTLMV3_MODE", "disabled").strip().lower()


def get_layoutlmv3_onnx() -> LayoutLMv3ONNXPredictor | None:
    global _predictor, _load_error
    if onnx_mode() == "disabled":
        return None
    if _predictor is not None:
        return _predictor
    if _load_error is not None:
        return None
    try:
        _predictor = LayoutLMv3ONNXPredictor.load()
    except Exception as exc:
        _load_error = str(exc)
        return None
    return _predictor


def load_error() -> str | None:
    return _load_error
