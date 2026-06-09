"""Optional fine-tuned OCR recognizer adapter (WP-3).

Loads the MC-OCR fine-tuned CRNN ONNX + its Vietnamese charset and runs greedy
CTC decode. Owns its own dictionary (RapidOCR's onnxruntime build has no
rec_keys_path override — its dict is hardcoded Chinese), so this is a standalone
recognizer, not a RapidOCR rec swap. Pure onnxruntime + numpy: no torch/paddle at
runtime, runs on CPU / RTX 1650.
"""
from __future__ import annotations
import os
from pathlib import Path

import cv2
import numpy as np

IMG_H, IMG_W = 32, 256
_inst = None
_load_error: str | None = None


class FineTunedRecognizer:
    def __init__(self, session, chars: list[str]):
        self.session = session
        self.chars = chars                       # idx0 = blank
        self.itos = {i + 1: c for i, c in enumerate(chars)}
        self.input_name = session.get_inputs()[0].name
        self.version = "vi_mcocr_crnn_ft"

    @classmethod
    def load(cls, onnx_path: str | None = None, dict_path: str | None = None):
        import onnxruntime as ort
        from .config import MODELS_DIR
        onnx_path = Path(onnx_path or os.environ.get(
            "DOCAI_OCR_REC_MODEL", MODELS_DIR / "ocr/vi_mcocr_crnn_ft/model.onnx"))
        dict_path = Path(dict_path or os.environ.get(
            "DOCAI_OCR_REC_DICT", MODELS_DIR / "ocr/vi_mcocr_crnn_ft/vi_dict.txt"))
        if not onnx_path.exists():
            raise FileNotFoundError(f"missing recognizer ONNX: {onnx_path}")
        if not dict_path.exists():
            raise FileNotFoundError(f"missing recognizer dict: {dict_path}")
        chars = [c for c in dict_path.read_text(encoding="utf-8").split("\n") if c != ""]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                    providers=["CPUExecutionProvider"])
        return cls(sess, chars)

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
        h, w = g.shape[:2]
        new_w = max(1, min(IMG_W, int(round(w * IMG_H / max(h, 1)))))
        r = cv2.resize(g, (new_w, IMG_H), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((IMG_H, IMG_W), np.uint8)
        canvas[:, :new_w] = r
        x = (canvas.astype(np.float32) / 255.0 - 0.5) / 0.5
        return x[None, :, :]

    def _decode(self, logits_tb_c: np.ndarray) -> tuple[str, float]:
        # logits: (T, C) for one sample
        probs = _softmax(logits_tb_c)
        ids = probs.argmax(1).tolist()
        confs = probs.max(1)
        out, prev, kept = [], 0, []
        for i, idx in enumerate(ids):
            if idx != prev and idx != 0:
                out.append(self.itos.get(idx, ""))
                kept.append(confs[i])
            prev = idx
        text = "".join(out)
        conf = float(np.mean(kept)) if kept else 0.0
        return text, conf

    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        if not crops:
            return []
        batch = np.stack([self._preprocess(c) for c in crops]).astype(np.float32)
        logits = self.session.run(None, {self.input_name: batch})[0]   # (T, B, C)
        results = []
        for b in range(logits.shape[1]):
            results.append(self._decode(logits[:, b, :]))
        return results


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def get_recognizer() -> "FineTunedRecognizer | None":
    global _inst, _load_error
    if _inst is not None:
        return _inst
    if _load_error is not None:
        return None
    try:
        _inst = FineTunedRecognizer.load()
    except Exception as exc:
        _load_error = str(exc)
        return None
    return _inst
