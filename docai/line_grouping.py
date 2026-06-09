"""Line-grouping / reading-order post-processing (WP-3 Task B).

Detector error analysis (docs/logs/detector_mcocr_*.md) showed the full-image gap is
NOT detection (recall 0.978) but OVERMERGE (one box spanning >=2 fields) — e.g.
`Ngày: 11/08/2020 08:06 Tổng tiền: 74,000` returned as one box. This module:

  1. sorts tokens into reading order (y-center, then x-center);
  2. splits a token whose text contains >=2 DIFFERENT field anchors (a horizontal
     two-field merge) at the second anchor, allocating bbox-x proportionally.

Flag-gated (DOCAI_LINE_REGROUP). Honest scope: this fixes HORIZONTAL anchor merges
(the "Ngày…Tổng tiền" case). It does NOT split VERTICAL multi-line ADDRESS merges —
that needs in-box horizontal-projection splitting (documented as next step).
"""
from __future__ import annotations
import re

import cv2
import numpy as np

from .kie import ANCHORS

# anchor keyword -> field-type, for detecting two-field horizontal merges
_ANCHOR_FIELD = []
for _f in ("date", "total_amount", "payment_method", "invoice_id"):
    for _kw in ANCHORS.get(_f, []):
        if _kw:
            _ANCHOR_FIELD.append((_kw, _f))
_ANCHOR_FIELD.sort(key=lambda kv: -len(kv[0]))   # match longer anchors first


def _anchor_hits(text_low: str):
    """Return [(char_idx, field)] for anchor keywords found, earliest per field-type."""
    hits = []
    seen = set()
    for kw, field in _ANCHOR_FIELD:
        i = text_low.find(kw)
        if i >= 0 and field not in seen:
            hits.append((i, field))
            seen.add(field)
    return sorted(hits)


def split_merged_token(tok: dict) -> list[dict]:
    text = tok.get("text", "") or ""
    if len(text) < 8:
        return [tok]
    hits = _anchor_hits(text.lower())
    # need >=2 different fields, and the 2nd anchor not at the very start
    cut_points = [i for i, _ in hits if i > 3]
    if len({f for _, f in hits}) < 2 or len(cut_points) < 1:
        return [tok]
    x0, y0, x1, y1 = tok["bbox"]
    width = max(x1 - x0, 1)
    L = max(len(text), 1)
    bounds = [0] + sorted(set(cut_points)) + [L]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = text[a:b].strip()
        if not seg:
            continue
        sx0 = x0 + width * (a / L)
        sx1 = x0 + width * (b / L)
        out.append({"text": seg, "bbox": [sx0, y0, sx1, y1], "conf": tok.get("conf", 0.0)})
    return out or [tok]


def projection_split_box(image_bgr, bbox, min_band_h: int = 8, gap_frac: float = 0.12):
    """Split a tall over-merged box into per-row sub-boxes via horizontal projection.

    WP-3 Task E: the real fix for ADDRESS vertical over-merge. Crop the box, binarize
    (ink=white), sum ink per row, find text bands separated by low-ink valleys, and
    emit one axis-aligned sub-box per band. Returns None if <2 bands (no split).
    """
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    H, W = image_bgr.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    crop = image_bgr[y0:y1, x0:x1]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    rowsum = th.sum(axis=1).astype(float) / 255.0      # ink pixels per row
    if rowsum.max() <= 0:
        return None
    text = rowsum >= gap_frac * rowsum.max()
    bands, s = [], None
    for i, v in enumerate(text):
        if v and s is None:
            s = i
        elif not v and s is not None:
            bands.append((s, i)); s = None
    if s is not None:
        bands.append((s, len(text)))
    bands = [(a, b) for a, b in bands if b - a >= min_band_h]
    if len(bands) < 2:
        return None
    return [[x0, y0 + a, x1, y0 + b] for a, b in bands]   # full-width per-row sub-boxes


def regroup_tokens(tokens: list[dict]) -> list[dict]:
    """Reading-order sort + split horizontal two-field merges."""
    if not tokens:
        return tokens
    ordered = sorted(tokens, key=lambda t: ((t["bbox"][1] + t["bbox"][3]) / 2,
                                            (t["bbox"][0] + t["bbox"][2]) / 2))
    out = []
    for t in ordered:
        out.extend(split_merged_token(t))
    return out
