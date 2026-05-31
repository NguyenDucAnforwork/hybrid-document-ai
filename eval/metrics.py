"""Stronger evaluation metrics for document KIE.

- CER: character error rate (OCR-style).
- ANLS: Average Normalized Levenshtein Similarity (standard for doc VQA/KIE;
  tolerant of minor OCR noise, thresholded at 0.5).
- field exact-match / F1.
- ECE: Expected Calibration Error — are confidences trustworthy? (critical for a
  confidence-routed system: a miscalibrated router sends wrong docs to humans/VLM).
"""
from __future__ import annotations
import Levenshtein


def _s(x):
    return "" if x is None else str(x)


def cer(pred, gold) -> float:
    g = _s(gold)
    if not g:
        return 0.0 if not _s(pred) else 1.0
    return Levenshtein.distance(_s(pred), g) / len(g)


def norm_sim(pred, gold) -> float:
    p, g = _s(pred), _s(gold)
    if not p and not g:
        return 1.0
    return 1.0 - Levenshtein.distance(p, g) / max(len(p), len(g), 1)


def anls(pred, gold, tau: float = 0.5) -> float:
    sim = norm_sim(pred, gold)
    return sim if sim >= tau else 0.0


def f1(tp, fp, fn) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


def ece(confidences, corrects, n_bins: int = 10) -> float:
    """Expected Calibration Error over confidence bins."""
    if not confidences:
        return 0.0
    n = len(confidences)
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi) or (b == 0 and c == 0)]
        if not idx:
            continue
        conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(corrects[i] for i in idx) / len(idx)
        total += (len(idx) / n) * abs(acc - conf)
    return round(total, 4)
