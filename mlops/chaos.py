"""Chaos engineering (MLOps - More and beyond).

Inject failures/out-of-distribution inputs and assert the system degrades
GRACEFULLY (no crash; bad inputs -> needs_human_review, never silent wrong data).
"""
from __future__ import annotations
import numpy as np
import cv2
from docai.pipeline import process_document


def _png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def run_chaos() -> dict:
    cases = {}

    # 1. Out-of-distribution: pure noise (not a document)
    noise = np.random.randint(0, 255, (600, 600, 3), np.uint8)
    r = process_document("ood_noise", _png(noise))
    cases["ood_noise"] = {"ok": r.needs_human_review, "route": r.route}

    # 2. Blank page
    blank = np.full((600, 600, 3), 255, np.uint8)
    r = process_document("blank", _png(blank))
    cases["blank"] = {"ok": r.needs_human_review}

    # 3. Corrupted bytes -> must raise cleanly, not hang
    try:
        process_document("corrupt", b"\x00\x01not-an-image")
        cases["corrupt"] = {"ok": False, "note": "should have raised"}
    except Exception as e:
        cases["corrupt"] = {"ok": True, "raised": type(e).__name__}

    # 4. Tiny image
    tiny = np.random.randint(0, 255, (12, 12, 3), np.uint8)
    try:
        r = process_document("tiny", _png(tiny))
        cases["tiny"] = {"ok": True, "needs_review": r.needs_human_review}
    except Exception as e:
        cases["tiny"] = {"ok": False, "raised": str(e)}

    passed = all(c.get("ok") for c in cases.values())
    return {"passed": passed, "cases": cases}


if __name__ == "__main__":
    import json
    print(json.dumps(run_chaos(), indent=2))
