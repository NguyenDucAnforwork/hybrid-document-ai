"""Quality check (Processing Layer): blur/brightness/resolution/skew."""
from __future__ import annotations
import cv2
import numpy as np
from .schemas import QualityReport
from .config import BLUR_THRESHOLD, DARK_THRESHOLD, MIN_DIM


def check_quality(image_bgr: np.ndarray) -> QualityReport:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    is_blurry = blur < BLUR_THRESHOLD
    is_dark = brightness < DARK_THRESHOLD
    low_res = w < MIN_DIM or h < MIN_DIM

    # Skew estimate via minAreaRect on thresholded foreground.
    skew = 0.0
    try:
        th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(th > 0))
        if len(coords) > 50:
            angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
            skew = angle - 90 if angle > 45 else angle
    except Exception:
        skew = 0.0
    is_rotated = abs(skew) > 5

    issues = []
    if is_blurry:
        issues.append("blurry_image")
    if is_dark:
        issues.append("dark_image")
    if low_res:
        issues.append("low_resolution")
    if is_rotated:
        issues.append("rotated_image")

    # low_res is a FLAG, not a hard blocker: real bank docs are often small
    # scans — we still attempt OCR (with upscaling) and let the confidence
    # router decide human review. Only severe blur fails the gate.
    quality_pass = not is_blurry
    return QualityReport(
        blur_score=round(blur, 2), is_blurry=is_blurry, is_dark=is_dark,
        low_resolution=low_res, is_rotated=is_rotated, quality_pass=quality_pass,
        issues=issues, action=None if quality_pass else "request_reupload",
    )
