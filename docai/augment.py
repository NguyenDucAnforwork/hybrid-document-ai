"""Banking-realistic image degradations (data layer).

Real bank documents are dark, skewed, torn, blurred, faded, low-res, noisy,
photographed at an angle. We apply controlled degradations so we can measure a
ROBUSTNESS CURVE (accuracy vs severity per degradation) instead of a single
clean number. Severity in [0,1].
"""
from __future__ import annotations
import cv2
import numpy as np


def _rng(seed):
    return np.random.RandomState(seed)


def darken(img, s, seed=0):
    return np.clip(img * (1.0 - 0.7 * s), 0, 255).astype(np.uint8)


def low_contrast(img, s, seed=0):
    m = img.mean()
    return np.clip((img - m) * (1.0 - 0.8 * s) + m, 0, 255).astype(np.uint8)


def gaussian_blur(img, s, seed=0):
    k = int(1 + 2 * round(6 * s))           # odd kernel
    return cv2.GaussianBlur(img, (k, k), 0) if k > 1 else img


def motion_blur(img, s, seed=0):
    k = max(3, int(2 * round(10 * s) + 1))
    kern = np.zeros((k, k)); kern[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kern)


def rotate(img, s, seed=0):
    ang = (_rng(seed).rand() * 2 - 1) * 12 * s     # up to ~12 deg
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))


def perspective(img, s, seed=0):
    h, w = img.shape[:2]; r = _rng(seed); d = 0.18 * s
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (src + r.uniform(-d, d, src.shape) * [w, h]).astype(np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                               borderValue=(255, 255, 255))


def low_res(img, s, seed=0):
    f = 1.0 - 0.7 * s
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def jpeg(img, s, seed=0):
    q = int(90 - 80 * s)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, max(5, q)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def noise(img, s, seed=0):
    n = _rng(seed).normal(0, 60 * s, img.shape)
    return np.clip(img + n, 0, 255).astype(np.uint8)


def tear(img, s, seed=0):
    """Occlusion bars simulating torn/stained/folded receipts."""
    out = img.copy(); h, w = img.shape[:2]; r = _rng(seed)
    for _ in range(int(1 + 3 * s)):
        x, y = r.randint(0, w), r.randint(0, h)
        bw, bh = int(w * 0.25 * s) + 5, int(h * 0.06 * s) + 5
        color = int(r.choice([0, 255]))
        out[y:y + bh, x:x + bw] = color
    return out


def fade(img, s, seed=0):
    """Faded ink: blend toward white."""
    white = np.full_like(img, 255)
    return cv2.addWeighted(img, 1 - 0.6 * s, white, 0.6 * s, 0)


DEGRADATIONS = {
    "dark": darken, "low_contrast": low_contrast, "blur": gaussian_blur,
    "motion_blur": motion_blur, "rotate": rotate, "perspective": perspective,
    "low_res": low_res, "jpeg": jpeg, "noise": noise, "tear": tear, "fade": fade,
}


def mixed_hard(img, s, seed=0):
    """Compose several degradations — the worst-case 'photo of a crumpled bill'."""
    r = _rng(seed)
    chain = ["rotate", "perspective", "dark", "blur", "noise", "jpeg"]
    out = img
    for i, name in enumerate(chain):
        if r.rand() < 0.7:
            out = DEGRADATIONS[name](out, s * r.uniform(0.5, 1.0), seed + i)
    return out
