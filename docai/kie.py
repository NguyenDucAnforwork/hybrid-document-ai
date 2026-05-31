"""KIE (Processing Layer) — 2-tier, multi-model.

Tier-1 (this file): candidate generation (regex/keyword/layout-graph) ->
feature vector -> scikit-learn field classifier -> calibrated confidence.
This is a LEARNED model, not pure regex (addresses the production requirement).
Tier-2 (router.py + vlm): VLM OCR-free fallback for hard cases.
"""
from __future__ import annotations
import re
import joblib
from .config import ALL_FIELDS

KIE_VERSION_FALLBACK = "rule-only-baseline"

# ---- anchors (VN + EN) ----------------------------------------------------
ANCHORS = {
    "merchant_name": [],
    "date": ["date", "ngay", "ngày"],
    "total_amount": ["total", "amount", "sum", "tong cong", "tổng cộng",
                     "thanh tien", "thành tiền", "tong", "tổng"],
    "invoice_id": ["invoice", "bill", "receipt", "no", "hoa don", "hóa đơn",
                   "so hd", "số hđ", "ma gd", "mã gd"],
    "payment_method": ["cash", "card", "qr", "visa", "momo", "tien mat",
                       "tiền mặt", "the", "thẻ", "payment", "chuyen khoan"],
}
DATE_RE = re.compile(r"\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{2,4}")
ID_RE = re.compile(r"[A-Z]{1,4}\d{3,}")
# Money: thousands-grouped (VND "235,000"), decimal cents (SROIE "9.00",
# "1,234.56"), or a bare >=3-digit run. Dates are stripped first so "25.12.2018"
# is not parsed as money. Convention: ',' = thousands (drop), '.' = decimal.
MONEY_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+\.\d{2}|\d{3,}")


# ---- shared normalization (MUST match between train / infer / eval) -------
def norm_money(s: str):
    """Return a canonical float amount, or None. Robust across VND/decimal."""
    t = DATE_RE.sub(" ", s or "")            # don't mistake a date for money
    m = MONEY_RE.search(t)
    if not m:
        return None
    raw = m.group().replace(",", "")          # commas are thousands separators
    try:
        return round(float(raw), 2)
    except ValueError:
        return None


def norm_date(s: str):
    m = DATE_RE.search(s or "")
    if not m:
        return None
    parts = re.split(r"[/\-.]", re.sub(r"\s", "", m.group()))
    if len(parts) != 3:
        return None
    d, mth, y = parts
    y = ("20" + y) if len(y) == 2 else y
    try:
        return f"{int(y):04d}-{int(mth):02d}-{int(d):02d}"
    except ValueError:
        return None


def norm_text(s):
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def norm_field(field: str, value: str):
    if field == "total_amount":
        return norm_money(value)
    if field == "date":
        return norm_date(value)
    if field == "payment_method":
        t = norm_text(value)
        for kw in ANCHORS["payment_method"]:
            if kw in t:
                return {"qr": "QR", "card": "CARD", "visa": "CARD", "cash": "CASH",
                        "tien mat": "CASH", "tiền mặt": "CASH", "momo": "MOMO"}.get(kw, kw.upper())
        return None
    if field == "invoice_id":
        m = ID_RE.search((value or "").upper())
        return m.group() if m else None
    return norm_text(value)  # merchant_name


# ---- feature extraction ---------------------------------------------------
def _kw_proximity(token, field, tokens):
    """1 - normalized distance from token to nearest field anchor keyword."""
    anchors = ANCHORS[field]
    if not anchors:
        return 0.0
    tx = (token["bbox"][0] + token["bbox"][2]) / 2
    ty = (token["bbox"][1] + token["bbox"][3]) / 2
    best = 1e9
    for t in tokens:
        low = t["text"].lower()
        if any(a in low for a in anchors):
            ax = (t["bbox"][0] + t["bbox"][2]) / 2
            ay = (t["bbox"][1] + t["bbox"][3]) / 2
            best = min(best, abs(tx - ax) + abs(ty - ay))
    if best > 1e8:
        return 0.0
    return max(0.0, 1.0 - best / 1000.0)


def group_lines(tokens):
    """Layout-graph: merge tokens sharing a text row into one line candidate.

    Fixes the train/serve gap where OCR splits a multi-word title ("ABC MART")
    into separate tokens. Training uses line-level tokens, so grouping makes
    inference candidates consistent with training.
    """
    if not tokens:
        return []
    toks = sorted(tokens, key=lambda t: (t["bbox"][1] + t["bbox"][3]) / 2)
    lines, cur = [], [toks[0]]
    for t in toks[1:]:
        cy = (t["bbox"][1] + t["bbox"][3]) / 2
        ref = cur[-1]
        ref_cy = (ref["bbox"][1] + ref["bbox"][3]) / 2
        ref_h = ref["bbox"][3] - ref["bbox"][1]
        if abs(cy - ref_cy) <= 0.6 * max(ref_h, 1):
            cur.append(t)
        else:
            lines.append(cur)
            cur = [t]
    lines.append(cur)
    merged = []
    for grp in lines:
        grp = sorted(grp, key=lambda t: t["bbox"][0])
        merged.append({
            "text": " ".join(g["text"].strip() for g in grp),
            "bbox": [min(g["bbox"][0] for g in grp), min(g["bbox"][1] for g in grp),
                     max(g["bbox"][2] for g in grp), max(g["bbox"][3] for g in grp)],
            "conf": min(g["conf"] for g in grp),
        })
    return merged


def token_features(token, field_idx, tokens, W, H, max_money, max_height):
    txt = token["text"]
    cx = (token["bbox"][0] + token["bbox"][2]) / 2
    cy = (token["bbox"][1] + token["bbox"][3]) / 2
    height = token["bbox"][3] - token["bbox"][1]
    digits = sum(c.isdigit() for c in txt)
    money_val = norm_money(txt) if MONEY_RE.search(txt) else None
    feats = [
        token["conf"],
        cy / max(H, 1),                       # vertical position (merchant=top, total=bottom)
        cx / max(W, 1),                       # horizontal position
        height / max(H, 1),                   # relative font size
        digits / max(len(txt), 1),            # digit ratio
        1.0 if MONEY_RE.search(txt) else 0.0,
        1.0 if DATE_RE.search(txt) else 0.0,
        min(len(txt), 40) / 40.0,
        1.0 if (money_val is not None and max_money and money_val >= max_money) else 0.0,
        _kw_proximity(token, ALL_FIELDS[field_idx], tokens),
        1.0 if (max_height and height >= max_height - 1e-6) else 0.0,  # is_largest_font -> title/merchant
    ]
    onehot = [0.0] * len(ALL_FIELDS)
    onehot[field_idx] = 1.0
    return feats + onehot


def candidates(tokens):
    """Layout-graph candidate generation: group into lines, then each line is a candidate."""
    tokens = group_lines(tokens)
    money_vals = [norm_money(t["text"]) for t in tokens if MONEY_RE.search(t["text"])]
    max_money = max([m for m in money_vals if m], default=0)
    W = max((t["bbox"][2] for t in tokens), default=1)
    H = max((t["bbox"][3] for t in tokens), default=1)
    max_height = max((t["bbox"][3] - t["bbox"][1] for t in tokens), default=1)
    return tokens, W, H, max_money, max_height


# ---- model wrapper --------------------------------------------------------
class KIEModel:
    def __init__(self, clf=None, version=KIE_VERSION_FALLBACK):
        self.clf = clf
        self.version = version

    @classmethod
    def load(cls, path):
        obj = joblib.load(path)
        return cls(clf=obj["clf"], version=obj["version"])

    def save(self, path, version):
        self.version = version
        joblib.dump({"clf": self.clf, "version": version}, path)

    def _score(self, feats):
        if self.clf is None:
            return None
        return float(self.clf.predict_proba([feats])[0][1])

    def extract(self, tokens) -> dict:
        """Return {field: (value, confidence, route_hint)}."""
        toks, W, H, max_money, max_height = candidates(tokens)
        out = {}
        for fi, field in enumerate(ALL_FIELDS):
            best, best_p = None, -1.0
            for t in toks:
                nv = norm_field(field, t["text"])
                if nv is None:
                    continue
                feats = token_features(t, fi, toks, W, H, max_money, max_height)
                p = self._score(feats)
                # rule-only baseline fallback when no classifier
                if p is None:
                    p = _rule_score(t, field, toks, W, H, max_money)
                if p > best_p:
                    best_p, best = p, (nv, t)
            if best is None:
                out[field] = (None, 0.0)
            else:
                nv, tok = best
                ocr_c = tok["conf"]
                pattern_ok = 1.0
                ens = 0.6 * best_p + 0.4 * (0.5 * ocr_c + 0.5 * pattern_ok)
                out[field] = (nv, round(min(1.0, ens), 3))
        return out


def _rule_score(token, field, tokens, W, H, max_money):
    """Heuristic score used as Setting-A baseline (no learned model)."""
    cy = (token["bbox"][1] + token["bbox"][3]) / 2 / max(H, 1)
    prox = _kw_proximity(token, field, tokens)
    if field == "merchant_name":
        return 1.0 - cy                          # higher = nearer top
    if field == "total_amount":
        mv = norm_money(token["text"]) or 0
        return 0.5 * prox + 0.5 * (1.0 if max_money and mv >= max_money else 0.0)
    return 0.4 + 0.6 * prox