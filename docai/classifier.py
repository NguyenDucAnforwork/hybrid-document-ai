"""Document-type router (Processing Layer, front of the pipeline).

A small learned classifier decides receipt vs bank_statement from global layout/
text features, then the pipeline dispatches to the right schema+extractor. This is
how the multi-model pipeline scales across document TYPES, not just fields.
"""
from __future__ import annotations
import re
import joblib
from .registry import active_path

DOCTYPE_VERSION_FALLBACK = "rule-only"
_STMT_KW = ["statement", "sao ke", "sao kê", "balance", "so du", "số dư",
            "account no", "so tai khoan", "số tài khoản", "opening", "closing"]
_RCPT_KW = ["total", "tong cong", "tổng cộng", "invoice", "hoa don", "hóa đơn",
            "thanh tien", "thành tiền", "receipt"]
_PO_KW = ["payment order", "uy nhiem chi", "ủy nhiệm chi", "beneficiary", "nguoi huong",
          "người hưởng", "remitter", "ben chuyen", "to account", "tk huong"]
_MONEY = re.compile(r"\d{1,3}(?:,\d{3})+|\d+\.\d{2}")


def global_features(tokens, W, H):
    txt = " ".join(t["text"].lower() for t in tokens)
    n = max(len(tokens), 1)
    stmt = sum(txt.count(k) for k in _STMT_KW)
    rcpt = sum(txt.count(k) for k in _RCPT_KW)
    po = sum(txt.count(k) for k in _PO_KW)
    money = sum(1 for t in tokens if _MONEY.search(t["text"]))
    # distinct text rows (statements are long, many rows)
    ys = sorted((t["bbox"][1] + t["bbox"][3]) / 2 for t in tokens)
    rows = 1 + sum(1 for a, b in zip(ys, ys[1:]) if b - a > 12)
    return [stmt, rcpt, po, money / n, rows / 20.0, money, H / max(W, 1), n / 50.0]


class DocTypeClassifier:
    def __init__(self, clf=None, version=DOCTYPE_VERSION_FALLBACK):
        self.clf = clf
        self.version = version
        self.classes_ = getattr(clf, "classes_", None)

    @classmethod
    def load(cls, path):
        o = joblib.load(path)
        return cls(clf=o["clf"], version=o["version"])

    def save(self, path, version):
        self.version = version
        joblib.dump({"clf": self.clf, "version": version}, path)

    def predict(self, tokens, W, H) -> tuple[str, float]:
        if self.clf is None:                      # rule fallback (stmt/rcpt/po keyword counts)
            f = global_features(tokens, W, H)
            lab = ["bank_statement", "receipt", "payment_order"][int(max(range(3), key=lambda i: f[i]))]
            return lab, 0.6
        feats = [global_features(tokens, W, H)]
        proba = self.clf.predict_proba(feats)[0]
        i = int(proba.argmax())
        return str(self.clf.classes_[i]), float(proba[i])


_loaded: DocTypeClassifier | None = None


def get_classifier() -> DocTypeClassifier:
    global _loaded
    if _loaded is None:
        p = active_path("doctype")
        _loaded = DocTypeClassifier.load(p) if p and p.exists() else DocTypeClassifier()
    return _loaded
