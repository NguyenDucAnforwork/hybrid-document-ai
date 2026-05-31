"""Train the document-type router (receipt vs bank_statement).

Small logistic regression on global layout/text features. Registered as model
'doctype' in the same registry (stage+lineage) — multi-document MLOps.
"""
from __future__ import annotations
import argparse
import json
import datetime as dt
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from docai.classifier import global_features, DocTypeClassifier  # noqa
from docai.config import MODELS_DIR  # noqa
from docai import registry  # noqa


def _xy(recs, label):
    X = []
    for r in recs:
        toks = r["tokens"]
        W = max((t["bbox"][2] for t in toks), default=1)
        H = max((t["bbox"][3] for t in toks), default=1)
        X.append(global_features(toks, W, H))
    return X, [label] * len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", required=True, nargs="+", help="one or more receipt label dirs")
    ap.add_argument("--statements", required=True, nargs="+")
    ap.add_argument("--payment-orders", nargs="+", default=[])
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    def load_many(dirs):
        recs = []
        for d in dirs:
            recs += json.loads((Path(d) / "labels.json").read_text())
        return recs

    rcpt = load_many(args.receipts)
    stmt = load_many(args.statements)
    po = load_many(args.payment_orders)
    Xr, yr = _xy(rcpt, "receipt")
    Xs, ys = _xy(stmt, "bank_statement")
    Xp, yp = _xy(po, "payment_order")
    X = np.array(Xr + Xs + Xp, float)
    y = np.array(yr + ys + yp)
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(y))
    sp = int(len(y) * 0.8)
    tr, te = idx[:sp], idx[sp:]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(X[tr], y[tr])
    acc = float((clf.predict(X[te]) == y[te]).mean())

    out = MODELS_DIR / "doctype" / args.version
    out.mkdir(parents=True, exist_ok=True)
    DocTypeClassifier(clf=clf, version=args.version).save(out / "model.joblib", args.version)
    metrics = {"test_accuracy": round(acc, 3), "n_receipt": len(rcpt),
               "n_statement": len(stmt), "n_payment_order": len(po)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    registry.register("doctype", args.version, str(out / "model.joblib"), metrics,
                      dt.datetime.now().isoformat(timespec="seconds"),
                      stage="production",
                      lineage={"receipts": args.receipts, "statements": args.statements,
                               "seed": args.seed})
    print(json.dumps(metrics, indent=2))
    print(f"Registered doctype:{args.version} (production) -> {out}")


if __name__ == "__main__":
    main()
