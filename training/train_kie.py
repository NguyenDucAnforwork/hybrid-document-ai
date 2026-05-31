"""MLOps training pipeline: train the KIE field classifier on labeled receipts.

data (synthetic/SROIE) -> features -> fit sklearn -> eval -> register version.
Reproducible (fixed seed). Inference-first: this is the ONLY trained component.
"""
from __future__ import annotations
import argparse
import json
import datetime as dt
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from docai.kie import token_features, candidates, norm_field, KIEModel  # noqa
from docai.config import ALL_FIELDS, MODELS_DIR  # noqa
from docai import registry  # noqa


def build_xy(records):
    X, y = [], []
    for r in records:
        toks, W, H, mm, mh = candidates(r["tokens"])
        for fi, field in enumerate(ALL_FIELDS):
            gold = r["gold"].get(field)
            for t in toks:
                nv = norm_field(field, t["text"])
                if nv is None:
                    continue
                X.append(token_features(t, fi, toks, W, H, mm, mh))
                y.append(1 if (gold is not None and nv == gold) else 0)
    return np.array(X, float), np.array(y, int)


def field_exact_match(model: KIEModel, records):
    """Field-level exact-match accuracy using the full extract() path."""
    hits = {f: 0 for f in ALL_FIELDS}
    tot = {f: 0 for f in ALL_FIELDS}
    for r in records:
        pred = model.extract(r["tokens"])
        for f in ALL_FIELDS:
            g = r["gold"].get(f)
            if g is None:
                continue
            tot[f] += 1
            if pred[f][0] == g:
                hits[f] += 1
    per = {f: round(hits[f] / tot[f], 3) if tot[f] else None for f in ALL_FIELDS}
    macro = round(np.mean([v for v in per.values() if v is not None]), 3)
    return per, macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with labels.json")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = json.loads((Path(args.data) / "labels.json").read_text())
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(records))
    split = int(len(records) * 0.8)
    train = [records[i] for i in idx[:split]]
    test = [records[i] for i in idx[split:]]

    Xtr, ytr = build_xy(train)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(Xtr, ytr)

    model = KIEModel(clf=clf, version=args.version)
    per_train, macro_train = field_exact_match(model, train)
    per_test, macro_test = field_exact_match(model, test)

    out = MODELS_DIR / "kie" / args.version
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.joblib", args.version)
    metrics = {
        "n_train": len(train), "n_test": len(test),
        "n_examples": int(len(ytr)), "pos_rate": round(float(ytr.mean()), 3),
        "field_exact_match_train": per_train, "macro_train": macro_train,
        "field_exact_match_test": per_test, "macro_test": macro_test,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    registry.register("kie", args.version, str(out / "model.joblib"), metrics,
                      dt.datetime.now().isoformat(timespec="seconds"))
    print(json.dumps(metrics, indent=2))
    print(f"\nRegistered kie:{args.version} (active). Saved -> {out}")


if __name__ == "__main__":
    main()
