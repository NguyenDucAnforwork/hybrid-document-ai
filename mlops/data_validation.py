"""Data validation step (MLOps Intro: data verification before training).

Great-expectations-style checks: schema presence, value sanity, distribution.
A training pipeline must GATE on data quality — bad data silently degrades models.
"""
from __future__ import annotations
import json
from pathlib import Path
from docai.config import REQUIRED_FIELDS


def validate(labels_path: str | Path) -> dict:
    recs = json.loads(Path(labels_path).read_text())
    n = len(recs)
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    add("non_empty", n > 0, f"{n} records")
    # gold field coverage
    cov = {f: sum(1 for r in recs if r["gold"].get(f) is not None) / max(n, 1)
           for f in REQUIRED_FIELDS}
    for f, c in cov.items():
        add(f"coverage:{f}", c >= 0.5, f"{c:.0%} have gold")
    # token sanity (records with OCR tokens)
    have_tokens = sum(1 for r in recs if r.get("tokens"))
    add("has_tokens", have_tokens >= 0.5 * n, f"{have_tokens}/{n} have tokens")
    # total_amount positivity
    totals = [r["gold"]["total_amount"] for r in recs
              if r["gold"].get("total_amount") is not None]
    add("total_positive", all(t > 0 for t in totals) if totals else False,
        f"{len(totals)} totals, min={min(totals) if totals else None}")

    passed = all(c["pass"] for c in checks)
    return {"passed": passed, "n": n, "coverage": cov, "checks": checks}


if __name__ == "__main__":
    import sys
    print(json.dumps(validate(sys.argv[1]), indent=2))
