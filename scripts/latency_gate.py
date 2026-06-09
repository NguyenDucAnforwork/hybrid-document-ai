"""SMOKE latency-regression gate for CI.

Deliberately a coarse smoke check, NOT an absolute SLA: GitHub Actions CPU
runners are noisy, so a tight gate would flake. We run a handful of synthetic
images, compare warm total p50 to a committed baseline, and fail only on a
LARGE regression (default +40%). The authoritative latency report is the local
profiler/sweep (profile_pipeline.py / bench_threads.py).

  python scripts/latency_gate.py --update-baseline   # record baseline
  python scripts/latency_gate.py                      # check (exit 1 on big regression)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASELINE = Path("docs/logs/latency_baseline.json")


def _pct(xs, p):
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def measure(n_images: int) -> float:
    from docai.synth import generate
    from docai.pipeline import process_document
    d = tempfile.mkdtemp(prefix="latgate_")
    generate(d, max(n_images + 1, 4), 42)
    imgs = sorted((Path(d) / "images").glob("*"))[: n_images + 1]
    totals = []
    for i, p in enumerate(imgs):
        res = process_document(p.name, p.read_bytes())
        if i == 0:                       # drop cold start
            continue
        totals.append((res.timings or {}).get("total", 0.0))
    return round(_pct(totals, 50), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=4, help="smoke: keep tiny (3-5)")
    ap.add_argument("--tolerance", type=float,
                    default=float(os.environ.get("DOCAI_LAT_TOL", "0.40")))
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()

    warm_p50 = measure(args.images)
    print(f"warm total p50 = {warm_p50} ms (n={args.images}, synthetic)")

    if args.update_baseline or not BASELINE.exists():
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps({"warm_total_p50_ms": warm_p50}, indent=2))
        print(f"baseline {'updated' if args.update_baseline else 'created'}: {BASELINE}")
        sys.exit(0)

    base = json.loads(BASELINE.read_text())["warm_total_p50_ms"]
    limit = base * (1 + args.tolerance)
    print(f"baseline={base} ms  limit(+{int(args.tolerance*100)}%)={limit:.1f} ms")
    if warm_p50 > limit:
        print(f"FAIL: latency regression {warm_p50} > {limit:.1f} ms")
        sys.exit(1)
    print("PASS: no large latency regression")
    sys.exit(0)


if __name__ == "__main__":
    main()
