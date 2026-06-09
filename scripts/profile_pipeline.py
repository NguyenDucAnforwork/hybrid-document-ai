"""Stage-level latency profiler — cold start vs warm p50/p95/p99 per stage.

Runs the FULL pipeline in-process (no HTTP) over real/synthetic images and reads
DocumentResult.timings. Reports cold_start_ms (first doc, includes lazy model
load) SEPARATELY from warm percentiles — the first OCR load is slow and would
otherwise pollute the steady-state numbers.

  python scripts/profile_pipeline.py --img-dir $WS/data/sroie/test/images --limit 40
  python scripts/profile_pipeline.py --synthetic 30          # no data needed
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return round(s[k], 1)


def _synth_images(n):
    import tempfile
    from docai.synth import generate
    d = tempfile.mkdtemp(prefix="profile_synth_")
    generate(d, n, 42)
    return sorted((Path(d) / "images").glob("*"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", default="docs/logs")
    args = ap.parse_args()

    from docai.pipeline import process_document

    if args.synthetic:
        imgs = _synth_images(args.synthetic)
    else:
        imgs = sorted(Path(args.img_dir).glob("*.jpg"))[: args.limit]
    if not imgs:
        print("no images"); sys.exit(1)

    per_stage: dict[str, list[float]] = {}
    cold = None
    for i, p in enumerate(imgs):
        res = process_document(p.name, p.read_bytes())
        t = res.timings or {}
        if i == 0:
            cold = t                      # first doc pays lazy model-load cold start
            continue
        for k, v in t.items():
            per_stage.setdefault(k, []).append(v)

    stages = sorted(per_stage, key=lambda k: -(_pct(per_stage[k], 50) or 0))
    report = {
        "n_warm": len(imgs) - 1,
        "cold_start_ms": cold,
        "warm": {k: {"p50": _pct(per_stage[k], 50),
                     "p95": _pct(per_stage[k], 95),
                     "p99": _pct(per_stage[k], 99)} for k in stages},
    }

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    (out / "profile_raw.json").write_text(json.dumps(report, indent=2))
    md = [f"# Pipeline profile {stamp}", "",
          f"- warm docs: {report['n_warm']}",
          f"- **cold start** total: {(cold or {}).get('total')} ms "
          f"(first doc, incl. lazy model load)", "",
          "| stage | warm p50 (ms) | warm p95 | warm p99 |", "|---|---|---|---|"]
    for k in stages:
        s = report["warm"][k]
        md.append(f"| {k} | {s['p50']} | {s['p95']} | {s['p99']} |")
    (out / f"profile_{stamp}.md").write_text("\n".join(md))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out / f'profile_{stamp}.md'}")


if __name__ == "__main__":
    main()
