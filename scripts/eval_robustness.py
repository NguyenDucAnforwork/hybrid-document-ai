"""Robustness evaluation under banking-realistic degradations (real SROIE).

Runs the full pipeline on clean + degraded images and reports, per degradation:
field-F1, CER, ANLS, needs-review-rate, VLM-fallback-rate, mean confidence, ECE.
This is the honest stress test: a single clean number hides production behaviour.
"""
from __future__ import annotations
import argparse
import json
import time
import datetime as dt
from pathlib import Path
import sys

import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from docai.pipeline import process_document      # noqa
from docai.augment import DEGRADATIONS, mixed_hard  # noqa
from eval.metrics import cer, anls, f1, ece        # noqa

SROIE_FIELDS = ["merchant_name", "date", "total_amount"]


def run_case(records, img_dir, transform, sev, seed0):
    stats = {f: {"tp": 0, "fp": 0, "fn": 0} for f in SROIE_FIELDS}
    confs, corrects, cers, anlss = [], [], [], []
    needs_review = fallback = 0
    lat = []
    for k, r in enumerate(records):
        img = cv2.imread(str(img_dir / r["image"]))
        if img is None:
            continue
        if transform is not None:
            try:
                img = transform(img, sev, seed0 + k)
            except Exception:
                continue   # a degradation failing on one image must not kill the run
        ok, enc = cv2.imencode(".jpg", img)
        t0 = time.perf_counter()
        res = process_document(r["image"], enc.tobytes())
        lat.append((time.perf_counter() - t0) * 1000)
        needs_review += int(res.needs_human_review)
        fallback += int(res.route == "vlm_fallback")
        for f in SROIE_FIELDS:
            gold = r["gold"].get(f)
            if gold is None:
                continue
            pred = res.fields[f].value
            conf = res.fields[f].confidence
            match = (pred == gold)
            confs.append(conf); corrects.append(int(match))
            cers.append(cer(pred, gold)); anlss.append(anls(pred, gold))
            if match:
                stats[f]["tp"] += 1
            else:
                stats[f]["fn"] += 1
                if pred is not None:
                    stats[f]["fp"] += 1
    per_f1 = {f: f1(s["tp"], s["fp"], s["fn"]) for f, s in stats.items()}
    macro = round(sum(per_f1.values()) / len(per_f1), 4)
    n = max(len(records), 1)
    return {
        "macro_f1": macro, "field_f1": per_f1,
        "cer": round(sum(cers) / max(len(cers), 1), 4),
        "anls": round(sum(anlss) / max(len(anlss), 1), 4),
        "needs_review_rate": round(needs_review / n, 3),
        "fallback_rate": round(fallback / n, 3),
        "mean_conf": round(sum(confs) / max(len(confs), 1), 3),
        "ece": ece(confs, corrects),
        "latency_p50_ms": round(sorted(lat)[len(lat) // 2], 1) if lat else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/nvidia-ai-workspace/data/sroie/test")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--severity", type=float, default=0.6)
    ap.add_argument("--out", default="docs/logs")
    args = ap.parse_args()

    data = Path(args.data)
    records = json.loads((data / "labels.json").read_text())[:args.limit]
    img_dir = data / "images"

    cases = {"clean": (None, 0.0)}
    for name, fn in DEGRADATIONS.items():
        cases[name] = (fn, args.severity)
    cases["mixed_hard"] = (mixed_hard, args.severity)

    results = {}
    for name, (fn, sev) in cases.items():
        results[name] = run_case(records, img_dir, fn, sev, seed0=1000)
        print(f"{name:14s} F1={results[name]['macro_f1']:.3f} "
              f"CER={results[name]['cer']:.3f} ANLS={results[name]['anls']:.3f} "
              f"review={results[name]['needs_review_rate']:.2f} "
              f"ECE={results[name]['ece']:.3f}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "robustness_raw.json").write_text(json.dumps(
        {"n": len(records), "severity": args.severity, "results": results}, indent=2))
    md = [f"# Robustness eval {stamp} (real SROIE, n={len(records)}, severity={args.severity})",
          "", "| degradation | macro-F1 | CER | ANLS | needs_review | fallback | mean_conf | ECE |",
          "|---|---|---|---|---|---|---|---|"]
    for name, m in results.items():
        md.append(f"| {name} | {m['macro_f1']} | {m['cer']} | {m['anls']} | "
                  f"{m['needs_review_rate']} | {m['fallback_rate']} | {m['mean_conf']} | {m['ece']} |")
    (out / f"robustness_{stamp}.md").write_text("\n".join(md))
    print("\nwrote", out / f"robustness_{stamp}.md")


if __name__ == "__main__":
    main()
