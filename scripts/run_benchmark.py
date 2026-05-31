"""Benchmark + eval-as-CI-gate (MLOps).

Runs the FULL pipeline (OCR -> KIE -> router) on rendered images and compares
to gold. Reports field exact-match/F1 + latency. Exits non-zero if macro-F1 <
threshold -> usable as a CI gate.
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
from docai.pipeline import process_document  # noqa
from docai.config import ALL_FIELDS, REQUIRED_FIELDS  # noqa
from eval.metrics import f1, anls as anls_metric, cer as cer_metric  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="docs/logs")
    ap.add_argument("--f1-threshold", type=float, default=0.6)
    args = ap.parse_args()

    data = Path(args.data)
    records = json.loads((data / "labels.json").read_text())
    stats = {f: {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "tot": 0, "anls": [], "cer": []}
             for f in ALL_FIELDS}
    lat = []
    raw = []
    all_correct = 0

    for r in records:
        img_path = data / "images" / r["image"]
        t0 = time.perf_counter()
        res = process_document(r["image"], img_path.read_bytes())
        lat.append((time.perf_counter() - t0) * 1000)
        rec = {"image": r["image"], "route": res.route, "fields": {}}
        req_ok = True
        for f in ALL_FIELDS:
            gold = r["gold"].get(f)
            pred = res.fields[f].value
            stats[f]["tot"] += 1 if gold is not None else 0
            match = (pred == gold)
            if gold is not None:
                stats[f]["anls"].append(anls_metric(pred, gold))
                stats[f]["cer"].append(cer_metric(pred, gold))
            if gold is not None and match:
                stats[f]["tp"] += 1
                stats[f]["exact"] += 1
            elif gold is not None and not match:
                stats[f]["fn"] += 1
                if pred is not None:
                    stats[f]["fp"] += 1
            if f in REQUIRED_FIELDS and not match:
                req_ok = False
            rec["fields"][f] = {"pred": pred, "gold": gold, "ok": match}
        all_correct += int(req_ok)
        raw.append(rec)

    per_f1 = {f: f1(s["tp"], s["fp"], s["fn"]) for f, s in stats.items()}
    per_em = {f: round(s["exact"] / s["tot"], 3) if s["tot"] else None
              for f, s in stats.items()}
    per_anls = {f: round(sum(s["anls"]) / len(s["anls"]), 3) if s["anls"] else None
                for f, s in stats.items()}
    per_cer = {f: round(sum(s["cer"]) / len(s["cer"]), 3) if s["cer"] else None
               for f, s in stats.items()}
    scored = [f for f in ALL_FIELDS if stats[f]["tot"]]
    macro_f1 = round(sum(per_f1[f] for f in scored) / max(len(scored), 1), 3)
    macro_anls = round(sum(per_anls[f] for f in scored) / max(len(scored), 1), 3)
    lat.sort()
    p50 = round(lat[len(lat) // 2], 1)
    p95 = round(lat[int(len(lat) * 0.95) - 1], 1)

    summary = {
        "n": len(records), "macro_f1": macro_f1, "macro_anls": macro_anls,
        "field_f1": per_f1, "field_exact_match": per_em,
        "field_anls": per_anls, "field_cer": per_cer, "scored_fields": scored,
        "all_required_correct_rate": round(all_correct / len(records), 3),
        "latency_ms_p50": p50, "latency_ms_p95": p95,
        "threshold": args.f1_threshold,
        "gate_pass": macro_f1 >= args.f1_threshold,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_raw.json").write_text(json.dumps({"summary": summary, "docs": raw}, indent=2))
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    md = [f"# Benchmark {stamp}", "", f"- N={summary['n']}  macro-F1={macro_f1}  "
          f"macro-ANLS={macro_anls}  gate(thr={args.f1_threshold})="
          f"{'PASS' if summary['gate_pass'] else 'FAIL'}",
          f"- scored fields={scored}  all-required-correct="
          f"{summary['all_required_correct_rate']}  latency p50={p50}ms p95={p95}ms",
          "", "| field | F1 | exact_match | ANLS | CER |", "|---|---|---|---|---|"]
    for f in ALL_FIELDS:
        md.append(f"| {f} | {per_f1[f]} | {per_em[f]} | {per_anls[f]} | {per_cer[f]} |")
    (out / f"benchmark_{stamp}.md").write_text("\n".join(md))
    print(json.dumps(summary, indent=2))
    sys.exit(0 if summary["gate_pass"] else 1)


if __name__ == "__main__":
    main()
