"""Setting B (OCR-only) vs Setting C (OCR + VLM hard-case fallback).

Shows the hybrid actually helps: on docs the confidence router flags as hard,
a real VLM (Qwen2.5-VL) re-extracts and (often) recovers fields the traditional
pipeline misses — at the cost of latency, so it runs ONLY on hard cases.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import datetime as dt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.metrics import anls, f1  # noqa
from docai.config import REQUIRED_FIELDS  # noqa

FIELDS = ["merchant_name", "date", "total_amount"]


def _maybe_degrade(img_bytes, degrade, severity, seed):
    if not degrade:
        return img_bytes
    import cv2, numpy as np
    from docai.augment import DEGRADATIONS
    arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    arr = DEGRADATIONS[degrade](arr, severity, seed)
    return cv2.imencode(".jpg", arr)[1].tobytes()


def evaluate(records, img_dir, degrade=None, severity=0.5):
    from docai import pipeline
    agg = {f: {"anls": [], "tp": 0, "fp": 0, "fn": 0} for f in FIELDS}
    vlm_used = 0
    lat = []
    for i, r in enumerate(records):
        raw = _maybe_degrade((img_dir / r["image"]).read_bytes(), degrade, severity, i)
        t0 = time.perf_counter()
        res = pipeline.process_document(r["image"], raw)
        lat.append(time.perf_counter() - t0)
        vlm_used += int(res.route == "vlm_fallback")
        for f in FIELDS:
            g = r["gold"].get(f)
            if g is None:
                continue
            p = res.fields[f].value
            agg[f]["anls"].append(anls(p, g))
            if p == g:
                agg[f]["tp"] += 1
            else:
                agg[f]["fn"] += 1
                if p is not None:
                    agg[f]["fp"] += 1
    out = {f: {"anls": round(sum(a["anls"]) / max(len(a["anls"]), 1), 3),
               "f1": f1(a["tp"], a["fp"], a["fn"])} for f, a in agg.items()}
    out["_vlm_used"] = vlm_used
    out["_mean_latency_s"] = round(sum(lat) / max(len(lat), 1), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/nvidia-ai-workspace/data/sroie/test")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--degrade", default=None, help="apply a degradation (e.g. blur) to create hard cases")
    ap.add_argument("--severity", type=float, default=0.45)
    ap.add_argument("--vlm-mode", default="local", choices=["local", "api"],
                    help="Setting C backend: local GPU/CPU, or api (Modal/managed via VLM_API_BASE)")
    ap.add_argument("--out", default="docs/logs")
    args = ap.parse_args()
    data = Path(args.data)
    records = json.loads((data / "labels.json").read_text())[:args.limit]
    img_dir = data / "images"

    os.environ["DOCAI_VLM_MODE"] = "disabled"
    print(f"Setting B (OCR-only), degrade={args.degrade} ...", flush=True)
    B = evaluate(records, img_dir, args.degrade, args.severity)
    os.environ["DOCAI_VLM_MODE"] = args.vlm_mode      # local | api (Modal/managed)
    if args.vlm_mode == "local":
        os.environ.setdefault("DOCAI_VLM_DEVICE", "cpu")
    print(f"Setting C (OCR + VLM hard-case, mode={args.vlm_mode}) ...", flush=True)
    C = evaluate(records, img_dir, args.degrade, args.severity)

    res = {"n": len(records), "setting_B_ocr_only": B, "setting_C_ocr_plus_vlm": C}
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "vlm_compare_raw.json").write_text(json.dumps(res, indent=2))
    md = [f"# OCR-only vs OCR+VLM (real SROIE, n={len(records)}) {stamp}", "",
          f"VLM used on {C['_vlm_used']}/{len(records)} hard cases. "
          f"Mean latency: B={B['_mean_latency_s']}s, C={C['_mean_latency_s']}s.", "",
          "| field | B ANLS | C ANLS | B F1 | C F1 |", "|---|---|---|---|---|"]
    for f in FIELDS:
        md.append(f"| {f} | {B[f]['anls']} | {C[f]['anls']} | {B[f]['f1']} | {C[f]['f1']} |")
    (Path(args.out) / f"vlm_compare_{stamp}.md").write_text("\n".join(md))
    print(json.dumps(res, indent=2))
    print("\nwrote", Path(args.out) / f"vlm_compare_{stamp}.md")


if __name__ == "__main__":
    main()
