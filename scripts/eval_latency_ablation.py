"""WP-3 Task B — latency ablation across recognizer configs (MC-OCR full-image).

Configs:
  1. default          RapidOCR only (no FT)
  2. ft_all           FT re-recognizes ALL detector boxes
  3. ft_critical      FT only field-critical boxes (top / date / money / anchor)
  4. auto             route by language (VN -> FT, EN -> default)

Per config: per-field CER, macro CER, p50/p95 run_ocr latency, mean #crops
re-recognized, needs_review rate. Goal: keep macro CER ~0.205 while cutting p50.
"""
from __future__ import annotations
import argparse
import ast
import csv
import datetime as dt
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.metrics import cer as cer_metric             # noqa

FIELDS = ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]


def _ws() -> Path:
    return Path(os.environ.get("DOCAI_WORKSPACE", "/data/nvidia-ai-workspace"))


def _gold(data: Path, limit: int):
    rows = []
    with open(data / "extracted" / "mcocr_train_df.csv", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            try:
                polys = ast.literal_eval(r["anno_polygons"])
            except Exception:
                continue
            texts = (r.get("anno_texts") or "").split("|||")
            labels = (r.get("anno_labels") or "").split("|||")
            regs = []
            for p, t, lb in zip(polys, texts, labels):
                bb = p.get("bbox")
                if bb and lb.strip() in FIELDS:
                    x, y, w, h = bb
                    regs.append({"bbox": [x, y, x + w, y + h], "text": t.strip(), "label": lb.strip()})
            if regs:
                rows.append({"img_id": r["img_id"], "regions": regs})
            if len(rows) >= limit:
                break
    return rows


def _match(region_bbox, tokens):
    x0, y0, x1, y1 = region_bbox
    hit = []
    for t in tokens:
        cx = (t["bbox"][0] + t["bbox"][2]) / 2; cy = (t["bbox"][1] + t["bbox"][3]) / 2
        if x0 - 5 <= cx <= x1 + 5 and y0 - 5 <= cy <= y1 + 5:
            hit.append((t["bbox"][0], t["text"]))
    hit.sort()
    return " ".join(tx for _, tx in hit)


CONFIGS = {
    "default":     {"OCR_RECOGNIZER": "rapidocr_default"},
    "ft_all":      {"OCR_RECOGNIZER": "ppocr_vi_mcocr_ft"},
    "ft_critical": {"OCR_RECOGNIZER": "ppocr_vi_mcocr_ft", "OCR_FIELD_CRITICAL": True},
    "auto":        {"OCR_RECOGNIZER": "auto"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_ws() / "data/processed/mcocr_ocr"))
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--out", default="docs/logs")
    ap.add_argument("--only", default=None, help="run a single config (for concurrency)")
    ap.add_argument("--with-review", action="store_true", help="also run process_document (2x OCR)")
    args = ap.parse_args()
    data = Path(args.data)
    gold = _gold(data, args.limit)
    img_dir = data / "train_images_sample"

    import docai.config as cfg
    from docai import ocr as ocrmod
    from docai.ocr import run_ocr
    if args.with_review:
        from docai.pipeline import process_document

    configs = {args.only: CONFIGS[args.only]} if args.only else CONFIGS
    results = {}
    for name, flags in configs.items():
        cfg.OCR_RECOGNIZER = flags.get("OCR_RECOGNIZER", "rapidocr_default")
        cfg.OCR_FIELD_CRITICAL = flags.get("OCR_FIELD_CRITICAL", False)
        per_field = defaultdict(list)
        lats, rerec, n_review, n = [], [], 0, 0
        for g in gold:
            p = img_dir / g["img_id"]
            img = cv2.imread(str(p))
            if img is None:
                continue
            t0 = time.perf_counter()
            toks = run_ocr(img)
            lats.append((time.perf_counter() - t0) * 1000)
            rerec.append(ocrmod._last_stats.get("rerec", 0))
            for r in g["regions"]:
                per_field[r["label"]].append(cer_metric(_match(r["bbox"], toks), r["text"]))
            if args.with_review:
                doc = process_document(g["img_id"], p.read_bytes())
                n_review += int(doc.needs_human_review)
            n += 1
        lats.sort()
        macro = float(np.mean([c for f in FIELDS for c in per_field[f]])) if per_field else None
        results[name] = {
            "field_cer": {f: round(float(np.mean(per_field[f])), 4) for f in FIELDS if per_field[f]},
            "macro_cer": round(macro, 4) if macro is not None else None,
            "p50_ms": round(lats[len(lats) // 2], 1) if lats else None,
            "p95_ms": round(lats[int(len(lats) * 0.95) - 1], 1) if lats else None,
            "mean_rerec": round(float(np.mean(rerec)), 1) if rerec else 0,
            "needs_review_rate": round(n_review / max(n, 1), 3),
        }
        print(name, json.dumps(results[name]))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.only:                                   # concurrency mode: dump just this config
        (out / f"latency_ablation_{args.only}.json").write_text(json.dumps(results, indent=2))
        print(f"wrote {out}/latency_ablation_{args.only}.json"); return
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    md = [f"# Latency ablation (MC-OCR full-image, n={len(gold)}) {stamp}", "",
          "| config | macro CER | SELLER | ADDRESS | TIMESTAMP | TOTAL_COST | p50 ms | p95 ms | mean #rerec | needs_review |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name in CONFIGS:
        r = results[name]; fc = r["field_cer"]
        md.append(f"| {name} | {r['macro_cer']} | " +
                  " | ".join(str(fc.get(f, '-')) for f in FIELDS) +
                  f" | {r['p50_ms']} | {r['p95_ms']} | {r['mean_rerec']} | {r['needs_review_rate']} |")
    md += ["", "Note: latency p50/p95 are noisy under shared-machine load; the FT path runs full "
           "RapidOCR (det+rec) THEN re-recognizes crops, so det+rec dominates. `auto` on this "
           "all-Vietnamese set behaves like ft_all (it routes VN→FT); its latency win is on "
           "English docs (Task C). `ft_critical` re-recognizes only field-critical boxes."]
    (out / f"latency_ablation_{stamp}.md").write_text("\n".join(md), encoding="utf-8")
    (out / "latency_ablation_raw.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\n".join(md))
    print(f"\nwrote {out / f'latency_ablation_{stamp}.md'}")


if __name__ == "__main__":
    main()
