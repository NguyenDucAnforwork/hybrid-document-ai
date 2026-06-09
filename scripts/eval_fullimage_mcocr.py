"""WP-3 Việc 1 — full-image MC-OCR pipeline eval (det + recognizer + matching).

Production runs: full receipt -> RapidOCR detector -> crops -> recognizer -> line
grouping -> KIE/router. This evaluates the WHOLE chain, comparing:
  default  = RapidOCR detector + RapidOCR default recognizer
  ft       = RapidOCR detector + fine-tuned CRNN recognizer
on full MC-OCR train receipts (the only images with field-level gold).

Per-field CER/ANLS via polygon(bbox) <-> detected-token matching, plus end-to-end
latency p50/p95 and needs_review rate (from process_document), plus failure examples.

CAVEAT: the recognizer was trained on crops from these same train receipts, so the
ft field numbers are OPTIMISTIC (in-domain). The default-vs-ft DELTA and the per-
field pattern are the takeaways, not absolute production accuracy.

Usage:
  python scripts/eval_fullimage_mcocr.py --limit 80
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
import zipfile
from pathlib import Path

import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.metrics import cer as cer_metric, anls as anls_metric   # noqa

FIELDS = ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]


def _ws() -> Path:
    return Path(os.environ.get("DOCAI_WORKSPACE", "/data/nvidia-ai-workspace"))


def _ensure_images(data: Path, ids: list[str]) -> Path:
    """Extract needed full train images from the zip (cached)."""
    dest = data / "train_images_sample"
    dest.mkdir(parents=True, exist_ok=True)
    missing = [i for i in ids if not (dest / i).exists()]
    if missing:
        zp = next((_ws() / "data/raw/mcocr").glob("*.zip"))
        want = {f"train_images/train_images/{i}" for i in missing}
        with zipfile.ZipFile(zp) as z:
            for m in z.namelist():
                if m in want:
                    data_bytes = z.read(m)
                    (dest / Path(m).name).write_bytes(data_bytes)
    return dest


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
            regions = []
            for p, t, lb in zip(polys, texts, labels):
                bb = p.get("bbox")
                if not bb:
                    continue
                x, y, w, h = bb
                regions.append({"bbox": [x, y, x + w, y + h], "text": t.strip(), "label": lb.strip()})
            if regions:
                rows.append({"img_id": r["img_id"], "regions": regions})
            if len(rows) >= limit:
                break
    return rows


def _match(region_bbox, tokens) -> str:
    """Concat tokens whose center falls inside the gold region bbox (reading order)."""
    x0, y0, x1, y1 = region_bbox
    hit = []
    for t in tokens:
        bx0, by0, bx1, by1 = t["bbox"]
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        if x0 - 5 <= cx <= x1 + 5 and y0 - 5 <= cy <= y1 + 5:
            hit.append((bx0, t["text"]))
    hit.sort()
    return " ".join(tx for _, tx in hit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_ws() / "data/processed/mcocr_ocr"))
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--out", default="docs/logs")
    args = ap.parse_args()
    data = Path(args.data)

    gold = _gold(data, args.limit)
    img_dir = _ensure_images(data, [g["img_id"] for g in gold])
    import docai.config as cfg
    from docai.ocr import run_ocr
    from docai.pipeline import process_document
    from collections import defaultdict

    results = {}
    failures = {"default": [], "ft": []}
    for mode in ["default", "ft"]:
        cfg.OCR_RECOGNIZER = "rapidocr_default" if mode == "default" else "ppocr_vi_mcocr_ft"
        per_field_cer = defaultdict(list)
        per_field_anls = defaultdict(list)
        lats, n_review, n_docs = [], 0, 0
        for g in gold:
            p = img_dir / g["img_id"]
            img = cv2.imread(str(p))
            if img is None:
                continue
            t0 = time.perf_counter()
            tokens = run_ocr(img)
            lats.append((time.perf_counter() - t0) * 1000)
            for r in g["regions"]:
                if r["label"] not in FIELDS:
                    continue
                pred = _match(r["bbox"], tokens)
                c = cer_metric(pred, r["text"])
                per_field_cer[r["label"]].append(c)
                per_field_anls[r["label"]].append(anls_metric(pred, r["text"]))
                if c > 0.5 and len(failures[mode]) < 8:
                    failures[mode].append({"img": g["img_id"], "label": r["label"],
                                           "gold": r["text"], "pred": pred, "cer": round(c, 2)})
            # needs_review + e2e via production pipeline
            doc = process_document(g["img_id"], p.read_bytes())
            n_review += int(doc.needs_human_review); n_docs += 1
        lats.sort()
        results[mode] = {
            "field_cer": {f: round(np.mean(per_field_cer[f]), 4) for f in FIELDS if per_field_cer[f]},
            "field_anls": {f: round(np.mean(per_field_anls[f]), 4) for f in FIELDS if per_field_anls[f]},
            "macro_cer": round(np.mean([c for f in FIELDS for c in per_field_cer[f]]), 4),
            "latency_ms_p50": round(lats[len(lats) // 2], 1) if lats else None,
            "latency_ms_p95": round(lats[int(len(lats) * 0.95) - 1], 1) if lats else None,
            "needs_review_rate": round(n_review / max(n_docs, 1), 3),
            "n_docs": n_docs,
        }
        print(json.dumps({mode: results[mode]}, indent=2, ensure_ascii=False))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    md = [f"# Full-image MC-OCR pipeline eval {stamp}", "",
          f"- n_docs={results['default']['n_docs']}  (RapidOCR detector shared; recognizer swapped)",
          "- **CAVEAT:** recognizer trained on crops from these train receipts → ft field numbers "
          "are optimistic (in-domain). Read the default→ft delta + per-field pattern.", "",
          "| field | default CER | ft CER | rel ↓ | default ANLS | ft ANLS |",
          "|---|---|---|---|---|---|"]
    for f in FIELDS:
        d = results["default"]["field_cer"].get(f); ft = results["ft"]["field_cer"].get(f)
        if d is None or ft is None:
            continue
        rel = round((d - ft) / d * 100, 1) if d else None
        da = results["default"]["field_anls"].get(f); fa = results["ft"]["field_anls"].get(f)
        md.append(f"| {f} | {d} | {ft} | {rel}% | {da} | {fa} |")
    md += ["",
           f"- macro field CER: default {results['default']['macro_cer']} → ft {results['ft']['macro_cer']}",
           f"- latency p50/p95 (full-image OCR): default {results['default']['latency_ms_p50']}/"
           f"{results['default']['latency_ms_p95']}ms · ft {results['ft']['latency_ms_p50']}/"
           f"{results['ft']['latency_ms_p95']}ms",
           f"- needs_review rate (process_document): default {results['default']['needs_review_rate']} · "
           f"ft {results['ft']['needs_review_rate']}", "",
           "### Failure examples (ft, CER>0.5)"]
    for fx in failures["ft"][:6]:
        md.append(f"- `{fx['label']}` cer={fx['cer']} gold=`{fx['gold']}` pred=`{fx['pred']}`")
    (out / f"fullimage_mcocr_{stamp}.md").write_text("\n".join(md), encoding="utf-8")
    (out / "fullimage_mcocr_raw.json").write_text(
        json.dumps({"results": results, "failures": failures}, indent=2, ensure_ascii=False))
    print("\n".join(md))
    print(f"\nwrote {out / f'fullimage_mcocr_{stamp}.md'}")


if __name__ == "__main__":
    main()
