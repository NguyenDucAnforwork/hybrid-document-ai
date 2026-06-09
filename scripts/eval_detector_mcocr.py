"""WP-3 Step 2 + Task A — detector + line-grouping error analysis (MC-OCR).

anno_polygons in mcocr_train_df.csv = gold field regions (SELLER/ADDRESS/TIMESTAMP/
TOTAL_COST), one bbox per labeled region. We compare them to RapidOCR's detected
boxes to quantify WHERE the full-image pipeline loses the crop-level CER gain.

Metrics (Step 2):
  det_field_recall      gold regions covered by >=1 detected box
  field_coverage        recall per label
  overmerge_rate        gold regions sharing one box with another gold region
  oversplit_rate        gold regions covered by >=2 detected boxes
  reading_order_error   images whose detected (y,x) order disagrees with gold order
  (precision vs ALL lines is NOT computable: gold = field regions only, not every
   text line — reported as boxes_per_img / unmatched_box_rate with that caveat.)

Per-region failure taxonomy (Task A), single dominant cause in priority order:
  DETECT_MISS > OVERMERGE > OVERSPLIT > REC_ERROR > OK
  (+ READING_ORDER flagged at image level; KIE_SELECT_ERROR not assessable here —
   MC-OCR labels don't map to the SROIE KIE schema; documented, not faked.)

Uses the fine-tuned recognizer for text so REC_ERROR isolates recognizer fault
given a clean 1:1 box.

Usage:
  python scripts/eval_detector_mcocr.py --limit 80
"""
from __future__ import annotations
import argparse
import ast
import csv
import datetime as dt
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.metrics import cer as cer_metric                  # noqa

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


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _inter(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return _area([x0, y0, x1, y1])


def _assoc(box, gold):
    """box associated with gold region if it covers >=30% of the gold area."""
    ga = _area(gold)
    return ga > 0 and _inter(box, gold) / ga >= 0.30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_ws() / "data/processed/mcocr_ocr"))
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--out", default="docs/logs")
    args = ap.parse_args()
    data = Path(args.data)

    gold = _gold(data, args.limit)
    img_dir = data / "train_images_sample"
    import docai.config as cfg
    cfg.OCR_RECOGNIZER = "ppocr_vi_mcocr_ft"
    from docai.ocr import run_ocr

    tax = defaultdict(int)                          # cause -> count
    tax_by_field = defaultdict(lambda: defaultdict(int))
    cov_by_field = defaultdict(lambda: [0, 0])      # label -> [covered, total]
    n_regions = covered = overmerge = oversplit = 0
    ro_imgs_bad = ro_imgs = 0
    boxes_per_img, unmatched_boxes, total_boxes = [], 0, 0
    n_imgs = 0

    for g in gold:
        p = img_dir / g["img_id"]
        img = cv2.imread(str(p))
        if img is None:
            continue
        n_imgs += 1
        toks = run_ocr(img)
        boxes = [{"bbox": [t["bbox"][0], t["bbox"][1], t["bbox"][2], t["bbox"][3]],
                  "text": t["text"], "cy": (t["bbox"][1] + t["bbox"][3]) / 2,
                  "cx": (t["bbox"][0] + t["bbox"][2]) / 2} for t in toks]
        boxes_per_img.append(len(boxes)); total_boxes += len(boxes)

        # association matrix
        box_to_golds = defaultdict(list)
        gold_to_boxes = defaultdict(list)
        for bi, b in enumerate(boxes):
            for gi, r in enumerate(g["regions"]):
                if _assoc(b["bbox"], r["bbox"]):
                    box_to_golds[bi].append(gi)
                    gold_to_boxes[gi].append(bi)
        unmatched_boxes += sum(1 for bi in range(len(boxes)) if not box_to_golds[bi])

        # reading-order: gold regions in csv order vs detector (y,x) order of their primary box
        order_pairs = []
        for gi, r in enumerate(g["regions"]):
            bxs = gold_to_boxes[gi]
            if bxs:
                pb = min(bxs, key=lambda bi: (boxes[bi]["cy"], boxes[bi]["cx"]))
                order_pairs.append((gi, boxes[pb]["cy"], boxes[pb]["cx"]))
        if len(order_pairs) >= 2:
            ro_imgs += 1
            det_order = [gi for gi, _, _ in sorted(order_pairs, key=lambda z: (z[1], z[2]))]
            gold_order = [gi for gi, _, _ in order_pairs]
            inv = sum(1 for i in range(len(gold_order)) for j in range(i + 1, len(gold_order))
                      if det_order.index(gold_order[i]) > det_order.index(gold_order[j]))
            if inv > 0:
                ro_imgs_bad += 1

        for gi, r in enumerate(g["regions"]):
            n_regions += 1
            lab = r["label"]
            cov_by_field[lab][1] += 1
            bxs = gold_to_boxes[gi]
            if not bxs:
                cause = "DETECT_MISS"
            else:
                covered += 1
                cov_by_field[lab][0] += 1
                shares = any(len(box_to_golds[bi]) >= 2 for bi in bxs)
                if shares:
                    cause = "OVERMERGE"; overmerge += 1
                elif len(bxs) >= 2:
                    cause = "OVERSPLIT"; oversplit += 1
                else:
                    txt = boxes[bxs[0]]["text"]
                    cause = "REC_ERROR" if cer_metric(txt, r["text"]) > 0.5 else "OK"
            tax[cause] += 1
            tax_by_field[lab][cause] += 1

    rep = {
        "n_imgs": n_imgs, "n_gold_regions": n_regions,
        "det_field_recall": round(covered / max(n_regions, 1), 3),
        "overmerge_rate": round(overmerge / max(n_regions, 1), 3),
        "oversplit_rate": round(oversplit / max(n_regions, 1), 3),
        "reading_order_error_img_rate": round(ro_imgs_bad / max(ro_imgs, 1), 3),
        "field_coverage": {f: round(cov_by_field[f][0] / max(cov_by_field[f][1], 1), 3) for f in FIELDS},
        "taxonomy": dict(tax),
        "taxonomy_by_field": {f: dict(tax_by_field[f]) for f in FIELDS},
        "boxes_per_img_mean": round(sum(boxes_per_img) / max(len(boxes_per_img), 1), 1),
        "unmatched_box_rate": round(unmatched_boxes / max(total_boxes, 1), 3),
        "note": "gold = labeled field regions only -> det precision vs ALL lines not computable; "
                "unmatched_box_rate is mostly unlabeled lines, NOT errors. KIE_SELECT_ERROR not "
                "assessable (MC-OCR labels != SROIE KIE schema).",
    }

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    causes = ["DETECT_MISS", "OVERMERGE", "OVERSPLIT", "REC_ERROR", "OK"]
    md = [f"# Detector + line-grouping error analysis (MC-OCR full-image) {stamp}", "",
          f"- n_imgs={n_imgs}, gold field regions={n_regions}",
          f"- **det_field_recall={rep['det_field_recall']}**  "
          f"overmerge_rate={rep['overmerge_rate']}  oversplit_rate={rep['oversplit_rate']}  "
          f"reading_order_error(img)={rep['reading_order_error_img_rate']}",
          f"- boxes/img≈{rep['boxes_per_img_mean']}, unmatched_box_rate={rep['unmatched_box_rate']} "
          "(unlabeled lines, not errors)", "",
          "### Failure taxonomy by field (dominant cause per gold region)",
          "| field | coverage | " + " | ".join(causes) + " |",
          "|---|---|" + "|".join(["---"] * len(causes)) + "|"]
    for f in FIELDS:
        d = tax_by_field[f]
        md.append(f"| {f} | {rep['field_coverage'][f]} | " +
                  " | ".join(str(d.get(c, 0)) for c in causes) + " |")
    md += ["", "### Overall taxonomy", "| cause | count |", "|---|---|"]
    for c in causes + [k for k in tax if k not in causes]:
        if tax.get(c):
            md.append(f"| {c} | {tax[c]} |")
    (out / f"detector_mcocr_{stamp}.md").write_text("\n".join(md), encoding="utf-8")
    (out / "detector_mcocr_raw.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\nwrote {out / f'detector_mcocr_{stamp}.md'}")


if __name__ == "__main__":
    main()
