"""WP-3 Task F — extract DETECTOR-STYLE crops for a field (default: TIMESTAMP).

TIMESTAMP full-image errors are REC_ERROR despite coverage 1.0 → the recognizer
hasn't seen crops the way the *detector* cuts them (looser/tighter than the clean
training crops). Fix = give it detector-style crops to fine-tune on.

For N train receipts: run RapidOCR detector, match boxes to the field's gold region
(best overlap), crop the DETECTOR box (not the gold box), label with the gold text.

Output: $WS/data/processed/mcocr_ocr/det_crops/<field>/*.jpg + det_crops_<field>.txt
(crop_path<TAB>transcription).

Usage:
  python scripts/wp3_extract_det_crops.py --field TIMESTAMP --limit 500
"""
from __future__ import annotations
import argparse
import ast
import csv
import os
import sys
import zipfile
from pathlib import Path

import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _ws() -> Path:
    return Path(os.environ.get("DOCAI_WORKSPACE", "/data/nvidia-ai-workspace"))


def _area(b): return max(0, b[2] - b[0]) * max(0, b[3] - b[1])
def _inter(a, b):
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="TIMESTAMP")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--data", default=str(_ws() / "data/processed/mcocr_ocr"))
    args = ap.parse_args()
    data = Path(args.data)

    # gold regions for the field
    gold = []
    with open(data / "extracted" / "mcocr_train_df.csv", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            try:
                polys = ast.literal_eval(r["anno_polygons"])
            except Exception:
                continue
            texts = (r.get("anno_texts") or "").split("|||")
            labels = (r.get("anno_labels") or "").split("|||")
            regs = []
            FIELDS = {"SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"}
            for p, t, lb in zip(polys, texts, labels):
                bb = p.get("bbox"); lbl = lb.strip()
                ok = (lbl in FIELDS) if args.field == "ALL" else (lbl == args.field)
                if bb and ok and t.strip():
                    x, y, w, h = bb
                    regs.append({"bbox": [x, y, x + w, y + h], "text": t.strip(), "label": lbl})
            if regs:
                gold.append({"img_id": r["img_id"], "regions": regs})
            if len(gold) >= args.limit:
                break

    out_dir = data / "det_crops" / args.field
    out_dir.mkdir(parents=True, exist_ok=True)
    zp = next((_ws() / "data/raw/mcocr").glob("*.zip"))
    from docai.ocr import _get_engine
    eng = _get_engine()

    lines, n = [], 0
    with zipfile.ZipFile(zp) as z:
        for g in gold:
            member = f"train_images/train_images/{g['img_id']}"
            try:
                buf = z.read(member)
            except KeyError:
                continue
            import numpy as np
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            res, _ = eng(img)
            boxes = []
            for box, _, _ in (res or []):
                xs = [p[0] for p in box]; ys = [p[1] for p in box]
                boxes.append([min(xs), min(ys), max(xs), max(ys)])
            for ri, reg in enumerate(g["regions"]):
                # detector box with max overlap of the gold region
                best, bi = 0, None
                for b in boxes:
                    ov = _inter(b, reg["bbox"])
                    if ov > best:
                        best, bi = ov, b
                if bi is None or best / max(_area(reg["bbox"]), 1) < 0.3:
                    continue
                x0, y0, x1, y1 = [int(max(0, v)) for v in bi]
                crop = img[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                name = f"{Path(g['img_id']).stem}_{ri}.jpg"
                cv2.imwrite(str(out_dir / name), crop)
                lines.append(f"{(out_dir / name).resolve()}\t{reg['text']}")
                n += 1

    label_file = data / f"det_crops_{args.field}.txt"
    label_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"extracted {n} detector-style {args.field} crops -> {out_dir}")
    print(f"label file: {label_file}")


if __name__ == "__main__":
    main()
