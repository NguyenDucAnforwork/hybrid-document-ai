"""WP-3 Việc 2 — per-field + subset CER breakdown (crop-level, MC-OCR val).

Val crops are HELD OUT for the recognizer, so this is leakage-free. Crop suffix
`_k` maps to region index k in mcocr_train_df.csv (validated 400/400 exact). We
bucket CER by field label (SELLER/ADDRESS/TIMESTAMP/TOTAL_COST) and by content
subsets (money/date-like digit-heavy lines; lines with Vietnamese diacritics),
comparing default RapidOCR vs the fine-tuned CRNN.

Usage:
  python scripts/eval_field_cer.py --limit 1300
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.metrics import cer as cer_metric            # noqa

DIACRITICS = set("ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
                 "ÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỹỴ")
DIGIT_RE = re.compile(r"\d")


def _ws() -> Path:
    return Path(os.environ.get("DOCAI_WORKSPACE", "/data/nvidia-ai-workspace"))


def _crop_label_map(data: Path) -> dict:
    """crop filename -> field label, via csv region index."""
    csvf = data / "extracted" / "mcocr_train_df.csv"
    parent = {}
    with open(csvf, encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            labels = (r.get("anno_labels") or "").split("|||")
            parent[r["img_id"]] = [l.strip() for l in labels]
    out = {}
    for ln in (data / "val.txt").read_text(encoding="utf-8").split("\n"):
        if not ln.strip():
            continue
        cp = ln.split("\t", 1)[0]
        stem = Path(cp).stem
        parts = stem.split("_")
        pid, k = "_".join(parts[:-1]) + ".jpg", int(parts[-1])
        if pid in parent and k < len(parent[pid]):
            out[Path(cp).name] = parent[pid][k]   # key by basename (lookup uses basename)
    return out


def _make_recognizer(kind: str):
    if kind == "ft":
        from docai.ocr_recognizer import FineTunedRecognizer
        rec = FineTunedRecognizer.load()
        return lambda crop: rec.recognize([crop])[0][0]
    from docai.ocr import _get_engine
    eng = _get_engine()
    return lambda crop: " ".join(t for _, t, _ in (eng(crop)[0] or []))


def _subsets(gold: str) -> list[str]:
    subs = []
    digits = sum(bool(DIGIT_RE.match(c)) for c in gold)
    if gold and sum(c.isdigit() for c in gold) / len(gold) >= 0.3:
        subs.append("digit_heavy(money/date)")
    if any(c in DIACRITICS for c in gold):
        subs.append("diacritics")
    return subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_ws() / "data/processed/mcocr_ocr"))
    ap.add_argument("--limit", type=int, default=1300)
    ap.add_argument("--out", default="docs/logs")
    args = ap.parse_args()
    data = Path(args.data)

    labels = _crop_label_map(data)
    items = []
    for ln in (data / "val.txt").read_text(encoding="utf-8").split("\n"):
        if ln.strip():
            p, t = ln.split("\t", 1)
            items.append((p, t))
    items = items[:args.limit]

    # bucket CER per recognizer
    from collections import defaultdict
    res = {}
    for kind in ["default", "ft"]:
        rec = _make_recognizer(kind)
        buckets = defaultdict(list)
        for p, gold in items:
            img = cv2.imread(p)
            if img is None:
                continue
            c = cer_metric(rec(img), gold)
            buckets["ALL"].append(c)
            lab = labels.get(Path(p).name)
            if lab:
                buckets[lab].append(c)
            for s in _subsets(gold):
                buckets[s].append(c)
        res[kind] = {k: (round(sum(v) / len(v), 4), len(v)) for k, v in buckets.items()}

    keys = ["ALL", "SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST",
            "digit_heavy(money/date)", "diacritics"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    md = [f"# Per-field + subset CER (MC-OCR val, crop-level) {stamp}", "",
          "> Leakage-free: val crops are held out for the recognizer. Labels via "
          "crop→csv region index (validated). default = RapidOCR, ft = fine-tuned CRNN.", "",
          "| bucket | n | default CER | ft CER | rel ↓ |", "|---|---|---|---|---|"]
    rows = {}
    for k in keys:
        d = res["default"].get(k); f = res["ft"].get(k)
        if not d or not f:
            continue
        rel = round((d[0] - f[0]) / d[0] * 100, 1) if d[0] else None
        rows[k] = {"n": f[1], "default_cer": d[0], "ft_cer": f[0], "rel_pct": rel}
        md.append(f"| {k} | {f[1]} | {d[0]} | {f[0]} | {rel}% |")
    (out / f"ocr_field_cer_{stamp}.md").write_text("\n".join(md), encoding="utf-8")
    (out / "ocr_field_cer_raw.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print("\n".join(md))
    print(f"\nwrote {out / f'ocr_field_cer_{stamp}.md'}")


if __name__ == "__main__":
    main()
