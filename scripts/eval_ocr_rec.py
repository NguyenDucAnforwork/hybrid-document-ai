"""WP-3 — crop-level OCR evaluation on text_recognition_val_data.txt.

Compares the fine-tuned CRNN recognizer ('ft') against the current RapidOCR
recognizer ('default') on the SAME crops. Crop-level eval needs NO detector.
Reports CER / exact-line acc / WER / latency, the relative CER improvement, and
an optional field-aware CER breakdown via mcocr_train_df.csv.

Usage:
  python scripts/eval_ocr_rec.py --recognizer ft        # or: default | both
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.metrics import cer as cer_metric          # noqa


def _ws() -> Path:
    return Path(os.environ.get("DOCAI_WORKSPACE", "/data/nvidia-ai-workspace"))


def wer(pred: str, gold: str) -> float:
    import Levenshtein
    p, g = pred.split(), gold.split()
    if not g:
        return 0.0 if not p else 1.0
    # word-level edit distance via Levenshtein on token-id strings
    vocab = {w: chr(i + 1) for i, w in enumerate(set(p + g))}
    ps = "".join(vocab[w] for w in p)
    gs = "".join(vocab[w] for w in g)
    return Levenshtein.distance(ps, gs) / len(g)


def _load_val(data: Path):
    items = []
    for ln in (data / "val.txt").read_text(encoding="utf-8").split("\n"):
        if ln.strip():
            p, t = ln.split("\t", 1)
            items.append((p, t))
    return items


def _crop_to_parent(crop_path: str) -> str:
    # mcocr_public_<id>_<k>.jpg -> mcocr_public_<id>.jpg
    stem = Path(crop_path).stem
    return "_".join(stem.split("_")[:-1]) + ".jpg"


def _field_labels(data: Path) -> dict:
    """parent img_id -> set of labels (analysis only)."""
    fm = data / "field_manifest.jsonl"
    out: dict[str, list[str]] = {}
    if not fm.exists():
        return out
    for ln in fm.read_text(encoding="utf-8").split("\n"):
        if not ln.strip():
            continue
        r = json.loads(ln)
        out.setdefault(r.get("img_id", ""), []).append(r.get("label", ""))
    return out


def _make_recognizer(kind: str):
    if kind == "ft":
        from docai.ocr_recognizer import FineTunedRecognizer
        rec = FineTunedRecognizer.load()
        return lambda crop: rec.recognize([crop])[0][0]
    else:  # default RapidOCR rec on the crop
        from docai.ocr import _get_engine
        eng = _get_engine()
        def run(crop):
            res, _ = eng(crop)
            return " ".join(t for _, t, _ in (res or []))
        return run


def evaluate(kind: str, items, limit):
    rec = _make_recognizer(kind)
    cers, exacts, wers, lats = [], [], [], []
    per_item = []
    for p, gold in items[:limit]:
        img = cv2.imread(p)
        if img is None:
            continue
        t0 = time.perf_counter()
        pred = rec(img)
        lats.append((time.perf_counter() - t0) * 1000)
        c = cer_metric(pred, gold)
        cers.append(c); exacts.append(int(pred == gold)); wers.append(wer(pred, gold))
        per_item.append({"crop": Path(p).name, "gold": gold, "pred": pred, "cer": round(c, 3)})
    n = len(cers)
    lats.sort()
    return {
        "recognizer": kind, "n": n,
        "cer": round(sum(cers) / max(n, 1), 4),
        "exact_line_acc": round(sum(exacts) / max(n, 1), 4),
        "wer": round(sum(wers) / max(n, 1), 4),
        "latency_ms_p50": round(lats[n // 2], 1) if n else None,
    }, per_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_ws() / "data/processed/mcocr_ocr"))
    ap.add_argument("--recognizer", choices=["ft", "default", "both"], default="both")
    ap.add_argument("--limit", type=int, default=1300)
    ap.add_argument("--out", default="docs/logs")
    args = ap.parse_args()

    data = Path(args.data)
    items = _load_val(data)
    kinds = ["default", "ft"] if args.recognizer == "both" else [args.recognizer]

    summaries, raws = {}, {}
    for k in kinds:
        try:
            s, per = evaluate(k, items, args.limit)
        except Exception as e:
            s, per = {"recognizer": k, "error": str(e)}, []
        summaries[k] = s
        raws[k] = per
        print(json.dumps(s, indent=2, ensure_ascii=False))

    rel = None
    if "default" in summaries and "ft" in summaries and \
       "cer" in summaries["default"] and "cer" in summaries["ft"]:
        d, f = summaries["default"]["cer"], summaries["ft"]["cer"]
        rel = round((d - f) / d * 100, 1) if d else None

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    md = [f"# OCR recognizer eval (MC-OCR val, crop-level) {stamp}", "",
          "> CAVEAT: `mcocr_val_sample_df.csv` is a placeholder stub, NOT downstream gold. "
          "This OCR-level eval uses `text_recognition_val_data.txt` (real transcriptions).", "",
          "| recognizer | n | CER | exact-line | WER | p50 ms |",
          "|---|---|---|---|---|---|"]
    for k in kinds:
        s = summaries[k]
        md.append(f"| {k} | {s.get('n','-')} | {s.get('cer','ERR')} | "
                  f"{s.get('exact_line_acc','-')} | {s.get('wer','-')} | {s.get('latency_ms_p50','-')} |")
    if rel is not None:
        md += ["", f"**Relative CER improvement (default→ft): {rel}%** "
               f"(good ≥15%, excellent ≥25%)."]
    (out / f"ocr_rec_eval_{stamp}.md").write_text("\n".join(md), encoding="utf-8")
    (out / "ocr_rec_eval_raw.json").write_text(
        json.dumps({"summaries": summaries, "rel_cer_improve_pct": rel}, indent=2, ensure_ascii=False))
    print("\n".join(md))
    print(f"\nwrote {out / f'ocr_rec_eval_{stamp}.md'}")


if __name__ == "__main__":
    main()
