"""Compare 3 KIE approaches on SROIE test set:
  1. Rule-only (regex/keyword candidates, no classifier)
  2. Logistic KIE v4 (current production)
  3. LayoutLMv3-base fine-tuned

Reports: field F1, ANLS, latency.  Writes docs/logs/model_comparison_*.md
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from docai.kie import candidates, norm_field, KIEModel, token_features  # noqa
from docai.ocr import run_ocr                               # noqa
from eval.metrics import anls as anls_fn, cer  # noqa

def exact_match(pred, gold):
    return float(str(pred or "").strip().lower() == str(gold or "").strip().lower())

WS = os.environ.get("DOCAI_WORKSPACE", "/workspace/docai-ws")

FIELDS = ["merchant_name", "date", "total_amount"]
REQUIRED = {"date", "total_amount"}


# ── Rule-only baseline (KIEModel with no classifier) ─────────────────────────

_rule_model = KIEModel()   # clf=None → uses _rule_score heuristic

def predict_rule(tokens: list[dict], gold: dict) -> dict[str, str | None]:
    out = _rule_model.extract(tokens)
    return {f: out.get(f, (None, 0))[0] for f in FIELDS}


# ── Logistic KIE (production) ─────────────────────────────────────────────────

def predict_logistic(tokens: list[dict], gold: dict, model: KIEModel) -> dict[str, str | None]:
    out = model.extract(tokens)
    return {f: out.get(f, (None, 0))[0] for f in FIELDS}


# ── LayoutLMv3 ────────────────────────────────────────────────────────────────

def load_layoutlmv3(model_dir: str):
    from transformers import AutoProcessor, LayoutLMv3ForTokenClassification
    import torch
    processor = AutoProcessor.from_pretrained(model_dir, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return processor, model, device


def predict_layoutlmv3(tokens, gold, processor, model, device, img_path=None):
    from PIL import Image
    import torch

    LABEL2ID = {"O": 0, "B-MERCHANT": 1, "I-MERCHANT": 2,
                "B-DATE": 3, "I-DATE": 4, "B-TOTAL": 5, "I-TOTAL": 6}
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}
    ENTITY_MAP = {"MERCHANT": "merchant_name", "DATE": "date", "TOTAL": "total_amount"}

    words = [t.get("text", "") or "" for t in tokens]

    # Normalize bboxes by actual image dimensions (required by LayoutLMv3)
    if img_path and Path(img_path).exists():
        image = Image.open(img_path).convert("RGB")
        W_img, H_img = image.size
    else:
        image = Image.new("RGB", (224, 224), (255, 255, 255))
        W_img, H_img = 1000, 1000

    def _norm(bbox):
        x0, y0, x1, y1 = bbox
        return [max(0, min(1000, int(x0 / W_img * 1000))),
                max(0, min(1000, int(y0 / H_img * 1000))),
                max(0, min(1000, int(x1 / W_img * 1000))),
                max(0, min(1000, int(y1 / H_img * 1000)))]

    boxes = [_norm(t.get("bbox", [0, 0, 10, 10])) for t in tokens]

    encoding = processor(image, words, boxes=boxes, truncation=True,
                         padding="max_length", max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**{k: v.to(device) for k, v in encoding.items()
                         if k != "word_labels"}).logits
    preds = logits.argmax(-1).squeeze(0).cpu().tolist()

    # Map subword preds back to words
    word_ids = encoding.word_ids(0)
    entity_tokens: dict[str, list[str]] = {"merchant_name": [], "date": [], "total_amount": []}
    prev_word = None
    for i, wid in enumerate(word_ids):
        if wid is None or wid == prev_word:
            continue
        prev_word = wid
        label = ID2LABEL.get(preds[i], "O")
        for ent, field in ENTITY_MAP.items():
            if ent in label:
                if wid < len(words):
                    entity_tokens[field].append(words[wid])

    raw = {f: " ".join(v) if v else None for f, v in entity_tokens.items()}
    # Apply same normalization as logistic KIE so scores are comparable
    return {f: norm_field(f, v) for f, v in raw.items()}


# ── Metrics ───────────────────────────────────────────────────────────────────

def score(preds: dict, gold: dict) -> dict:
    result = {}
    for f in FIELDS:
        g = str(gold.get(f) or "")
        p = str(preds.get(f) or "")
        result[f] = {
            "f1": exact_match(p, g),
            "anls": anls_fn(p, g),
            "cer": cer(p, g),
        }
    return result


def aggregate(scores: list[dict]) -> dict:
    out = {}
    for f in FIELDS:
        out[f] = {k: round(sum(s[f][k] for s in scores) / len(scores), 4)
                  for k in ("f1", "anls", "cer")}
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-data", default=f"{WS}/data/sroie/test/labels.json")
    ap.add_argument("--test-img-dir", default=f"{WS}/data/sroie/test/images")
    ap.add_argument("--train-data", default=f"{WS}/data/sroie/train/labels.json")
    ap.add_argument("--layoutlmv3-dir", default=f"{WS}/models/layoutlmv3/model")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--out", default="docs/logs")
    args = ap.parse_args()

    # Load test records (train-format has tokens; test-only has images+gold)
    train_recs = json.loads(Path(args.train_data).read_text())
    # Use the last --limit of train as a held-out comparison set
    # (same images that were processed in benchmark via OCR)
    # For a proper eval, we run OCR on test images
    test_recs_meta = json.loads(Path(args.test_data).read_text())[:args.limit]
    img_dir = Path(args.test_img_dir)

    print(f"Running OCR on {len(test_recs_meta)} test images...")
    test_recs = []
    for rec in test_recs_meta:
        img_path = img_dir / rec["image"]
        if not img_path.exists():
            continue
        tokens = run_ocr(img_path.read_bytes())
        test_recs.append({"tokens": tokens, "gold": rec["gold"], "image": rec["image"]})

    print(f"Loaded {len(test_recs)} records with OCR tokens")

    # ── Rule only ──
    rule_scores, rule_times = [], []
    for rec in test_recs:
        t0 = time.perf_counter()
        p = predict_rule(rec["tokens"], rec["gold"])
        rule_times.append(time.perf_counter() - t0)
        rule_scores.append(score(p, rec["gold"]))

    # ── Logistic KIE ──
    kie_model = KIEModel.load(f"{WS}/models/kie/v4/model.joblib")
    log_scores, log_times = [], []
    for rec in test_recs:
        t0 = time.perf_counter()
        p = predict_logistic(rec["tokens"], rec["gold"], kie_model)
        log_times.append(time.perf_counter() - t0)
        log_scores.append(score(p, rec["gold"]))

    # ── LayoutLMv3 ──
    lm_dir = args.layoutlmv3_dir
    lm_scores, lm_times = [], []
    if Path(lm_dir).exists():
        print("Loading LayoutLMv3...")
        processor, lm_model, device = load_layoutlmv3(lm_dir)
        for rec in test_recs:
            img_path = img_dir / rec["image"]
            t0 = time.perf_counter()
            p = predict_layoutlmv3(rec["tokens"], rec["gold"], processor, lm_model, device,
                                   str(img_path))
            lm_times.append(time.perf_counter() - t0)
            lm_scores.append(score(p, rec["gold"]))
    else:
        print(f"LayoutLMv3 model not found at {lm_dir}, skipping")

    # ── Report ──
    rule_agg = aggregate(rule_scores)
    log_agg = aggregate(log_scores)
    lm_agg = aggregate(lm_scores) if lm_scores else None

    def lat_ms(times): return round(sum(times) / len(times) * 1000, 1)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    md = [f"# Model Comparison (SROIE test, n={len(test_recs)}) {stamp}", "",
          "## F1 per field", "",
          "| field | rule-only | logistic-KIE (prod) | LayoutLMv3-base |",
          "|---|---|---|---|"]
    for f in FIELDS:
        r = rule_agg[f]["f1"]
        l = log_agg[f]["f1"]
        lm = lm_agg[f]["f1"] if lm_agg else "—"
        md.append(f"| {f} | {r} | {l} | {lm} |")

    md += ["", "## ANLS per field", "",
           "| field | rule-only | logistic-KIE | LayoutLMv3-base |",
           "|---|---|---|---|"]
    for f in FIELDS:
        r = rule_agg[f]["anls"]
        l = log_agg[f]["anls"]
        lm = lm_agg[f]["anls"] if lm_agg else "—"
        md.append(f"| {f} | {r} | {l} | {lm} |")

    lm_lat = lat_ms(lm_times) if lm_times else "—"
    md += ["", "## Latency (KIE inference only, ms/doc)", "",
           "| model | mean latency |",
           "|---|---|",
           f"| rule-only | {lat_ms(rule_times)}ms |",
           f"| logistic-KIE | {lat_ms(log_times)}ms |",
           f"| LayoutLMv3-base | {lm_lat}ms |",
           "",
           "> Note: latency above is KIE inference only (excludes OCR which is shared across models).",
           "> Full pipeline (OCR + KIE) p50 ≈ 1-2s; see load_test logs for end-to-end numbers."]

    out_path = Path(args.out) / f"model_comparison_{stamp}.md"
    out_path.write_text("\n".join(md))
    raw = {"rule": rule_agg, "logistic": log_agg, "layoutlmv3": lm_agg,
           "latency_ms": {"rule": lat_ms(rule_times), "logistic": lat_ms(log_times),
                          "layoutlmv3": lm_lat}, "n": len(test_recs)}
    (Path(args.out) / "model_comparison_raw.json").write_text(json.dumps(raw, indent=2))
    print("\n".join(md))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
