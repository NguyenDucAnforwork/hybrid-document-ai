"""Fine-tune LayoutLMv3-base on SROIE for token-level KIE (BIO).

Input:  SROIE train labels.json (tokens with bbox, gold fields)
Output: $DOCAI_WORKSPACE/models/layoutlmv3/{model, metrics.json}

BIO labels: B/I-DATE, B/I-TOTAL, B/I-MERCHANT, O
Compared against logistic KIE baseline on same SROIE test set.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from docai.kie import norm_text, norm_date, norm_money  # noqa

LABEL2ID = {"O": 0, "B-MERCHANT": 1, "I-MERCHANT": 2,
            "B-DATE": 3, "I-DATE": 4, "B-TOTAL": 5, "I-TOTAL": 6}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MODEL_NAME = "microsoft/layoutlmv3-base"


def _norm_bbox(bbox, W, H):
    """Normalise [x0,y0,x1,y1] to 0-1000 LayoutLM scale."""
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(1000, int(x0 / W * 1000))),
        max(0, min(1000, int(y0 / H * 1000))),
        max(0, min(1000, int(x1 / W * 1000))),
        max(0, min(1000, int(y1 / H * 1000))),
    ]


def _fuzzy_match(token_text: str, gold_str: str) -> bool:
    return gold_str and token_text.strip().lower() in gold_str.lower()


def build_bio(tokens: list[dict], gold: dict, img_w=1000, img_h=1000) -> list[str]:
    """Assign BIO label to each token by substring match against gold values."""
    date_val = norm_date(gold.get("date") or "")
    total_raw = gold.get("total_amount")
    total_val = str(total_raw) if total_raw is not None else ""
    merch_val = norm_text(gold.get("merchant_name") or "")

    labels = []
    prev = {}
    for tok in tokens:
        txt = (tok.get("text") or "").strip().lower()
        label = "O"
        if date_val and txt in date_val.lower().replace("-", "").replace("/", ""):
            label = "I-DATE" if prev.get("f") == "DATE" else "B-DATE"
            prev = {"f": "DATE"}
        elif total_val and txt.replace(".", "").replace(",", "") in total_val.replace(".", "").replace(",", ""):
            label = "I-TOTAL" if prev.get("f") == "TOTAL" else "B-TOTAL"
            prev = {"f": "TOTAL"}
        elif merch_val and _fuzzy_match(txt, merch_val):
            label = "I-MERCHANT" if prev.get("f") == "MERCHANT" else "B-MERCHANT"
            prev = {"f": "MERCHANT"}
        else:
            prev = {}
        labels.append(label)
    return labels


class SROIEDataset(Dataset):
    def __init__(self, records, img_dir: Path | None, processor, max_len=512):
        self.records = records
        self.img_dir = img_dir
        self.processor = processor
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        tokens = rec.get("tokens", [])
        gold = rec.get("gold", {})

        words = [t.get("text", "") or "" for t in tokens]
        # Use a uniform bbox if bbox missing; assume 1000×1000 virtual canvas
        boxes = [_norm_bbox(t.get("bbox", [0, 0, 10, 10]), 1000, 1000) for t in tokens]
        bio = build_bio(tokens, gold)
        word_labels = [LABEL2ID[l] for l in bio]

        # Load image if available, else white placeholder
        img_name = rec.get("image", "")
        image = None
        if self.img_dir and img_name and (self.img_dir / img_name).exists():
            image = Image.open(self.img_dir / img_name).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), (255, 255, 255))

        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


def compute_metrics_fn(p):
    preds_raw, labels_raw = p
    preds = np.argmax(preds_raw, axis=2)
    true_seqs, pred_seqs = [], []
    for pred_row, label_row in zip(preds, labels_raw):
        true_seq, pred_seq = [], []
        for p_id, l_id in zip(pred_row, label_row):
            if l_id == -100:
                continue
            true_seq.append(ID2LABEL[l_id])
            pred_seq.append(ID2LABEL[p_id])
        true_seqs.append(true_seq)
        pred_seqs.append(pred_seq)

    # Per-entity F1
    from seqeval.metrics import classification_report, f1_score
    report = classification_report(true_seqs, pred_seqs, output_dict=True, zero_division=0)
    result = {"f1": f1_score(true_seqs, pred_seqs, zero_division=0)}
    for ent in ["MERCHANT", "DATE", "TOTAL"]:
        result[f"f1_{ent.lower()}"] = report.get(ent, {}).get("f1-score", 0.0)
    return result


def main():
    ap = argparse.ArgumentParser()
    WS = os.environ.get("DOCAI_WORKSPACE", "/workspace/docai-ws")
    ap.add_argument("--train-data", default=f"{WS}/data/sroie/train/labels.json")
    ap.add_argument("--test-data", default=f"{WS}/data/sroie/test/labels.json")
    ap.add_argument("--test-img-dir", default=f"{WS}/data/sroie/test/images")
    ap.add_argument("--train-img-dir", default=f"{WS}/sroie_src/data/img")
    ap.add_argument("--out", default=f"{WS}/models/layoutlmv3")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading processor from {MODEL_NAME} ...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, apply_ocr=False)

    train_recs = json.loads(Path(args.train_data).read_text())
    test_recs = json.loads(Path(args.test_data).read_text())

    # Test records don't have tokens — need to load from train-compatible format
    # For eval we use train split's validation (last 10%)
    split = int(len(train_recs) * 0.9)
    val_recs = train_recs[split:]
    train_recs = train_recs[:split]
    print(f"Train: {len(train_recs)}, Val: {len(val_recs)}")

    train_img_dir = Path(args.train_img_dir) if Path(args.train_img_dir).exists() else None
    train_ds = SROIEDataset(train_recs, train_img_dir, processor)
    val_ds = SROIEDataset(val_recs, train_img_dir, processor)

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID
    )

    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_fn,
    )

    t0 = time.time()
    trainer.train()
    train_sec = time.time() - t0
    print(f"Training done in {train_sec:.0f}s")

    # Save model + processor
    model_dir = out / "model"
    trainer.save_model(str(model_dir))
    processor.save_pretrained(str(model_dir))

    # Final eval on val
    metrics = trainer.evaluate()
    metrics["train_seconds"] = round(train_sec, 1)
    metrics["model"] = "layoutlmv3-base"
    metrics["data"] = "sroie_train_90pct_val_10pct"
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
