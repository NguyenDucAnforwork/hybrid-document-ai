"""Synthetic receipt generator (data layer).

Produces labeled receipt images + ground-truth fields. Fully reproducible (seed),
no personal data. Used to TRAIN the KIE classifier and to BENCHMARK the pipeline.
SROIE is the real-world target (see scripts/), synthetic keeps the demo runnable
within the disk/time budget and gives us exact gold labels.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

MERCHANTS = ["ABC MART", "VINMART", "CIRCLE K", "COOP FOOD", "BIG C", "HIGHLANDS COFFEE",
             "THE COFFEE HOUSE", "LOTTE MART", "FAMILY MART", "GS25"]
PAYMENTS = [("CASH", "Cash"), ("CARD", "Card VISA"), ("QR", "QR Pay"),
            ("MOMO", "MoMo"), ("CASH", "Tien mat")]


def _font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def gen_one(rng: random.Random):
    merchant = rng.choice(MERCHANTS)
    day, mon, yr = rng.randint(1, 28), rng.randint(1, 12), rng.randint(2024, 2026)
    total = rng.choice([89000, 120000, 235000, 450000, 890000, 1250000])
    inv = f"HD{rng.randint(1000, 9999)}"
    pay_norm, pay_text = rng.choice(PAYMENTS)
    gold = {
        "merchant_name": merchant.lower(),
        "date": f"{yr:04d}-{mon:02d}-{day:02d}",
        "total_amount": total,
        "invoice_id": inv,
        "payment_method": pay_norm,
    }
    lines = [
        (merchant, 44),
        (f"Invoice No: {inv}", 26),
        (f"Date: {day:02d}/{mon:02d}/{yr}", 26),
        ("Item A .......... 12,000", 26),
        ("Item B .......... 23,000", 26),
        (f"TONG CONG: {total:,}", 34),
        (f"Payment: {pay_text}", 26),
    ]
    W, H = 720, 560
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    # ground-truth OCR tokens (text + bbox + conf) — used for training labels.
    tokens = []
    y = 24
    for text, sz in lines:
        f = _font(sz)
        d.text((32, y), text, fill="black", font=f)
        bbox = d.textbbox((32, y), text, font=f)
        tokens.append({"text": text, "bbox": [float(b) for b in bbox], "conf": 0.95})
        y += sz + 28
    return img, tokens, gold


def generate(out_dir: str | Path, n: int, seed: int = 42):
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records = []
    for i in range(n):
        img, tokens, gold = gen_one(rng)
        name = f"rcpt_{i:04d}.png"
        img.save(out / "images" / name)
        records.append({"image": name, "tokens": tokens, "gold": gold})
    (out / "labels.json").write_text(json.dumps(records, indent=2))
    return records
