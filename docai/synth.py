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


BANKS = ["VIETCOMBANK", "BIDV", "TECHCOMBANK", "VPBANK", "MB BANK", "ACB", "VNPAY BANK"]
DESCS = ["POS PURCHASE", "ATM WITHDRAWAL", "ONLINE TRANSFER", "QR PAYMENT",
         "SALARY CREDIT", "BILL PAYMENT", "FEE", "REFUND", "INTEREST"]


def gen_statement(rng: random.Random):
    bank = rng.choice(BANKS)
    acc = "".join(str(rng.randint(0, 9)) for _ in range(12))
    holder = rng.choice(["NGUYEN VAN A", "TRAN THI B", "LE VAN C", "PHAM THI D"])
    yr = rng.choice([2024, 2025, 2026])
    period = f"01/01/{yr} - 31/01/{yr}"
    bal = rng.randint(2_000_000, 50_000_000)
    opening = bal
    n_tx = rng.randint(6, 11)
    txns = []
    for i in range(n_tx):
        day = i + 2
        amt = rng.choice([-1, 1]) * rng.choice([50000, 120000, 350000, 1500000, 4500000])
        bal += amt
        txns.append({"date": f"{day:02d}/01/{yr}", "description": rng.choice(DESCS),
                     "amount": amt, "balance": bal})
    closing = bal
    gold = {
        "bank_name": bank.lower(), "account_number": acc,
        "account_holder": holder.lower(), "statement_period": period,
        "opening_balance": float(opening), "closing_balance": float(closing),
        "transactions": [{"date": f"{yr:04d}-01-{int(t['date'][:2]):02d}",
                          "description": t["description"].lower(),
                          "amount": float(t["amount"]), "balance": float(t["balance"])}
                         for t in txns],
    }
    # render
    W, H = 900, 560 + n_tx * 34
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    tokens = []

    def put(x, y, text, sz):
        f = _font(sz)
        d.text((x, y), text, fill="black", font=f)
        bb = d.textbbox((x, y), text, font=f)
        tokens.append({"text": text, "bbox": [float(b) for b in bb], "conf": 0.95})

    put(32, 24, bank, 40)
    put(32, 84, "ACCOUNT STATEMENT", 26)
    put(32, 130, f"Account No: {acc}", 24)
    put(32, 168, f"Account Holder: {holder}", 24)
    put(32, 206, f"Period: {period}", 24)
    put(32, 244, f"Opening Balance: {opening:,}", 24)
    put(32, 278, f"Closing Balance: {closing:,}", 24)   # own row (not merged with opening)
    # table header (column x-centers matter for parsing)
    hy = 326
    put(40, hy, "Date", 22); put(220, hy, "Description", 22)
    put(560, hy, "Amount", 22); put(760, hy, "Balance", 22)
    ry = hy + 40
    for t in txns:
        put(40, ry, t["date"], 20)
        put(220, ry, t["description"], 20)
        put(560, ry, f"{t['amount']:,}", 20)
        put(760, ry, f"{t['balance']:,}", 20)
        ry += 34
    return img, tokens, gold


def generate_statements(out_dir: str | Path, n: int, seed: int = 7):
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records = []
    for i in range(n):
        img, tokens, gold = gen_statement(rng)
        name = f"stmt_{i:04d}.png"
        img.save(out / "images" / name)
        records.append({"image": name, "tokens": tokens, "gold": gold,
                        "doc_type": "bank_statement"})
    (out / "labels.json").write_text(json.dumps(records, indent=2))
    return records


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
