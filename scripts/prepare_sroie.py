"""Ingest the real SROIE 2019 dataset (626 scanned receipts) into our format.

Source: github.com/zzzDavid/ICDAR-2019-SROIE (data/{img,box,key}).
- train records: GT OCR tokens (from box) + gold (from key) -> KIE training.
- test records: image copied + gold -> end-to-end eval (real RapidOCR).
Maps SROIE fields company/date/total -> merchant_name/date/total_amount.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from docai.kie import norm_text, norm_date, norm_money  # noqa


def parse_box(box_path: Path):
    tokens = []
    for line in box_path.read_text(errors="ignore").splitlines():
        parts = line.split(",", 8)
        if len(parts) < 9:
            continue
        try:
            c = list(map(float, parts[:8]))
        except ValueError:
            continue
        xs, ys = c[0::2], c[1::2]
        tokens.append({"text": parts[8], "bbox": [min(xs), min(ys), max(xs), max(ys)],
                       "conf": 1.0})
    return tokens


def gold_from_key(key_path: Path):
    k = json.loads(key_path.read_text(errors="ignore"))
    return {
        "merchant_name": norm_text(k.get("company", "")) or None,
        "date": norm_date(k.get("date", "")),
        "total_amount": norm_money(k.get("total", "")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/nvidia-ai-workspace/sroie_src/data")
    ap.add_argument("--out", default="/data/nvidia-ai-workspace/data/sroie")
    ap.add_argument("--n-test", type=int, default=80)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    ids = sorted(p.stem for p in (src / "key").glob("*.json"))
    ids = [i for i in ids if (src / "img" / f"{i}.jpg").exists()
           and (src / "box" / f"{i}.csv").exists()]

    train_ids, test_ids = ids[args.n_test:], ids[:args.n_test]
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test" / "images").mkdir(parents=True, exist_ok=True)

    def build(id_):
        return {"image": f"{id_}.jpg", "tokens": parse_box(src / "box" / f"{id_}.csv"),
                "gold": gold_from_key(src / "key" / f"{id_}.json")}

    train = [build(i) for i in train_ids]
    (out / "train" / "labels.json").write_text(json.dumps(train))

    test = []
    for i in test_ids:
        r = build(i)
        shutil.copy(src / "img" / f"{i}.jpg", out / "test" / "images" / f"{i}.jpg")
        test.append({"image": r["image"], "gold": r["gold"]})
    (out / "test" / "labels.json").write_text(json.dumps(test))
    print(f"SROIE: {len(train)} train, {len(test)} test -> {out}")
    print("sample gold:", json.dumps(train[0]["gold"], ensure_ascii=False))


if __name__ == "__main__":
    main()
