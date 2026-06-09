"""WP-3 — materialize the MC-OCR recognizer dataset + sanity report.

Selectively extracts ONLY the WP-3 subset from the downloaded Kaggle zip (avoids
unpacking the 36k full receipt images), validates lines, and writes:

  $WS/data/processed/mcocr_ocr/
    crops/                 (extracted recognition crops)
    train.txt val.txt      (crop_path<TAB>transcription, Kaggle split preserved)
    manifest.jsonl rejected.jsonl
    vi_dict.txt            (charset, one char per line; NFC-normalized)
    sanity_report.json     (accepted/missing/empty/corrupt counts)
    field_manifest.jsonl   (optional, from mcocr_train_df.csv)

Usage:
  python scripts/wp3_prepare_ocr.py            # auto-find zip under $WS/data/raw/mcocr
"""
from __future__ import annotations
import argparse
import json
import os
import unicodedata
import zipfile
from pathlib import Path

CROP_PREFIX = "text_recognition_mcocr_data/text_recognition_mcocr_data/"
TRAIN_TXT = "text_recognition_train_data.txt"
VAL_TXT = "text_recognition_val_data.txt"
TRAIN_CSV = "mcocr_train_df.csv"


def _ws() -> Path:
    return Path(os.environ.get("DOCAI_WORKSPACE", "/data/nvidia-ai-workspace"))


def _find_zip(raw_dir: Path) -> Path | None:
    zips = sorted(raw_dir.glob("*.zip"))
    return zips[0] if zips else None


def _extract_subset(zip_path: Path, dest: Path) -> Path:
    """Extract only labels + crops + train csv. Returns the extract root."""
    root = dest / "extracted"
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = z.namelist()
        want = [m for m in members
                if m in (TRAIN_TXT, VAL_TXT, TRAIN_CSV)
                or (m.startswith(CROP_PREFIX) and m.lower().endswith((".jpg", ".jpeg", ".png")))]
        for m in want:
            z.extract(m, root)
    return root


def _load_labels(path: Path) -> list[tuple[str, str]]:
    rows = []
    for ln in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        if not ln.strip():
            continue
        parts = ln.split("\t")
        if len(parts) < 2:
            rows.append((ln, None))          # malformed -> rejected downstream
            continue
        fname = parts[0].strip()
        text = "\t".join(parts[1:])          # keep any literal tabs in transcription
        rows.append((fname, text))
    return rows


def _validate(fname, text, crop_dir: Path):
    if text is None:
        return None, "malformed_line"
    text = unicodedata.normalize("NFC", text).strip()
    if not text:
        return None, "empty_label"
    p = crop_dir / fname
    if not p.exists():
        return None, "missing_image"
    try:
        from PIL import Image
        with Image.open(p) as im:
            im.verify()
    except Exception:
        return None, "corrupt_image"
    return text, None


def _build_split(rows, split, crop_dir, accepted_lines, manifest_fh, rejected_fh, charset):
    n_ok = counts = 0
    breakdown = {"missing_image": 0, "empty_label": 0, "corrupt_image": 0, "malformed_line": 0}
    for fname, text in rows:
        clean, reason = _validate(fname, text, crop_dir)
        if reason:
            breakdown[reason] = breakdown.get(reason, 0) + 1
            rejected_fh.write(json.dumps({"crop": fname, "split": split, "reason": reason}) + "\n")
            continue
        crop_path = str((crop_dir / fname).resolve())
        accepted_lines.append(f"{crop_path}\t{clean}")
        charset.update(clean)
        manifest_fh.write(json.dumps({"crop": fname, "text": clean, "split": split,
                                      "n_chars": len(clean)}, ensure_ascii=False) + "\n")
        n_ok += 1
    return n_ok, breakdown


def _field_manifest(csv_path: Path, out: Path):
    """Optional field-aware manifest from mcocr_train_df.csv (analysis only)."""
    try:
        import csv as _csv
        rows = 0
        with open(csv_path, encoding="utf-8", errors="replace") as fh, open(out, "w", encoding="utf-8") as w:
            reader = _csv.DictReader(fh)
            for r in reader:
                texts = (r.get("anno_texts") or "").split("|||")
                labels = (r.get("anno_labels") or "").split("|||")
                for t, lb in zip(texts, labels):
                    w.write(json.dumps({"img_id": r.get("img_id"), "text": t.strip(),
                                        "label": lb.strip(),
                                        "image_quality": r.get("anno_image_quality")},
                                       ensure_ascii=False) + "\n")
                rows += 1
        return rows
    except Exception as e:
        return f"skipped: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip")
    ap.add_argument("--raw-dir", default=str(_ws() / "data/raw/mcocr"))
    ap.add_argument("--out", default=str(_ws() / "data/processed/mcocr_ocr"))
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    zip_path = Path(args.zip) if args.zip else _find_zip(raw_dir)
    if not zip_path or not zip_path.exists():
        raise SystemExit(f"no dataset zip found under {raw_dir}")

    print(f"extracting WP-3 subset from {zip_path.name} ...")
    root = _extract_subset(zip_path, out)
    crop_dir = root / CROP_PREFIX
    train_rows = _load_labels(root / TRAIN_TXT)
    val_rows = _load_labels(root / VAL_TXT)

    charset: set[str] = set()
    train_lines, val_lines = [], []
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as mf, \
         open(out / "rejected.jsonl", "w", encoding="utf-8") as rf:
        n_train, bd_train = _build_split(train_rows, "train", crop_dir, train_lines, mf, rf, charset)
        n_val, bd_val = _build_split(val_rows, "val", crop_dir, val_lines, mf, rf, charset)

    (out / "train.txt").write_text("\n".join(train_lines), encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val_lines), encoding="utf-8")
    (out / "vi_dict.txt").write_text("\n".join(sorted(charset)), encoding="utf-8")

    fm_rows = _field_manifest(root / TRAIN_CSV, out / "field_manifest.jsonl") \
        if (root / TRAIN_CSV).exists() else "csv_absent"

    report = {
        "zip": zip_path.name,
        "accepted_train": n_train, "accepted_val": n_val,
        "raw_train_lines": len(train_rows), "raw_val_lines": len(val_rows),
        "rejected_train": bd_train, "rejected_val": bd_val,
        "missing_image": bd_train["missing_image"] + bd_val["missing_image"],
        "empty_label": bd_train["empty_label"] + bd_val["empty_label"],
        "corrupt_image": bd_train["corrupt_image"] + bd_val["corrupt_image"],
        "charset_size": len(charset),
        "field_manifest_rows": fm_rows,
        "caveat": "mcocr_val_sample_df.csv is a placeholder stub (anno_texts='abc abc abc'), "
                  "NOT a trustworthy downstream gold validation set.",
    }
    (out / "sanity_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
