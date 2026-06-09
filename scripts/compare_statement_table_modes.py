"""Compare rule parser vs zero-shot table-structure parser on hard bank statements."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import time
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docai.ocr import run_ocr  # noqa
from docai.statement import extract_statement  # noqa
from eval.metrics import anls  # noqa


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return round(2 * p * r / (p + r), 3) if p + r else 0.0


def eval_mode(records, img_dir: Path, mode: str, debug_dir: Path | None = None):
    row_tp = row_fp = row_fn = amt_ok = amt_tot = desc_ok = desc_tot = 0
    lat_ms = []
    row_recon = []
    close_recon = []
    debug_written = []

    if records:
        warm = records[0]
        warm_img = cv2.imread(str(img_dir / warm["image"]))
        if warm_img is not None:
            ok, enc = cv2.imencode(".png", warm_img)
            if ok:
                warm_tokens = run_ocr(enc.tobytes())
                extract_statement(warm_tokens, image_bgr=warm_img, mode=mode, return_meta=True)

    for idx, rec in enumerate(records):
        img_path = img_dir / rec["image"]
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        ok, enc = cv2.imencode(".png", image_bgr)
        if not ok:
            continue
        tokens = run_ocr(enc.tobytes())
        dbg = None
        if debug_dir is not None and idx < 3:
            dbg = debug_dir / f"{Path(rec['image']).stem}_{mode}"
        t0 = time.perf_counter()
        _, items, meta = extract_statement(tokens, image_bgr=image_bgr, mode=mode,
                                           debug_path=dbg, return_meta=True)
        lat_ms.append((time.perf_counter() - t0) * 1000)
        row_recon.append(meta.get("row_reconcile", 0.0))
        close_recon.append(meta.get("closing_reconcile", 0.0))
        if dbg is not None and idx < 9999:
            debug_written.append(str(dbg.with_suffix(".json")))

        gold_by_date = {}
        for t in rec["gold"]["transactions"]:
            gold_by_date.setdefault(t["date"], []).append(t)
        seen = set()
        for pred in items:
            cand = gold_by_date.get(pred.get("date"))
            if cand:
                row_tp += 1
                gt = cand[0]
                amt_tot += 1
                amt_ok += int(abs((pred.get("amount") or 0) - gt["amount"]) < 1)
                desc_tot += 1
                desc_ok += int(anls(pred.get("description"), gt["description"]) >= 0.5)
                seen.add(pred.get("date"))
            else:
                row_fp += 1
        row_fn += sum(1 for t in rec["gold"]["transactions"] if t["date"] not in seen)

    return {
        "mode": mode,
        "n": len(records),
        "table_row_f1": _f1(row_tp, row_fp, row_fn),
        "table_amount_accuracy": round(amt_ok / max(amt_tot, 1), 3),
        "table_description_anls@.5": round(desc_ok / max(desc_tot, 1), 3),
        "latency_ms_mean": round(statistics.mean(lat_ms), 1) if lat_ms else None,
        "latency_ms_p50": round(statistics.median(lat_ms), 1) if lat_ms else None,
        "row_reconcile_mean": round(statistics.mean(row_recon), 3) if row_recon else None,
        "closing_reconcile_rate": round(sum(close_recon) / max(len(close_recon), 1), 3) if close_recon else None,
        "debug_examples": debug_written[:6],
    }


def main():
    ap = argparse.ArgumentParser()
    ws = os.environ.get("DOCAI_WORKSPACE", "C:/docai-demo-ws")
    ap.add_argument("--data", default=f"{ws}/data/statements_test_hard")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--out", default="docs/logs")
    ap.add_argument("--debug-count", type=int, default=3)
    args = ap.parse_args()

    data = Path(args.data)
    records = json.loads((data / "labels.json").read_text(encoding="utf-8"))[:args.limit]
    img_dir = data / "images"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    debug_dir = Path(args.out) / f"statement_table_debug_{stamp}"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Keep debug artifacts manageable.
    rules_debug = debug_dir / "rules"
    tatr_debug = debug_dir / "tatr"
    hybrid_debug = debug_dir / "hybrid"

    subset = records[:args.debug_count]
    res_rules = eval_mode(records, img_dir, "rules", debug_dir=rules_debug if args.debug_count else None)
    res_tatr = eval_mode(records, img_dir, "tatr", debug_dir=tatr_debug if args.debug_count else None)
    res_hybrid = eval_mode(records, img_dir, "hybrid", debug_dir=hybrid_debug if args.debug_count else None)

    summary = {
        "dataset": str(data),
        "limit": args.limit,
        "debug_count": args.debug_count,
        "results": {
            "rules": res_rules,
            "tatr": res_tatr,
            "hybrid": res_hybrid,
        },
        "note": "Metrics use OCR tokens from run_ocr(image), then compare statement-table extraction only.",
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "statement_table_compare_raw.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [
        f"# Statement Table Parser Comparison {stamp}",
        "",
        f"- Dataset: `{data}`",
        f"- N={args.limit} hard statements (OCR tokens + parser only)",
        f"- Debug artifacts: `{debug_dir}`",
        "",
        "| mode | row-F1 | amount-acc | description-acc | mean latency (ms) | p50 latency (ms) | row reconcile | closing reconcile |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in ("rules", "tatr", "hybrid"):
        r = summary["results"][key]
        md.append(
            f"| {key} | {r['table_row_f1']} | {r['table_amount_accuracy']} | "
            f"{r['table_description_anls@.5']} | {r['latency_ms_mean']} | {r['latency_ms_p50']} | "
            f"{r['row_reconcile_mean']} | {r['closing_reconcile_rate']} |"
        )
    md += [
        "",
        "## Interpretation",
        "",
        "- `rules`: current heuristic row/column parser.",
        "- `tatr`: zero-shot Table Transformer structure recognition only.",
        "- `hybrid`: use `tatr` when structure assignment looks trustworthy, else fall back to `rules`.",
        "",
        "## Debug",
        "",
        f"- rules: `{rules_debug}`",
        f"- tatr: `{tatr_debug}`",
        f"- hybrid: `{hybrid_debug}`",
    ]
    (out / f"statement_table_compare_{stamp}.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
