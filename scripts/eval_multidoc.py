"""Multi-document eval: doc-type routing + bank-statement header + TABLE metrics.

- doc-type routing accuracy on a mix of real SROIE receipts + synthetic statements.
- statement header field exact-match.
- transaction table: row recall/precision (matched by date) + amount accuracy.
"""
from __future__ import annotations
import argparse
import json
import datetime as dt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from docai.pipeline import process_document  # noqa

HEADER = ["bank_name", "account_number", "account_holder", "statement_period",
          "opening_balance", "closing_balance"]


def eval_statements(recs, img_dir):
    hh = {f: {"hit": 0, "tot": 0} for f in HEADER}
    row_tp = row_fp = row_fn = amt_ok = amt_tot = 0
    routed_ok = 0
    for r in recs:
        res = process_document(r["image"], (img_dir / r["image"]).read_bytes())
        routed_ok += int(res.document_type == "bank_statement")
        g = r["gold"]
        for f in HEADER:
            if g.get(f) is None:
                continue
            hh[f]["tot"] += 1
            hh[f]["hit"] += int(res.fields.get(f) and res.fields[f].value == g[f])
        # table rows matched by date
        gold_by_date = {}
        for t in g["transactions"]:
            gold_by_date.setdefault(t["date"], []).append(t)
        pred = res.line_items
        for p in pred:
            cand = gold_by_date.get(p.get("date"))
            if cand:
                row_tp += 1
                amt_tot += 1
                amt_ok += int(abs((p.get("amount") or 0) - cand[0]["amount"]) < 1)
            else:
                row_fp += 1
        matched_dates = {p.get("date") for p in pred}
        row_fn += sum(1 for t in g["transactions"] if t["date"] not in matched_dates)

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        return round(2 * p * r / (p + r), 3) if p + r else 0.0

    return {
        "routing_to_statement": round(routed_ok / max(len(recs), 1), 3),
        "header_exact_match": {f: round(hh[f]["hit"] / hh[f]["tot"], 3) if hh[f]["tot"] else None
                               for f in HEADER},
        "table_row_f1": f1(row_tp, row_fp, row_fn),
        "table_amount_accuracy": round(amt_ok / max(amt_tot, 1), 3),
        "n": len(recs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--statements", default="/data/nvidia-ai-workspace/data/statements_test")
    ap.add_argument("--receipts", default="/data/nvidia-ai-workspace/data/sroie/test")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--out", default="docs/logs")
    args = ap.parse_args()

    srec = json.loads((Path(args.statements) / "labels.json").read_text())[:args.limit]
    S = eval_statements(srec, Path(args.statements) / "images")

    # doc-type routing on receipts (should NOT be classified statement)
    rrec = json.loads((Path(args.receipts) / "labels.json").read_text())[:args.limit]
    rdir = Path(args.receipts) / "images"
    rec_ok = sum(int(process_document(r["image"], (rdir / r["image"]).read_bytes()).document_type
                     == "receipt") for r in rrec)
    routing_acc = round((S["routing_to_statement"] * len(srec) + rec_ok) / (len(srec) + len(rrec)), 3)

    summary = {"doctype_routing_accuracy": routing_acc,
               "routing_receipt_correct": round(rec_ok / max(len(rrec), 1), 3),
               "statement": S}
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "multidoc_raw.json").write_text(json.dumps(summary, indent=2))
    md = [f"# Multi-document eval {stamp}", "",
          f"- **doc-type routing accuracy**: {routing_acc} (receipts {summary['routing_receipt_correct']}, statements {S['routing_to_statement']})",
          f"- **statement table**: row-F1 {S['table_row_f1']}, amount-acc {S['table_amount_accuracy']}", "",
          "| statement header field | exact-match |", "|---|---|"]
    for f in HEADER:
        md.append(f"| {f} | {S['header_exact_match'][f]} |")
    (Path(args.out) / f"multidoc_{stamp}.md").write_text("\n".join(md))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
