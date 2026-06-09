"""Multi-document eval (HARD): doc-type routing + statement table + payment order.

Statements use the HARD generator (varied column schema/order, parentheses/CR-DR
negatives, VN/EN, footer distractors, jitter) to avoid the easy-data illusion.
Reports per-field, per-row, and per-column accuracy so weaknesses are visible.
"""
from __future__ import annotations
import argparse
import json
import datetime as dt
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from docai.pipeline import process_document  # noqa
from docai.doctypes import BANK_STATEMENT, PAYMENT_ORDER  # noqa
from eval.metrics import anls  # noqa

HEADER = BANK_STATEMENT.fields


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return round(2 * p * r / (p + r), 3) if p + r else 0.0


def eval_statements(recs, img_dir):
    hh = {f: [0, 0] for f in HEADER}
    row_tp = row_fp = row_fn = amt_ok = amt_tot = desc_ok = desc_tot = 0
    routed = 0
    for r in recs:
        res = process_document(r["image"], (img_dir / r["image"]).read_bytes())
        routed += int(res.document_type == "bank_statement")
        g = r["gold"]
        for f in HEADER:
            if g.get(f) is None:
                continue
            hh[f][1] += 1
            hh[f][0] += int(res.fields.get(f) and res.fields[f].value == g[f])
        gold_by_date = {}
        for t in g["transactions"]:
            gold_by_date.setdefault(t["date"], []).append(t)
        seen = set()
        for p in res.line_items:
            cand = gold_by_date.get(p.get("date"))
            if cand:
                row_tp += 1
                gt = cand[0]
                amt_tot += 1; amt_ok += int(abs((p.get("amount") or 0) - gt["amount"]) < 1)
                desc_tot += 1; desc_ok += int(anls(p.get("description"), gt["description"]) >= 0.5)
                seen.add(p.get("date"))
            else:
                row_fp += 1
        row_fn += sum(1 for t in g["transactions"] if t["date"] not in seen)
    return {
        "routing_to_statement": round(routed / max(len(recs), 1), 3),
        "header_exact_match": {f: round(hh[f][0] / hh[f][1], 3) if hh[f][1] else None for f in HEADER},
        "table_row_f1": _f1(row_tp, row_fp, row_fn),
        "table_amount_accuracy": round(amt_ok / max(amt_tot, 1), 3),
        "table_description_anls@.5": round(desc_ok / max(desc_tot, 1), 3),
        "n": len(recs),
    }


def eval_kv(recs, img_dir, fields, true_type):
    acc = {f: [0, 0] for f in fields}
    routed = 0
    for r in recs:
        res = process_document(r["image"], (img_dir / r["image"]).read_bytes())
        routed += int(res.document_type == true_type)
        for f in fields:
            g = r["gold"].get(f)
            if g is None:
                continue
            acc[f][1] += 1
            acc[f][0] += int(res.fields.get(f) and res.fields[f].value == g)
    return {"routing_correct": round(routed / max(len(recs), 1), 3),
            "field_exact_match": {f: round(acc[f][0] / acc[f][1], 3) if acc[f][1] else None for f in fields},
            "n": len(recs)}


def main():
    ap = argparse.ArgumentParser()
    WS = "/data/nvidia-ai-workspace"
    ap.add_argument("--statements", default=f"{WS}/data/statements_test_hard")
    ap.add_argument("--receipts", default=f"{WS}/data/sroie/test")
    ap.add_argument("--payment-orders", default=f"{WS}/data/payment_orders_test")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--out", default="docs/logs")
    ap.add_argument("--statement-table-mode", default=None,
                    help="override DOCAI_STATEMENT_TABLE_MODE for this eval: rules|tatr|hybrid")
    args = ap.parse_args()
    if args.statement_table_mode:
        os.environ["DOCAI_STATEMENT_TABLE_MODE"] = args.statement_table_mode

    def load(d):
        return json.loads((Path(d) / "labels.json").read_text())[:args.limit]

    S = eval_statements(load(args.statements), Path(args.statements) / "images")
    P = eval_kv(load(args.payment_orders), Path(args.payment_orders) / "images",
                PAYMENT_ORDER.fields, "payment_order")
    rrec = load(args.receipts); rdir = Path(args.receipts) / "images"
    rec_ok = sum(int(process_document(r["image"], (rdir / r["image"]).read_bytes()).document_type
                     == "receipt") for r in rrec)
    n = len(rrec) + S["n"] + P["n"]
    routing = round((rec_ok + S["routing_to_statement"] * S["n"] + P["routing_correct"] * P["n"]) / n, 3)

    summary = {"doctype_routing_accuracy_3way": routing,
               "routing": {"receipt": round(rec_ok / max(len(rrec), 1), 3),
                           "bank_statement": S["routing_to_statement"],
                           "payment_order": P["routing_correct"]},
               "bank_statement_HARD": S, "payment_order": P}
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "multidoc_raw.json").write_text(json.dumps(summary, indent=2))
    mode_note = os.environ.get("DOCAI_STATEMENT_TABLE_MODE", "rules")
    md = [f"# Multi-document eval (HARD statements) {stamp}", "",
          f"- **statement-table mode**: `{mode_note}`",
          f"- **3-way routing accuracy**: {routing} "
          f"(receipt {summary['routing']['receipt']}, statement {S['routing_to_statement']}, payment_order {P['routing_correct']})",
          f"- **statement table (HARD)**: row-F1 {S['table_row_f1']}, amount-acc {S['table_amount_accuracy']}, "
          f"description-acc {S['table_description_anls@.5']}", "",
          "| statement header | exact | | payment_order field | exact |", "|---|---|---|---|---|"]
    pf = list(PAYMENT_ORDER.fields)
    for i, f in enumerate(HEADER):
        pof = pf[i] if i < len(pf) else ""
        pov = P["field_exact_match"].get(pof, "") if pof else ""
        md.append(f"| {f} | {S['header_exact_match'][f]} | | {pof} | {pov} |")
    (Path(args.out) / f"multidoc_{stamp}.md").write_text("\n".join(md))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
