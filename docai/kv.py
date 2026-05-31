"""Generic anchor-based key-value extractor for simple doc types (e.g. payment order).

Driven by a DocType's anchors; infers per-field value kind from the field name
(account->digits, amount->money, date->date, else text). Reused so adding a new
key-value document type needs only a registry entry, no new extractor code.
"""
from __future__ import annotations
import re
from .kie import norm_date, group_lines
from .statement import signed_money

_DIGITS = re.compile(r"\d{6,}")
_DATE = re.compile(r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}")


def _value(field: str, line: str):
    after = line.split(":", 1)[-1].strip() if ":" in line else line
    if "account" in field:
        m = _DIGITS.search(line.replace(" ", "")); return m.group() if m else None
    if "amount" in field or "balance" in field:
        return signed_money(line)
    if field == "date":
        return norm_date(line)
    return after.lower() or None


def kv_extract(tokens, doctype) -> dict:
    """Return {field: (value, confidence)} using the doctype's anchors."""
    lines = group_lines(tokens)
    out = {}
    out["bank_name"] = (lines[0]["text"].lower(), 0.85) if lines else (None, 0.0)
    for f in doctype.fields:
        if f == "bank_name":
            continue
        anchors = doctype.anchors.get(f, [])
        val, conf = None, 0.0
        for ln in lines:
            if any(a in ln["text"].lower() for a in anchors):
                val = _value(f, ln["text"]); conf = 0.82 if val is not None else 0.0
                break
        out[f] = (val, conf)
    return out
