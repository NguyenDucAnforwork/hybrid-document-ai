"""Bank statement extractor (Processing Layer) — header KIE + TABLE parsing.

Statements are table-heavy, so beyond key-value header fields we parse the
transaction table via a layout-graph: cluster tokens into rows, detect columns
from the header row, assign each cell to its nearest column. Returns header
fields + a list of transaction rows.
"""
from __future__ import annotations
import re
from .doctypes import BANK_STATEMENT
from .kie import norm_date, group_lines

_ACC = re.compile(r"\d{8,}")
_MONEY = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+\.\d{2}|\d{3,}")
_DATE = re.compile(r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}")
_FOOTER = ["total", "tong", "tổng", "closing", "ending", "so du cuoi", "số dư cuối",
           "c/f", "carried"]
_COLS = {"date": ["date", "ngay", "ngày"],
         "description": ["desc", "description", "dien giai", "diễn giải", "noi dung", "nội dung"],
         "amount": ["amount", "so tien", "số tiền"],
         "debit": ["debit", "withdrawal", "no", "nợ"],
         "credit": ["credit", "deposit", "co", "có"],
         "balance": ["balance", "so du", "số dư"],
         "ref": ["ref", "soct", "so ct", "số ct"]}


def _col_match(low, kws):
    """Word-boundary match so short VN headers 'No'/'Co' match without 'content'
    falsely matching 'co'."""
    return any(re.search(rf"\b{re.escape(k)}\b", low) for k in kws)


def signed_money(s: str):
    """Parse amount with sign. Handles -x, (x)=neg, 'x DR'=neg, 'x CR'=pos."""
    s = (s or "").strip()
    neg = (s.startswith("(") and ")" in s) or s.lstrip().startswith("-")
    up = s.upper()
    if re.search(r"\bDR\b", up):
        neg = True
    m = _MONEY.search(s.replace(" ", ""))
    if not m:
        return None
    try:
        v = round(float(m.group().replace(",", "")), 2)
    except ValueError:
        return None
    return -v if neg else v


def _rows_raw(tokens):
    """Cluster raw tokens into rows (keep individual cells), sorted top->bottom, left->right."""
    toks = sorted(tokens, key=lambda t: (t["bbox"][1] + t["bbox"][3]) / 2)
    rows, cur = [], []
    last_cy = None
    for t in toks:
        cy = (t["bbox"][1] + t["bbox"][3]) / 2
        h = t["bbox"][3] - t["bbox"][1]
        if last_cy is not None and abs(cy - last_cy) > 0.7 * max(h, 1):
            rows.append(sorted(cur, key=lambda z: z["bbox"][0])); cur = []
        cur.append(t); last_cy = cy
    if cur:
        rows.append(sorted(cur, key=lambda z: z["bbox"][0]))
    return rows


def _cx(t):
    return (t["bbox"][0] + t["bbox"][2]) / 2


def extract_table(tokens):
    rows = _rows_raw(tokens)
    # find header row + column x-centers
    header_idx, centers = None, {}
    for i, row in enumerate(rows):
        hits = {}
        for t in row:
            low = t["text"].lower()
            for col, kws in _COLS.items():
                if col not in hits and _col_match(low, kws):
                    hits[col] = _cx(t)
        if len(hits) >= 3:
            header_idx, centers = i, hits
            break
    if header_idx is None:
        return []
    order = sorted(centers, key=lambda c: centers[c])
    cs = [centers[c] for c in order]
    bounds = [(cs[i] + cs[i + 1]) / 2 for i in range(len(cs) - 1)]

    def col_of(x0):
        i = 0
        while i < len(bounds) and x0 >= bounds[i]:
            i += 1
        return order[i]

    items = []
    for row in rows[header_idx + 1:]:
        cells = {c: [] for c in order}
        for t in row:
            # assign by LEFT edge (cells are left-aligned) -> robust to right-
            # extending suffixes like " CR"/" DR" that shift the token center.
            cells[col_of(t["bbox"][0])].append(t["text"])
        joined = {c: " ".join(cells[c]).strip() for c in order}
        desc = joined.get("description", "")
        # skip footer/summary rows (Total / Closing / Tổng ...) and non-date rows
        if any(k in desc.lower() for k in _FOOTER) or not _DATE.search(joined.get("date", "")):
            continue
        # amount: single column, or derive from debit/credit pair
        if joined.get("amount"):
            amt = signed_money(joined["amount"])
        else:
            deb = signed_money(joined.get("debit", "")) or 0.0
            cred = signed_money(joined.get("credit", "")) or 0.0
            amt = (abs(cred) - abs(deb)) if (deb or cred) else None
        bal = signed_money(joined.get("balance", ""))
        if amt is None and bal is None:
            continue
        items.append({"date": norm_date(joined.get("date", "")),
                      "description": desc.lower() or None,
                      "amount": amt, "balance": bal})
    return items


def reconcile(items, opening, closing) -> float:
    """Domain check: does running balance match amounts? Returns fraction of rows
    where balance[i] ≈ balance[i-1] + amount[i]. Low score => parser got the table
    wrong => route to VLM. (A signal the rule parser can't self-assess otherwise.)"""
    if not items:
        return 0.0
    ok, prev = 0, opening
    for it in items:
        amt, bal = it.get("amount"), it.get("balance")
        if amt is not None and bal is not None and prev is not None and abs(prev + amt - bal) < 1:
            ok += 1
        prev = bal if bal is not None else prev
    return round(ok / len(items), 3)


def extract_statement(tokens):
    """Return (fields: {name:(value,conf)}, line_items)."""
    lines = group_lines(tokens)
    fields = {}

    def find_line(anchors):
        for ln in lines:
            low = ln["text"].lower()
            if any(a in low for a in anchors):
                return ln["text"]
        return None

    # bank_name = top line
    fields["bank_name"] = (lines[0]["text"].lower(), 0.85) if lines else (None, 0.0)
    for f in BANK_STATEMENT.fields:
        if f == "bank_name":
            continue
        line = find_line(BANK_STATEMENT.anchors[f])
        val, conf = None, 0.0
        if line:
            conf = 0.82
            if f == "account_number":
                m = _ACC.search(line.replace(" ", "")); val = m.group() if m else None
            elif f == "statement_period":
                ds = _DATE.findall(line); val = " - ".join(ds[:2]) if ds else line.split(":")[-1].strip()
            elif f in ("opening_balance", "closing_balance"):
                val = signed_money(line)
            else:  # account_holder
                val = line.split(":")[-1].strip().lower() or None
        if val is None:
            conf = 0.0
        fields[f] = (val, conf)

    # account_number fallback: OCR may garble the "Account" anchor (e.g. "Acc0unt").
    # A bare run of >=8 consecutive digits in the header is the account number
    # (money has comma separators, dates have slashes -> no long digit run).
    if fields["account_number"][0] is None:
        for ln in lines:
            mm = _ACC.search(ln["text"].replace(" ", ""))
            if mm:
                fields["account_number"] = (mm.group(), 0.78)
                break

    return fields, extract_table(tokens)
