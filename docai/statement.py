"""Bank statement extractor (Processing Layer) — header KIE + TABLE parsing.

Statements are table-heavy, so beyond key-value header fields we parse the
transaction table via a layout-graph: cluster tokens into rows, detect columns
from the header row, assign each cell to its nearest column. Returns header
fields + a list of transaction rows.
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path

from .doctypes import BANK_STATEMENT
from .kie import norm_date, group_lines
from .table_structure import detect_structure, save_debug_overlay

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


def statement_table_mode() -> str:
    return os.environ.get("DOCAI_STATEMENT_TABLE_MODE", "rules").strip().lower()


def _debug_path(doc_id: str | None = None):
    root = os.environ.get("DOCAI_STATEMENT_DEBUG_DIR")
    if not root:
        return None
    stem = Path(doc_id or "statement").stem
    return Path(root) / stem


def _write_debug(base_path: Path | None, payload: dict, image_bgr=None, structure=None):
    if base_path is None:
        return
    base_path.parent.mkdir(parents=True, exist_ok=True)
    (base_path.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if image_bgr is not None and structure is not None:
        save_debug_overlay(image_bgr, structure, base_path.with_suffix(".png"))


def _semantic_map_from_header(header_cells: dict[int, str]) -> dict[str, int]:
    semantic = {}
    for col_idx, text in header_cells.items():
        low = (text or "").lower()
        for col_name, kws in _COLS.items():
            if col_name in semantic:
                continue
            if _col_match(low, kws):
                semantic[col_name] = col_idx
                break
    return semantic


def _required_statement_columns_ok(semantic: dict[str, int]) -> bool:
    has_amount = "amount" in semantic or ("debit" in semantic and "credit" in semantic)
    return "date" in semantic and "description" in semantic and has_amount and "balance" in semantic


def _header_contaminated(header_cells: dict[int, str], semantic: dict[str, int]) -> bool:
    for name, idx in semantic.items():
        text = header_cells.get(idx, "") or ""
        low = text.lower()
        if name == "date" and _DATE.search(text):
            # A header should say "date/ngay", not already contain a transaction date.
            return True
        if name in {"amount", "balance", "debit", "credit"} and signed_money(text) is not None:
            return True
        if name == "ref" and _ACC.search(text.replace(" ", "")):
            return True
        if name == "description" and any(k in low for k in _FOOTER):
            return True
    return False


def _join_rule_rows(rows, order, bounds):
    def col_of(x0):
        i = 0
        while i < len(bounds) and x0 >= bounds[i]:
            i += 1
        return order[i]

    joined_rows = []
    for row in rows:
        cells = {c: [] for c in order}
        for t in row:
            cells[col_of(t["bbox"][0])].append(t["text"])
        joined_rows.append({c: " ".join(cells[c]).strip() for c in order})
    return joined_rows


def _extract_table_rules(tokens, debug_path: Path | None = None):
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
        meta = {
            "mode": "rules",
            "ok": False,
            "reason": "header_not_found",
            "rows_detected": len(rows),
            "column_order": [],
            "joined_rows": [],
            "semantic_columns": {},
            "line_items_count": 0,
        }
        _write_debug(debug_path, meta)
        return [], meta
    order = sorted(centers, key=lambda c: centers[c])
    cs = [centers[c] for c in order]
    bounds = [(cs[i] + cs[i + 1]) / 2 for i in range(len(cs) - 1)]
    joined_rows = _join_rule_rows(rows, order, bounds)

    items = []
    for joined in joined_rows[header_idx + 1:]:
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
    semantic = {c: idx for idx, c in enumerate(order)}
    meta = {
        "mode": "rules",
        "ok": True,
        "reason": None,
        "rows_detected": len(rows),
        "column_order": order,
        "semantic_columns": semantic,
        "required_columns_ok": _required_statement_columns_ok(semantic),
        "joined_rows": joined_rows,
        "line_items_count": len(items),
        "assignment_rate": 1.0,
    }
    _write_debug(debug_path, meta)
    return items, meta


def _extract_table_tatr(tokens, image_bgr, debug_path: Path | None = None):
    structure = detect_structure(image_bgr, tokens)
    if not structure["ok"]:
        meta = {
            "mode": "tatr",
            "ok": False,
            "reason": structure.get("reason"),
            "table_bbox": structure.get("table_bbox"),
            "rows_detected": len(structure.get("rows", [])),
            "columns_detected": len(structure.get("columns", [])),
            "assignment_rate": structure.get("assignment_rate", 0.0),
            "semantic_columns": {},
            "grid_text": structure.get("grid_text", []),
            "line_items_count": 0,
        }
        _write_debug(debug_path, meta, image_bgr=image_bgr, structure=structure)
        return [], meta

    header_idx = structure.get("header_row_index") or 0
    grid = structure.get("grid_text", [])
    header_cells = {}
    if 0 <= header_idx < len(grid):
        for cell in grid[header_idx]:
            header_cells[cell["col_index"]] = cell["text"]
    semantic = _semantic_map_from_header(header_cells)
    items = []
    semantic_inv = {name: idx for name, idx in semantic.items()}

    for ridx, row in enumerate(grid):
        if ridx <= header_idx:
            continue
        joined = {name: row[idx]["text"] for name, idx in semantic_inv.items() if idx < len(row)}
        desc = joined.get("description", "")
        if any(k in desc.lower() for k in _FOOTER) or not _DATE.search(joined.get("date", "")):
            continue
        if joined.get("amount"):
            amt = signed_money(joined["amount"])
        else:
            deb = signed_money(joined.get("debit", "")) or 0.0
            cred = signed_money(joined.get("credit", "")) or 0.0
            amt = (abs(cred) - abs(deb)) if (deb or cred) else None
        bal = signed_money(joined.get("balance", ""))
        if amt is None and bal is None:
            continue
        items.append({
            "date": norm_date(joined.get("date", "")),
            "description": desc.lower() or None,
            "amount": amt,
            "balance": bal,
        })

    meta = {
        "mode": "tatr",
        "ok": True,
        "reason": None,
        "table_bbox": structure.get("table_bbox"),
        "rows_detected": len(structure.get("rows", [])),
        "columns_detected": len(structure.get("columns", [])),
        "header_row_index": header_idx,
        "header_cells": header_cells,
        "semantic_columns": semantic,
        "required_columns_ok": _required_statement_columns_ok(semantic),
        "header_contaminated": _header_contaminated(header_cells, semantic),
        "assignment_rate": structure.get("assignment_rate", 0.0),
        "grid_text": grid,
        "line_items_count": len(items),
    }
    _write_debug(debug_path, meta, image_bgr=image_bgr, structure=structure)
    return items, meta


def extract_table(tokens, image_bgr=None, mode: str | None = None, debug_path: Path | None = None,
                  return_meta: bool = False):
    mode = (mode or statement_table_mode()).lower()
    if mode == "rules":
        items, meta = _extract_table_rules(tokens, debug_path=debug_path)
    elif mode == "tatr":
        if image_bgr is None:
            items, meta = _extract_table_rules(tokens, debug_path=debug_path)
            meta["reason"] = "image_missing_for_tatr"
        else:
            items, meta = _extract_table_tatr(tokens, image_bgr, debug_path=debug_path)
    elif mode == "hybrid":
        if image_bgr is None:
            items, meta = _extract_table_rules(tokens, debug_path=debug_path)
            meta["reason"] = "image_missing_for_hybrid"
        else:
            t_items, t_meta = _extract_table_tatr(tokens, image_bgr, debug_path=debug_path)
            use_tatr = (
                t_meta.get("ok")
                and t_meta.get("required_columns_ok")
                and not t_meta.get("header_contaminated", False)
                and t_meta.get("assignment_rate", 0.0) >= 0.75
            )
            if use_tatr:
                items, meta = t_items, t_meta
                meta["selected_mode"] = "tatr"
            else:
                items, meta = _extract_table_rules(tokens, debug_path=debug_path)
                meta["selected_mode"] = "rules"
                meta["fallback_from_tatr"] = t_meta
            _write_debug(debug_path, meta)
    else:
        raise ValueError(f"unknown statement table mode: {mode}")
    if return_meta:
        return items, meta
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


def reconcile_closing(items, opening, closing) -> float:
    """Global balance consistency: opening + Σ(amount) ≈ closing."""
    if not items or opening is None or closing is None:
        return 0.0
    amounts = [it.get("amount") for it in items if it.get("amount") is not None]
    if not amounts:
        return 0.0
    est = round(opening + sum(amounts), 2)
    return 1.0 if abs(est - closing) < 1.0 else 0.0


def extract_statement(tokens, image_bgr=None, mode: str | None = None,
                      debug_path: Path | None = None, return_meta: bool = False):
    """Return (fields: {name:(value,conf)}, line_items[, meta])."""
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

    line_items, table_meta = extract_table(tokens, image_bgr=image_bgr, mode=mode,
                                           debug_path=debug_path, return_meta=True)
    meta = {
        "table_mode": (mode or statement_table_mode()).lower(),
        "table": table_meta,
        "row_reconcile": reconcile(
            line_items,
            fields.get("opening_balance", (None,))[0],
            fields.get("closing_balance", (None,))[0],
        ),
        "closing_reconcile": reconcile_closing(
            line_items,
            fields.get("opening_balance", (None,))[0],
            fields.get("closing_balance", (None,))[0],
        ),
    }
    if return_meta:
        return fields, line_items, meta
    return fields, line_items
