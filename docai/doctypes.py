"""Document-type registry — makes the pipeline multi-document (not receipt-only).

Each bank document type declares its KIE schema (fields/required/anchors), whether
it has a transaction TABLE, and a VLM prompt. The pipeline classifies the doc type
first, then dispatches to the right extractor — same serving/MLOps backbone.
Receipt KIE keeps using the trained sklearn model in docai/kie.py untouched.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class DocType:
    name: str
    fields: list[str]
    required: list[str]
    anchors: dict
    kind: str = "keyvalue"                 # keyvalue | keyvalue+table
    table_columns: list[str] = field(default_factory=list)
    vlm_prompt: str = ""


RECEIPT = DocType(
    name="receipt",
    fields=["merchant_name", "date", "total_amount", "invoice_id", "payment_method"],
    required=["date", "total_amount"],
    anchors={},                            # receipt uses the trained classifier (kie.py)
    kind="keyvalue",
    vlm_prompt=("Extract merchant_name, invoice_id, date, total_amount, payment_method "
                "from this receipt. Return ONLY valid JSON, null if not visible."),
)

BANK_STATEMENT = DocType(
    name="bank_statement",
    fields=["bank_name", "account_number", "account_holder",
            "statement_period", "opening_balance", "closing_balance"],
    required=["account_number", "closing_balance"],
    anchors={
        "bank_name": [],
        "account_number": ["account no", "account number", "a/c no", "so tai khoan",
                           "số tài khoản", "stk", "acc no"],
        "account_holder": ["account holder", "name", "chu tai khoan", "chủ tài khoản",
                           "ten", "tên"],
        "statement_period": ["period", "statement period", "from", "ky sao ke", "kỳ", "tu ngay"],
        "opening_balance": ["opening balance", "beginning balance", "so du dau", "số dư đầu"],
        "closing_balance": ["closing balance", "ending balance", "so du cuoi", "số dư cuối"],
    },
    kind="keyvalue+table",
    table_columns=["date", "description", "amount", "balance"],
    vlm_prompt=("This is a bank statement. Return ONLY JSON with keys: bank_name, "
                "account_number, account_holder, statement_period, opening_balance, "
                "closing_balance, and transactions (a list of {date, description, "
                "amount, balance}). Use null if not visible. Do not guess."),
)

REGISTRY: dict[str, DocType] = {RECEIPT.name: RECEIPT, BANK_STATEMENT.name: BANK_STATEMENT}
DOC_TYPES = list(REGISTRY.keys())


def get(name: str) -> DocType:
    return REGISTRY.get(name, RECEIPT)
