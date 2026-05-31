"""Tests for multi-document support: doc-type router + statement table parsing."""
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docai.synth import gen_statement, gen_one, gen_payment_order, gen_statement_hard
from docai.classifier import get_classifier
from docai.statement import extract_statement, signed_money
from docai.kv import kv_extract
from docai.doctypes import PAYMENT_ORDER


def _wh(tokens):
    return (max(t["bbox"][2] for t in tokens), max(t["bbox"][3] for t in tokens))


def test_signed_money():
    assert signed_money("-4,500,000") == -4500000.0
    assert signed_money("30,063,058") == 30063058.0


def test_doctype_router_separates():
    rng = random.Random(1)
    _, stmt_tokens, _ = gen_statement(rng)
    _, rcpt_tokens, _ = gen_one(rng)
    clf = get_classifier()
    sW, sH = _wh(stmt_tokens); rW, rH = _wh(rcpt_tokens)
    assert clf.predict(stmt_tokens, sW, sH)[0] == "bank_statement"
    assert clf.predict(rcpt_tokens, rW, rH)[0] == "receipt"


def test_statement_table_parsed():
    rng = random.Random(2)
    _, tokens, gold = gen_statement(rng)
    fields, items = extract_statement(tokens)
    assert fields["account_number"][0] == gold["account_number"]
    assert len(items) >= 1
    assert all("amount" in it for it in items)   # table rows have an amount column


def test_signed_money_formats():
    assert signed_money("(1,500,000)") == -1500000.0     # parentheses = negative
    assert signed_money("9,900,000 CR") == 9900000.0      # CR = positive
    assert signed_money("4,500,000 DR") == -4500000.0     # DR = negative


def test_payment_order_router_and_kv():
    rng = random.Random(3)
    _, tokens, gold = gen_payment_order(rng)
    W = max(t["bbox"][2] for t in tokens); H = max(t["bbox"][3] for t in tokens)
    assert get_classifier().predict(tokens, W, H)[0] == "payment_order"
    f = kv_extract(tokens, PAYMENT_ORDER)
    assert f["beneficiary_account"][0] == gold["beneficiary_account"]
    assert f["amount"][0] == gold["amount"]


def test_hard_statement_still_classified():
    rng = random.Random(4)
    _, tokens, _ = gen_statement_hard(rng)
    W = max(t["bbox"][2] for t in tokens); H = max(t["bbox"][3] for t in tokens)
    assert get_classifier().predict(tokens, W, H)[0] == "bank_statement"
