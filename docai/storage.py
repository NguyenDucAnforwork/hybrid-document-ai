"""SQLite metadata store: job/doc state + results (production parity)."""
from __future__ import annotations
import json
import sqlite3
import threading
from .config import META_DB

_lock = threading.Lock()


def _conn():
    c = sqlite3.connect(META_DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _lock, _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs(
                job_id TEXT PRIMARY KEY, status TEXT, total INTEGER, summary TEXT);
            CREATE TABLE IF NOT EXISTS docs(
                doc_id TEXT PRIMARY KEY, job_id TEXT, state TEXT,
                retries INTEGER DEFAULT 0, result TEXT, error TEXT);
            """
        )


def create_job(job_id, total):
    with _lock, _conn() as c:
        c.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?)",
                  (job_id, "queued", total, json.dumps({})))


def add_doc(doc_id, job_id):
    with _lock, _conn() as c:
        c.execute("INSERT OR REPLACE INTO docs(doc_id,job_id,state) VALUES (?,?,?)",
                  (doc_id, job_id, "uploaded"))


def set_doc(doc_id, *, state=None, result=None, error=None, inc_retry=False):
    with _lock, _conn() as c:
        if state:
            c.execute("UPDATE docs SET state=? WHERE doc_id=?", (state, doc_id))
        if result is not None:
            c.execute("UPDATE docs SET result=? WHERE doc_id=?", (json.dumps(result), doc_id))
        if error is not None:
            c.execute("UPDATE docs SET error=? WHERE doc_id=?", (error, doc_id))
        if inc_retry:
            c.execute("UPDATE docs SET retries=retries+1 WHERE doc_id=?", (doc_id,))


def get_doc_retries(doc_id) -> int:
    with _lock, _conn() as c:
        r = c.execute("SELECT retries FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
        return r["retries"] if r else 0


def set_job(job_id, status=None, summary=None):
    with _lock, _conn() as c:
        if status:
            c.execute("UPDATE jobs SET status=? WHERE job_id=?", (status, job_id))
        if summary is not None:
            c.execute("UPDATE jobs SET summary=? WHERE job_id=?", (json.dumps(summary), job_id))


def get_job(job_id) -> dict | None:
    with _lock, _conn() as c:
        j = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if not j:
            return None
        return {"job_id": j["job_id"], "status": j["status"], "total": j["total"],
                "summary": json.loads(j["summary"] or "{}")}


def get_results(job_id) -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute("SELECT result FROM docs WHERE job_id=? AND result IS NOT NULL",
                         (job_id,)).fetchall()
        return [json.loads(r["result"]) for r in rows]
