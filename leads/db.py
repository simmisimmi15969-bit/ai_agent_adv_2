"""Leads database: save leads once (no duplicates), store AI scores, and query the best ones."""
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "leads.db"
STATUSES = {"new", "shortlisted", "ignored", "contacted"}


def _connect() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row  # rows behave like dicts: row["title"]
    db.execute(
        """CREATE TABLE IF NOT EXISTS leads (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            url       TEXT NOT NULL UNIQUE,
            source    TEXT NOT NULL,
            kind      TEXT NOT NULL,
            title     TEXT NOT NULL,
            text      TEXT,
            budget    TEXT,
            services  TEXT,
            posted_at TEXT,
            found_at  TEXT NOT NULL,
            score     INTEGER,
            why       TEXT,
            status    TEXT NOT NULL DEFAULT 'new'
        )"""
    )
    return db


def _as_dict(row: sqlite3.Row) -> dict:
    lead = dict(row)
    lead["services"] = json.loads(lead["services"] or "[]")
    return lead


def save_new(leads: list[dict]) -> list[dict]:
    """Save leads whose URL we have not seen before; return only the new ones (with their id)."""
    today = date.today().isoformat()
    new = []
    with _connect() as db:
        for lead in leads:
            cursor = db.execute(
                """INSERT OR IGNORE INTO leads (url, source, kind, title, text, budget, services, posted_at, found_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (lead["url"], lead["source"], lead["kind"], lead["title"], lead["text"], lead["budget"],
                 json.dumps(lead.get("services", [])), lead["posted_at"], today),
            )
            if cursor.rowcount:  # 1 = inserted, 0 = this URL was already saved
                new.append({**lead, "id": cursor.lastrowid})
    return new


def unscored(limit: int = 50) -> list[dict]:
    """Leads the AI has not scored yet, newest first."""
    with _connect() as db:
        rows = db.execute("SELECT * FROM leads WHERE score IS NULL ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_as_dict(r) for r in rows]


def set_score(lead_id: int, score: int, why: str) -> None:
    with _connect() as db:
        db.execute("UPDATE leads SET score = ?, why = ? WHERE id = ?", (score, why, lead_id))


def set_status(lead_id: int, status: str) -> None:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}")
    with _connect() as db:
        db.execute("UPDATE leads SET status = ? WHERE id = ?", (status, lead_id))


def top(kind: str | None = None, min_score: int = 6, service: str | None = None,
        days: int = 7, limit: int = 10) -> list[dict]:
    """Best scored leads found in the last `days` days, optionally only one kind or one service."""
    since = (date.today() - timedelta(days=days)).isoformat()
    sql = "SELECT * FROM leads WHERE score >= ? AND found_at >= ? AND status != 'ignored'"
    params: list = [min_score, since]
    if kind:
        sql += " AND kind = ?"
        params.append(kind)
    if service:
        sql += " AND services LIKE ?"
        params.append(f'%"{service}"%')
    sql += " ORDER BY score DESC, id DESC LIMIT ?"
    params.append(limit)
    with _connect() as db:
        rows = db.execute(sql, params).fetchall()
    return [_as_dict(r) for r in rows]


if __name__ == "__main__":
    # Test on a temporary database so real data is never touched.
    import tempfile

    DB_PATH = Path(tempfile.mkdtemp()) / "test_leads.db"
    fake = [
        {"url": "https://example.com/1", "source": "test", "kind": "project", "title": "Build an AI chatbot",
         "text": "Need a RAG chatbot", "budget": "5000 USD", "services": ["ai agents & chatbots"], "posted_at": "2026-10-15"},
        {"url": "https://example.com/2", "source": "test", "kind": "company", "title": "Startup raises $5M",
         "text": "Fintech startup", "budget": None, "services": [], "posted_at": "2026-10-15"},
    ]
    print("First save, new leads:   ", [l["id"] for l in save_new(fake)])
    print("Second save, new leads:  ", [l["id"] for l in save_new(fake)], "(duplicates skipped)")
    print("Waiting for AI score:    ", [l["title"] for l in unscored()])

    set_score(1, 9, "Needs a RAG chatbot with a good budget")
    set_score(2, 5, "Fintech, not sure they need us")
    print("Top leads (score >= 6):  ", [(l["id"], l["score"], l["title"]) for l in top()])
    print("Top AI-agent projects:   ", [l["title"] for l in top(kind="project", service="ai agents & chatbots")])

    set_status(1, "shortlisted")
    print("Status of lead 1:        ", top()[0]["status"])

    DB_PATH.unlink()
    print("Test database deleted:   ", not DB_PATH.exists())
