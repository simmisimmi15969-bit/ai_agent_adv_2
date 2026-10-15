"""Long-term memory: facts about each user (name, language, past topics), saved in a SQLite file."""
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "memory" / "users.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.execute(
        """CREATE TABLE IF NOT EXISTS facts (
            user_id    TEXT NOT NULL,
            key        TEXT NOT NULL,
            value      TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, key)
        )"""
    )
    return db


def save_facts(user_id: str, facts: dict) -> None:
    """Save or update facts about a user, e.g. {"name": "Seema", "language": "Hindi"}."""
    now = datetime.now().isoformat(timespec="seconds")
    with _connect() as db:
        for key, value in facts.items():
            db.execute(
                "INSERT OR REPLACE INTO facts (user_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
                (user_id, key, str(value), now),
            )


def load_facts(user_id: str) -> dict:
    """Return everything remembered about a user as a dict (empty if nothing yet)."""
    with _connect() as db:
        rows = db.execute("SELECT key, value FROM facts WHERE user_id = ? ORDER BY key", (user_id,)).fetchall()
    return dict(rows)


def forget_user(user_id: str) -> None:
    """Delete everything remembered about a user."""
    with _connect() as db:
        db.execute("DELETE FROM facts WHERE user_id = ?", (user_id,))


if __name__ == "__main__":
    save_facts("test_user", {"name": "Seema", "language": "Hindi"})
    print("After saving:  ", load_facts("test_user"))

    save_facts("test_user", {"last_topic": "work from home policy", "language": "English"})
    print("After updating:", load_facts("test_user"))

    forget_user("test_user")
    print("After forget:  ", load_facts("test_user"))
    print("Database file: ", DB_PATH)
