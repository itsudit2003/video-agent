import os
import sqlite3
import time
from contextlib import closing

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "video_agent.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings(
  id TEXT PRIMARY KEY, title TEXT, source TEXT, language TEXT,
  transcript TEXT, summary TEXT, action_items TEXT, key_decisions TEXT, open_questions TEXT,
  chunks INTEGER, seconds INTEGER, created_at REAL
);
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  recording_id TEXT REFERENCES recordings(id) ON DELETE CASCADE,
  role TEXT, content TEXT, created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_messages_recording ON messages(recording_id);
"""


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    c.executescript(SCHEMA)
    return c


def save_recording(rid: str, r: dict, source: str, language: str) -> None:
    with closing(_conn()) as c, c:
        c.execute(
            """INSERT INTO recordings
               (id, title, source, language, transcript, summary, action_items,
                key_decisions, open_questions, chunks, seconds, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 title=excluded.title, transcript=excluded.transcript, summary=excluded.summary,
                 action_items=excluded.action_items, key_decisions=excluded.key_decisions,
                 open_questions=excluded.open_questions, chunks=excluded.chunks, seconds=excluded.seconds""",
            (rid, r["title"], source, language, r["transcript"], r["summary"], r["action_items"],
             r["key_decisions"], r["open_questions"], r["chunks"], r["seconds"], time.time()),
        )


def get_recording(rid: str) -> dict | None:
    with closing(_conn()) as c:
        row = c.execute("SELECT * FROM recordings WHERE id=?", (rid,)).fetchone()
        return dict(row) if row else None


def list_recordings() -> list[dict]:
    with closing(_conn()) as c:
        rows = c.execute(
            "SELECT id, title, language, created_at FROM recordings ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def add_message(rid: str, role: str, content: str) -> None:
    with closing(_conn()) as c, c:
        c.execute(
            "INSERT INTO messages (recording_id, role, content, created_at) VALUES (?,?,?,?)",
            (rid, role, content, time.time()),
        )


def get_messages(rid: str) -> list[dict]:
    with closing(_conn()) as c:
        rows = c.execute(
            "SELECT role, content FROM messages WHERE recording_id=? ORDER BY id", (rid,)
        ).fetchall()
        return [dict(r) for r in rows]


def delete_recording(rid: str) -> None:
    """Also call vector_store.delete_recording_vectors(rid)."""
    with closing(_conn()) as c, c:
        c.execute("DELETE FROM recordings WHERE id=?", (rid,))