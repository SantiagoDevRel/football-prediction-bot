"""One-shot migration: add picks.source/parlay_group_id/note + chat_history.

Idempotent: checks for existing columns/tables before altering.
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402


def column_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == col for r in rows)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def main() -> None:
    conn = sqlite3.connect(settings.db_path)
    try:
        if not column_exists(conn, "picks", "source"):
            conn.execute("ALTER TABLE picks ADD COLUMN source TEXT DEFAULT 'model'")
            conn.execute("UPDATE picks SET source = 'model' WHERE source IS NULL")
            print("[+] picks.source added")
        else:
            print("[=] picks.source already exists")

        if not column_exists(conn, "picks", "parlay_group_id"):
            conn.execute("ALTER TABLE picks ADD COLUMN parlay_group_id INTEGER")
            print("[+] picks.parlay_group_id added")
        else:
            print("[=] picks.parlay_group_id already exists")

        if not column_exists(conn, "picks", "note"):
            conn.execute("ALTER TABLE picks ADD COLUMN note TEXT")
            print("[+] picks.note added")
        else:
            print("[=] picks.note already exists")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_picks_source ON picks(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_picks_parlay ON picks(parlay_group_id)")

        if not table_exists(conn, "chat_history"):
            conn.execute("""
                CREATE TABLE chat_history (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id      TEXT NOT NULL,
                    role         TEXT NOT NULL,
                    content      TEXT NOT NULL,
                    tool_calls   TEXT,
                    tokens_in    INTEGER,
                    tokens_out   INTEGER,
                    model        TEXT,
                    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX idx_chat_chat_time ON chat_history(chat_id, created_at)")
            print("[+] chat_history table created")
        else:
            print("[=] chat_history table already exists")

        conn.commit()
        print("[OK] migration complete")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
