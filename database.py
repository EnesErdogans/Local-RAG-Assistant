import sqlite3
import json
from config import DB_PATH


def init_db():
    """Create the documents table if it does not exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source      TEXT    NOT NULL,
                chunk_index INTEGER NOT NULL,
                content     TEXT    NOT NULL,
                embedding   TEXT    NOT NULL
            )
        """)
        conn.commit()


def insert_chunk(source: str, chunk_index: int, content: str, embedding: list) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO documents (source, chunk_index, content, embedding) VALUES (?, ?, ?, ?)",
            (source, chunk_index, content, json.dumps(embedding)),
        )
        conn.commit()


def get_all_chunks() -> list[dict]:
    """Return every row as a dict with keys: id, source, content, embedding."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, source, content, embedding FROM documents"
        ).fetchall()
    return [
        {
            "id": r[0],
            "source": r[1],
            "content": r[2],
            "embedding": json.loads(r[3]),
        }
        for r in rows
    ]


def count_chunks() -> int:
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


def list_sources() -> list[str]:
    """Return distinct source filenames stored in the DB."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT DISTINCT source FROM documents ORDER BY source"
        ).fetchall()
    return [r[0] for r in rows]


def clear_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM documents")
        conn.commit()
