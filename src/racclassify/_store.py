"""SQLite-backed learning store for exact-match classification lookup.

Each time a document is classified (by any method), the result can be
stored here. On subsequent calls with the same doc_id, the stored label
is returned instantly at confidence 1.0 — no embedding needed.

This implements the experience-replay pattern from Lin (1992): past
decisions inform future ones, making the classifier faster and more
consistent over time.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


class LearningStore:
    """Thread-safe SQLite store for classification results."""

    _CREATE = """
    CREATE TABLE IF NOT EXISTS classifications (
        doc_id      TEXT NOT NULL,
        namespace   TEXT NOT NULL,
        label       TEXT NOT NULL,
        confidence  REAL NOT NULL,
        text_prefix TEXT,
        recorded_at TEXT NOT NULL,
        PRIMARY KEY (doc_id, namespace)
    )
    """
    _INDEX = "CREATE INDEX IF NOT EXISTS idx_ns ON classifications (namespace)"

    def __init__(self, path: str | Path, namespace: str = "default") -> None:
        self._path = Path(path)
        self._ns = namespace
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._conn()
        conn.execute(self._CREATE)
        conn.execute(self._INDEX)
        conn.commit()

    def lookup(self, doc_id: str) -> tuple[str, float] | None:
        """Return (label, confidence) for a previously classified doc, or None."""
        row = self._conn().execute(
            "SELECT label, confidence FROM classifications WHERE doc_id=? AND namespace=?",
            (doc_id, self._ns),
        ).fetchone()
        return (row["label"], row["confidence"]) if row else None

    def record(
        self,
        doc_id: str,
        label: str,
        confidence: float,
        text: str | None = None,
    ) -> None:
        """Store or update a classification result."""
        now = datetime.now(timezone.utc).isoformat()
        prefix = (text or "")[:200]
        self._conn().execute(
            """
            INSERT INTO classifications (doc_id, namespace, label, confidence, text_prefix, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id, namespace) DO UPDATE SET
                label=excluded.label,
                confidence=excluded.confidence,
                text_prefix=excluded.text_prefix,
                recorded_at=excluded.recorded_at
            """,
            (doc_id, self._ns, label, confidence, prefix, now),
        )
        self._conn().commit()

    def delete(self, doc_id: str) -> None:
        """Remove a stored classification (e.g. to force re-evaluation)."""
        self._conn().execute(
            "DELETE FROM classifications WHERE doc_id=? AND namespace=?",
            (doc_id, self._ns),
        )
        self._conn().commit()

    def stats(self) -> dict[str, object]:
        """Return distribution of stored labels for monitoring."""
        rows = self._conn().execute(
            "SELECT label, COUNT(*) as n FROM classifications WHERE namespace=? GROUP BY label ORDER BY n DESC",
            (self._ns,),
        ).fetchall()
        total = sum(r["n"] for r in rows)
        return {
            "total": total,
            "namespace": self._ns,
            "distribution": {r["label"]: r["n"] for r in rows},
        }
