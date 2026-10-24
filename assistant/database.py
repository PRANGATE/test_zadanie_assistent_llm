"""SQLite repository using a short-lived connection per operation.

This avoids SQLite's cross-thread connection error in a Tkinter application.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import DB_PATH


def re_words(text):
    return [word.strip(".,!?;:()\"'«»") for word in text.casefold().split()]


class Repository:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        con = sqlite3.connect(str(self.db_path), timeout=15)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout=15000")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    @contextmanager
    def _connection(self):
        con = self._connect()
        try:
            with con:
                yield con
        finally:
            con.close()

    def _initialize(self):
        with self._connection() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL CHECK(kind IN ('task','reminder','meeting')),
                    title TEXT NOT NULL CHECK(length(trim(title)) > 0),
                    due_at TEXT,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','done'))
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_entries_due ON entries(due_at)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_entries_status ON entries(status)")

    def create(self, kind, title, due_at=None):
        if kind not in {"task", "reminder", "meeting"}:
            raise ValueError("Неизвестный тип записи")
        title = (title or "").strip()
        if not title:
            raise ValueError("Название не может быть пустым")
        created_at = datetime.now().isoformat(timespec="seconds")
        with self._connection() as con:
            cur = con.execute(
                "INSERT INTO entries(kind,title,due_at,created_at,status) VALUES(?,?,?,?, 'open')",
                (kind, title, due_at, created_at),
            )
            entry_id = cur.lastrowid
        return self.get(entry_id)

    def get(self, entry_id):
        with self._connection() as con:
            row = con.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()
        return dict(row) if row else None

    def all(self, limit=200, include_done=True):
        sql = "SELECT * FROM entries"
        args = []
        if not include_done:
            sql += " WHERE status='open'"
        sql += " ORDER BY CASE WHEN due_at IS NULL THEN 1 ELSE 0 END, due_at ASC, created_at DESC LIMIT ?"
        args.append(max(1, min(int(limit), 1000)))
        with self._connection() as con:
            rows = con.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def update_due(self, entry_id, due_at):
        with self._connection() as con:
            cur = con.execute("UPDATE entries SET due_at=? WHERE id=?", (due_at, entry_id))
        return cur.rowcount > 0

    def update_title(self, entry_id, title):
        """Rename an entry without changing its date, kind, or status."""
        title = (title or "").strip()
        if not title:
            raise ValueError("Название не может быть пустым")
        with self._connection() as con:
            cur = con.execute("UPDATE entries SET title=? WHERE id=?", (title, entry_id))
        return cur.rowcount > 0

    def delete(self, entry_id):
        with self._connection() as con:
            cur = con.execute("DELETE FROM entries WHERE id=?", (entry_id,))
        return cur.rowcount > 0

    def complete(self, entry_id):
        with self._connection() as con:
            cur = con.execute("UPDATE entries SET status='done' WHERE id=?", (entry_id,))
        return cur.rowcount > 0

    def find(self, query, kind=None, limit=10, include_done=True):
        terms = [term.casefold().strip(".,!?;:()\"'«»") for term in (query or "").split()]
        terms = [t for t in terms if len(t) > 2]
        rows = self.all(limit=1000, include_done=include_done)
        scored = []
        for row in rows:
            if kind and kind != "any" and row["kind"] != kind:
                continue
            haystack = f"{row['title']} {row['kind']}".casefold()
            title_words = re_words(haystack)
            score = 0
            for term in terms:
                if term in haystack:
                    score += 1
                elif len(term) >= 6 and any(word.startswith(term[:6]) or term.startswith(word[:6]) for word in title_words if len(word) >= 6):
                    score += 1
            if score:
                scored.append((score, row["created_at"], row))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:max(1, min(int(limit), 100))]]

    def between(self, start_iso, end_iso, kind=None, include_done=True):
        sql = "SELECT * FROM entries WHERE due_at IS NOT NULL AND due_at >= ? AND due_at < ?"
        args = [start_iso, end_iso]
        if kind and kind != "any":
            sql += " AND kind=?"
            args.append(kind)
        if not include_done:
            sql += " AND status='open'"
        sql += " ORDER BY due_at ASC"
        with self._connection() as con:
            rows = con.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        """Compatibility no-op: every method owns and closes its own connection."""
        return None
