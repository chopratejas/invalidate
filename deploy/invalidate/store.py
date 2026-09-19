"""Storage. SQLite by default; implement `Store` to put invalidate in front of anything else."""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from typing import Any, Protocol

from .types import Disposition, Event, Memory, Status, Verdict, Votes, now


class Store(Protocol):
    def add_memory(self, m: Memory) -> None: ...
    def get_memory(self, memory_id: str) -> Memory | None: ...
    def update_memory(self, m: Memory) -> None: ...
    def list_memories(
        self, namespace: str | None = None, statuses: Iterable[Status] | None = None, limit: int | None = None
    ) -> list[Memory]: ...
    def add_event(self, e: Event) -> None: ...
    def get_event(self, event_id: str) -> Event | None: ...
    def list_events(self, namespace: str | None = None, limit: int | None = None) -> list[Event]: ...
    def add_verdicts(self, verdicts: Iterable[Verdict]) -> None: ...
    def list_verdicts(self, memory_id: str | None = None, event_id: str | None = None) -> list[Verdict]: ...
    def close(self) -> None: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  namespace TEXT NOT NULL DEFAULT 'default',
  fact TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'fact',
  source TEXT NOT NULL DEFAULT 'unknown',
  status TEXT NOT NULL DEFAULT 'active',
  p_true REAL NOT NULL DEFAULT 1.0,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  last_checked REAL,
  expires_at REAL,
  superseded_by TEXT,
  metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS memories_ns_status ON memories(namespace, status);

CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  namespace TEXT NOT NULL DEFAULT 'default',
  text TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'unknown',
  created_at REAL NOT NULL,
  metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS verdicts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL,
  memory_id TEXT NOT NULL,
  bears REAL NOT NULL,
  still_true REAL NOT NULL,
  replaces REAL NOT NULL,
  hypothetical REAL NOT NULL,
  directive REAL NOT NULL DEFAULT 0,
  partial REAL NOT NULL DEFAULT 0,
  disposition TEXT NOT NULL,
  from_status TEXT NOT NULL,
  to_status TEXT NOT NULL,
  applied INTEGER NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS verdicts_memory ON verdicts(memory_id, created_at);
CREATE INDEX IF NOT EXISTS verdicts_event ON verdicts(event_id);
"""


class SQLiteStore:
    """Thread-safe SQLite store. `path=":memory:"` for tests."""

    def __init__(self, path: str = "invalidate.db") -> None:
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(verdicts)")}
        for col in ("directive", "partial"):
            if col not in cols:
                self._conn.execute(f"ALTER TABLE verdicts ADD COLUMN {col} REAL NOT NULL DEFAULT 0")

    # -- memories -----------------------------------------------------------
    def add_memory(self, m: Memory) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories (id, namespace, fact, kind, source, status, p_true, created_at, updated_at,"
                " last_checked, expires_at, superseded_by, metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    m.id, m.namespace, m.fact, m.kind, m.source, m.status.value, m.p_true, m.created_at,
                    m.updated_at, m.last_checked, m.expires_at, m.superseded_by, json.dumps(m.metadata),
                ),
            )

    def get_memory(self, memory_id: str) -> Memory | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return _row_to_memory(row) if row else None

    def update_memory(self, m: Memory) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET namespace=?, fact=?, kind=?, source=?, status=?, p_true=?, updated_at=?,"
                " last_checked=?, expires_at=?, superseded_by=?, metadata=? WHERE id=?",
                (
                    m.namespace, m.fact, m.kind, m.source, m.status.value, m.p_true, m.updated_at,
                    m.last_checked, m.expires_at, m.superseded_by, json.dumps(m.metadata), m.id,
                ),
            )

    def list_memories(
        self, namespace: str | None = None, statuses: Iterable[Status] | None = None, limit: int | None = None
    ) -> list[Memory]:
        sql = "SELECT * FROM memories"
        clauses: list[str] = []
        args: list[Any] = []
        if namespace is not None:
            clauses.append("namespace = ?")
            args.append(namespace)
        if statuses is not None:
            vals = [s.value for s in statuses]
            if not vals:
                return []
            clauses.append(f"status IN ({','.join('?' * len(vals))})")
            args.extend(vals)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at ASC, rowid ASC"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [_row_to_memory(r) for r in rows]

    # -- events ---------------------------------------------------------------
    def add_event(self, e: Event) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (id, namespace, text, source, created_at, metadata) VALUES (?,?,?,?,?,?)",
                (e.id, e.namespace, e.text, e.source, e.created_at, json.dumps(e.metadata)),
            )

    def get_event(self, event_id: str) -> Event | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return _row_to_event(row) if row else None

    def list_events(self, namespace: str | None = None, limit: int | None = None) -> list[Event]:
        sql = "SELECT * FROM events"
        args: list[Any] = []
        if namespace is not None:
            sql += " WHERE namespace = ?"
            args.append(namespace)
        sql += " ORDER BY created_at DESC, rowid DESC"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [_row_to_event(r) for r in rows]

    # -- verdicts -------------------------------------------------------------
    def add_verdicts(self, verdicts: Iterable[Verdict]) -> None:
        rows = [
            (
                v.event_id, v.memory_id, v.votes.bears, v.votes.still_true, v.votes.replaces, v.votes.hypothetical,
                v.votes.directive, v.votes.partial, v.disposition.value, v.from_status.value, v.to_status.value, int(v.applied), v.created_at,
            )
            for v in verdicts
        ]
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT INTO verdicts (event_id, memory_id, bears, still_true, replaces, hypothetical, directive,"
                " partial, disposition, from_status, to_status, applied, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )

    def list_verdicts(self, memory_id: str | None = None, event_id: str | None = None) -> list[Verdict]:
        sql = "SELECT * FROM verdicts"
        clauses: list[str] = []
        args: list[Any] = []
        if memory_id is not None:
            clauses.append("memory_id = ?")
            args.append(memory_id)
        if event_id is not None:
            clauses.append("event_id = ?")
            args.append(event_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at ASC, id ASC"
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [_row_to_verdict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _row_to_memory(r: sqlite3.Row) -> Memory:
    return Memory(
        id=r["id"], namespace=r["namespace"], fact=r["fact"], kind=r["kind"], source=r["source"],
        status=Status(r["status"]), p_true=r["p_true"], created_at=r["created_at"], updated_at=r["updated_at"],
        last_checked=r["last_checked"], expires_at=r["expires_at"], superseded_by=r["superseded_by"],
        metadata=json.loads(r["metadata"] or "{}"),
    )


def _row_to_event(r: sqlite3.Row) -> Event:
    return Event(
        id=r["id"], namespace=r["namespace"], text=r["text"], source=r["source"], created_at=r["created_at"],
        metadata=json.loads(r["metadata"] or "{}"),
    )


def _row_to_verdict(r: sqlite3.Row) -> Verdict:
    return Verdict(
        id=r["id"], event_id=r["event_id"], memory_id=r["memory_id"],
        votes=Votes(bears=r["bears"], still_true=r["still_true"], replaces=r["replaces"], hypothetical=r["hypothetical"],
                    directive=r["directive"], partial=r["partial"]),
        disposition=Disposition(r["disposition"]), from_status=Status(r["from_status"]), to_status=Status(r["to_status"]),
        applied=bool(r["applied"]), created_at=r["created_at"],
    )


__all__ = ["Store", "SQLiteStore", "now"]
