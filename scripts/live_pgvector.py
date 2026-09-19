"""Govern a real Postgres + pgvector table with the real Jev.

Creates a temp table with a vector(4) column, seeds the README demo facts, syncs a flag-mode
Governor, observes "we migrated to SQLite last Tuesday", prints each row's JSONB receipt, a
`live_where()` query, then a delete-mode run with a successor insert. Drops the table at the end.

Usage:
    INVALIDATE_PG_DSN=postgresql://user:pass@localhost:5432/db TYPESAFE_API_KEY=... python scripts/live_pgvector.py
    (the key may live in .env; the DSN is never printed)

Needs: pip install "psycopg[binary]" and the pgvector extension installed on the server.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import invalidate  # noqa: E402

invalidate.load_dotenv()
if not os.environ.get("TYPESAFE_API_KEY"):
    print("TYPESAFE_API_KEY not set (export it or put it in .env)")
    sys.exit(2)
DSN = os.environ.get("INVALIDATE_PG_DSN")
if not DSN:
    print("INVALIDATE_PG_DSN not set (e.g. postgresql://user:pass@localhost:5432/db)")
    sys.exit(2)

try:
    import psycopg  # noqa: E402
except ImportError:
    print('psycopg not installed: pip install "psycopg[binary]"')
    sys.exit(2)

from invalidate.adapters import Governor  # noqa: E402
from invalidate.adapters.pgvector import PgvectorAdapter, governed_rows  # noqa: E402

FACTS = [
    ("user prefers Postgres", {"kind": "preference"}),
    ("deploys run at 2pm UTC", {"kind": "schedule"}),
    ("Alice owns the billing service", None),
    ("users.email is nullable", {"kind": "config"}),
    ("lunch is at noon on Fridays", {"kind": "fact"}),
    ("prod reads go through the Postgres replica", {"kind": "config"}),
]
EVENT = "we migrated to SQLite last Tuesday"
QUERY = "which database?"
DIMS = 4


def embed(text: str) -> list[float]:
    """A tiny deterministic bag-of-words stand-in so the demo needs no model."""
    v = [0.0] * DIMS
    for tok in text.lower().replace("?", "").split():
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIMS] += 1.0
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def vec_literal(v: list[float]) -> str:
    return "[" + ",".join(repr(x) for x in v) + "]"


def main() -> int:
    conn = psycopg.connect(DSN)
    table = f"invalidate_live_{uuid.uuid4().hex[:8]}"
    q = f'"{table}"'
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"CREATE TABLE {q} (id bigserial PRIMARY KEY, text text NOT NULL, metadata jsonb, embedding vector({DIMS}))")
        for text, meta in FACTS:
            cur.execute(f"INSERT INTO {q} (text, metadata, embedding) VALUES (%s, %s::jsonb, %s)",
                        (text, None if meta is None else json.dumps(meta), vec_literal(embed(text))))
    conn.commit()
    print(f"table {table}: {len(FACTS)} rows seeded\n")

    try:
        # -- flag mode -----------------------------------------------------------------
        adapter = PgvectorAdapter(conn, table, vector_fn=embed)
        gov = Governor(adapter, ":memory:", mode="flag")
        print("sync:", gov.sync())
        rep = gov.observe(EVENT, source="slack")
        print("observe:", rep.summary())
        for p in rep.pushes:
            print(f"  push {p.action} id={p.host_id} -> {p.status.value}" + (f"  ERROR {p.error}" if p.error else ""))
        with conn.cursor() as cur:
            cur.execute(f"SELECT id, text, metadata FROM {q} ORDER BY id")
            for rid, text, meta in cur.fetchall():
                status = (meta or {}).get("invalidate_status", "-")
                print(f"  [{rid}] {status:12} {text}")
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, text FROM {q} WHERE {adapter.live_where()} ORDER BY embedding <-> %s LIMIT 5",
                (vec_literal(embed(QUERY)),),
            )
            hits = governed_rows(cur.fetchall(), gov)
        conn.commit()
        print(f"\ngoverned query {QUERY!r}:")
        for rid, text in hits:
            print(f"  [{rid}] {text}")

        # -- delete mode with a successor -----------------------------------------------
        gov2 = Governor(adapter, ":memory:", mode="delete", successors=True)
        gov2.sync()
        rep2 = gov2.observe("we are on Postgres 16 now, replica retired", source="ops")
        print("\ndelete-mode observe:", rep2.summary(), "successor:", rep2.successor_host_id)
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {q}")
            print("rows now:", cur.fetchone()[0])
        conn.commit()
        gov.close()
        gov2.close()
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {q}")
        conn.commit()
        conn.close()
        print(f"\ndropped {table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
