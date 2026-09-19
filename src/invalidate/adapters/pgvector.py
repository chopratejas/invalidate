"""Adapter for a Postgres table with a JSONB metadata column (pgvector or plain Postgres).

    import psycopg
    from invalidate.adapters import Governor
    from invalidate.adapters.pgvector import PgvectorAdapter, governed_rows

    conn = psycopg.connect(dsn)
    adapter = PgvectorAdapter(conn, "memories", id_col="id", text_col="text", meta_col="metadata")
    gov = Governor(adapter, "ledger.db", mode="flag")
    gov.sync()
    gov.observe("we migrated to SQLite last Tuesday", source="slack")
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, text FROM memories WHERE {adapter.live_where()} "
                    f"ORDER BY embedding <-> %s LIMIT 5", (embed("which database?"),))
        hits = governed_rows(cur.fetchall(), gov)          # dead rows excluded, even in ledger mode

The adapter never imports a driver. It takes any DB-API 2 connection whose module uses the
`format` / `pyformat` parameter style (`%s`): psycopg 3, psycopg2, pg8000. Every statement is
built from validated, double-quoted identifiers and `%s` placeholders; values are never
interpolated. Statements issued (identifiers shown unquoted, `t` = your table):

    pull    SELECT id, text, meta FROM t [WHERE scope] ORDER BY id LIMIT %s
            SELECT id, text, meta FROM t WHERE [scope AND] id > %s ORDER BY id LIMIT %s   (next pages)
    flag    UPDATE t SET meta = COALESCE(meta, '{}'::jsonb) || %s::jsonb WHERE id = %s
    delete  DELETE FROM t WHERE id = %s
    insert  INSERT INTO t (text, meta, embedding) VALUES (%s, %s::jsonb, %s) RETURNING id

Semantics, verified against the PostgreSQL and psycopg docs:

* `jsonb || jsonb` is a key-union with the RIGHT operand winning on duplicates, so `flag()` merges
  the `invalidate_*` receipt into whatever metadata the row has. `NULL || x` is NULL, hence the
  `COALESCE`.
* `jsonb ? 'key'` is "top-level key exists". `NULL ? 'key'` is NULL, so `live_where()` guards with
  `IS NULL` first; rows that were never flagged (no key, or a NULL column) stay live.
* psycopg 3 dumps a Python `str` with the unknown OID and psycopg2 inlines it as a literal, so the
  string host id compares against uuid and bigint columns without a cast, and a `json.dumps` string
  with `%s::jsonb` is a jsonb value. A vector goes in as the pgvector text form `'[0.1,0.2,...]'`.
"""
from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .base import DEAD, Governor, HostMemory, Reason

DEAD_VALUES = sorted(s.value for s in DEAD)  # ["contradicted", "superseded"]

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_QUALIFIED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")
_SUPPORTED_PARAMSTYLES = ("format", "pyformat")


def quote_ident(name: str, *, qualified: bool = False) -> str:
    """Validate `name` as a SQL identifier (optionally `schema.table`) and double-quote it."""
    pat = _QUALIFIED if qualified else _IDENT
    if not isinstance(name, str) or not pat.match(name):
        raise ValueError(f"not a valid SQL identifier: {name!r}")
    return ".".join(f'"{part}"' for part in name.split("."))


def live_where(meta_col: str = "metadata", prefix: str = "invalidate_") -> str:
    """A WHERE fragment (parenthesised, no parameters) that keeps every row not contradicted/superseded.

    `(m IS NULL OR NOT (m ? 'invalidate_status') OR m->>'invalidate_status' NOT IN ('contradicted', 'superseded'))`
    """
    m = quote_ident(meta_col)
    key = f"{quote_ident(prefix)[1:-1]}status"  # validated as an identifier, so safe as a literal
    dead = ", ".join(f"'{v}'" for v in DEAD_VALUES)
    return f"({m} IS NULL OR NOT ({m} ? '{key}') OR {m}->>'{key}' NOT IN ({dead}))"


def paramstyle_of(conn: Any) -> str | None:
    """The DB-API `paramstyle` of the module that owns `conn`, or None if it cannot be found."""
    mod = type(conn).__module__ or ""
    while mod:
        try:
            style = getattr(importlib.import_module(mod), "paramstyle", None)
        except Exception:  # noqa: BLE001 - a private submodule may not import cleanly
            style = None
        if isinstance(style, str):
            return style
        mod = mod.rpartition(".")[0]
    return None


def format_vector(vec: Any) -> Any:
    """A sequence of numbers becomes pgvector's text form `[a,b,c]`; anything else passes through
    (a `pgvector.Vector` / numpy array after `register_vector()` is adapted by the driver itself)."""
    if isinstance(vec, (list, tuple)) and all(isinstance(x, (int, float)) for x in vec):
        return "[" + ",".join(repr(float(x)) for x in vec) + "]"
    return vec


class PgvectorAdapter:
    """Governs one Postgres table. Ids are the table's ids (uuid or bigint, handled as strings);
    receipts land in the row's JSONB metadata column. The text column is never rewritten.

    `scope_sql` is a WHERE fragment (with its own `%s` placeholders, values in `scope_params`) that
    limits pull() to the rows you govern, e.g. `"user_id" = %s`. `vector_fn(text) -> sequence` embeds
    successor inserts; without it the embedding column is inserted as NULL. `insert_columns` adds
    fixed column values to every insert (e.g. `{"user_id": "alice"}` so successors land in scope).
    """

    def __init__(
        self,
        conn: Any,
        table: str,
        *,
        id_col: str = "id",
        text_col: str = "text",
        meta_col: str = "metadata",
        embedding_col: str = "embedding",
        scope_sql: str | None = None,
        scope_params: Sequence[Any] = (),
        vector_fn: Callable[[str], Any] | None = None,
        autocommit: bool = False,
        insert_columns: dict[str, Any] | None = None,
        name: str | None = None,
        metadata_prefix: str = "invalidate_",
        source: str = "pgvector",
        batch_size: int = 1000,
        paramstyle: str | None = None,
    ) -> None:
        style = paramstyle or paramstyle_of(conn) or "format"
        if style not in _SUPPORTED_PARAMSTYLES:
            raise ValueError(
                f"connection uses DB-API paramstyle {style!r}; PgvectorAdapter needs 'format' or 'pyformat' "
                "(psycopg 3, psycopg2, pg8000)"
            )
        self.conn = conn
        self.paramstyle = style
        self.table = table
        self.id_col, self.text_col, self.meta_col, self.embedding_col = id_col, text_col, meta_col, embedding_col
        self.q_table = quote_ident(table, qualified=True)
        self.q_id, self.q_text, self.q_meta, self.q_emb = (quote_ident(c) for c in (id_col, text_col, meta_col, embedding_col))
        self.scope_sql = scope_sql
        self.scope_params = tuple(scope_params)
        self.vector_fn = vector_fn
        self.autocommit = autocommit
        self.insert_columns = dict(insert_columns or {})
        self.q_insert_cols = [quote_ident(c) for c in self.insert_columns]
        self.prefix = quote_ident(metadata_prefix)[1:-1]
        self.source = source
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.name = name or f"pgvector:{table}"

    # -- SQL -------------------------------------------------------------------------
    def pull_sql(self, *, after: bool) -> str:
        where = [f"({self.scope_sql})"] if self.scope_sql else []
        if after:
            where.append(f"{self.q_id} > %s")
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        return f"SELECT {self.q_id}, {self.q_text}, {self.q_meta} FROM {self.q_table}{clause} ORDER BY {self.q_id} LIMIT %s"

    @property
    def flag_sql(self) -> str:
        return f"UPDATE {self.q_table} SET {self.q_meta} = COALESCE({self.q_meta}, '{{}}'::jsonb) || %s::jsonb WHERE {self.q_id} = %s"

    @property
    def delete_sql(self) -> str:
        return f"DELETE FROM {self.q_table} WHERE {self.q_id} = %s"

    @property
    def insert_sql(self) -> str:
        cols = ", ".join([self.q_text, self.q_meta, self.q_emb, *self.q_insert_cols])
        vals = ", ".join(["%s", "%s::jsonb", "%s", *(["%s"] * len(self.q_insert_cols))])
        return f"INSERT INTO {self.q_table} ({cols}) VALUES ({vals}) RETURNING {self.q_id}"

    # -- plumbing --------------------------------------------------------------------
    def _commit(self) -> None:
        if not self.autocommit:
            self.conn.commit()

    def _execute(self, sql: str, params: Sequence[Any]) -> Any:
        cur = self.conn.cursor()
        cur.execute(sql, tuple(params))
        return cur

    @staticmethod
    def _meta(raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, (str, bytes, bytearray)):  # a driver that hands jsonb back as text
            try:
                raw = json.loads(raw)
            except ValueError:
                return {}
        return dict(raw) if isinstance(raw, dict) else {}

    # -- Adapter protocol ------------------------------------------------------------
    def pull(self) -> Iterable[HostMemory]:
        """Keyset-paged read in id order (`batch_size` rows per SELECT). Rows without text are skipped."""
        out: list[HostMemory] = []
        try:
            self._pull_pages(out)
        except Exception:
            rollback = getattr(self.conn, "rollback", None)
            if callable(rollback) and not self.autocommit:
                rollback()
            raise
        self._commit()  # a SELECT opens a transaction on psycopg; do not leave it idle
        return out

    def _pull_pages(self, out: list[HostMemory]) -> None:
        last: Any = None
        first = True
        while True:
            params = [*self.scope_params] + ([] if first else [last]) + [self.batch_size]
            cur = self._execute(self.pull_sql(after=not first), params)
            rows = cur.fetchall()
            self._close(cur)
            for rid, text, raw in rows:
                last = rid
                if rid is None or not isinstance(text, str) or not text.strip():
                    continue
                meta = self._meta(raw)
                out.append(
                    HostMemory(
                        id=str(rid),
                        text=text,
                        kind=str(meta.get("kind", "fact")),
                        source=str(meta.get("source", self.source)),
                        metadata={k: v for k, v in meta.items() if not str(k).startswith(self.prefix)},
                    )
                )
            first = False
            if len(rows) < self.batch_size:
                break

    def flag(self, host_id: str, reason: Reason) -> None:
        patch = json.dumps(reason.as_metadata(self.prefix), default=str)
        cur = self._execute(self.flag_sql, (patch, host_id))
        self._close(cur)
        self._commit()

    def delete(self, host_id: str, reason: Reason) -> None:
        cur = self._execute(self.delete_sql, (host_id,))
        self._close(cur)
        self._commit()

    def insert(self, text: str, source: str, metadata: dict[str, Any]) -> str | None:
        meta = {"source": source, f"{self.prefix}status": "active", **metadata}
        emb = format_vector(self.vector_fn(text)) if self.vector_fn is not None else None
        params = (text, json.dumps(meta, default=str), emb, *self.insert_columns.values())
        cur = self._execute(self.insert_sql, params)
        row = cur.fetchone()
        self._close(cur)
        self._commit()
        return None if not row or row[0] is None else str(row[0])

    # -- helpers ---------------------------------------------------------------------
    def live_where(self) -> str:
        """WHERE fragment for "not invalidated", on this adapter's metadata column and prefix."""
        return live_where(self.meta_col, self.prefix)

    @staticmethod
    def _close(cur: Any) -> None:
        close = getattr(cur, "close", None)
        if callable(close):
            close()


def _default_id(row: Any) -> Any:
    if isinstance(row, dict):
        return row["id"]
    if isinstance(row, (list, tuple)):
        return row[0]
    return getattr(row, "id")


def governed_rows(rows: Iterable[Any], gov: Governor, *, id_of: Callable[[Any], Any] | None = None,
                  include_review: bool = True, annotate: bool = False) -> list[Any]:
    """Drop rows the ledger knows are dead (or under review) from a result set you fetched yourself.

    `id_of` extracts the host id; the default takes column 0 of a tuple, `row["id"]` of a dict, or
    `row.id`. Combine with `live_where()` in the query: the SQL hides what the adapter flagged in the
    host, the ledger prune hides what the Governor knows but did not write (mode="ledger", push errors).

    `annotate=True` keeps every row and returns `[(row, note), ...]` instead, `note` being
    `Governor.annotate`'s label for retired rows and None otherwise. Pairs because cursor rows are usually
    tuples, which cannot carry a key; query without `live_where()` so the retired rows are fetched at all.
    """
    pick = id_of or _default_id
    if annotate:
        return gov.annotate(rows, id_of=lambda r: str(pick(r)))
    return gov.filter(rows, id_of=lambda r: str(pick(r)), include_review=include_review)


__all__ = ["PgvectorAdapter", "governed_rows", "live_where", "quote_ident", "paramstyle_of", "format_vector", "DEAD_VALUES"]
