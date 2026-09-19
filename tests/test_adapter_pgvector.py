"""PgvectorAdapter against a fake DB-API connection (always) and a real Postgres (when reachable).

The fake records every `(sql, params)` and simulates only the statements the adapter issues: keyset
SELECT, `UPDATE ... COALESCE(meta,'{}'::jsonb) || %s::jsonb`, DELETE, `INSERT ... RETURNING id` and a
SELECT filtered by `live_where()`. The SQL text and parameters asserted here are the contract.

Set INVALIDATE_PG_DSN (and have psycopg importable) to also run every test against a real database:
a temp table with a vector(4) column is created per test and dropped afterwards.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from typing import Any

import pytest
from conftest import CONFIRM, CONTRADICT, SUPERSEDE, FakeJudge

from invalidate import Status
from invalidate.adapters import Governor, HostMemory, Reason
from invalidate.adapters.pgvector import (
    PgvectorAdapter,
    format_vector,
    governed_rows,
    live_where,
    paramstyle_of,
    quote_ident,
)

paramstyle = "format"  # DB-API attribute: makes this module the "driver" of FakeConnection

FACTS = {
    "pg": ("user prefers Postgres", {"kind": "preference", "n": 1}),
    "deploy": ("deploys run at 2pm UTC", None),                      # NULL metadata
    "alice": ("Alice owns the billing service", {}),
    "replica": ("prod reads go through the Postgres replica", {"kind": "config", "invalidate_status": "active"}),
}
EVENT = "we migrated to SQLite last Tuesday"
DEAD = {"contradicted", "superseded"}
LIVE_SQL = "(\"metadata\" IS NULL OR NOT (\"metadata\" ? 'invalidate_status') OR \"metadata\"->>'invalidate_status' NOT IN ('contradicted', 'superseded'))"


def make_judge() -> FakeJudge:
    return FakeJudge().script("prefers Postgres", SUPERSEDE).script("Postgres replica", CONTRADICT)


def vec4(_text: str) -> list[float]:
    return [0.1, 0.2, 0.3, 0.4]


# --- fake DB-API connection ------------------------------------------------------------------

class FakeCursor:
    _SELECT = re.compile(r'^SELECT (?P<cols>.+?) FROM "(?P<table>\w+)"(?: WHERE (?P<where>.+?))?(?: ORDER BY "(?P<order>\w+)")?(?: LIMIT %s)?$')
    _UPDATE = re.compile(r'^UPDATE "(?P<table>\w+)" SET "(?P<meta>\w+)" = COALESCE\("(?P=meta)", \'\{\}\'::jsonb\) \|\| %s::jsonb WHERE "(?P<id>\w+)" = %s$')
    _DELETE = re.compile(r'^DELETE FROM "(?P<table>\w+)" WHERE "(?P<id>\w+)" = %s$')
    _INSERT = re.compile(r'^INSERT INTO "(?P<table>\w+)" \((?P<cols>.+)\) VALUES \((?P<vals>.+)\) RETURNING "(?P<id>\w+)"$')
    _SCOPE = re.compile(r'^\("(?P<col>\w+)" = %s\)$')
    _AFTER = re.compile(r'^"(?P<col>\w+)" > %s$')
    _LIVE = re.compile(r'^\("(?P<m>\w+)" IS NULL OR NOT \("(?P=m)" \? \'(?P<key>\w+)\'\) OR "(?P=m)"->>\'(?P=key)\' NOT IN \((?P<dead>[^)]+)\)\)$')

    def __init__(self, conn: "FakeConnection") -> None:
        self.conn = conn
        self._rows: list[tuple] = []
        self.rowcount = -1
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self.closed = True

    def execute(self, sql: str, params: Any = None) -> None:
        if self.closed:
            raise RuntimeError("cursor closed")
        params = tuple(params or ())
        self.conn.executed.append((sql, params))
        if "%s" in sql.replace("%%", "") and sql.count("%s") != len(params):
            raise TypeError(f"placeholder/parameter mismatch: {sql.count('%s')} vs {len(params)}")
        if sql.split()[0] not in ("SELECT",) and not self.conn.autocommit:
            self.conn.in_txn = True
        for handler, pat in ((self._select, self._SELECT), (self._update, self._UPDATE),
                             (self._delete, self._DELETE), (self._insert, self._INSERT)):
            m = pat.match(sql)
            if m:
                handler(m, list(params))
                return
        raise ValueError(f"fake cannot simulate: {sql}")

    # -- statements ------------------------------------------------------------------
    def _table(self, name: str) -> "FakeTable":
        if name != self.conn.table.name:
            raise ValueError(f'relation "{name}" does not exist')
        return self.conn.table

    def _select(self, m: re.Match, params: list) -> None:
        t = self._table(m["table"])
        cols = [c.strip().strip('"') for c in m["cols"].split(",")]
        preds = []
        for cond in (m["where"].split(" AND ") if m["where"] else []):
            if (s := self._SCOPE.match(cond)):
                v = params.pop(0)
                preds.append(lambda r, c=s["col"], v=v: r[c] == v)
            elif (a := self._AFTER.match(cond)):
                v = params.pop(0)
                preds.append(lambda r, c=a["col"], v=v: r[c] > v)
            elif (lv := self._LIVE.match(cond)):
                dead = {d.strip().strip("'") for d in lv["dead"].split(",")}
                preds.append(lambda r, mc=lv["m"], k=lv["key"], dead=dead: self._live(r[mc], k, dead))
            else:
                raise ValueError(f"fake cannot simulate predicate: {cond}")
        rows = [r for r in t.rows.values() if all(p(r) for p in preds)]
        if m["order"]:
            rows.sort(key=lambda r: r[m["order"]])
        if m.group(0).endswith("LIMIT %s"):
            rows = rows[: int(params.pop(0))]
        assert not params, "unconsumed parameters"
        self._rows = [tuple(self._out(r[c]) for c in cols) for r in rows]
        self.rowcount = len(self._rows)

    @staticmethod
    def _live(meta: Any, key: str, dead: set[str]) -> bool:
        # SQL three-valued logic: NULL ? k is NULL; the IS NULL guard makes the row live.
        if meta is None:
            return True
        if key not in meta:
            return True
        v = meta[key]
        return v is not None and str(v) not in dead

    @staticmethod
    def _out(v: Any) -> Any:
        return json.loads(json.dumps(v)) if isinstance(v, dict) else v  # psycopg hands jsonb back as a fresh dict

    def _update(self, m: re.Match, params: list) -> None:
        t = self._table(m["table"])
        patch = json.loads(params[0])                     # %s::jsonb
        if not isinstance(patch, dict):
            raise ValueError("jsonb || expects an object here")
        n = 0
        for r in t.rows.values():
            if str(r[m["id"]]) == str(params[1]):
                cur = r[m["meta"]] if r[m["meta"]] is not None else {}   # COALESCE(meta, '{}')
                r[m["meta"]] = {**cur, **patch}            # right operand wins on duplicate keys
                n += 1
        self.rowcount = n

    def _delete(self, m: re.Match, params: list) -> None:
        t = self._table(m["table"])
        gone = [k for k, r in t.rows.items() if str(r[m["id"]]) == str(params[0])]
        for k in gone:
            del t.rows[k]
        self.rowcount = len(gone)

    def _insert(self, m: re.Match, params: list) -> None:
        t = self._table(m["table"])
        cols = [c.strip().strip('"') for c in m["cols"].split(",")]
        vals = [v.strip() for v in m["vals"].split(",")]
        assert len(cols) == len(vals) == len(params)
        row = t.new_row()
        for c, v, p in zip(cols, vals, params):
            if c not in t.columns:
                raise ValueError(f'column "{c}" does not exist')
            if v == "%s::jsonb":
                p = json.loads(p)
            elif v != "%s":
                raise ValueError(f"fake cannot simulate value {v}")
            if c == "embedding" and p is not None:
                nums = re.fullmatch(r"\[([^\]]*)\]", str(p))
                if not nums or len(nums.group(1).split(",")) != t.dims:
                    raise ValueError(f"expected {t.dims} dimensions")
            row[c] = p
        t.rows[row["id"]] = row
        self._rows = [(row[m["id"]],)]
        self.rowcount = 1

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None

    def fetchmany(self, n: int = 1):
        out, self._rows = self._rows[:n], self._rows[n:]
        return out

    def fetchall(self):
        out, self._rows = self._rows, []
        return out


class FakeTable:
    columns = ("id", "text", "metadata", "embedding", "user_id")

    def __init__(self, name: str, id_kind: str, dims: int = 4) -> None:
        self.name, self.id_kind, self.dims = name, id_kind, dims
        self.rows: dict[Any, dict[str, Any]] = {}
        self._seq = 0

    def new_row(self) -> dict[str, Any]:
        if self.id_kind == "bigint":
            self._seq += 1
            rid: Any = self._seq
        else:
            rid = uuid.uuid4()
        return {"id": rid, "text": None, "metadata": None, "embedding": None, "user_id": None}

    def add(self, text: Any, meta: Any, user_id: str = "alice") -> Any:
        row = self.new_row()
        row.update(text=text, metadata=None if meta is None else dict(meta), user_id=user_id)
        self.rows[row["id"]] = row
        return row["id"]


class FakeConnection:
    def __init__(self, table: FakeTable, autocommit: bool = False) -> None:
        self.table = table
        self.autocommit = autocommit
        self.executed: list[tuple[str, tuple]] = []
        self.commits = 0
        self.in_txn = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1
        self.in_txn = False

    def rollback(self) -> None:
        self.in_txn = False

    def close(self) -> None:
        pass


# --- real Postgres (optional) -----------------------------------------------------------------

PG_DSN = os.environ.get("INVALIDATE_PG_DSN")
try:
    import psycopg  # noqa: F401
    HAVE_PSYCOPG = True
except ImportError:
    HAVE_PSYCOPG = False
needs_pg = pytest.mark.skipif(not (PG_DSN and HAVE_PSYCOPG), reason="INVALIDATE_PG_DSN unset or psycopg missing")


class RecordingCursor:
    def __init__(self, cur: Any, log: list) -> None:
        self._cur, self._log = cur, log

    def execute(self, sql: str, params: Any = None) -> Any:
        self._log.append((sql, tuple(params or ())))
        return self._cur.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cur, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._cur.close()


class RecordingConnection:
    """Wraps a psycopg connection; `type(self).__module__` resolves to this test module, paramstyle 'format'."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self.executed: list[tuple[str, tuple]] = []
        self.commits = 0

    def cursor(self, *a: Any, **k: Any) -> RecordingCursor:
        return RecordingCursor(self._conn.cursor(*a, **k), self.executed)

    def commit(self) -> None:
        self.commits += 1
        self._conn.commit()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


# --- host fixture: same interface over both ------------------------------------------------------

class Host:
    """conn + table name + a way to read rows back independently of the adapter."""

    def __init__(self, conn: Any, table: str, id_kind: str, kind: str) -> None:
        self.conn, self.table, self.id_kind, self.kind = conn, table, id_kind, kind
        self.ids: dict[str, str] = {}
        self.junk: set[str] = set()          # rows pull() must skip; live in SQL (no receipt) so hidden from select_live

    def seed(self) -> None:
        for key, (text, meta) in FACTS.items():
            self.ids[key] = str(self.add(text, meta))
        self.junk.add(str(self.add("   ", {"kind": "blank"})))     # skipped by pull: blank text
        self.junk.add(str(self.add(None, None)))                    # skipped by pull: NULL text

    # -- fake --------------------------------------------------------------------------
    def add(self, text: Any, meta: Any, user_id: str = "alice") -> Any:
        if self.kind == "fake":
            return self.conn.table.add(text, meta, user_id)
        with self.conn._conn.cursor() as cur:
            cur.execute(f'INSERT INTO "{self.table}" (text, metadata, user_id) VALUES (%s, %s::jsonb, %s) RETURNING id',
                        (text, None if meta is None else json.dumps(meta), user_id))
            rid = cur.fetchone()[0]
        self.conn._conn.commit()
        return rid

    def row(self, host_id: str) -> dict[str, Any] | None:
        if self.kind == "fake":
            for r in self.conn.table.rows.values():
                if str(r["id"]) == str(host_id):
                    return dict(r)
            return None
        with self.conn._conn.cursor() as cur:
            cur.execute(f'SELECT id, text, metadata, embedding::text, user_id FROM "{self.table}" WHERE id = %s', (host_id,))
            r = cur.fetchone()
        self.conn._conn.commit()
        return None if r is None else dict(zip(("id", "text", "metadata", "embedding", "user_id"), r))

    def meta(self, host_id: str) -> dict[str, Any] | None:
        r = self.row(host_id)
        assert r is not None, host_id
        return r["metadata"]

    def count(self) -> int:
        if self.kind == "fake":
            return len(self.conn.table.rows)
        with self.conn._conn.cursor() as cur:
            cur.execute(f'SELECT count(*) FROM "{self.table}"')
            n = cur.fetchone()[0]
        self.conn._conn.commit()
        return int(n)

    def select_live(self, adapter: PgvectorAdapter) -> list[tuple]:
        """A user query with the live filter, through the adapter's own connection."""
        cur = self.conn.cursor()
        cur.execute(f'SELECT "id", "text" FROM "{self.table}" WHERE {adapter.live_where()} ORDER BY "id"')
        rows = cur.fetchall()
        cur.close()
        self.conn.commit()
        return [r for r in rows if str(r[0]) not in self.junk]


HOSTS = [
    pytest.param(("fake", "bigint"), id="fake-bigint"),
    pytest.param(("fake", "uuid"), id="fake-uuid"),
    pytest.param(("pg", "bigint"), id="pg-bigint", marks=needs_pg),
    pytest.param(("pg", "uuid"), id="pg-uuid", marks=needs_pg),
]


@pytest.fixture(params=HOSTS)
def host(request):
    kind, id_kind = request.param
    if kind == "fake":
        h = Host(FakeConnection(FakeTable("memories", id_kind)), "memories", id_kind, kind)
        h.seed()
        yield h
        return
    import psycopg

    raw = psycopg.connect(PG_DSN)
    table = f"invalidate_t_{uuid.uuid4().hex[:8]}"
    id_ddl = "bigserial PRIMARY KEY" if id_kind == "bigint" else "uuid PRIMARY KEY DEFAULT gen_random_uuid()"
    try:
        with raw.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f'CREATE TABLE "{table}" (id {id_ddl}, text text, metadata jsonb, embedding vector(4), user_id text)')
        raw.commit()
    except Exception as e:  # noqa: BLE001
        raw.rollback()
        raw.close()
        pytest.skip(f"cannot create pgvector table: {e!r}")
    h = Host(RecordingConnection(raw), table, id_kind, kind)
    h.seed()
    h.conn.executed.clear()
    try:
        yield h
    finally:
        raw.rollback()
        with raw.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{table}"')
        raw.commit()
        raw.close()


def make_adapter(host: Host, **kw: Any) -> PgvectorAdapter:
    return PgvectorAdapter(host.conn, host.table, **kw)


def reason(status: Status = Status.SUPERSEDED) -> Reason:
    return Reason(status=status, disposition=status.value, event_text=EVENT, event_source="slack",
                  event_id="ev1", still_true=0.05, at=1700000000.0)


# --- construction ------------------------------------------------------------------------------

def test_identifiers_are_validated_and_quoted():
    assert quote_ident("memories") == '"memories"'
    assert quote_ident("public.memories", qualified=True) == '"public"."memories"'
    for bad in ("mem ories", 'x"; DROP TABLE t; --', "1abc", "", "a.b"):
        with pytest.raises(ValueError):
            quote_ident(bad)
    conn = FakeConnection(FakeTable("memories", "bigint"))
    with pytest.raises(ValueError):
        PgvectorAdapter(conn, "memories; DROP TABLE x")
    with pytest.raises(ValueError):
        PgvectorAdapter(conn, "memories", meta_col="meta data")
    with pytest.raises(ValueError):
        PgvectorAdapter(conn, "memories", metadata_prefix="inv'")
    with pytest.raises(ValueError):
        PgvectorAdapter(conn, "memories", batch_size=0)
    a = PgvectorAdapter(conn, "public.memories", id_col="pk", text_col="body", meta_col="meta", embedding_col="vec")
    assert a.name == "pgvector:public.memories" and a.q_table == '"public"."memories"'
    assert a.flag_sql == 'UPDATE "public"."memories" SET "meta" = COALESCE("meta", \'{}\'::jsonb) || %s::jsonb WHERE "pk" = %s'
    assert a.delete_sql == 'DELETE FROM "public"."memories" WHERE "pk" = %s'
    assert a.insert_sql == 'INSERT INTO "public"."memories" ("body", "meta", "vec") VALUES (%s, %s::jsonb, %s) RETURNING "pk"'
    assert a.pull_sql(after=False) == 'SELECT "pk", "body", "meta" FROM "public"."memories" ORDER BY "pk" LIMIT %s'
    assert a.pull_sql(after=True) == 'SELECT "pk", "body", "meta" FROM "public"."memories" WHERE "pk" > %s ORDER BY "pk" LIMIT %s'
    assert a.live_where() == "(\"meta\" IS NULL OR NOT (\"meta\" ? 'invalidate_status') OR \"meta\"->>'invalidate_status' NOT IN ('contradicted', 'superseded'))"


def test_paramstyle_detection():
    assert paramstyle_of(FakeConnection(FakeTable("m", "bigint"))) == "format"   # this module's attribute
    lite = sqlite3.connect(":memory:")
    assert paramstyle_of(lite) == "qmark"
    with pytest.raises(ValueError, match="qmark"):
        PgvectorAdapter(lite, "memories")
    assert PgvectorAdapter(lite, "memories", paramstyle="pyformat").paramstyle == "pyformat"   # explicit override
    with pytest.raises(ValueError):
        PgvectorAdapter(FakeConnection(FakeTable("m", "bigint")), "memories", paramstyle="numeric")
    assert paramstyle_of(object()) is None                                          # builtins: no paramstyle


def test_format_vector():
    assert format_vector([1, 0.5, 2]) == "[1.0,0.5,2.0]"
    assert format_vector((0.25,)) == "[0.25]"
    assert format_vector("[1,2]") == "[1,2]"           # already text: pass through
    assert format_vector(None) is None


def test_live_where_helper():
    assert live_where() == LIVE_SQL
    assert live_where("meta", "inv_") == "(\"meta\" IS NULL OR NOT (\"meta\" ? 'inv_status') OR \"meta\"->>'inv_status' NOT IN ('contradicted', 'superseded'))"
    with pytest.raises(ValueError):
        live_where("meta; --")


# --- each operation: exact SQL + effect --------------------------------------------------------

def test_pull_sql_paging_and_shape(host):
    adapter = make_adapter(host, batch_size=2)
    host.conn.executed.clear()
    pulled = list(adapter.pull())
    t = f'"{host.table}"'
    first = f'SELECT "id", "text", "metadata" FROM {t} ORDER BY "id" LIMIT %s'
    nxt = f'SELECT "id", "text", "metadata" FROM {t} WHERE "id" > %s ORDER BY "id" LIMIT %s'
    sqls = [s for s, _ in host.conn.executed]
    assert sqls[0] == first and set(sqls[1:]) == {nxt} and len(sqls) == 4      # 6 rows / 2 per page, last page short
    assert host.conn.executed[0][1] == (2,)
    assert [p[1] for _, p in host.conn.executed[1:]] == [2, 2, 2]
    assert all(isinstance(hm, HostMemory) for hm in pulled)
    by_id = {hm.id: hm for hm in pulled}
    assert set(by_id) == set(host.ids.values())                                   # blank / NULL text skipped
    pg, deploy, alice, replica = (by_id[host.ids[k]] for k in ("pg", "deploy", "alice", "replica"))
    assert (pg.text, pg.kind, pg.source, pg.metadata) == (FACTS["pg"][0], "preference", "pgvector", {"kind": "preference", "n": 1})
    assert deploy.metadata == {} and alice.metadata == {} and deploy.kind == "fact"
    assert replica.kind == "config" and "invalidate_status" not in replica.metadata  # receipt keys stripped
    key = (lambda i: int(i)) if host.id_kind == "bigint" else (lambda i: uuid.UUID(i))
    assert [hm.id for hm in pulled] == sorted((hm.id for hm in pulled), key=key)   # id order
    assert host.conn.commits == 1                                                   # the read transaction is closed


def test_pull_with_scope(host):
    bob = str(host.add("Bob's fact", {"kind": "fact"}, user_id="bob"))
    adapter = make_adapter(host, scope_sql='"user_id" = %s', scope_params=("alice",), batch_size=3)
    host.conn.executed.clear()
    ids = {hm.id for hm in adapter.pull()}
    assert ids == set(host.ids.values()) and bob not in ids
    t = f'"{host.table}"'
    assert host.conn.executed[0] == (f'SELECT "id", "text", "metadata" FROM {t} WHERE ("user_id" = %s) ORDER BY "id" LIMIT %s', ("alice", 3))
    sql2, p2 = host.conn.executed[1]
    assert sql2 == f'SELECT "id", "text", "metadata" FROM {t} WHERE ("user_id" = %s) AND "id" > %s ORDER BY "id" LIMIT %s'
    assert p2[0] == "alice" and p2[2] == 3 and str(p2[1]) in ids | set(host.junk)  # a junk row may end page 1
    assert {hm.id for hm in make_adapter(host, scope_sql='"user_id" = %s', scope_params=("bob",)).pull()} == {bob}


def test_flag_sql_merges_into_jsonb(host):
    adapter = make_adapter(host)
    host.conn.executed.clear()
    pg = host.ids["pg"]
    adapter.flag(pg, reason())
    patch = reason().as_metadata()
    assert host.conn.executed == [(
        f'UPDATE "{host.table}" SET "metadata" = COALESCE("metadata", \'{{}}\'::jsonb) || %s::jsonb WHERE "id" = %s',
        (json.dumps(patch, default=str), pg),
    )]
    assert host.conn.commits == 1
    meta = host.meta(pg)
    assert meta["kind"] == "preference" and meta["n"] == 1                 # existing keys kept
    assert meta["invalidate_status"] == "superseded" and meta["invalidate_event"] == EVENT
    assert meta["invalidate_still_true"] == 0.05 and meta["invalidate_at"] == 1700000000.0
    assert host.row(pg)["text"] == FACTS["pg"][0]                              # text never rewritten
    # NULL metadata: COALESCE makes the merge work
    adapter.flag(host.ids["deploy"], reason(Status.CONTRADICTED))
    assert host.meta(host.ids["deploy"]) == reason(Status.CONTRADICTED).as_metadata()
    # re-flag: right operand wins
    adapter.flag(pg, reason(Status.ACTIVE))
    assert host.meta(pg)["invalidate_status"] == "active" and host.meta(pg)["kind"] == "preference"
    assert host.conn.commits == 3


def test_flag_without_commit_when_autocommit(host):
    adapter = make_adapter(host, autocommit=True)
    host.conn.executed.clear()
    adapter.flag(host.ids["pg"], reason())
    assert host.conn.commits == 0
    if host.kind == "fake":
        assert host.meta(host.ids["pg"])["invalidate_status"] == "superseded"


def test_delete_sql(host):
    adapter = make_adapter(host)
    host.conn.executed.clear()
    pg = host.ids["pg"]
    adapter.delete(pg, reason())
    assert host.conn.executed == [(f'DELETE FROM "{host.table}" WHERE "id" = %s', (pg,))]
    assert host.row(pg) is None and host.count() == 5 and host.conn.commits == 1
    adapter.delete(pg, reason())                                              # idempotent, no error
    assert host.count() == 5


def test_insert_sql_with_and_without_vector(host):
    adapter = make_adapter(host)
    host.conn.executed.clear()
    new_id = adapter.insert(EVENT, "slack", {"invalidate_supersedes": [host.ids["pg"]], "invalidate_event_id": "ev1"})
    meta = {"source": "slack", "invalidate_status": "active", "invalidate_supersedes": [host.ids["pg"]], "invalidate_event_id": "ev1"}
    assert host.conn.executed == [(
        f'INSERT INTO "{host.table}" ("text", "metadata", "embedding") VALUES (%s, %s::jsonb, %s) RETURNING "id"',
        (EVENT, json.dumps(meta), None),
    )]
    assert isinstance(new_id, str) and new_id and host.conn.commits == 1
    row = host.row(new_id)
    assert row["text"] == EVENT and row["metadata"] == meta and row["embedding"] is None
    if host.id_kind == "uuid":
        uuid.UUID(new_id)
    else:
        assert int(new_id) > int(host.ids["replica"])

    embedded = make_adapter(host, vector_fn=vec4, insert_columns={"user_id": "alice"})
    host.conn.executed.clear()
    sid = embedded.insert("SQLite it is", "slack", {})
    assert host.conn.executed == [(
        f'INSERT INTO "{host.table}" ("text", "metadata", "embedding", "user_id") VALUES (%s, %s::jsonb, %s, %s) RETURNING "id"',
        ("SQLite it is", json.dumps({"source": "slack", "invalidate_status": "active"}), "[0.1,0.2,0.3,0.4]", "alice"),
    )]
    row = host.row(sid)
    assert row["embedding"] == "[0.1,0.2,0.3,0.4]" and row["user_id"] == "alice"
    with pytest.raises(Exception, match="dimension"):
        make_adapter(host, vector_fn=lambda t: [1.0, 2.0]).insert("wrong dims", "x", {})
    if host.kind == "pg":
        host.conn._conn.rollback()


def test_live_where_keeps_unflagged_and_null_rows(host):
    adapter = make_adapter(host)
    ids = host.ids
    assert {str(r[0]) for r in host.select_live(adapter)} >= set(ids.values())   # nothing flagged: all live
    adapter.flag(ids["pg"], reason(Status.SUPERSEDED))
    adapter.flag(ids["replica"], reason(Status.CONTRADICTED))
    adapter.flag(ids["alice"], reason(Status.NEEDS_REVIEW))
    live = {str(r[0]) for r in host.select_live(adapter)}
    assert ids["pg"] not in live and ids["replica"] not in live
    assert ids["deploy"] in live and ids["alice"] in live                       # NULL meta / needs_review stay live in SQL
    assert host.conn.executed[-1][0].startswith(f'SELECT "id", "text" FROM "{host.table}" WHERE {LIVE_SQL} ORDER BY')


# --- Governor end-to-end (FakeJudge) ---------------------------------------------------------

def test_governor_flag_mode_end_to_end(host):
    adapter = make_adapter(host)
    gov = Governor(adapter, ":memory:", judge=make_judge(), mode="flag")
    rep = gov.sync()
    assert (rep.total, rep.added) == (4, 4) and gov.sync().unchanged == 4
    m = gov.mem.get(gov.our_id(host.ids["pg"]))
    assert m.fact == FACTS["pg"][0] and m.kind == "preference" and m.metadata["host_id"] == host.ids["pg"]
    assert not any(k.startswith("invalidate_") for k in gov.mem.get(gov.our_id(host.ids["replica"])).metadata)

    host.conn.executed.clear()
    out = gov.observe(EVENT, source="slack")
    assert not out.errors and {(p.host_id, p.action) for p in out.pushes} == {(host.ids["pg"], "flag"), (host.ids["replica"], "flag")}
    flag_sql = f'UPDATE "{host.table}" SET "metadata" = COALESCE("metadata", \'{{}}\'::jsonb) || %s::jsonb WHERE "id" = %s'
    assert [s for s, _ in host.conn.executed] == [flag_sql, flag_sql]
    assert {p[1] for _, p in host.conn.executed} == {host.ids["pg"], host.ids["replica"]}
    for _, p in host.conn.executed:
        receipt = json.loads(p[0])
        assert receipt["invalidate_event"] == EVENT and receipt["invalidate_event_source"] == "slack"
        assert set(receipt) == {"invalidate_status", "invalidate_disposition", "invalidate_event", "invalidate_event_source",
                                "invalidate_event_id", "invalidate_still_true", "invalidate_at"}

    pg_meta, rep_meta = host.meta(host.ids["pg"]), host.meta(host.ids["replica"])
    assert pg_meta["invalidate_status"] == "superseded" and pg_meta["kind"] == "preference" and pg_meta["n"] == 1
    assert rep_meta["invalidate_status"] == "contradicted" and rep_meta["kind"] == "config"
    assert host.meta(host.ids["deploy"]) is None                                 # untouched rows untouched
    assert host.row(host.ids["pg"])["text"] == FACTS["pg"][0]
    assert gov.status_of(host.ids["pg"]) is Status.SUPERSEDED and gov.dead_ids() == {host.ids["pg"], host.ids["replica"]}

    live = host.select_live(adapter)
    assert {str(r[0]) for r in live} == {host.ids["deploy"], host.ids["alice"]}
    assert {str(r[0]) for r in governed_rows(live, gov)} == {host.ids["deploy"], host.ids["alice"]}
    rep2 = gov.sync()
    assert rep2.unchanged == 4 and rep2.updated == 0 and rep2.removed == 0    # receipts do not change the text

    gov.keep(host.ids["pg"])
    assert host.meta(host.ids["pg"])["invalidate_status"] == "active"
    assert host.ids["pg"] in {str(r[0]) for r in host.select_live(adapter)}
    gov.forget(host.ids["deploy"])
    assert host.row(host.ids["deploy"]) is None
    gov.close()


def test_governor_ledger_mode_prunes_via_governed_rows(host):
    adapter = make_adapter(host)
    gov = Governor(adapter, ":memory:", judge=make_judge(), mode="ledger")
    gov.sync()
    host.conn.executed.clear()
    out = gov.observe(EVENT, source="slack")
    assert out.pushes == [] and host.conn.executed == []                           # host untouched
    assert host.meta(host.ids["pg"]) == FACTS["pg"][1]
    rows = host.select_live(adapter)                                             # SQL hides nothing...
    assert {str(r[0]) for r in rows} == set(host.ids.values())
    kept = governed_rows(rows, gov)                                              # ...the ledger does
    assert {str(r[0]) for r in kept} == {host.ids["deploy"], host.ids["alice"]}
    dicts = [{"id": r[0], "text": r[1]} for r in rows]
    assert {str(d["id"]) for d in governed_rows(dicts, gov)} == {host.ids["deploy"], host.ids["alice"]}
    assert [str(r[1]) for r in governed_rows(rows, gov, id_of=lambda r: r[0])] == [
        r[1] for r in rows if str(r[0]) in {host.ids["deploy"], host.ids["alice"]}]
    gov.close()


def test_governor_delete_mode_and_successor(host):
    adapter = make_adapter(host, vector_fn=vec4)
    gov = Governor(adapter, ":memory:", judge=make_judge(), mode="delete", successors=True)
    gov.sync()
    host.conn.executed.clear()
    out = gov.observe(EVENT, source="slack")
    assert not out.errors
    assert {(p.host_id, p.action) for p in out.pushes if p.action == "delete"} == {(host.ids["pg"], "delete"), (host.ids["replica"], "delete")}
    sqls = [s for s, _ in host.conn.executed]
    assert sqls.count(f'DELETE FROM "{host.table}" WHERE "id" = %s') == 2
    assert sqls[-1] == f'INSERT INTO "{host.table}" ("text", "metadata", "embedding") VALUES (%s, %s::jsonb, %s) RETURNING "id"'
    sid = out.successor_host_id
    assert sid and host.row(host.ids["pg"]) is None and host.row(host.ids["replica"]) is None
    row = host.row(sid)
    assert row["text"] == EVENT and row["embedding"] == "[0.1,0.2,0.3,0.4]"
    assert row["metadata"]["invalidate_supersedes"] == [host.ids["pg"]] and row["metadata"]["invalidate_status"] == "active"
    assert row["metadata"]["source"] == "slack"
    assert gov.mem.get(gov.our_id(host.ids["pg"])).superseded_by == gov.our_id(sid)
    rep2 = gov.sync()
    assert rep2.total == 3 and rep2.removed == 0 and rep2.added == 0 and rep2.unchanged == 3
    assert {str(r[0]) for r in host.select_live(adapter)} == {host.ids["deploy"], host.ids["alice"], sid}
    gov.close()


def test_governor_push_error_is_reported_and_ledger_moves(host):
    class Broken(PgvectorAdapter):
        def flag(self, host_id, reason):
            if host_id == host.ids["pg"]:
                raise ConnectionError("pool exhausted")
            super().flag(host_id, reason)

    gov = Governor(Broken(host.conn, host.table), ":memory:", judge=make_judge())
    gov.sync()
    out = gov.observe(EVENT, source="slack")
    assert [p.host_id for p in out.errors] == [host.ids["pg"]] and "pool exhausted" in out.errors[0].error
    assert gov.status_of(host.ids["pg"]) is Status.SUPERSEDED
    assert host.meta(host.ids["replica"])["invalidate_status"] == "contradicted"
    assert host.meta(host.ids["pg"]) == FACTS["pg"][1]
    assert host.ids["pg"] not in {str(r[0]) for r in governed_rows(host.select_live(gov.adapter), gov)}
    gov.close()


def test_confirm_does_not_touch_host(host):
    gov = Governor(make_adapter(host), ":memory:", judge=FakeJudge().script("prefers Postgres", CONFIRM))
    gov.sync()
    host.conn.executed.clear()
    out = gov.observe("yep, still on Postgres", source="slack")
    assert out.pushes == [] and host.conn.executed == [] and host.meta(host.ids["pg"]) == FACTS["pg"][1]
    gov.close()


def test_fake_rejects_unknown_sql():
    conn = FakeConnection(FakeTable("memories", "bigint"))
    with pytest.raises(ValueError, match="cannot simulate"):
        conn.cursor().execute('TRUNCATE "memories"')
    with pytest.raises(TypeError):
        conn.cursor().execute('DELETE FROM "memories" WHERE "id" = %s', ())
