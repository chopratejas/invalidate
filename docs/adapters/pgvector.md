# Postgres / pgvector

Governs one Postgres table that holds memory text plus a JSONB metadata column, with or without a
pgvector `embedding` column. Postgres keeps the rows; invalidate keeps the ledger and writes each verdict
back as `invalidate_*` keys merged into the JSONB. The text column is never rewritten.

The adapter takes a plain DB-API connection: psycopg 3, psycopg2 or pg8000 (any driver whose
`paramstyle` is `format` / `pyformat`). It imports no driver itself, so the base package is enough.

## Install

```sh
pip install invalidate "psycopg[binary]"   # or psycopg2-binary
# on the server: CREATE EXTENSION vector;  (only needed for the embedding column)
```

## Usage

```python
import psycopg
from invalidate.adapters import Governor
from invalidate.adapters.pgvector import PgvectorAdapter, governed_rows

conn = psycopg.connect("postgresql://user:pass@localhost:5432/app")
adapter = PgvectorAdapter(
    conn, "memories",                       # or "public.memories"
    id_col="id", text_col="text", meta_col="metadata", embedding_col="embedding",
    scope_sql='"user_id" = %s', scope_params=("alice",),   # optional: govern a slice of the table
    vector_fn=embed,                        # optional: text -> list[float], used for successor inserts
    insert_columns={"user_id": "alice"},    # optional: fixed values so successors land in scope
)
gov = Governor(adapter, "ledger.db", mode="flag", successors=True)

gov.sync()                                                    # pull rows into the ledger (idempotent)
report = gov.observe("we migrated to SQLite", source="slack") # judge every row, flag the losers in Postgres
print(report.summary())

with conn.cursor() as cur:                                    # your own query, minus dead rows
    cur.execute(
        f"SELECT id, text FROM memories WHERE {adapter.live_where()} ORDER BY embedding <-> %s LIMIT 5",
        ("[0.1,0.2,...]",),
    )
    hits = governed_rows(cur.fetchall(), gov)                 # also prunes what only the ledger knows
    pairs = governed_rows(cur.fetchall(), gov, annotate=True) # keep all: [(row, note), ...]; query without live_where()
```

`gov.keep(id)` writes `invalidate_status="active"` back; `gov.forget(id)` deletes the row.

## The SQL issued

Identifiers are validated with `^[A-Za-z_][A-Za-z0-9_]*$` (table may be `schema.table`) and double-quoted;
values are always bound as `%s` parameters, never interpolated. With the defaults and table `memories`:

| action | statement | parameters |
|---|---|---|
| pull (page 1) | `SELECT "id", "text", "metadata" FROM "memories" ORDER BY "id" LIMIT %s` | `(batch_size,)` |
| pull (next pages) | `SELECT "id", "text", "metadata" FROM "memories" WHERE "id" > %s ORDER BY "id" LIMIT %s` | `(last_id, batch_size)` |
| pull with scope | `... FROM "memories" WHERE ("user_id" = %s) AND "id" > %s ORDER BY "id" LIMIT %s` | `(*scope_params, last_id, batch_size)` |
| flag | `UPDATE "memories" SET "metadata" = COALESCE("metadata", '{}'::jsonb) \|\| %s::jsonb WHERE "id" = %s` | `(json.dumps(reason.as_metadata()), host_id)` |
| delete | `DELETE FROM "memories" WHERE "id" = %s` | `(host_id,)` |
| insert | `INSERT INTO "memories" ("text", "metadata", "embedding") VALUES (%s, %s::jsonb, %s) RETURNING "id"` | `(text, json.dumps(meta), "[0.1,0.2,...]" or None)` |

`insert_columns={"user_id": "alice"}` appends `, "user_id"` / `, %s` / `"alice"` to the insert. The inserted
metadata is `{"source": ..., "invalidate_status": "active", "invalidate_supersedes": [...], "invalidate_event_id": ...}`.

`live_where()` returns the fragment used at query time:

```sql
("metadata" IS NULL OR NOT ("metadata" ? 'invalidate_status')
 OR "metadata"->>'invalidate_status' NOT IN ('contradicted', 'superseded'))
```

Every write is followed by `conn.commit()` unless `autocommit=True`; `pull()` commits too, so a SELECT never
leaves the connection idle in transaction (on error it rolls back and re-raises).

## Why it is written this way

- `jsonb || jsonb` is a key union where the right operand wins, so a flag merges the receipt over the row's
  existing metadata and a re-flag overwrites the earlier receipt. `NULL || x` is NULL, hence `COALESCE`.
- `jsonb ? 'key'` is "top-level key exists"; `NULL ? 'key'` is NULL and would drop the row, hence the `IS NULL`
  guard. Rows that were never flagged (no key, or a NULL column) stay live without any stamping step.
- psycopg 3 sends a Python `str` with the unknown OID and psycopg2 inlines it as a literal, so the host id
  (kept as a string in the ledger) compares against `uuid` and `bigint` columns without a cast, and a
  `json.dumps` string with `%s::jsonb` is a jsonb value. A vector is passed in pgvector's text form
  `'[0.1,0.2,...]'`; if `vector_fn` returns something the driver adapts natively (a `pgvector.Vector` after
  `register_vector(conn)`, a numpy array) it is passed through unchanged.
- Keyset paging (`id > last ORDER BY id LIMIT n`) streams large tables in `batch_size` rows per SELECT on any
  driver, without a server-side cursor.

## Limits

- Ids are `uuid` or `bigint` (anything with a total order that round-trips through `str()`). Composite keys
  are not supported.
- `scope_sql` is inserted verbatim (in parentheses) with its own `%s` placeholders; write it with quoted
  identifiers yourself. It applies to `pull()` only. To keep successor rows in scope use `insert_columns`.
- Without `vector_fn`, successor inserts write `NULL` into the embedding column; a `NOT NULL` constraint there
  turns the insert into a reported push error (the ledger still moves).
- Rows whose text is NULL or blank are skipped by `pull()`; they are still returned by a `live_where()` query,
  since they carry no receipt.
- Drivers with `qmark` / `numeric` / `named` paramstyle (sqlite3, asyncpg) are rejected at construction. Async
  connections are not supported; wrap a sync connection.
- One connection per adapter; the adapter does no pooling or reconnection.

## Testing

`tests/test_adapter_pgvector.py` runs against a fake DB-API connection that records every statement and
simulates the JSONB merge and `?` operator; the SQL text and parameters it asserts are the contract above.
Set `INVALIDATE_PG_DSN` with `psycopg` installed and the same tests also run against a real database (a temp
table with `vector(4)` per test). `scripts/live_pgvector.py` does the same end to end with the real judge.
