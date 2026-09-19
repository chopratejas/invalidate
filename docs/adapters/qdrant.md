# Qdrant

Governs one Qdrant collection (or a filtered slice of it) through `qdrant-client`. Qdrant keeps the points, vectors
and payloads; invalidate keeps the ledger and pushes verdicts back as `invalidate_*` payload keys. The memory text is
never rewritten.

## Install

```sh
pip install invalidate qdrant-client        # tested against qdrant-client 1.19.1 signatures
```

Works against a server (`QdrantClient("http://localhost:6333")`), Qdrant Cloud, or the embedded `QdrantClient(":memory:")`
/ `QdrantClient(path=...)` modes; the tests use `:memory:`.

## Usage

```python
from qdrant_client import QdrantClient, models
from invalidate.adapters import Governor
from invalidate.adapters.qdrant import QdrantAdapter, governed_query

client = QdrantClient("http://localhost:6333")
adapter = QdrantAdapter(
    client, "memories",
    text_key="text",                                   # payload key holding the memory text
    scope_filter=models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value="alice"))]),
    scope_payload={"user_id": "alice"},                # written onto successors so they stay in scope
    vector_fn=embed,                                   # text -> list[float]; only needed for successors
)
gov = Governor(adapter, "ledger.db", mode="flag", successors=True)

gov.sync()                                                       # scroll alice's points into the ledger
report = gov.observe("we migrated to SQLite", source="slack")    # judge, then set_payload / delete in Qdrant
print(report.summary())

hits = governed_query(client, gov, "memories", embed("which database?"), limit=5)   # QueryResponse minus dead points
hits = governed_query(client, gov, "memories", embed("which database?"), limit=5, annotate=True)  # keep all; payload["invalidate_note"]
for p in hits.points:
    print(p.score, p.payload["text"])
```

`QdrantAdapter(client, collection, *, text_key="text", scope_filter=None, vector_fn=None, vector_name=None,
scope_payload=None, metadata_prefix="invalidate_", name=None, page_size=256, max_rows=5000, source="qdrant")`.
Point ids become host ids as strings (`"42"`, `"3f2b…-…"`) and are converted back on flag/delete.

## What each push does in Qdrant

Line numbers are in qdrant-client 1.19.1 under `site-packages/qdrant_client/`.

| action | call | notes |
|---|---|---|
| pull | `client.scroll(collection, scroll_filter=scope_filter, limit=page_size, offset=next, with_payload=True, with_vectors=False)` (`qdrant_client.py:705`) | returns `(records, next_offset)`; loops until `next_offset is None` or `max_rows`. Points whose `payload[text_key]` is not a non-empty string are skipped. `kind` / `source` payload keys map onto `HostMemory`; other non-`invalidate_*` keys ride along as metadata. |
| flag | `client.set_payload(collection, payload=reason.as_metadata(), points=[id])` (`qdrant_client.py:1189`) | **merges** top-level keys, text untouched; receipt keys `invalidate_status`, `invalidate_disposition`, `invalidate_event`, `invalidate_event_source`, `invalidate_event_id`, `invalidate_still_true`, `invalidate_at` |
| delete | `client.delete(collection, points_selector=models.PointIdsList(points=[id]))` (`qdrant_client.py:1136`, `http/models/models.py:2303`) | mode `"delete"` only, for contradicted / superseded; unknown ids are no-ops |
| insert | `client.upsert(collection, points=[models.PointStruct(id=uuid4, vector=vector_fn(text), payload={...})])` (`qdrant_client.py:867`, `http/models/models.py:2320`) | payload is `{**scope_payload, text_key: text (verbatim), "source": ..., "invalidate_status": "active", "invalidate_supersedes": [...], "invalidate_event_id": ...}`; returns the new UUID string. Without `vector_fn` the adapter has no `insert` and the Governor writes no successors. |
| query | `client.query_points(collection, query=vector, query_filter=..., limit=..., using=..., ...)` (`qdrant_client.py:269`) | via `governed_query`; there is no `QdrantClient.search` in 1.19.1 |

### Payload merge semantics (verified)

`set_payload` merges at the top level: `local/local_collection.py:3061` is literally
`self.payload[idx] = {**self.payload[idx], **jsonable_payload}`, and the server behaves the same (the docstring at
`qdrant_client.py:1209` says an existing key "will be overwritten", i.e. per key). `tests/test_adapter_qdrant.py::
test_set_payload_merges_where_overwrite_payload_replaces` pins this down and contrasts it with `overwrite_payload`
(`qdrant_client.py:1291`), which replaces the whole payload and is never used here. Consequences:

- Unrelated keys (`text`, `kind`, `user_id`, ...) survive a flag; the adapter does not need a read-merge-write.
- A same-named top-level key is replaced, not deep-merged: a nested dict under `invalidate_status` would be swapped
  out wholesale (the receipt uses flat scalar values, so this never matters in practice).
- A point with a `None` payload gets the receipt as its whole payload.
- Flagging an id that no longer exists raises (`KeyError` in `:memory:` mode, `UnexpectedResponse` 404 from a server).
  The Governor catches it, records a push error, and keeps the ledger verdict.

### Live filter

`adapter.live_filter()` / `live_filter(prefix)` returns

```python
models.Filter(must_not=[models.FieldCondition(key="invalidate_status", match=models.MatchAny(any=["contradicted", "superseded"]))])
```

Verified on 1.19.1: `must_not MatchAny` keeps points that have **no** `invalidate_status` key at all, so unflagged points
stay visible without any stamping step (there is no `stamp_live()` here because none is needed). The alternative
`must=[FieldCondition(match=MatchExcept(except=[...]))]` does *not* match key-less points and is not used.
`needs_review` is not a dead value, so reviewed points remain queryable by filter; `Governor.filter()` hides them at the
application layer if you want that.

`merge_filter(base, extra)` ANDs two filters by nesting (`Filter(must=[base, extra])`; a `Filter` is a valid
`Condition`, `http/models/models.py:4120`). `adapter.scoped_live_filter()` is `scope_filter AND live_filter()`.

`governed_query(client, gov, collection, query_vector, *, query_filter=None, **kw)` ANDs the live filter into
`query_filter`, calls `query_points`, then drops any point whose id is in `gov.dead_ids()`. The ledger prune matters in
`mode="ledger"` and after push errors, where the host never received the flag. It returns the `QueryResponse` with
`.points` pruned; `using=`, `limit=`, `with_payload=`, `score_threshold=`, ... pass straight through.

## Limits

- **String point ids must be UUIDs** (`ExtendedPointId`, `http/models/models.py:4143`); `"pg"` is rejected by the client.
  Integer ids are fine and round-trip as `"42"`.
- One collection per adapter. Use `scope_filter` to govern a tenant slice, and `scope_payload` so successors inherit the
  tenant keys; otherwise the next `sync()` will not see them and will mark them gone.
- `pull` stops at `max_rows` (default 5000) with a warning; raise it for larger collections.
- `vector_fn` must produce the collection's vector shape: a `list[float]` for an unnamed vector, or set `vector_name=`
  (or return `{name: vector}`) for named-vector collections. Sparse and multi-vector layouts are not handled.
- Receipt writes go through `set_payload` with `wait=True` (the client default), so a flag is visible to the next query.
- Payload lists are allowed (`invalidate_supersedes` is stored as a real list), unlike Chroma.
