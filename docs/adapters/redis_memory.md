# Redis Agent Memory Server

Governs the long-term memories of one scope (`namespace` / `user_id` / `session_id`, any subset) in a
[Redis Agent Memory Server](https://github.com/redis/agent-memory-server) through its Python client
`agent-memory-client` (`MemoryAPIClient`). The server keeps its memories; invalidate keeps the ledger and
pushes verdicts back as **marker topics**, because a memory there has no free-form metadata field.

## Install

```sh
pip install invalidate agent-memory-client   # tested against agent-memory-client 0.14.0 (== tag client/v0.14.0)
```

The server must be 0.14.0 or newer: `pull` relies on the empty-query filter-only listing that
`search_long_term_memories` performs when `text` is blank (present in server/v0.14.0, server/v0.15.2 and main).

## Usage

```python
from agent_memory_client import MemoryAPIClient, MemoryClientConfig
from invalidate.adapters import Governor
from invalidate.adapters.redis_memory import RedisMemoryAdapter, governed_search, dead_markers

client = MemoryAPIClient(MemoryClientConfig(base_url="http://localhost:8000"))
adapter = RedisMemoryAdapter(client, namespace="app", user_id="alice")
gov = Governor(adapter, "ledger.db", mode="flag", successors=True)

gov.sync()                                                     # list alice's memories into the ledger
report = gov.observe("we migrated to SQLite", source="slack")  # judge, then flag/delete on the server
print(report.summary())

hits = governed_search(client, gov, "which database?")         # search minus dead memories (sync)
hits = governed_search(client, gov, "which database?", topics={"none": dead_markers()})  # also pre-filter host-side
hits = governed_search(client, gov, "which database?", annotate=True)  # keep all; each record gets .invalidate_note
```

The client is async. The adapter (and `governed_search`) run each coroutine with `asyncio.run` when no event
loop is running in the calling thread; inside a running loop they use a fresh loop on a helper thread. From
async code prefer `await agoverned_search(client, gov, query)` for the read side; `gov.sync()` / `gov.observe()`
are sync by protocol and will use the helper thread. Create the `MemoryAPIClient` outside any event loop if you
mix both, so its `httpx.AsyncClient` is not bound to a loop that later disappears.

## What each push does on the server

| action | call (agent-memory-client 0.14.0) | notes |
|---|---|---|
| pull | `search_long_term_memory(text="", namespace={"eq": ...}, user_id={"eq": ...}, session_id={"eq": ...}, limit=100, offset=n)` (client.py:1033) | blank text = filter-only listing; pages by `offset` until a short page; `limit` is clamped to the server max of 100; `memory_types=[...]` adds a `memory_type={"in_": [...]}` filter |
| flag | `get_long_term_memory(id)` (client.py:753) then `edit_long_term_memory(id, {"topics": [...existing minus old markers, *receipt]})` (client.py:773) | topics-only PATCH; text never sent |
| delete | `delete_long_term_memories([id])` (client.py:731) | mode `"delete"` only, for contradicted / superseded |
| insert | `create_long_term_memory([MemoryRecord(id=<ULID>, text=<verbatim>, memory_type="semantic", topics=[markers], discrete_memory_extracted="t", <scope>)], deduplicate=False)` (client.py:661) | id minted client-side and returned; `deduplicate=False` so the server never LLM-merges the text |

### The receipt: marker topics

`MemoryRecord` (models.py:168) has `topics` and `entities` but no metadata dict, and the server's PATCH accepts
only `text, topics, entities, memory_type, namespace, user_id, session_id, event_date, pinned` and **replaces**
each field it is given (`long_term_memory.py: update_long_term_memory`). So a flag is written as topics, all
prefixed `invalidate:`:

```
invalidate:superseded                              # the status: active | needs_review | contradicted | superseded
invalidate:disposition:replaces
invalidate:event:we migrated to sqlite             # event text, <=160 chars, commas replaced (tags may not contain commas)
invalidate:event_source:slack
invalidate:event_id:evt_01J...
invalidate:still_true:0.05
invalidate:at:1758240000.0
```

`flag` reads the current topics, drops every `invalidate:*` marker, appends the new receipt and PATCHes the
list, so your own topics survive and a memory carries exactly one status marker. Successors are created with
`invalidate:successor`, `invalidate:source:<source>`, `invalidate:supersedes:<old id>` (one per superseded id)
and `invalidate:event_id:<id>`. `pull` strips the markers out of `HostMemory.metadata["topics"]` (they are
kept under `metadata["invalidate_markers"]`), so a flag never looks like a rewrite to the ledger.

Because the markers are indexed tags, you can pre-filter on the server too:
`search_long_term_memory(..., topics={"none": dead_markers()})`. `governed_search` still applies the ledger
filter on top (it also hides `needs_review` memories by default, which a topic filter would not).

`governed_search(client, gov, query, **kw)` returns the `MemoryRecordResults` the client returned with
`memories` filtered and `total` / `next_offset` untouched; the adapter's scope filters are injected when `kw`
names none of `namespace`, `user_id`, `session_id`. `agoverned_search` is the awaitable twin.

## Limits

- **Background indexing.** `create_long_term_memory` returns before the server has indexed the record
  (`api.py` schedules `index_long_term_memories` as a background task). The adapter remembers each inserted
  successor locally and serves it from `pull` until the server lists it, so a `sync()` right after
  `observe()` does not mark the successor as gone. That memory of pending inserts lives only in the adapter
  instance: a fresh process that syncs before the server has caught up will drop the row and re-add it later.
- **Server-side topic extraction.** When the server extracts topics/entities for a new memory it *merges*
  them with the stored list (`extract_memory_structure`), so markers survive; inserted successors are sent
  with `discrete_memory_extracted="t"` so the server does not derive extra (rewritten) memories from them.
- **No metadata field.** The receipt is limited to what fits in tags: the event text is truncated to 160
  characters and commas become spaces. The full event lives in the ledger (`gov.mem.store.get_event(id)`).
  A server version that exposes a PATCH-able metadata field would be a better home; as of `main` the
  `metadata` field on `MemoryRecord` exists but is not updatable.
- **`flag` needs a read first.** If `get_long_term_memory` fails (404, network) the flag is not written and
  the governor records a push error; the ledger is still updated.
- **Scope.** `namespace` / `user_id` / `session_id` are all optional. With none given, the client's
  `MemoryClientConfig.default_namespace` (if set) is applied by the client to `pull` and `insert`; otherwise
  the adapter governs every long-term memory the server returns. `memory_types=("semantic", "episodic")`
  narrows a pull to those types (the default pulls `message` memories too).
- **Sync wrappers.** Calling the sync adapter from inside a running event loop runs the coroutine on a
  helper thread with its own loop; `httpx.AsyncClient` tolerates this in practice but it is not the intended
  use of the client. Async apps should keep governor calls on a worker thread or use `agoverned_search`.
