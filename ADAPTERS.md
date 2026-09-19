# invalidate as a layer over your memory system

You do not replace Mem0, Letta, Zep, LangGraph, Chroma, or your Markdown notes.
You put invalidate in front of them. The host keeps the data; invalidate keeps
the truth about it: which memories are still true, which died, which need a
human, and every vote that led there.

```
                 events (user messages, Slack, GitHub, migrations, webhooks)
                                   │
                                   ▼
   ┌────────────┐   pull()   ┌────────────┐  6 votes per (event, memory)  ┌──────┐
   │  host      │ ─────────▶ │  Governor  │ ─────────────────────────────▶ │ Jev  │
   │  (Mem0,    │            │  + ledger  │ ◀───────────────────────────── │      │
   │  Chroma,   │ ◀───────── │  (SQLite)  │         probabilities          └──────┘
   │  files…)   │ flag/delete│            │
   └────────────┘ /insert    └────────────┘
         ▲                          │
         │ filter(results)          │ policy: code decides the write
         └──────── your app ◀───────┘
```

## The contract an adapter implements

```python
class Adapter(Protocol):
    name: str
    def pull(self) -> Iterable[HostMemory]: ...          # id + verbatim text, read-only
    def flag(self, host_id: str, reason: Reason) -> None  # mirror a status + receipt into the host
    def delete(self, host_id: str, reason: Reason) -> None
    # optional
    def insert(self, text: str, source: str, metadata: dict) -> str | None   # store a successor verbatim
```

`Reason` carries the status, the event text and source, the `still_true` vote
and a timestamp. `Reason.as_metadata()` is what metadata-capable hosts get;
`Reason.line()` is the one-line receipt for hosts that only hold text.

## What the Governor does

- `sync()`: pulls the host and reconciles the ledger by host id. New text is a
  new claim (a host rewrite resets the row to active). Ids that vanish while
  live are marked deleted. Idempotent.
- `observe(text, source)`: judges the event against every live host memory
  (batched, screened above 200) and pushes each status change into the host.
- `mode="flag"` (default): write `invalidate_status`, `invalidate_event`, ... into
  the host record. Reversible. `mode="delete"`: dead rows are deleted from the
  host; reviews are still flagged. `mode="ledger"`: judge and log, touch nothing.
- `successors=True`: when an event supersedes memories, insert the event text
  verbatim into the host and link the dead rows to it.
- `filter(results, id_of=...)`: drop dead and reviewed ids from any host search result
  (reviewed rows are hidden until a human calls `keep()` or `forget()`).
- `guard(host.add)`: wrap the host's write path so incoming user text is judged
  against memory before it is stored.
- `keep(id)` / `forget(id)`: human overrides, mirrored both ways.
- Push errors never lose the ledger: they are reported on the `GovernorReport`.

## Adapters

| host | module | pull | flag | delete | insert | verified |
|---|---|---|---|---|---|---|
| any dict / test double | `adapters.InMemoryAdapter` | yes | metadata | yes | yes | unit |
| Markdown files (CLAUDE.md, memory notes, docs) | `adapters.markdown` | claim lines, content-addressed ids | HTML comment on the line | line removed | appended section | live |
| Chroma | `adapters.chroma` | collection.get | metadata merge | collection.delete | collection.add | live |
| any vector store (pgvector, Pinecone, Qdrant, ...) | `adapters.vectorstore` | your `list_fn` | your `update_metadata_fn` | your `delete_fn` | your `insert_fn` | unit |
| LangGraph store | `adapters.langgraph` | store.search | put (merged) | store.delete | put | live (InMemoryStore) |
| Mem0 (OSS and platform) | `adapters.mem0` | get_all | metadata (see limits) | delete | add(infer=False) | live (OSS 2.1.0) |
| Letta | `adapters.letta` | archival passages, block lines | note passage / annotated block line | passage or line removed | passage | signatures mirrored |
| Zep / Graphiti | `adapters.graphiti` | entity edges' `fact` | `invalid_at` set | same as flag (history kept) | add_episode | signatures mirrored |
| Qdrant | `adapters.qdrant` | client.scroll | set_payload (merged) | client.delete | client.upsert (needs `vector_fn`) | live (`:memory:`, 1.19.1) |
| Postgres / pgvector (any DB-API conn) | `adapters.pgvector` | keyset SELECT by id | `metadata = COALESCE(metadata,'{}') \|\| receipt::jsonb` | DELETE row | INSERT ... RETURNING id (embedding via `vector_fn`) | exact-SQL fake; real PG via `INVALIDATE_PG_DSN` |
| LlamaIndex (FactExtractionMemoryBlock) | `adapters.llamaindex` | block.facts, content-addressed ids | fact moved out of the list into `adapter.hidden` | fact removed | fact appended | live (llama-index-core 0.14.24, MockLLM) |
| MCP memory server (`@modelcontextprotocol/server-memory` JSONL) | `adapters.mcp_memory` | observations + relations | marker suffix inside the observation string | observation / relation removed | observation appended to an entity | live (server 0.6.3 file format) |
| Redis Agent Memory Server | `adapters.redis_memory` | search_long_term_memory paged | `invalidate:*` marker topics (no metadata field) | delete_long_term_memories | create_long_term_memory verbatim, dedup off | signatures mirrored (agent-memory-client 0.14.0) |
| Cognee | `adapters.cognee` | one dataset's Data rows (verbatim text) | ledger-only by default; opt-in `flag_in_host=True` writes `external_metadata` | datasets.delete_data | cognee.add(DataItem) verbatim, no cognify | signatures mirrored (cognee 1.6.0); governed_search filters by document_id |

Per-adapter notes live in `docs/adapters/`.

## Lazy mode through the layer

`Governor(adapter, db, lazy=True)`: `observe()` appends events only; `filter(results, id_of=...)` validates
exactly the results your host returned against the events they have not seen, pushes flags or deletes for
the ones that died, and hides them. `validate(budget_requests=N)` drains backlogs from a background job;
`observe_many(texts)` is the batch-ingest path. See [SCALING.md](SCALING.md).

## Why a layer and not a feature of each host

Each host either adds and never removes (Mem0), checks only the ten nearest by
similarity with a generative LLM (Graphiti), or hopes the agent notices (Letta).
None can afford to re-judge every memory on every event with an LLM, and none
ingests events that are not memories (a migration, an outage, a PR). Jev makes
exhaustive judgment cost about $0.00006 per pair (about $0.00001 with screening) at 150 ms, and the Governor
makes it host-agnostic. See COMPETITIVE.md.
