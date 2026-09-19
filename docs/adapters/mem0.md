# mem0

Governs one mem0 scope (`user_id` / `agent_id` / `run_id`) in either the OSS `mem0.Memory` or the platform
`mem0.MemoryClient`. mem0 keeps its memories; invalidate keeps the ledger and pushes verdicts back.

## Install

```sh
pip install invalidate mem0ai        # tested against mem0ai 2.1.0 signatures
export MEM0_TELEMETRY=false          # optional: mem0 phones home by default
```

## Usage

```python
from mem0 import Memory                      # or: from mem0 import MemoryClient
from invalidate.adapters import Governor
from invalidate.adapters.mem0 import Mem0Adapter, governed_search, guard_add

memory = Memory()
adapter = Mem0Adapter(memory, user_id="alice")
gov = Governor(adapter, "ledger.db", mode="flag", successors=True)

gov.sync()                                                    # pull alice's memories into the ledger
report = gov.observe("we migrated to SQLite", source="slack") # judge, then flag/delete in mem0
print(report.summary(), adapter.flagged_in_host)

hits = governed_search(memory, gov, "which database?")        # search() minus dead memories
hits = governed_search(memory, gov, "which database?", annotate=True)  # keep all; each result dict gains "invalidate_note"
add = guard_add(memory, gov)                                  # judge-before-write add()
add([{"role": "user", "content": "we're on Postgres 16 now"}], user_id="alice")
```

## What each push does in mem0

| action | call | notes |
|---|---|---|
| pull | `get_all(filters={"user_id": ...}, top_k=5000)` (2.x / platform) or `get_all(user_id=..., limit=5000)` (0.1.x) | handles `{"results": [...]}`, bare lists, and platform `next` pagination |
| flag | `update(memory_id, metadata={**existing, **reason.as_metadata()})` | text untouched; receipt keys `invalidate_status`, `invalidate_event`, ... |
| delete | `delete(memory_id)` | mode `"delete"` only, for contradicted / superseded |
| insert | `add(text, <scope>, infer=False, metadata={"invalidate_source": ..., "invalidate_supersedes": ...})` | `infer=False` stores the event **verbatim**; returns `results[0]["id"]` |

`governed_search` returns what `search()` returned (the `{"results": [...]}` dict with other keys preserved, or a
list) with dead ids dropped; the adapter's scope is injected when you pass no `filters=`. `guard_add` wraps
`memory.add` with `gov.guard`, judging every user-role message before mem0 stores anything.

## Limits

- **Old OSS releases whose `update(memory_id, data)` has no `metadata` parameter cannot carry a flag.** The
  adapter never rewrites the memory text to smuggle a receipt in, so on those versions a flag is ledger-only:
  `flag()` returns without error and `adapter.flagged_in_host` is `False`. `gov.status_of(id)` and
  `governed_search` still work because they read the ledger.
- The platform `update` replaces metadata; the adapter reads the current metadata with `get(memory_id)` and
  merges before writing. If `get` fails the receipt is written alone.
- Receipt values are kept scalar (lists are joined with `,`) because some vector stores behind mem0 (Chroma)
  reject list-valued payloads.
- `add(..., infer=True)` (mem0's default) rewrites text through its LLM; successors use `infer=False`, but
  memories your app adds with inference are pulled as whatever mem0 extracted.
- One scope per adapter. Memories in other `user_id`s are invisible to the governor.
