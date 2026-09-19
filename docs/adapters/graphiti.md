# Graphiti

Governs the entity edges (facts) of one graphiti-core `group_id`. Graphiti already has a native "no longer
true" representation (`invalid_at` / `expired_at` on an edge); the adapter uses it instead of deleting.

## Install

```sh
pip install invalidate graphiti-core   # tested against graphiti-core 0.30.2 signatures; needs a Neo4j/FalkorDB/Kuzu driver
```

## Usage

```python
from graphiti_core import Graphiti
from invalidate.adapters import Governor
from invalidate.adapters.graphiti import GraphitiAdapter

graphiti = Graphiti("bolt://localhost:7687", "neo4j", "password")
adapter = GraphitiAdapter(graphiti, group_id="alice")        # driver defaults to graphiti.driver
gov = Governor(adapter, "ledger.db", mode="flag", successors=True)

gov.sync()                                                   # live RELATES_TO edges -> ledger
report = gov.observe("Alice migrated to SQLite", source="slack")
print(report.summary())                                      # stale edges now carry invalid_at/expired_at
gov.keep("<edge uuid>")                                      # human override: window cleared again
labelled = gov.annotate(edges, id_of=lambda e: e.uuid)       # serve all: [(edge, "OUTDATED, replaced as of ..." | None)]

# the adapter is sync; it runs graphiti's coroutines for you, inside or outside an event loop
```

## What each push does in Graphiti

| action | call | notes |
|---|---|---|
| pull | `EntityEdge.get_by_group_ids(driver, [group_id])` | edges with `expired_at is None`; id = `edge.uuid`, text = `edge.fact`. Episodes invalidate inserted (name `invalidate:<event id>`) are pulled too, as `kind="event"` |
| flag | `edge.attributes.update(reason.as_metadata()); await edge.save(driver)` | for contradicted / superseded / deleted also `edge.invalid_at = invalid_at or now`, `edge.expired_at = now`. needs_review only writes the attributes. A restore clears both timestamps |
| delete | identical to a dead flag | Graphiti keeps history; the adapter never hard-deletes an edge. `mode="delete"` == invalidate |
| insert | `graphiti.add_episode(name="invalidate:<event id>", episode_body=text, source_description=source, reference_time=now, group_id=group_id)` | returns `results.episode.uuid` |

## Limits

- **Successors are not verbatim edges.** `add_episode` runs Graphiti's own extraction, so the facts it derives
  are its wording and appear as new memories on the next `sync()`. The episode node itself keeps the event
  verbatim, which is why the adapter pulls its own episodes: the successor id stays stable across syncs.
  Set `include_episodes=False` to pull edges only (then `insert` still returns the episode uuid, but the next
  sync forgets it).
- Verdicts on an episode (a successor later contradicted) stay in the ledger; episodes have no validity
  window in Graphiti.
- `attributes` round-trip as edge properties on Neo4j/FalkorDB and as JSON on Kuzu; values are the scalar
  receipt keys from `Reason.as_metadata()`. `get_entity_edge_from_record` strips the reserved keys, so the
  receipt never collides with `fact`, `valid_at`, etc.
- `pull` needs `graphiti_core.edges.EntityEdge` (and `nodes.EpisodicNode`) importable; both are imported
  lazily inside the adapter, or you can inject `edge_class=` / `episode_class=`.
- Invalidated edges drop out of `pull`, so the ledger row stays at its dead status and is not re-judged. If
  Graphiti later re-derives the same fact it becomes a new edge (new uuid) and a new memory.
- Sync-over-async: when a loop is already running in the calling thread, each call runs on a short-lived
  helper thread with its own loop. Drivers that are bound to one loop (some async Neo4j sessions) should be
  used from a thread without a running loop, i.e. plain scripts or a worker thread.
