# Letta

Two adapters over one Letta agent: `LettaAdapter` for archival memory (one passage = one memory) and
`LettaBlockAdapter` for a core memory block (one line = one memory).

## Install

```sh
pip install invalidate letta-client   # tested against letta-client 1.12.1 signatures
```

## Usage

```python
from letta_client import Letta
from invalidate.adapters import Governor
from invalidate.adapters.letta import LettaAdapter, LettaBlockAdapter, block_lines

client = Letta(api_key="...")           # or Letta(base_url="http://localhost:8283")
agent_id = "agent-..."

gov = Governor(LettaAdapter(client, agent_id), "ledger.db", mode="flag", successors=True)
gov.sync()                                                # every archival passage into the ledger
gov.observe("we migrated to SQLite", source="slack")      # stale passages get a sidecar note
gov.keep("passage-...")                                   # human override: note removed
labelled = gov.annotate(passages, id_of=lambda p: p.id)   # serve all: [(passage, "OUTDATED, replaced as of ..." | None)]

human = Governor(LettaBlockAdapter(client, agent_id, "human"), "ledger.db", mode="flag")
human.sync(); human.observe("Alice moved to Lisbon", source="crm")
print(block_lines(client, agent_id, "human"))             # one HostMemory per line, id = human#<sha1>
```

## What each push does in Letta

### Archival passages (`LettaAdapter`)

| action | call | notes |
|---|---|---|
| pull | `client.agents.passages.list(agent_id=..., limit=100, after=<last id>)` | paginated; invalidate's own notes are skipped |
| flag | `client.agents.passages.create(agent_id=..., text="[invalidate] “<fact>” is stale: superseded by “<event>” (<source>, still true 5%) ref:<passage id>", tags=["invalidate", "<status>"])` | the original passage is untouched; the note id is kept in `adapter.notes` and rebuilt from `ref:` on the next pull |
| restore (keep) | `client.agents.passages.delete(memory_id=<note id>, agent_id=...)` | the note goes away |
| delete | `client.agents.passages.delete(memory_id=<passage id>, agent_id=...)` | mode `"delete"`: the stale passage itself is removed, plus its note |
| insert | `client.agents.passages.create(agent_id=..., text=<event verbatim>, tags=["invalidate-successor", "source:<source>"])` | returns the first chunk's id |

### Core memory block (`LettaBlockAdapter`)

| action | call | notes |
|---|---|---|
| pull | `client.agents.blocks.retrieve(block_label=..., agent_id=...)` | each non-empty line; an annotated line is reported under its original text and id |
| flag | `client.agents.blocks.update(block_label=..., agent_id=..., value=<whole block>)` | the line becomes `~~line~~ (invalidate: superseded by “…” (source, still true 5%))`; a restore puts the plain line back |
| delete | same `update` | the line is removed |
| insert | same `update` | the event is appended as a new line; id `<label>#<sha1 of line>` |

## Limits

- **Passages cannot be edited or tagged after creation** (the SDK has create / list / delete / search only), so
  a flag is a new passage next to the stale one. The agent will see both in archival search; the note says
  which one is stale and why. Use `mode="delete"` if you would rather the stale passage disappear.
- `passages.create` may split long text into several chunks and returns a list; the adapter keeps the first
  id. Successors should be short facts.
- The `ref:` suffix is what lets a new process map notes back to passages. If you hand-edit notes, keep it.
- Blocks are rewritten whole. The block's character `limit` still applies, so an annotation that pushes the
  block over its limit fails and shows up as a push error in `GovernorReport.errors`. Lines are identified by
  their sha1, so editing a line's text elsewhere makes it a new memory on the next sync.
- Older `letta-client` releases named the block PATCH `modify` instead of `update`; both are tried. All calls
  are keyword-only so argument-order changes between SDK versions do not matter.
