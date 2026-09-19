# MCP memory server

Governs the JSONL file behind `@modelcontextprotocol/server-memory`, the knowledge-graph memory server that
Claude Desktop, Claude Code and other MCP clients use. Every observation string (and every relation) is
one memory; invalidate keeps the ledger and writes its verdicts back into the same strings, so the server
and the model see them without any change on the server side. No Python dependency beyond invalidate.

## The file (verified upstream)

Source: <https://raw.githubusercontent.com/modelcontextprotocol/servers/main/src/memory/index.ts>
(`@modelcontextprotocol/server-memory` 0.6.3, checked 2026-09-19).

- **Path.** `MEMORY_FILE_PATH` (a leading `~` is expanded; a relative path is resolved against the server's
  package directory, so use an absolute one). Default: `memory.jsonl` next to the package's `dist/index.js`.
  A legacy `memory.json` in that directory is renamed to `memory.jsonl` on first start.
- **One JSON object per line**, written by `saveGraph` with compact `JSON.stringify`, entities first, then
  relations, `"\n"`-joined with a trailing newline, through a temp file + rename:

  ```
  {"type":"entity","name":"alice","entityType":"person","observations":["user prefers Postgres","deploys run at 2pm UTC"]}
  {"type":"relation","from":"alice","to":"billing","relationType":"owns"}
  ```

- **Field names** (zod schemas in `index.ts`): entity = `type: "entity"`, `name: string`, `entityType: string`,
  `observations: string[]`; relation = `type: "relation"`, `from: string`, `to: string`, `relationType: string`.
  There are no ids, timestamps or metadata fields: the observation string is the whole memory.
- **Loading** (`loadGraph`): lines are split on `"\n"`, blank lines skipped, malformed JSON skipped with a
  log line, objects whose `type` is neither `entity` nor `relation` ignored, and entity/relation objects that
  fail the schema skipped. Unknown extra keys are stripped by zod and are not written back.
- **Semantics** that matter to us: `add_observations` deduplicates by exact string inside an entity;
  `delete_observations` deletes by exact string; `create_entities` deduplicates by `name`; `search_nodes`
  substring-matches names, types and observation text (so a marker is searchable too).

## What the adapter does

| action | what happens in the file |
|---|---|
| pull | one `HostMemory` per observation, id `<entity name>#<sha1(normalized text)[:10]>`, text = the observation with any marker stripped; one per relation, id `relation#<sha1("<from> <relationType> <to>")[:10]>`, text `"<from> <relationType> <to>"` |
| flag | the observation becomes `"<text> [invalidate: superseded by “<event>” (<source>, still true 5%), <YYYY-MM-DD>]"` (same wording as the Markdown adapter's comment; `]` in the event is replaced). A relation is flagged on its `relationType`. `Status.ACTIVE` (`gov.keep`) strips the marker. Flagging again replaces the marker |
| delete | the observation is removed from its entity's array (an entity with no observations left is kept); a relation line is removed |
| insert | appends `"<text> [invalidate: from <source>, <date>]"` to the entity named in `metadata["entity"]`, else the entity of the single observation it supersedes (`follow_superseded=True`), else `default_entity` (created with `default_entity_type` when missing, placed before the first relation line). Same text in the same entity returns the existing id |

Ids are derived from the **original** text, so a flagged observation keeps its id across pulls; the marker
is parsed back out and reported in `HostMemory.metadata` as `invalidate_status` (`superseded`, `contradicted`,
`needs_review`, ...) and `invalidate_note`; successors carry `invalidate_from`. Every other line (unknown
`type`s, malformed JSON, blank lines, extra keys on an entity, `\r\n` endings) is preserved byte for byte
when the adapter writes; writes go through a temp file + `os.replace`, like the server's own.

`governed_read(path, gov)` returns `{"entities": [...], "relations": [...]}` as the server would load it,
minus dead observations and relations (and minus memories under review unless `include_review=True`).

## Usage

```python
from invalidate.adapters import Governor
from invalidate.adapters.mcp_memory import McpMemoryAdapter, governed_read

adapter = McpMemoryAdapter("~/.claude/memory.jsonl")          # or McpMemoryAdapter() to read MEMORY_FILE_PATH
gov = Governor(adapter, "ledger.db", mode="flag", successors=True)

gov.sync()                                                     # every observation and relation into the ledger
report = gov.observe("we migrated to SQLite last Tuesday", source="slack")
print(report.summary())                                        # marks "user prefers Postgres [...]" in the file
graph = governed_read(None, gov)                               # the graph with dead memories removed
gov.keep("alice#75cc9d30cc"); gov.forget("billing#1a2b3c4d5e") # human overrides, mirrored into the file
```

Constructor: `McpMemoryAdapter(path=None, *, default_entity="invalidate", default_entity_type="invalidate",
follow_superseded=True, relations=True, annotate=True)`. `relations=False` pulls observations only;
`annotate=False` makes `flag` a no-op (ledger only) while `delete` and `insert` still edit the file.

## With Claude Desktop / Claude Code

Point the server and the adapter at the same absolute file. Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "env": { "MEMORY_FILE_PATH": "/Users/you/.claude/memory.jsonl" }
    }
  }
}
```

Claude Code: `claude mcp add memory -e MEMORY_FILE_PATH=/Users/you/.claude/memory.jsonl -- npx -y @modelcontextprotocol/server-memory`
(or the same `mcpServers` block in `.mcp.json` / `~/.claude.json`). Then run the governor with
`MEMORY_FILE_PATH=/Users/you/.claude/memory.jsonl` set (or pass the path). Because the marker lives inside
the observation, `read_graph` / `search_nodes` / `open_nodes` return it to the model as-is, so Claude sees
"user prefers Postgres [invalidate: superseded by “we migrated to SQLite” (slack, still true 5%), 2026-09-19]"
and can act on it; `mode="delete"` removes dead observations instead.

## Limits

- **No locking against the server.** The server loads, mutates and saves the whole file on every tool call; the
  adapter does the same. Both writes are atomic, but a governor write that lands between the server's load and
  save is overwritten (and vice versa). Run `observe()` when the server is idle, or between sessions.
- **Markers are text.** The model can quote, copy or ask the server to delete the marked string; the server's
  `delete_observations` needs the exact string including the marker. `search_nodes` matches marker words.
- **The server drops what it does not know.** The adapter preserves unknown lines and extra keys, but the next
  `saveGraph` on the server side discards them; do not rely on the file for anything but entities and relations.
- **Rewritten observations are new claims.** The server has no ids, so if the model rewrites an observation
  the content-addressed id changes: the old row keeps its verdict in the ledger and the new text starts active.
- **Relations flag on `relationType`.** After a flag the server sees `"owns [invalidate: ...]"` as the relation
  type; `create_relations` with the clean type would then add a second relation. Pass `relations=False` to
  leave relations out entirely.
- **One file per adapter.** The server's default path lives inside the npm package directory and moves with
  every `npx` cache refresh; always set `MEMORY_FILE_PATH`.
