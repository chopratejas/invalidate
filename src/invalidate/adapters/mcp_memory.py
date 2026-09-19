"""The MCP knowledge-graph memory server (`@modelcontextprotocol/server-memory`) as a memory host.

The server persists its graph to one JSONL file (`MEMORY_FILE_PATH`, default `memory.jsonl` next to the
package's `dist/index.js`; a legacy `memory.json` is renamed on first start). Verified against
https://raw.githubusercontent.com/modelcontextprotocol/servers/main/src/memory/index.ts (server 0.6.3):

    {"type":"entity","name":"alice","entityType":"person","observations":["user prefers Postgres", ...]}
    {"type":"relation","from":"alice","to":"billing","relationType":"owns"}

`loadGraph` splits on "\\n", skips blank and malformed lines, keeps `type == "entity"` lines that pass
`{name: string, entityType: string, observations: string[]}` and `type == "relation"` lines that pass
`{from: string, to: string, relationType: string}`, and silently ignores every other `type`. `saveGraph`
rewrites the whole file (entities first, then relations, compact `JSON.stringify`, "\\n"-joined) through a
temp file + rename. Observations are deduplicated and deleted by exact string match.

Every observation string is one host memory; every relation is one host memory with the text
"<from> <relationType> <to>". Ids are content-addressed and derived from the ORIGINAL text:

    <entity name>#<sha1(normalized observation)[:10]>      observation
    relation#<sha1(normalized "<from> <type> <to>")[:10]>   relation

Receipts stay inside the string the server already stores, as a trailing bracketed marker the server
loads and shows verbatim:

    "user prefers Postgres [invalidate: superseded by “we migrated to SQLite” (slack, still true 5%), 2026-09-19]"

`pull` parses the marker back out, so the id never changes and `HostMemory.metadata["invalidate_status"]`
reports what the marker says. `flag(..., Status.ACTIVE)` strips the marker, `delete` removes the
observation (or relation), `insert` appends an observation to an entity. Lines the adapter does not
understand are preserved byte for byte (the server itself drops them on its next save).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..types import Status
from .base import Governor, HostMemory, Reason
from .markdown import content_hash, normalize

RELATION_PREFIX = "relation"
ENV_VAR = "MEMORY_FILE_PATH"

_MARK = re.compile(r"^(.*?)\s*\[invalidate: ([^\]]*)\]$", re.DOTALL)
_VERBS = {"superseded by": Status.SUPERSEDED.value, "contradicted by": Status.CONTRADICTED.value,
          "unclear after": Status.NEEDS_REVIEW.value, "restored after": Status.ACTIVE.value}


def split_marker(s: str) -> tuple[str, str | None]:
    """('original text', 'marker body' | None) for a stored string."""
    m = _MARK.match(s)
    if not m:
        return s, None
    return m.group(1), m.group(2)


def _safe(text: str) -> str:
    # "]" would end the marker early; keep it well-formed whatever the event says.
    return text.replace("]", "）").replace("\n", " ")


def _marker(reason: Reason) -> str:
    when = _dt.datetime.fromtimestamp(reason.at, _dt.timezone.utc).date() if reason.at > 0 else _dt.date.today()
    body = reason.line()[len("invalidate: "):]
    return f"[invalidate: {_safe(body)}, {when.isoformat()}]"


def _note_meta(note: str | None) -> dict[str, Any]:
    if note is None:
        return {}
    if note.startswith("from "):
        return {"invalidate_from": note}
    status = next((v for verb, v in _VERBS.items() if note.startswith(verb)), None) or note.split(" ", 1)[0]
    return {"invalidate_status": status, "invalidate_note": note}


def _dump(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _is_entity(obj: Any) -> bool:
    return (isinstance(obj, dict) and obj.get("type") == "entity" and isinstance(obj.get("name"), str)
            and isinstance(obj.get("entityType"), str) and isinstance(obj.get("observations"), list))


def _is_relation(obj: Any) -> bool:
    return (isinstance(obj, dict) and obj.get("type") == "relation" and isinstance(obj.get("from"), str)
            and isinstance(obj.get("to"), str) and isinstance(obj.get("relationType"), str))


def _split_line(raw: str) -> tuple[str, str]:
    for end in ("\r\n", "\n", "\r"):
        if raw.endswith(end):
            return raw[: -len(end)], end
    return raw, ""


@dataclass
class _Line:
    raw: str                      # exactly what was read, terminator included
    end: str                      # its terminator ("" on an unterminated last line)
    obj: dict[str, Any] | None    # parsed entity / relation, None for anything else
    kind: str | None              # "entity" | "relation" | None
    dirty: bool = False

    def render(self) -> str:
        return self.raw if not self.dirty else _dump(self.obj) + (self.end or "\n")


def relation_text(rel: dict[str, Any]) -> str:
    return f"{rel['from']} {split_marker(rel['relationType'])[0]} {rel['to']}"


class McpMemoryAdapter:
    """Govern the observations (and relations) in one MCP memory-server JSONL file.

        gov = Governor(McpMemoryAdapter("~/.claude/memory.jsonl"), "ledger.db", mode="flag", successors=True)
        gov.sync(); gov.observe("we migrated to SQLite", source="slack")

    `path=None` reads `MEMORY_FILE_PATH` from the environment (the server's own variable). Successors go
    to the entity named in `insert` metadata (`entity`), else to the entity of the observation they
    supersede when `follow_superseded` is on and exactly one entity is involved, else to `default_entity`
    (created with `default_entity_type` when missing). `relations=False` hides relations from `pull`.
    `annotate=False` turns `flag` into a no-op so only the ledger records verdicts.
    """

    name = "mcp_memory"

    def __init__(self, path: str | os.PathLike[str] | None = None, *, default_entity: str = "invalidate",
                 default_entity_type: str = "invalidate", follow_superseded: bool = True, relations: bool = True,
                 annotate: bool = True) -> None:
        if path is None:
            path = os.environ.get(ENV_VAR)
            if not path:
                raise ValueError(f"McpMemoryAdapter needs a path or the {ENV_VAR} environment variable")
        self.path = os.path.abspath(os.path.expanduser(os.fspath(path)))
        self.default_entity = default_entity
        self.default_entity_type = default_entity_type
        self.follow_superseded = follow_superseded
        self.relations = relations
        self.annotate = annotate

    # -- ids --------------------------------------------------------------------
    @staticmethod
    def id_for(entity: str, text: str) -> str:
        return f"{entity}#{content_hash(text)}"

    @staticmethod
    def relation_id(from_: str, relation_type: str, to: str) -> str:
        return f"{RELATION_PREFIX}#{content_hash(f'{from_} {relation_type} {to}')}"

    # -- file -------------------------------------------------------------------
    def _load(self) -> list[_Line]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8", newline="") as fh:
            raws = fh.read().splitlines(keepends=True)
        out: list[_Line] = []
        for raw in raws:
            body, end = _split_line(raw)
            obj: Any = None
            if body.strip():
                try:
                    obj = json.loads(body)
                except ValueError:
                    obj = None
            kind = "entity" if _is_entity(obj) else "relation" if _is_relation(obj) else None
            out.append(_Line(raw=raw, end=end, obj=obj if kind else None, kind=kind))
        return out

    def _write(self, lines: list[_Line]) -> None:
        d = os.path.dirname(self.path) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".invalidate-", suffix=".tmp", dir=d)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
                fh.write("".join(l.render() for l in lines))
            try:
                os.chmod(tmp, os.stat(self.path).st_mode & 0o7777)
            except OSError:
                pass
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def graph(self) -> dict[str, list[dict[str, Any]]]:
        """The graph as the server would load it: {"entities": [...], "relations": [...]} (copies)."""
        lines = self._load()
        return {"entities": [dict(l.obj) for l in lines if l.kind == "entity"],
                "relations": [dict(l.obj) for l in lines if l.kind == "relation"]}

    # -- Adapter protocol ------------------------------------------------------
    def pull(self) -> Iterable[HostMemory]:
        out: list[HostMemory] = []
        seen: set[str] = set()
        for l in self._load():
            if l.kind == "entity":
                name = l.obj["name"]
                for i, obs in enumerate(l.obj["observations"]):
                    if not isinstance(obs, str):
                        continue
                    text, note = split_marker(obs)
                    hid = self.id_for(name, text)
                    if hid in seen or not normalize(text):
                        continue
                    seen.add(hid)
                    meta = {"record": "observation", "entity": name, "entity_type": l.obj["entityType"], "index": i,
                            **_note_meta(note)}
                    out.append(HostMemory(id=hid, text=text, source=self.name, metadata=meta))
            elif l.kind == "relation" and self.relations:
                rtype, note = split_marker(l.obj["relationType"])
                hid = self.relation_id(l.obj["from"], rtype, l.obj["to"])
                if hid in seen:
                    continue
                seen.add(hid)
                meta = {"record": "relation", "from": l.obj["from"], "to": l.obj["to"], "relation_type": rtype,
                        **_note_meta(note)}
                out.append(HostMemory(id=hid, text=relation_text(l.obj), source=self.name, metadata=meta))
        return out

    def _edit(self, host_id: str, fn) -> int:
        """Apply `fn(original) -> new string | None (drop)` to every observation / relation with this id."""
        prefix, _, want = host_id.rpartition("#")
        lines = self._load()
        hits = 0
        keep: list[_Line] = []
        for l in lines:
            if l.kind == "entity" and l.obj["name"] == prefix:
                obs_out: list[Any] = []
                for obs in l.obj["observations"]:
                    text = split_marker(obs)[0] if isinstance(obs, str) else None
                    if text is None or content_hash(text) != want:
                        obs_out.append(obs)
                        continue
                    hits += 1
                    new = fn(text)
                    if new is not None:
                        obs_out.append(new)
                if obs_out != l.obj["observations"]:
                    l.obj = {**l.obj, "observations": obs_out}
                    l.dirty = True
            elif l.kind == "relation" and prefix == RELATION_PREFIX and self.relations:
                rtype = split_marker(l.obj["relationType"])[0]
                if content_hash(f"{l.obj['from']} {rtype} {l.obj['to']}") == want:
                    hits += 1
                    new = fn(rtype)
                    if new is None:
                        continue  # drop the line
                    if new != l.obj["relationType"]:
                        l.obj = {**l.obj, "relationType": new}
                        l.dirty = True
            keep.append(l)
        if not hits:
            raise KeyError(host_id)
        if len(keep) != len(lines) or any(l.dirty for l in keep):
            self._write(keep)
        return hits

    def flag(self, host_id: str, reason: Reason) -> None:
        if not self.annotate:
            return
        if reason.status is Status.ACTIVE:
            self._edit(host_id, lambda text: text)
            return
        note = _marker(reason)
        self._edit(host_id, lambda text: f"{text} {note}")

    def delete(self, host_id: str, reason: Reason) -> None:
        self._edit(host_id, lambda text: None)

    def _target_entity(self, metadata: dict[str, Any]) -> str:
        entity = metadata.get("entity")
        if isinstance(entity, str) and entity:
            return entity
        if self.follow_superseded:
            owners = {str(h).rpartition("#")[0] for h in metadata.get("invalidate_supersedes") or []}
            owners.discard(RELATION_PREFIX)
            if len(owners) == 1:
                return owners.pop()
        return self.default_entity

    def insert(self, text: str, source: str, metadata: dict[str, Any]) -> str:
        entity = self._target_entity(metadata)
        hid = self.id_for(entity, text)
        want = hid.rpartition("#")[2]
        lines = self._load()
        target: _Line | None = None
        for l in lines:
            if l.kind == "entity" and l.obj["name"] == entity:
                for obs in l.obj["observations"]:
                    if isinstance(obs, str) and content_hash(split_marker(obs)[0]) == want:
                        return hid  # already there: content-addressed hosts hand back the same id
                if target is None:
                    target = l
        stamp = _dt.date.today().isoformat()
        stored = f"{text} [invalidate: from {_safe(source)}, {stamp}]"
        if target is None:
            obj = {"type": "entity", "name": entity, "entityType": self.default_entity_type, "observations": [stored]}
            # Entities go before relations, as the server writes them; keep unknown lines where they are.
            at = next((i for i, l in enumerate(lines) if l.kind == "relation"), len(lines))
            if at == len(lines) and lines and not lines[-1].end:
                lines[-1].raw += "\n"
                lines[-1].end = "\n"
            lines.insert(at, _Line(raw="", end="\n", obj=obj, kind="entity", dirty=True))
        else:
            target.obj = {**target.obj, "observations": [*target.obj["observations"], stored]}
            target.dirty = True
        self._write(lines)
        return hid


def governed_read(path: str | os.PathLike[str] | None, gov: Governor, *, include_review: bool = False) -> dict[str, list[dict[str, Any]]]:
    """The graph at `path` (default: the governor's adapter file) with dead observations and relations removed:
    {"entities": [{"type","name","entityType","observations"}], "relations": [...]}, strings as stored.
    Memories under review are hidden unless `include_review=True`, mirroring `Governor.filter`."""
    adapter = gov.adapter if isinstance(gov.adapter, McpMemoryAdapter) and path is None else McpMemoryAdapter(path)
    g = adapter.graph()
    for ent in g["entities"]:
        obs = [o for o in ent["observations"] if isinstance(o, str)]
        ent["observations"] = gov.filter(obs, id_of=lambda o, n=ent["name"]: McpMemoryAdapter.id_for(n, split_marker(o)[0]),
                                         include_review=include_review)
    g["relations"] = gov.filter(
        g["relations"], id_of=lambda r: McpMemoryAdapter.relation_id(r["from"], split_marker(r["relationType"])[0], r["to"]),
        include_review=include_review,
    )
    return g


__all__ = ["McpMemoryAdapter", "governed_read", "split_marker", "relation_text", "RELATION_PREFIX", "ENV_VAR"]
