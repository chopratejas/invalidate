"""Letta adapters: archival passages and core-memory blocks of one agent.

    from letta_client import Letta
    from invalidate.adapters import Governor
    from invalidate.adapters.letta import LettaAdapter, LettaBlockAdapter

    client = Letta(api_key=...)
    gov = Governor(LettaAdapter(client, agent_id), "ledger.db", mode="flag", successors=True)
    gov.sync(); gov.observe("we migrated to SQLite", source="slack")

Nothing from letta-client is imported; every call is keyword-only so it resolves on both the current SDK and
older ones that ordered the arguments differently. Signatures mirrored (letta-client 1.12.1):

  resources/agents/passages.py:102  client.agents.passages.list(agent_id, *, after=, ascending=, before=, limit=,
                                    search=) -> List[Passage]  (GET /v1/agents/{agent_id}/archival-memory)
  resources/agents/passages.py:50   client.agents.passages.create(agent_id, *, text, created_at=, tags=)
                                    -> List[Passage]  (one text may come back as several chunks)
  resources/agents/passages.py:166  client.agents.passages.delete(memory_id, *, agent_id) -> object
  types/passage.py:7                Passage(id, text, tags, metadata, created_at, ...)
                                    there is NO update/modify endpoint for an agent's passages
  resources/agents/blocks.py        client.agents.blocks.retrieve(block_label, *, agent_id) -> BlockResponse
                                    client.agents.blocks.update(block_label, *, agent_id, value=, ...) -> BlockResponse
                                    (PATCH /v1/agents/{agent_id}/core-memory/blocks/{block_label};
                                    older SDKs named it `modify`, which is tried as a fallback)
  types/block_response.py:10        BlockResponse(id, value, label, limit, ...)
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any

from ..types import Status
from .base import HostMemory, Reason

NOTE_PREFIX = "[invalidate]"
_NOTE_REF = re.compile(r"ref:(\S+)\s*$")
_ANNOTATED = re.compile(r"^~~(?P<line>.*)~~ \(invalidate: .*\)$", re.DOTALL)


def _first_id(created: Any) -> str | None:
    items = created if isinstance(created, list) else [created]
    for p in items:
        pid = getattr(p, "id", None) if not isinstance(p, dict) else p.get("id")
        if pid:
            return str(pid)
    return None


def _text(p: Any) -> str:
    return p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "") or ""


def _id(p: Any) -> str | None:
    pid = p.get("id") if isinstance(p, dict) else getattr(p, "id", None)
    return str(pid) if pid else None


def note_text(fact: str, reason: Reason, ref: str) -> str:
    verb = {"superseded": "superseded by", "contradicted": "contradicted by", "needs_review": "unclear after",
            "deleted": "forgotten after"}.get(reason.status.value, reason.status.value)
    return (f"{NOTE_PREFIX} “{fact[:300]}” is stale: {verb} “{reason.event_text[:300]}” "
            f"({reason.event_source}, still true {reason.still_true:.0%}) ref:{ref}")


class LettaAdapter:
    """Archival memory of one Letta agent, one passage = one memory.

    pull   -> every passage (paginated with `after`), minus invalidate's own notes.
    flag   -> passages cannot be edited or tagged after creation, so the flag is a NEW sidecar passage
              "[invalidate] “<fact>” is stale: superseded by “<event>” (<source>) ref:<passage id>";
              the original stays untouched. The note id is kept in `notes` so keep()/restore removes it,
              and `ref:` lets a fresh process rebuild that map from a pull.
    delete -> `passages.delete(memory_id=..., agent_id=...)` on the stale passage itself (mode="delete").
    insert -> `passages.create(agent_id=..., text=<event verbatim>)`; returns the first chunk's id.
    """

    name = "letta"

    def __init__(self, client: Any, agent_id: str, mode: str = "archival", *, page_size: int = 100) -> None:
        if mode != "archival":
            raise ValueError("LettaAdapter supports mode='archival'; use LettaBlockAdapter for core memory blocks")
        self.client = client
        self.agent_id = agent_id
        self.page_size = page_size
        self.notes: dict[str, str] = {}  # host passage id -> note passage id
        self._texts: dict[str, str] = {}

    @property
    def _passages(self) -> Any:
        return self.client.agents.passages

    def _list_all(self) -> list[Any]:
        out: list[Any] = []
        after: str | None = None
        while True:
            kw: dict[str, Any] = {"agent_id": self.agent_id, "limit": self.page_size}
            if after:
                kw["after"] = after
            page = self._passages.list(**kw)
            page = list(page or [])
            out.extend(page)
            if len(page) < self.page_size:
                break
            last = _id(page[-1])
            if not last or last == after:
                break
            after = last
        return out

    def pull(self) -> Iterable[HostMemory]:
        out: list[HostMemory] = []
        self._texts = {}
        for p in self._list_all():
            pid, text = _id(p), _text(p)
            if not pid:
                continue
            if text.startswith(NOTE_PREFIX):
                m = _NOTE_REF.search(text)
                if m:
                    self.notes[m.group(1)] = pid
                continue
            self._texts[pid] = text
            out.append(HostMemory(id=pid, text=text, source="letta"))
        return out

    def _drop_note(self, host_id: str) -> None:
        note_id = self.notes.pop(host_id, None)
        if note_id:
            self._passages.delete(memory_id=note_id, agent_id=self.agent_id)

    def _create(self, text: str, tags: list[str] | None = None) -> str | None:
        kw: dict[str, Any] = {"agent_id": self.agent_id, "text": text}
        if tags:
            kw["tags"] = tags
        try:
            created = self._passages.create(**kw)
        except TypeError:
            kw.pop("tags", None)  # SDKs before `tags` existed
            created = self._passages.create(**kw)
        return _first_id(created)

    def flag(self, host_id: str, reason: Reason) -> None:
        self._drop_note(host_id)  # one note per passage: replace, or clear on restore
        if reason.status is Status.ACTIVE:
            return
        fact = self._texts.get(host_id, host_id)
        note_id = self._create(note_text(fact, reason, host_id), tags=["invalidate", reason.status.value])
        if note_id:
            self.notes[host_id] = note_id

    def delete(self, host_id: str, reason: Reason) -> None:
        self._drop_note(host_id)
        self._passages.delete(memory_id=host_id, agent_id=self.agent_id)
        self._texts.pop(host_id, None)

    def insert(self, text: str, source: str, metadata: dict[str, Any]) -> str | None:
        pid = self._create(text, tags=["invalidate-successor", f"source:{source}"])
        if pid:
            self._texts[pid] = text
        return pid


# -- core memory blocks ---------------------------------------------------------
def _line_id(label: str, line: str) -> str:
    return f"{label}#{hashlib.sha1(line.encode('utf-8')).hexdigest()}"


def _block_value(client: Any, agent_id: str, label: str) -> str:
    block = client.agents.blocks.retrieve(block_label=label, agent_id=agent_id)
    value = block.get("value") if isinstance(block, dict) else getattr(block, "value", None)
    return value or ""


def _set_block_value(client: Any, agent_id: str, label: str, value: str) -> None:
    blocks = client.agents.blocks
    update = getattr(blocks, "update", None) or getattr(blocks, "modify")
    update(block_label=label, agent_id=agent_id, value=value)


def _split(value: str) -> list[str]:
    return [ln for ln in value.split("\n") if ln.strip()]


def _plain(line: str) -> str:
    """The original text of a line, whether or not a flag already annotated it."""
    m = _ANNOTATED.match(line)
    return m.group("line") if m else line


def block_lines(client: Any, agent_id: str, label: str = "human") -> list[HostMemory]:
    """Each non-empty line of a core memory block as a HostMemory with id `<label>#<sha1 of line>`.
    A line invalidate annotated earlier is reported under its ORIGINAL text and id so the ledger stays
    stable across syncs; `metadata["annotated"]` says so."""
    out: list[HostMemory] = []
    for raw in _split(_block_value(client, agent_id, label)):
        line = _plain(raw)
        out.append(HostMemory(id=_line_id(label, line), text=line, source=f"letta:{label}",
                              metadata={"annotated": raw != line} if raw != line else {}))
    return out


class LettaBlockAdapter:
    """One core memory block (e.g. "human" or "persona") as a list of line-memories.

    flag   -> the line is rewritten in place as `~~line~~ (invalidate: superseded by “…” (source, still true 5%))`;
              a restore (status active) puts the plain line back.
    delete -> the line is removed from the block.
    insert -> the event text is appended as a new line; its id is `<label>#<sha1>`.
    The whole block value is re-sent with `blocks.update(block_label, agent_id=, value=)` each time; the block's
    character `limit` still applies, so long annotations can be rejected by the server.
    """

    def __init__(self, client: Any, agent_id: str, label: str = "human") -> None:
        self.client = client
        self.agent_id = agent_id
        self.label = label
        self.name = f"letta_block_{label}"

    def pull(self) -> Iterable[HostMemory]:
        return block_lines(self.client, self.agent_id, self.label)

    def _rewrite(self, host_id: str, fn: Any) -> None:
        lines = _split(_block_value(self.client, self.agent_id, self.label))
        hit = False
        new: list[str] = []
        for raw in lines:
            plain = _plain(raw)
            if _line_id(self.label, plain) == host_id:
                hit = True
                replacement = fn(plain)
                if replacement is not None:
                    new.append(replacement)
            else:
                new.append(raw)
        if not hit:
            raise KeyError(f"no line {host_id!r} in block {self.label!r}")
        _set_block_value(self.client, self.agent_id, self.label, "\n".join(new))

    def flag(self, host_id: str, reason: Reason) -> None:
        if reason.status is Status.ACTIVE:
            self._rewrite(host_id, lambda plain: plain)
        else:
            self._rewrite(host_id, lambda plain: f"~~{plain}~~ ({reason.line()})")

    def delete(self, host_id: str, reason: Reason) -> None:
        self._rewrite(host_id, lambda plain: None)

    def insert(self, text: str, source: str, metadata: dict[str, Any]) -> str | None:
        line = " ".join(text.split("\n")).strip()
        if not line:
            return None
        value = _block_value(self.client, self.agent_id, self.label)
        new = f"{value.rstrip()}\n{line}" if value.strip() else line
        _set_block_value(self.client, self.agent_id, self.label, new)
        return _line_id(self.label, line)


__all__ = ["LettaAdapter", "LettaBlockAdapter", "block_lines", "note_text", "NOTE_PREFIX"]
