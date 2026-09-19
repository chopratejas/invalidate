"""Hosted-memory adapters (mem0, Letta, Graphiti) against faithful fakes.

Each fake mirrors the method signatures and return shapes of the real SDK as read from its source:
  mem0ai 2.1.0        mem0/memory/main.py (Memory) and mem0/client/main.py (MemoryClient)
  letta-client 1.12.1 letta_client/resources/agents/{passages,blocks}.py and types/passage.py
  graphiti-core 0.30.2 graphiti_core/edges.py (EntityEdge), nodes.py (EpisodicNode), graphiti.py (add_episode)
No network, no SDK imports.
"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pytest
from conftest import CONTRADICT, FakeJudge, SUPERSEDE, UNCERTAIN

from invalidate.adapters import Governor, Reason
from invalidate.adapters.graphiti import EPISODE_PREFIX, GraphitiAdapter, _run
from invalidate.adapters.letta import NOTE_PREFIX, LettaAdapter, LettaBlockAdapter, block_lines
from invalidate.adapters.mem0 import Mem0Adapter, governed_search, guard_add, user_texts
from invalidate.types import Status, now

# =============================================================================================
# mem0 fakes
# =============================================================================================
_ENTITY = frozenset({"user_id", "agent_id", "run_id"})
_UNSET = object()


class FakeMem0OSS:
    """mem0ai 2.1.0 `mem0.Memory` (mem0/memory/main.py). Signatures copied verbatim from the source."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._n = 0

    def _new_id(self) -> str:
        self._n += 1
        return f"m{self._n}"

    def seed(self, text: str, user_id: str = "u1", metadata: dict | None = None) -> str:
        mid = self._new_id()
        self.rows[mid] = {"id": mid, "memory": text, "user_id": user_id, "metadata": dict(metadata or {})}
        return mid

    # main.py:760
    def add(self, messages, *, user_id: Optional[str] = None, agent_id: Optional[str] = None,
            run_id: Optional[str] = None, metadata: Optional[dict] = None, timestamp: Any = None,
            expiration_date: Any = None, infer: bool = True, memory_type: Optional[str] = None,
            prompt: Optional[str] = None):
        self.calls.append(("add", {"messages": messages, "user_id": user_id, "metadata": metadata, "infer": infer}))
        if not any((user_id, agent_id, run_id)):
            raise ValueError("One of the filters: user_id, agent_id or run_id is required!")
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]
        results = []
        for msg in messages:
            if msg.get("role") == "system":
                continue
            content = msg["content"] if not infer else f"User {msg['content'].lower()}"  # "LLM extraction"
            mid = self._new_id()
            self.rows[mid] = {"id": mid, "memory": content, "user_id": user_id, "metadata": dict(metadata or {})}
            results.append({"id": mid, "memory": content, "event": "ADD", "actor_id": None, "role": msg["role"]})
        return {"results": results}  # main.py:877

    # main.py:1208
    def get(self, memory_id):
        row = self.rows.get(memory_id)
        return dict(row) if row else None

    # main.py:1255 - rejects top-level entity params (main.py:165)
    def get_all(self, *, filters: Optional[dict] = None, top_k: int = 20, show_expired: bool = False, **kwargs):
        self.calls.append(("get_all", {"filters": filters, "top_k": top_k, **kwargs}))
        if _ENTITY & set(kwargs):
            raise ValueError("Top-level entity parameters are not supported in get_all(). Use filters={'user_id': '...'} instead.")
        if not filters or not any(k in filters for k in _ENTITY):
            raise ValueError("filters must contain at least one of: user_id, agent_id, run_id.")
        uid = filters.get("user_id")
        rows = [dict(r) for r in self.rows.values() if uid is None or r["user_id"] == uid]
        return {"results": rows[:top_k]}

    # main.py:1379
    def search(self, query: str, *, top_k: int = 20, filters: Optional[dict] = None, threshold: float = 0.1,
               rerank: bool = False, explain: bool = False, reference_date: Any = None, show_expired: bool = False,
               **kwargs):
        if _ENTITY & set(kwargs):
            raise ValueError("Top-level entity parameters are not supported in search().")
        self.calls.append(("search", {"query": query, "filters": filters}))
        hits = [{**r, "score": 0.9} for r in self.rows.values() if query.lower() in r["memory"].lower()]
        return {"results": hits[:top_k]}

    # main.py:1815 - metadata-only update merges (main.py:2059) and keeps the text
    def update(self, memory_id, text: Optional[str] = None, metadata: Optional[dict] = None,
               expiration_date: Any = _UNSET, data: Optional[str] = None):
        self.calls.append(("update", {"memory_id": memory_id, "text": text, "metadata": metadata}))
        if data is not None and text is None:
            text = data
        if text is None and metadata is None and expiration_date is _UNSET:
            raise ValueError("At least one of text, metadata, or expiration_date must be provided.")
        row = self.rows.get(memory_id)
        if row is None:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        if metadata is not None:
            row["metadata"].update({k: v for k, v in metadata.items() if k not in _ENTITY})
        if text is not None:
            row["memory"] = text
        return {"message": "Memory updated successfully!"}

    # main.py:1869
    def delete(self, memory_id):
        self.calls.append(("delete", {"memory_id": memory_id}))
        if memory_id not in self.rows:
            raise ValueError(f"Memory with id {memory_id} not found")
        del self.rows[memory_id]
        return {"message": "Memory deleted successfully!"}


class FakeMem0Old(FakeMem0OSS):
    """mem0ai 0.1.x-style `Memory`: top-level user_id=, list return, text-only `update(memory_id, data)`."""

    def get_all(self, user_id=None, agent_id=None, run_id=None, limit=100):  # type: ignore[override]
        self.calls.append(("get_all", {"user_id": user_id, "limit": limit}))
        return [dict(r) for r in self.rows.values() if user_id is None or r["user_id"] == user_id][:limit]

    def update(self, memory_id, data):  # type: ignore[override]
        self.calls.append(("update", {"memory_id": memory_id, "data": data}))
        self.rows[memory_id]["memory"] = data
        return {"message": "Memory updated successfully!"}

    def add(self, messages, user_id=None, agent_id=None, run_id=None, metadata=None, infer=True, prompt=None):  # type: ignore[override]
        return super().add(messages, user_id=user_id, agent_id=agent_id, run_id=run_id, metadata=metadata, infer=infer)


class FakeMem0Platform:
    """mem0ai 2.1.0 `mem0.MemoryClient` (mem0/client/main.py). Paginated get_all, filters-only identity."""

    def __init__(self, page_size: int = 2) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.page_size = page_size
        self._n = 0
        self.users = object()  # capability marker the real client exposes (client.users())

    def seed(self, text: str, user_id: str = "u1", metadata: dict | None = None) -> str:
        self._n += 1
        mid = f"p{self._n}"
        self.rows[mid] = {"id": mid, "memory": text, "user_id": user_id, "metadata": dict(metadata or {})}
        return mid

    # client/main.py:265 - identity inside filters (client/types.py:18)
    def add(self, messages, options=None, **kwargs):
        self.calls.append(("add", {"messages": messages, **kwargs}))
        filters = kwargs.get("filters") or {}
        if _ENTITY & set(kwargs) or not filters.get("user_id"):
            raise ValueError("identity fields must be inside filters")
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        results = []
        for msg in messages:
            self._n += 1
            mid = f"p{self._n}"
            content = msg["content"] if kwargs.get("infer") is False else f"User {msg['content'].lower()}"
            self.rows[mid] = {"id": mid, "memory": content, "user_id": filters["user_id"], "metadata": dict(kwargs.get("metadata") or {})}
            results.append({"id": mid, "memory": content, "event": "ADD"})
        return {"results": results}

    def get(self, memory_id: str):
        return dict(self.rows[memory_id])

    # client/main.py:330 - paginated {"count", "next", "previous", "results"}
    def get_all(self, options=None, **kwargs):
        self.calls.append(("get_all", dict(kwargs)))
        if _ENTITY & set(kwargs):
            raise ValueError("Top-level entity parameters are not supported in get_all(). Use filters={'user_id': '...'} instead.")
        uid = (kwargs.get("filters") or {}).get("user_id")
        rows = [dict(r) for r in self.rows.values() if r["user_id"] == uid]
        page = int(kwargs.get("page") or 1)
        size = int(kwargs.get("page_size") or self.page_size)
        chunk = rows[(page - 1) * size: page * size]
        has_next = page * size < len(rows)
        return {"count": len(rows), "next": f"?page={page + 1}" if has_next else None,
                "previous": f"?page={page - 1}" if page > 1 else None, "results": chunk}

    # client/main.py:378
    def search(self, query: str, options=None, **kwargs):
        self.calls.append(("search", {"query": query, **kwargs}))
        return {"results": [{**r, "score": 0.5} for r in self.rows.values() if query.lower() in r["memory"].lower()]}

    # client/main.py:425
    def update(self, memory_id: str, options=None, **kwargs):
        payload = {k: v for k, v in kwargs.items() if v is not None or k == "expiration_date"}
        if not payload:
            raise ValueError("At least one of text, metadata, timestamp, or expiration_date must be provided for update.")
        self.calls.append(("update", {"memory_id": memory_id, **payload}))
        row = self.rows[memory_id]
        if "metadata" in payload:
            row["metadata"] = dict(payload["metadata"])  # the platform REPLACES metadata; the adapter merges first
        if "text" in payload:
            row["memory"] = payload["text"]
        return {"message": "Memory updated successfully!"}

    # client/main.py:464
    def delete(self, memory_id: str, delete_linked: bool = False):
        self.calls.append(("delete", {"memory_id": memory_id}))
        del self.rows[memory_id]
        return {"message": "Memory deleted successfully!"}


def _gov(adapter, fake: FakeJudge, **kw) -> Governor:
    return Governor(adapter, ":memory:", judge=fake, **kw)


def _reason(status: Status, event: str = "we moved to sqlite") -> Reason:
    return Reason(status, status.value, event, "slack", "evt_x", 0.05, now())


# =============================================================================================
# mem0 tests
# =============================================================================================
class TestMem0:
    def test_requires_a_scope(self):
        with pytest.raises(ValueError):
            Mem0Adapter(FakeMem0OSS())

    def test_sync_uses_filters_and_large_top_k_on_oss_2x(self, fake):
        m = FakeMem0OSS()
        m.seed("user prefers postgres")
        m.seed("user lives in berlin")
        m.seed("other user's memory", user_id="u2")
        gov = _gov(Mem0Adapter(m, user_id="u1"), fake)
        rep = gov.sync()
        assert rep.added == 2 and rep.total == 2
        name, kw = m.calls[-1]
        assert name == "get_all" and kw["filters"] == {"user_id": "u1"} and kw["top_k"] >= 1000
        assert gov.status_of("m1") is Status.ACTIVE

    def test_sync_falls_back_to_top_level_ids_and_list_shape_on_old_oss(self, fake):
        m = FakeMem0Old()
        m.seed("user prefers postgres")
        gov = _gov(Mem0Adapter(m, user_id="u1"), fake)
        assert gov.sync().added == 1
        assert m.calls[-1] == ("get_all", {"user_id": "u1", "limit": 5000})

    def test_sync_follows_platform_pagination(self, fake):
        m = FakeMem0Platform(page_size=2)
        for i in range(5):
            m.seed(f"fact {i}")
        gov = _gov(Mem0Adapter(m, user_id="u1"), fake)
        assert gov.sync().added == 5
        pages = [kw.get("page", 1) for n, kw in m.calls if n == "get_all"]
        assert pages == [1, 2, 3]
        assert all("user_id" not in kw for n, kw in m.calls if n == "get_all")

    def test_flag_merges_receipt_into_metadata_and_keeps_text(self, fake):
        m = FakeMem0OSS()
        mid = m.seed("user prefers postgres", metadata={"topic": "db"})
        fake.script("postgres", SUPERSEDE)
        gov = _gov(Mem0Adapter(m, user_id="u1"), fake)
        gov.sync()
        rep = gov.observe("we migrated to sqlite", source="slack")
        assert [p.action for p in rep.pushes] == ["flag"] and not rep.errors
        row = m.rows[mid]
        assert row["memory"] == "user prefers postgres"
        assert row["metadata"]["topic"] == "db"
        assert row["metadata"]["invalidate_status"] == "superseded"
        assert row["metadata"]["invalidate_event"] == "we migrated to sqlite"
        name, kw = [c for c in m.calls if c[0] == "update"][-1]
        assert kw["text"] is None  # metadata-only update: never rewrite the text

    def test_flag_on_platform_merges_before_replacing(self, fake):
        m = FakeMem0Platform()
        mid = m.seed("user prefers postgres", metadata={"topic": "db"})
        fake.script("postgres", CONTRADICT)
        gov = _gov(Mem0Adapter(m, user_id="u1"), fake)
        gov.sync()
        gov.observe("we dropped postgres", source="slack")
        assert m.rows[mid]["metadata"]["topic"] == "db"
        assert m.rows[mid]["metadata"]["invalidate_status"] == "contradicted"

    def test_old_oss_flag_is_ledger_only_and_never_rewrites_text(self, fake):
        m = FakeMem0Old()
        mid = m.seed("user prefers postgres")
        fake.script("postgres", SUPERSEDE)
        adapter = Mem0Adapter(m, user_id="u1")
        assert adapter.flagged_in_host is False
        gov = _gov(adapter, fake)
        gov.sync()
        rep = gov.observe("we migrated to sqlite", source="slack")
        assert not rep.errors and rep.pushes[0].action == "flag"
        assert m.rows[mid]["memory"] == "user prefers postgres"
        assert not any(c[0] == "update" for c in m.calls)
        assert gov.status_of(mid) is Status.SUPERSEDED  # the truth lives in the ledger

    def test_delete_mode_deletes_the_memory(self, fake):
        m = FakeMem0OSS()
        mid = m.seed("user prefers postgres")
        fake.script("postgres", CONTRADICT)
        gov = _gov(Mem0Adapter(m, user_id="u1"), fake, mode="delete")
        gov.sync()
        rep = gov.observe("we dropped postgres", source="slack")
        assert rep.pushes[0].action == "delete" and mid not in m.rows

    def test_successor_is_inserted_verbatim_with_infer_false(self, fake):
        m = FakeMem0OSS()
        mid = m.seed("user prefers postgres")
        fake.script("postgres", SUPERSEDE)
        gov = _gov(Mem0Adapter(m, user_id="u1"), fake, successors=True)
        gov.sync()
        rep = gov.observe("We migrated to SQLite", source="slack")
        assert rep.successor_host_id and rep.successor_host_id in m.rows
        new = m.rows[rep.successor_host_id]
        assert new["memory"] == "We migrated to SQLite"  # not the "extracted" rewrite
        assert new["metadata"]["invalidate_supersedes"] == mid  # lists flattened to scalars
        assert new["metadata"]["invalidate_source"] == "slack"
        name, kw = [c for c in m.calls if c[0] == "add"][-1]
        assert kw["infer"] is False and kw["user_id"] == "u1"
        assert gov.mem.get(gov.our_id(mid)).superseded_by == gov.our_id(rep.successor_host_id)
        assert gov.sync().unchanged == 2  # the successor is stable across syncs

    def test_successor_on_platform_uses_filters(self, fake):
        m = FakeMem0Platform()
        m.seed("user prefers postgres")
        fake.script("postgres", SUPERSEDE)
        gov = _gov(Mem0Adapter(m, user_id="u1"), fake, successors=True)
        gov.sync()
        rep = gov.observe("we migrated to sqlite", source="slack")
        name, kw = [c for c in m.calls if c[0] == "add"][-1]
        assert kw["filters"] == {"user_id": "u1"} and kw["infer"] is False
        assert m.rows[rep.successor_host_id]["memory"] == "we migrated to sqlite"

    def test_governed_search_drops_dead_and_keeps_shape(self, fake):
        m = FakeMem0OSS()
        dead = m.seed("user prefers postgres")
        live = m.seed("postgres tips from the wiki")
        fake.script("prefers postgres", SUPERSEDE)
        gov = _gov(Mem0Adapter(m, user_id="u1"), fake)
        gov.sync()
        gov.observe("we migrated to sqlite", source="slack")
        res = governed_search(m, gov, "postgres")
        assert isinstance(res, dict) and [r["id"] for r in res["results"]] == [live]
        assert m.calls[-1] == ("search", {"query": "postgres", "filters": {"user_id": "u1"}})
        res2 = governed_search(m, gov, "postgres", filters={"user_id": "u1"}, top_k=5)
        assert [r["id"] for r in res2["results"]] == [live]
        assert dead not in {r["id"] for r in res2["results"]}

    def test_governed_search_handles_list_shape(self, fake):
        class ListSearch(FakeMem0OSS):
            def search(self, query, **kw):  # type: ignore[override]
                return super().search(query, **kw)["results"]

        m = ListSearch()
        m.seed("user prefers postgres")
        fake.script("postgres", CONTRADICT)
        gov = _gov(Mem0Adapter(m, user_id="u1"), fake)
        gov.sync()
        gov.observe("we dropped postgres", source="slack")
        assert governed_search(m, gov, "postgres", filters={"user_id": "u1"}) == []

    def test_guard_add_judges_user_messages_then_adds_and_syncs(self, fake):
        m = FakeMem0OSS()
        old = m.seed("user prefers postgres")
        fake.script("prefers postgres", SUPERSEDE)
        gov = _gov(Mem0Adapter(m, user_id="u1"), fake)
        gov.sync()
        add = guard_add(m, gov)
        out = add([{"role": "assistant", "content": "noted"}, {"role": "user", "content": "we migrated to sqlite"}],
                  user_id="u1", infer=False)
        assert [r["memory"] for r in out["results"]] == ["noted", "we migrated to sqlite"]
        assert gov.status_of(old) is Status.SUPERSEDED
        assert m.rows[old]["metadata"]["invalidate_status"] == "superseded"
        assert gov.status_of(out["results"][1]["id"]) is Status.ACTIVE  # synced after the add
        assert {e.text for e, _ in fake.observe_calls} == {"we migrated to sqlite"}

    def test_user_texts_shapes(self):
        assert user_texts("hi") == ["hi"]
        assert user_texts({"role": "user", "content": "a"}) == ["a"]
        assert user_texts([{"role": "system", "content": "x"}, {"role": "user", "content": "b"}]) == ["b"]
        assert user_texts(messages=[{"role": "user", "content": "c"}]) == ["c"]

    def test_keep_and_forget_mirror_to_host(self, fake):
        m = FakeMem0OSS()
        mid = m.seed("user prefers postgres")
        fake.script("postgres", CONTRADICT)
        gov = _gov(Mem0Adapter(m, user_id="u1"), fake)
        gov.sync()
        gov.observe("we dropped postgres", source="slack")
        gov.keep(mid)
        assert m.rows[mid]["metadata"]["invalidate_status"] == "active"
        gov.forget(mid)
        assert mid not in m.rows


# =============================================================================================
# Letta fakes
# =============================================================================================
_OMIT = object()


@dataclass
class FakePassage:
    """letta_client/types/passage.py:7 (the fields the adapter reads)."""

    text: str
    id: Optional[str] = None
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    created_at: Optional[datetime] = None


@dataclass
class FakeBlock:
    """letta_client/types/block_response.py:10."""

    id: str
    value: str
    label: Optional[str] = None
    limit: Optional[int] = 5000


class FakePassages:
    """letta_client/resources/agents/passages.py: create (:50), list (:102), delete (:166)."""

    def __init__(self) -> None:
        self.store: dict[str, list[FakePassage]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._n = 0

    def create(self, agent_id: str, *, text: str, created_at: Any = _OMIT, tags: Any = _OMIT) -> list[FakePassage]:
        self.calls.append(("create", {"agent_id": agent_id, "text": text, "tags": None if tags is _OMIT else tags}))
        self._n += 1
        p = FakePassage(text=text, id=f"passage-{self._n:04d}", tags=None if tags is _OMIT else list(tags),
                        created_at=datetime.now(timezone.utc))
        self.store.setdefault(agent_id, []).append(p)
        return [p]

    def list(self, agent_id: str, *, after: Any = _OMIT, ascending: Any = _OMIT, before: Any = _OMIT,
             limit: Any = _OMIT, search: Any = _OMIT) -> list[FakePassage]:
        self.calls.append(("list", {"agent_id": agent_id, "after": None if after is _OMIT else after,
                                    "limit": None if limit is _OMIT else limit}))
        rows = list(self.store.get(agent_id, []))
        if after is not _OMIT and after is not None:
            ids = [p.id for p in rows]
            rows = rows[ids.index(after) + 1:] if after in ids else []
        if limit is not _OMIT and limit is not None:
            rows = rows[:limit]
        return rows

    def delete(self, memory_id: str, *, agent_id: str) -> object:
        self.calls.append(("delete", {"memory_id": memory_id, "agent_id": agent_id}))
        rows = self.store.get(agent_id, [])
        idx = [p.id for p in rows].index(memory_id)  # ValueError like a 404 would raise
        del rows[idx]
        return {}


class FakeBlocks:
    """letta_client/resources/agents/blocks.py: retrieve(block_label, *, agent_id), update(block_label, *, agent_id, value=...)."""

    def __init__(self, values: dict[str, str]) -> None:
        self.blocks = {label: FakeBlock(id=f"block-{i}", value=v, label=label) for i, (label, v) in enumerate(values.items())}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def retrieve(self, block_label: str, *, agent_id: str) -> FakeBlock:
        self.calls.append(("retrieve", {"block_label": block_label, "agent_id": agent_id}))
        return self.blocks[block_label]

    def update(self, block_label: str, *, agent_id: str, value: Any = _OMIT, label: Any = _OMIT, **kw: Any) -> FakeBlock:
        self.calls.append(("update", {"block_label": block_label, "agent_id": agent_id, "value": value}))
        b = self.blocks[block_label]
        if value is not _OMIT:
            if b.limit and len(value) > b.limit:
                raise ValueError("block value exceeds limit")
            b.value = value
        return b


class FakeOldBlocks(FakeBlocks):
    """letta-client 0.x named the PATCH `modify`."""

    def modify(self, block_label: str, *, agent_id: str, value: Any = _OMIT, **kw: Any) -> FakeBlock:
        return super().update(block_label, agent_id=agent_id, value=value)

    update = None  # type: ignore[assignment]


@dataclass
class FakeAgents:
    passages: FakePassages
    blocks: FakeBlocks


@dataclass
class FakeLetta:
    agents: FakeAgents


def _letta(texts: list[str], blocks: dict[str, str] | None = None, old_blocks: bool = False) -> FakeLetta:
    passages = FakePassages()
    for t in texts:
        passages.create("agent-1", text=t)
    blk = (FakeOldBlocks if old_blocks else FakeBlocks)(blocks or {"human": ""})
    return FakeLetta(agents=FakeAgents(passages=passages, blocks=blk))


# =============================================================================================
# Letta tests
# =============================================================================================
class TestLettaArchival:
    def test_only_archival_mode(self):
        with pytest.raises(ValueError):
            LettaAdapter(_letta([]), "agent-1", mode="core")

    def test_sync_paginates_with_after(self, fake):
        c = _letta([f"fact {i}" for i in range(5)])
        gov = _gov(LettaAdapter(c, "agent-1", page_size=2), fake)
        assert gov.sync().added == 5
        lists = [kw for n, kw in c.agents.passages.calls if n == "list"]
        assert [kw["after"] for kw in lists] == [None, "passage-0002", "passage-0004"]
        assert all(kw["limit"] == 2 for kw in lists)

    def test_flag_adds_a_note_and_leaves_the_passage_alone(self, fake):
        c = _letta(["user prefers postgres", "user lives in berlin"])
        fake.script("postgres", SUPERSEDE)
        adapter = LettaAdapter(c, "agent-1")
        gov = _gov(adapter, fake)
        gov.sync()
        rep = gov.observe("we migrated to sqlite", source="slack")
        assert not rep.errors and rep.pushes[0].action == "flag"
        rows = c.agents.passages.store["agent-1"]
        assert [p.text for p in rows][:2] == ["user prefers postgres", "user lives in berlin"]
        note = rows[2]
        assert note.text.startswith(f"{NOTE_PREFIX} “user prefers postgres” is stale: superseded by “we migrated to sqlite” (slack")
        assert note.text.endswith("ref:passage-0001") and note.tags == ["invalidate", "superseded"]
        assert adapter.notes == {"passage-0001": note.id}
        # notes are never pulled as memories, and a second flag replaces the note
        assert gov.sync().total == 2
        gov.observe("actually we use mysql", source="slack")
        assert [p.text for p in rows if p.text.startswith(NOTE_PREFIX)][0].endswith("ref:passage-0001")
        assert sum(p.text.startswith(NOTE_PREFIX) for p in rows) == 1

    def test_notes_are_rebuilt_from_a_fresh_pull_and_removed_on_keep(self, fake):
        c = _letta(["user prefers postgres"])
        fake.script("postgres", CONTRADICT)
        gov = _gov(LettaAdapter(c, "agent-1"), fake)
        gov.sync()
        gov.observe("we dropped postgres", source="slack")
        fresh = LettaAdapter(c, "agent-1")
        list(fresh.pull())
        assert fresh.notes == {"passage-0001": "passage-0002"}
        gov2 = _gov(fresh, fake)
        gov2.sync()
        gov2.keep("passage-0001")
        assert [p.id for p in c.agents.passages.store["agent-1"]] == ["passage-0001"]

    def test_delete_mode_removes_the_passage_and_its_note(self, fake):
        c = _letta(["user prefers postgres"])
        fake.script("postgres", UNCERTAIN)
        gov = _gov(LettaAdapter(c, "agent-1"), fake, mode="delete")
        gov.sync()
        gov.observe("maybe postgres is gone", source="slack")  # needs_review -> note
        assert len(c.agents.passages.store["agent-1"]) == 2
        fake.script("postgres", CONTRADICT)
        rep = gov.observe("we dropped postgres", source="slack")
        assert rep.pushes[0].action == "delete"
        assert c.agents.passages.store["agent-1"] == []

    def test_successor_is_a_verbatim_passage(self, fake):
        c = _letta(["user prefers postgres"])
        fake.script("prefers postgres", SUPERSEDE)
        gov = _gov(LettaAdapter(c, "agent-1"), fake, successors=True)
        gov.sync()
        rep = gov.observe("We migrated to SQLite.", source="slack")
        succ = [p for p in c.agents.passages.store["agent-1"] if p.id == rep.successor_host_id][0]
        assert succ.text == "We migrated to SQLite." and "invalidate-successor" in succ.tags
        assert gov.sync().unchanged == 2

    def test_create_without_tags_on_old_sdk(self):
        class OldPassages(FakePassages):
            def create(self, agent_id: str, *, text: str) -> list[FakePassage]:  # type: ignore[override]
                return super().create(agent_id, text=text)

        c = FakeLetta(agents=FakeAgents(passages=OldPassages(), blocks=FakeBlocks({})))
        assert LettaAdapter(c, "agent-1").insert("x", "slack", {}) == "passage-0001"


class TestLettaBlocks:
    HUMAN = "Name: Alice\nPrefers postgres\nLives in Berlin"

    def test_block_lines_ids_and_stability(self):
        c = _letta([], {"human": self.HUMAN})
        lines = block_lines(c, "agent-1", "human")
        assert [l.text for l in lines] == ["Name: Alice", "Prefers postgres", "Lives in Berlin"]
        assert lines[1].id == "human#" + hashlib.sha1(b"Prefers postgres").hexdigest()
        assert c.agents.blocks.calls[-1] == ("retrieve", {"block_label": "human", "agent_id": "agent-1"})

    def test_flag_annotates_the_line_in_place_and_survives_resync(self, fake):
        c = _letta([], {"human": self.HUMAN})
        fake.script("postgres", SUPERSEDE)
        adapter = LettaBlockAdapter(c, "agent-1", "human")
        gov = _gov(adapter, fake)
        gov.sync()
        rep = gov.observe("we migrated to sqlite", source="slack")
        assert not rep.errors
        value = c.agents.blocks.blocks["human"].value
        assert value.split("\n")[1].startswith("~~Prefers postgres~~ (invalidate: superseded by “we migrated to sqlite” (slack")
        assert value.split("\n")[0] == "Name: Alice"
        n, kw = c.agents.blocks.calls[-1]
        assert n == "update" and kw["block_label"] == "human" and kw["agent_id"] == "agent-1"
        # the annotated line is still the same memory to the ledger
        pulled = {l.id: l for l in adapter.pull()}
        lid = "human#" + hashlib.sha1(b"Prefers postgres").hexdigest()
        assert pulled[lid].text == "Prefers postgres" and pulled[lid].metadata == {"annotated": True}
        s = gov.sync()
        assert s.unchanged == 3 and gov.status_of(lid) is Status.SUPERSEDED

    def test_keep_restores_the_plain_line(self, fake):
        c = _letta([], {"human": self.HUMAN})
        fake.script("postgres", CONTRADICT)
        gov = _gov(LettaBlockAdapter(c, "agent-1", "human"), fake)
        gov.sync()
        gov.observe("we dropped postgres", source="slack")
        gov.keep("human#" + hashlib.sha1(b"Prefers postgres").hexdigest())
        assert c.agents.blocks.blocks["human"].value == self.HUMAN

    def test_delete_mode_removes_the_line(self, fake):
        c = _letta([], {"human": self.HUMAN})
        fake.script("postgres", CONTRADICT)
        gov = _gov(LettaBlockAdapter(c, "agent-1", "human"), fake, mode="delete")
        gov.sync()
        gov.observe("we dropped postgres", source="slack")
        assert c.agents.blocks.blocks["human"].value == "Name: Alice\nLives in Berlin"

    def test_insert_appends_a_line_and_links_the_successor(self, fake):
        c = _letta([], {"human": self.HUMAN})
        fake.script("Prefers postgres", SUPERSEDE)
        gov = _gov(LettaBlockAdapter(c, "agent-1", "human"), fake, successors=True)
        gov.sync()
        rep = gov.observe("Prefers sqlite now", source="slack")
        assert c.agents.blocks.blocks["human"].value.endswith("\nPrefers sqlite now")
        assert rep.successor_host_id == "human#" + hashlib.sha1(b"Prefers sqlite now").hexdigest()
        assert gov.sync().unchanged == 4

    def test_missing_line_is_a_push_error_not_a_crash(self, fake):
        c = _letta([], {"human": self.HUMAN})
        adapter = LettaBlockAdapter(c, "agent-1", "human")
        with pytest.raises(KeyError):
            adapter.flag("human#nope", _reason(Status.CONTRADICTED))

    def test_modify_fallback_on_old_sdk(self, fake):
        c = _letta([], {"human": self.HUMAN}, old_blocks=True)
        adapter = LettaBlockAdapter(c, "agent-1", "human")
        adapter.delete("human#" + hashlib.sha1(b"Lives in Berlin").hexdigest(), _reason(Status.CONTRADICTED))
        assert c.agents.blocks.blocks["human"].value == "Name: Alice\nPrefers postgres"


# =============================================================================================
# Graphiti fakes
# =============================================================================================
class FakeDriver:
    def __init__(self) -> None:
        self.edges: dict[str, "FakeEntityEdge"] = {}
        self.episodes: dict[str, "FakeEpisodicNode"] = {}
        self.saves: list[str] = []


@dataclass
class FakeEntityEdge:
    """graphiti_core/edges.py:263 EntityEdge — fields, save (:335), get_by_uuid (:378), get_by_group_ids (:481)."""

    uuid: str
    group_id: str
    fact: str
    name: str = "RELATES_TO"
    source_node_uuid: str = "n1"
    target_node_uuid: str = "n2"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    episodes: list[str] = field(default_factory=list)
    expired_at: Optional[datetime] = None
    valid_at: Optional[datetime] = None
    invalid_at: Optional[datetime] = None
    reference_time: Optional[datetime] = None
    attributes: dict[str, Any] = field(default_factory=dict)

    async def save(self, driver: FakeDriver):
        driver.saves.append(self.uuid)
        driver.edges[self.uuid] = self
        return None

    @classmethod
    async def get_by_uuid(cls, driver: FakeDriver, uuid: str):
        return driver.edges[uuid]

    @classmethod
    async def get_by_group_ids(cls, driver: FakeDriver, group_ids: list[str], limit: Optional[int] = None,
                               uuid_cursor: Optional[str] = None, with_embeddings: bool = False):
        return [e for e in driver.edges.values() if e.group_id in group_ids]


@dataclass
class FakeEpisodicNode:
    """graphiti_core/nodes.py EpisodicNode — uuid, name, content, source_description, group_id."""

    uuid: str
    name: str
    content: str
    source_description: str
    group_id: str

    @classmethod
    async def get_by_group_ids(cls, driver: FakeDriver, group_ids: list[str], limit: Optional[int] = None,
                               uuid_cursor: Optional[str] = None):
        return [e for e in driver.episodes.values() if e.group_id in group_ids]


@dataclass
class FakeAddEpisodeResults:
    """graphiti_core/graphiti.py:114 AddEpisodeResults."""

    episode: FakeEpisodicNode
    episodic_edges: list = field(default_factory=list)
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)


class FakeGraphiti:
    """graphiti_core/graphiti.py Graphiti: `.driver` (:208) and add_episode(name, episode_body,
    source_description, reference_time, source=..., group_id=None, ...) -> AddEpisodeResults."""

    def __init__(self) -> None:
        self.driver = FakeDriver()
        self.calls: list[dict[str, Any]] = []
        self._n = 0

    def edge(self, fact: str, group_id: str = "g1", **kw: Any) -> FakeEntityEdge:
        self._n += 1
        e = FakeEntityEdge(uuid=f"e{self._n}", group_id=group_id, fact=fact, **kw)
        self.driver.edges[e.uuid] = e
        return e

    async def add_episode(self, name: str, episode_body: str, source_description: str, reference_time: datetime,
                          source: Any = "message", group_id: Optional[str] = None, uuid: Optional[str] = None,
                          update_communities: bool = False, **kw: Any) -> FakeAddEpisodeResults:
        self.calls.append({"name": name, "episode_body": episode_body, "source_description": source_description,
                           "reference_time": reference_time, "group_id": group_id})
        await asyncio.sleep(0)
        self._n += 1
        ep = FakeEpisodicNode(uuid=f"ep{self._n}", name=name, content=episode_body,
                              source_description=source_description, group_id=group_id or "")
        self.driver.episodes[ep.uuid] = ep
        # Graphiti's extraction: NOT verbatim
        extracted = self.edge(f"extracted: {episode_body.lower()}", group_id=group_id or "", episodes=[ep.uuid])
        return FakeAddEpisodeResults(episode=ep, edges=[extracted])


def _graphiti_adapter(g: FakeGraphiti, **kw: Any) -> GraphitiAdapter:
    return GraphitiAdapter(g, "g1", edge_class=FakeEntityEdge, episode_class=FakeEpisodicNode, **kw)


# =============================================================================================
# Graphiti tests
# =============================================================================================
class TestGraphiti:
    def test_needs_a_driver(self):
        class NoDriver:
            pass

        with pytest.raises(ValueError):
            GraphitiAdapter(NoDriver(), "g1", edge_class=FakeEntityEdge)

    def test_pull_skips_expired_edges_and_other_groups(self, fake):
        g = FakeGraphiti()
        g.edge("Alice prefers Postgres")
        g.edge("Alice lives in Berlin")
        g.edge("old fact", expired_at=datetime.now(timezone.utc))
        g.edge("Bob's fact", group_id="g2")
        gov = _gov(_graphiti_adapter(g), fake)
        rep = gov.sync()
        assert rep.added == 2
        assert {gov.host_id(m) for m in gov.mem.list()} == {"e1", "e2"}

    def test_flag_superseded_sets_invalid_and_expired_and_receipt(self, fake):
        g = FakeGraphiti()
        e = g.edge("Alice prefers Postgres")
        fake.script("Postgres", SUPERSEDE)
        gov = _gov(_graphiti_adapter(g), fake)
        gov.sync()
        rep = gov.observe("Alice migrated to SQLite", source="slack")
        assert not rep.errors and rep.pushes[0].action == "flag"
        assert e.invalid_at is not None and e.expired_at is not None
        assert e.attributes["invalidate_status"] == "superseded"
        assert e.attributes["invalidate_event"] == "Alice migrated to SQLite"
        assert g.driver.saves == ["e1"]
        assert e.fact == "Alice prefers Postgres"  # never rewritten, never deleted
        s = gov.sync()  # gone from pull (expired) but kept in the graph; a dead row is not "removed" by sync
        assert s.total == 0 and s.removed == 0 and "e1" in g.driver.edges and gov.status_of("e1") is Status.SUPERSEDED

    def test_needs_review_only_writes_attributes(self, fake):
        g = FakeGraphiti()
        e = g.edge("Alice prefers Postgres")
        fake.script("Postgres", UNCERTAIN)
        gov = _gov(_graphiti_adapter(g), fake)
        gov.sync()
        gov.observe("Alice might switch databases", source="slack")
        assert gov.status_of("e1") is Status.NEEDS_REVIEW
        assert e.expired_at is None and e.invalid_at is None
        assert e.attributes["invalidate_status"] == "needs_review"

    def test_delete_mode_invalidates_instead_of_deleting(self, fake):
        g = FakeGraphiti()
        e = g.edge("Alice prefers Postgres")
        fake.script("Postgres", CONTRADICT)
        gov = _gov(_graphiti_adapter(g), fake, mode="delete")
        gov.sync()
        rep = gov.observe("Alice dropped Postgres", source="slack")
        assert rep.pushes[0].action == "delete"
        assert "e1" in g.driver.edges and e.expired_at is not None and e.invalid_at is not None

    def test_keep_clears_the_validity_window(self, fake):
        g = FakeGraphiti()
        e = g.edge("Alice prefers Postgres")
        fake.script("Postgres", CONTRADICT)
        gov = _gov(_graphiti_adapter(g), fake)
        gov.sync()
        gov.observe("Alice dropped Postgres", source="slack")
        gov.keep("e1")
        assert e.expired_at is None and e.invalid_at is None and e.attributes["invalidate_status"] == "active"

    def test_preexisting_invalid_at_is_preserved(self):
        g = FakeGraphiti()
        earlier = datetime(2024, 1, 1, tzinfo=timezone.utc)
        e = g.edge("x", invalid_at=earlier)
        _graphiti_adapter(g).flag("e1", _reason(Status.CONTRADICTED))
        assert e.invalid_at == earlier and e.expired_at is not None

    def test_insert_goes_through_add_episode_and_the_episode_is_the_successor(self, fake):
        g = FakeGraphiti()
        g.edge("Alice prefers Postgres")
        fake.script("prefers Postgres", SUPERSEDE)
        gov = _gov(_graphiti_adapter(g), fake, successors=True)
        gov.sync()
        rep = gov.observe("Alice migrated to SQLite", source="slack")
        call = g.calls[0]
        assert call["episode_body"] == "Alice migrated to SQLite" and call["source_description"] == "slack"
        assert call["group_id"] == "g1" and call["name"] == f"{EPISODE_PREFIX}{rep.report.event.id}"
        assert call["reference_time"].tzinfo is not None
        assert rep.successor_host_id == "ep2"
        assert gov.mem.get(gov.our_id("e1")).superseded_by == gov.our_id("ep2")
        # next sync: the episode (verbatim) stays; Graphiti's own extracted edge shows up as a new memory
        s = gov.sync()
        assert s.removed == 0 and s.added == 1 and s.unchanged == 1  # e1 is dead, so its absence is not a removal
        succ = gov.mem.get(gov.our_id("ep2"))
        assert succ.fact == "Alice migrated to SQLite" and succ.status is Status.ACTIVE
        extracted = gov.mem.get(gov.our_id("e3"))
        assert extracted.fact == "extracted: alice migrated to sqlite"

    def test_verdicts_on_episodes_stay_in_the_ledger(self, fake):
        g = FakeGraphiti()
        g.edge("Alice prefers Postgres")
        fake.script("prefers Postgres", SUPERSEDE)
        gov = _gov(_graphiti_adapter(g), fake, successors=True)
        gov.sync()
        gov.observe("Alice migrated to SQLite", source="slack")
        gov.sync()
        fake.script("migrated to SQLite", CONTRADICT)
        rep = gov.observe("Alice never migrated", source="slack")
        assert not rep.errors and gov.status_of("ep2") is Status.CONTRADICTED
        assert g.driver.saves == ["e1"]  # the episode itself was never "saved" as an edge

    def test_episodes_can_be_excluded(self, fake):
        g = FakeGraphiti()
        adapter = _graphiti_adapter(g, include_episodes=False)
        assert adapter.insert("hello", "slack", {}) == "ep1"
        assert list(adapter.pull()) != [] and all(m.id != "ep1" for m in adapter.pull())

    def test_run_inside_a_running_loop_uses_a_thread(self):
        async def inner():
            g = FakeGraphiti()
            g.edge("x")
            return [m.id for m in _graphiti_adapter(g).pull()]

        assert asyncio.run(inner()) == ["e1"]

    def test_run_propagates_errors(self):
        async def boom():
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            _run(boom())

        async def outer():
            with pytest.raises(RuntimeError):
                _run(boom())

        asyncio.run(outer())


def test_mem0_governed_search_annotate_keeps_dead_and_adds_note_key(fake):
    m = FakeMem0OSS()
    dead = m.seed("user prefers postgres")
    live = m.seed("postgres tips from the wiki")
    fake.script("prefers postgres", SUPERSEDE)
    gov = _gov(Mem0Adapter(m, user_id="u1"), fake)
    gov.sync()
    gov.observe("we migrated to sqlite", source="slack")
    res = governed_search(m, gov, "postgres", annotate=True)
    assert isinstance(res, dict) and {r["id"] for r in res["results"]} == {dead, live}
    notes = {r["id"]: r["invalidate_note"] for r in res["results"]}
    assert notes[dead] == "OUTDATED, replaced as of slack: we migrated to sqlite" and notes[live] is None
    assert all("memory" in r and "score" in r for r in res["results"])            # host keys preserved
    assert "invalidate_note" not in m.rows[dead]                                    # host row untouched
    assert [r["id"] for r in governed_search(m, gov, "postgres")["results"]] == [live]   # default still filters
