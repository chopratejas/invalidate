"""Redis Agent Memory Server adapter against a faithful async fake of `agent_memory_client.MemoryAPIClient`.

The fake mirrors agent-memory-client 0.14.0 (client.py / models.py / filters.py) and the server rules that
matter: empty search text is a filter-only listing, `limit` is capped at 100, PATCH accepts a fixed set of
fields and replaces them, tags must not contain commas, and creates are indexed in the background.
No network, no SDK imports.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Optional

import pytest
from conftest import CONTRADICT, FakeJudge, SUPERSEDE, UNCERTAIN

from invalidate.adapters import Governor, Reason
from invalidate.adapters.redis_memory import (
    MARKER_PREFIX,
    RedisMemoryAdapter,
    _run,
    agoverned_search,
    dead_markers,
    governed_search,
    receipt_topics,
)
from invalidate.types import Status, now

# =============================================================================================
# fakes
# =============================================================================================
_UPDATABLE = {"text", "topics", "entities", "memory_type", "namespace", "user_id", "session_id", "event_date", "pinned"}


class FakeNotFound(Exception):
    """agent_memory_client.exceptions.MemoryNotFoundError (raised by _handle_http_error on 404, client.py:186)."""


def _no_commas(values: Optional[list[str]], name: str) -> None:
    for i, v in enumerate(values or []):
        if "," in v:
            raise ValueError(f"{name}[{i}] contains a comma: {v!r}")


@dataclass
class FakeMemoryRecord:
    """models.py:168 MemoryRecord (the fields the adapter reads and writes) with pydantic-style helpers."""

    id: str
    text: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    namespace: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    topics: Optional[list[str]] = None
    entities: Optional[list[str]] = None
    discrete_memory_extracted: str = "f"
    memory_type: str = "message"
    pinned: bool = False

    def model_dump(self, *, exclude_none: bool = False, mode: str = "python") -> dict[str, Any]:
        d = dict(self.__dict__)
        if mode == "json":
            d = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in d.items()}
        return {k: v for k, v in d.items() if not (exclude_none and v is None)}

    def model_copy(self, update: Optional[dict[str, Any]] = None) -> "FakeMemoryRecord":
        return replace(self, **(update or {}))


@dataclass
class FakeMemoryRecordResults:
    """models.py:388 MemoryRecordResults."""

    memories: list[FakeMemoryRecord]
    total: int
    next_offset: Optional[int] = None

    def model_copy(self, update: Optional[dict[str, Any]] = None) -> "FakeMemoryRecordResults":
        return replace(self, **(update or {}))


@dataclass
class FakeAck:
    status: str = "ok"


@dataclass
class FakeConfig:
    """client.py:126 MemoryClientConfig."""

    base_url: str = "http://memory.test"
    default_namespace: Optional[str] = None


def _eq(f: Any, value: Optional[str]) -> bool:
    """Apply an eq/in_/not_eq/not_in filter given as a dict or a filter object (filters.py:18-43)."""
    if f is None:
        return True
    get = f.get if isinstance(f, dict) else lambda k, d=None: getattr(f, k, d)
    if get("eq") is not None and value != get("eq"):
        return False
    if get("in_") is not None and value not in get("in_"):
        return False
    if get("not_eq") is not None and value == get("not_eq"):
        return False
    if get("not_in") is not None and value in get("not_in"):
        return False
    return True


def _tags(f: Any, values: Optional[list[str]]) -> bool:
    """Topics/Entities any/all/none (filters.py:45-58)."""
    if f is None:
        return True
    get = f.get if isinstance(f, dict) else lambda k, d=None: getattr(f, k, d)
    have = set(values or [])
    if get("any") is not None and not have & set(get("any")):
        return False
    if get("all") is not None and not set(get("all")) <= have:
        return False
    if get("none") is not None and have & set(get("none")):
        return False
    return True


class FakeMemoryAPIClient:
    """agent-memory-client 0.14.0 `MemoryAPIClient`, long-term-memory surface only. All methods are async."""

    def __init__(self, default_namespace: Optional[str] = None, lag: bool = False) -> None:
        self.config = FakeConfig(default_namespace=default_namespace)
        self.rows: dict[str, FakeMemoryRecord] = {}
        self.queued: list[FakeMemoryRecord] = []  # created but not yet indexed (background task)
        self.lag = lag
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._n = 0

    def seed(self, text: str, *, namespace: str = "app", user_id: str = "alice", session_id: Optional[str] = None,
             topics: Optional[list[str]] = None, memory_type: str = "semantic") -> str:
        self._n += 1
        mid = f"01J{self._n:023d}"
        self.rows[mid] = FakeMemoryRecord(id=mid, text=text, namespace=namespace, user_id=user_id, session_id=session_id,
                                          topics=list(topics) if topics else None, memory_type=memory_type)
        return mid

    def flush(self) -> None:
        """The server's background `index_long_term_memories` finishing."""
        for r in self.queued:
            self.rows[r.id] = r
        self.queued = []

    # client.py:1033
    async def search_long_term_memory(self, text: str, session_id: Any = None, namespace: Any = None, topics: Any = None,
                                      entities: Any = None, created_at: Any = None, last_accessed: Any = None,
                                      user_id: Any = None, distance_threshold: Optional[float] = None,
                                      memory_type: Any = None, recency: Any = None, limit: int = 10, offset: int = 0,
                                      optimize_query: bool = False) -> FakeMemoryRecordResults:
        await asyncio.sleep(0)
        if namespace is None and self.config.default_namespace is not None:  # client.py:1113
            namespace = {"eq": self.config.default_namespace}
        self.calls.append(("search", {"text": text, "namespace": namespace, "user_id": user_id, "session_id": session_id,
                                      "topics": topics, "memory_type": memory_type, "limit": limit, "offset": offset}))
        if not (1 <= limit <= 100) or offset < 0:  # server SearchRequest: limit ge=1 le=100, offset ge=0
            raise ValueError("HTTP 422: limit must be between 1 and 100")
        rows = [r for r in self.rows.values()
                if _eq(namespace, r.namespace) and _eq(user_id, r.user_id) and _eq(session_id, r.session_id)
                and _eq(memory_type, r.memory_type) and _tags(topics, r.topics) and _tags(entities, r.entities)]
        if (text or "").strip():  # otherwise: filter-only listing (server long_term_memory.py)
            rows = [r for r in rows if text.lower() in r.text.lower()]
        page = rows[offset: offset + limit]
        nxt = offset + limit if offset + limit < len(rows) else None
        return FakeMemoryRecordResults(memories=[replace(r) for r in page], total=len(rows), next_offset=nxt)

    # client.py:753
    async def get_long_term_memory(self, memory_id: str) -> FakeMemoryRecord:
        await asyncio.sleep(0)
        self.calls.append(("get", {"memory_id": memory_id}))
        if memory_id not in self.rows:
            raise FakeNotFound(f"Resource not found: /v1/long-term-memory/{memory_id}")
        return replace(self.rows[memory_id])

    # client.py:773 -> server api.py update_long_term_memory / long_term_memory.py update_long_term_memory
    async def edit_long_term_memory(self, memory_id: str, updates: dict[str, Any]) -> FakeMemoryRecord:
        await asyncio.sleep(0)
        self.calls.append(("edit", {"memory_id": memory_id, "updates": updates}))
        clean = {k: v for k, v in updates.items() if v is not None}
        if not clean:
            raise ValueError("HTTP 400: No fields provided for update")
        bad = set(clean) - _UPDATABLE
        if bad:
            raise ValueError(f"HTTP 400: Cannot update fields: {bad}")
        _no_commas(clean.get("topics"), "topics")
        _no_commas(clean.get("entities"), "entities")
        if memory_id not in self.rows:
            raise FakeNotFound(f"Resource not found: /v1/long-term-memory/{memory_id}")
        self.rows[memory_id] = self.rows[memory_id].model_copy(update={**clean, "updated_at": datetime.now(timezone.utc)})
        return replace(self.rows[memory_id])

    # client.py:731
    async def delete_long_term_memories(self, memory_ids: Sequence[str]) -> FakeAck:
        await asyncio.sleep(0)
        self.calls.append(("delete", {"memory_ids": list(memory_ids)}))
        for mid in memory_ids:
            self.rows.pop(mid, None)
        return FakeAck()

    # client.py:661 -> server api.py create_long_term_memory (background indexing)
    async def create_long_term_memory(self, memories: Sequence[Any], deduplicate: bool = True) -> FakeAck:
        await asyncio.sleep(0)
        if self.config.default_namespace is not None:  # client.py:704
            for m in memories:
                if m.namespace is None:
                    m.namespace = self.config.default_namespace
        for m in memories:  # client.py:709
            if not m.id:
                self._n += 1
                m.id = f"01J{self._n:023d}"
        payload = [m.model_dump(exclude_none=True, mode="json") for m in memories]
        self.calls.append(("create", {"memories": payload, "deduplicate": deduplicate}))
        for p in payload:
            if not p.get("id") or not p.get("text"):
                raise ValueError("HTTP 422: id and text are required")
            _no_commas(p.get("topics"), "topics")
            rec = FakeMemoryRecord(
                id=p["id"], text=p["text"], namespace=p.get("namespace"), user_id=p.get("user_id"),
                session_id=p.get("session_id"), topics=p.get("topics"), entities=p.get("entities"),
                memory_type=p.get("memory_type", "message"), discrete_memory_extracted=p.get("discrete_memory_extracted", "f"),
            )
            if self.lag:
                self.queued.append(rec)
            else:
                self.rows[rec.id] = rec
        return FakeAck()


def _gov(adapter, fake: FakeJudge, **kw) -> Governor:
    return Governor(adapter, ":memory:", judge=fake, **kw)


def _reason(status: Status, event: str = "we moved to sqlite") -> Reason:
    return Reason(status, status.value, event, "slack", "evt_x", 0.05, now())


def _adapter(c: FakeMemoryAPIClient, **kw) -> RedisMemoryAdapter:
    return RedisMemoryAdapter(c, namespace="app", user_id="alice", **kw)


# =============================================================================================
# pull
# =============================================================================================
class TestPull:
    def test_lists_with_empty_text_scope_filters_and_pages(self, fake):
        c = FakeMemoryAPIClient()
        for i in range(5):
            c.seed(f"fact {i}")
        c.seed("bob's fact", user_id="bob")
        c.seed("other app", namespace="other")
        gov = _gov(_adapter(c, page_size=2), fake)
        rep = gov.sync()
        assert rep.added == 5 and rep.total == 5
        searches = [kw for n, kw in c.calls if n == "search"]
        assert [kw["offset"] for kw in searches] == [0, 2, 4]
        assert all(kw["text"] == "" and kw["limit"] == 2 for kw in searches)
        assert searches[0]["namespace"] == {"eq": "app"} and searches[0]["user_id"] == {"eq": "alice"}
        assert searches[0]["session_id"] is None
        assert gov.sync().unchanged == 5

    def test_page_size_is_clamped_to_the_server_max(self):
        c = FakeMemoryAPIClient()
        c.seed("x")
        adapter = _adapter(c, page_size=5000)
        assert adapter.page_size == 100
        assert [m.text for m in adapter.pull()] == ["x"]

    def test_markers_are_stripped_from_topics_into_metadata(self):
        c = FakeMemoryAPIClient()
        mid = c.seed("user prefers postgres", topics=["db", "invalidate:superseded", "invalidate:event_id:e1"])
        hm = {m.id: m for m in _adapter(c).pull()}[mid]
        assert hm.text == "user prefers postgres" and hm.source == "redis_memory"
        assert hm.metadata["topics"] == ["db"]
        assert hm.metadata["invalidate_markers"] == ["invalidate:superseded", "invalidate:event_id:e1"]
        assert hm.metadata["memory_type"] == "semantic" and hm.metadata["namespace"] == "app"
        assert hm.metadata["user_id"] == "alice" and "created_at" in hm.metadata

    def test_memory_types_filter(self):
        c = FakeMemoryAPIClient()
        c.seed("a message", memory_type="message")
        c.seed("a fact", memory_type="semantic")
        assert [m.text for m in _adapter(c, memory_types=["semantic", "episodic"]).pull()] == ["a fact"]
        assert c.calls[-1][1]["memory_type"] == {"in_": ["semantic", "episodic"]}
        assert len(list(_adapter(c).pull())) == 2

    def test_unscoped_adapter_uses_the_client_default_namespace(self):
        c = FakeMemoryAPIClient(default_namespace="app")
        c.seed("in app")
        c.seed("elsewhere", namespace="other")
        adapter = RedisMemoryAdapter(c)
        assert adapter.scope == {} and [m.text for m in adapter.pull()] == ["in app"]

    def test_pull_inside_a_running_loop_uses_a_thread(self):
        async def inner():
            c = FakeMemoryAPIClient()
            c.seed("x")
            return [m.text for m in _adapter(c).pull()]

        assert asyncio.run(inner()) == ["x"]

    def test_run_propagates_errors(self):
        async def boom():
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            _run(boom())


# =============================================================================================
# flag / delete
# =============================================================================================
class TestFlagDelete:
    def test_flag_writes_receipt_topics_keeps_text_and_plain_topics(self, fake):
        c = FakeMemoryAPIClient()
        mid = c.seed("user prefers postgres", topics=["db"])
        fake.script("postgres", SUPERSEDE)
        gov = _gov(_adapter(c), fake)
        gov.sync()
        rep = gov.observe("we migrated to sqlite", source="slack")
        assert [p.action for p in rep.pushes] == ["flag"] and not rep.errors
        row = c.rows[mid]
        assert row.text == "user prefers postgres"
        assert row.topics[0] == "db" and "invalidate:superseded" in row.topics
        assert "invalidate:event:we migrated to sqlite" in row.topics
        assert "invalidate:event_source:slack" in row.topics
        assert f"invalidate:event_id:{rep.report.event.id}" in row.topics
        assert any(t.startswith("invalidate:disposition:") for t in row.topics)
        assert any(t.startswith("invalidate:still_true:") for t in row.topics)
        n, kw = [x for x in c.calls if x[0] == "edit"][-1]
        assert set(kw["updates"]) == {"topics"}  # topics-only PATCH: the text is never sent
        assert gov.status_of(mid) is Status.SUPERSEDED

    def test_second_flag_replaces_old_markers(self, fake):
        c = FakeMemoryAPIClient()
        mid = c.seed("user prefers postgres", topics=["db"])
        fake.script("postgres", UNCERTAIN)
        gov = _gov(_adapter(c), fake)
        gov.sync()
        gov.observe("postgres might be going away", source="slack")
        assert "invalidate:needs_review" in c.rows[mid].topics
        fake.script("postgres", CONTRADICT)
        gov.observe("we dropped postgres", source="slack")
        topics = c.rows[mid].topics
        assert topics[0] == "db"
        assert "invalidate:needs_review" not in topics and "invalidate:contradicted" in topics
        assert sum(t.startswith("invalidate:event_id:") for t in topics) == 1
        # the marker change does not look like a rewrite to the ledger
        s = gov.sync()
        assert s.unchanged == 1 and gov.status_of(mid) is Status.CONTRADICTED

    def test_receipt_topics_never_contain_commas(self):
        r = _reason(Status.SUPERSEDED, "we moved to sqlite, then to duckdb, honestly")
        topics = receipt_topics(r)
        assert topics[0] == "invalidate:superseded"
        assert not any("," in t for t in topics)
        assert "invalidate:event:we moved to sqlite then to duckdb honestly" in topics
        assert not any(t.startswith("invalidate:status") for t in topics)
        assert dead_markers() == ["invalidate:contradicted", "invalidate:superseded"]

    def test_flag_on_a_missing_memory_is_a_push_error_not_a_crash(self, fake):
        c = FakeMemoryAPIClient()
        mid = c.seed("user prefers postgres")
        fake.script("postgres", CONTRADICT)
        gov = _gov(_adapter(c), fake)
        gov.sync()
        del c.rows[mid]  # vanished between sync and observe
        rep = gov.observe("we dropped postgres", source="slack")
        assert len(rep.errors) == 1 and "FakeNotFound" in rep.errors[0].error
        assert gov.status_of(mid) is Status.CONTRADICTED  # the ledger is still right

    def test_delete_mode_deletes_the_memory(self, fake):
        c = FakeMemoryAPIClient()
        mid = c.seed("user prefers postgres")
        fake.script("postgres", CONTRADICT)
        gov = _gov(_adapter(c), fake, mode="delete")
        gov.sync()
        rep = gov.observe("we dropped postgres", source="slack")
        assert rep.pushes[0].action == "delete" and mid not in c.rows
        assert c.calls[-1] == ("delete", {"memory_ids": [mid]})

    def test_keep_and_forget_mirror_to_host(self, fake):
        c = FakeMemoryAPIClient()
        mid = c.seed("user prefers postgres", topics=["db"])
        fake.script("postgres", CONTRADICT)
        gov = _gov(_adapter(c), fake)
        gov.sync()
        gov.observe("we dropped postgres", source="slack")
        gov.keep(mid)
        topics = c.rows[mid].topics
        assert "invalidate:active" in topics and "invalidate:contradicted" not in topics and topics[0] == "db"
        assert not any(t.startswith("invalidate:event_id:") for t in topics)  # empty values are skipped
        gov.forget(mid)
        assert mid not in c.rows

    def test_marker_prefix_with_a_comma_is_rejected(self):
        with pytest.raises(ValueError):
            RedisMemoryAdapter(FakeMemoryAPIClient(), marker_prefix="inv,")


# =============================================================================================
# insert
# =============================================================================================
class TestInsert:
    def test_successor_is_verbatim_semantic_scoped_and_not_deduplicated(self, fake):
        c = FakeMemoryAPIClient()
        old = c.seed("user prefers postgres")
        fake.script("postgres", SUPERSEDE)
        gov = _gov(_adapter(c), fake, successors=True)
        gov.sync()
        rep = gov.observe("We migrated to SQLite.", source="slack")
        assert rep.successor_host_id and rep.successor_host_id in c.rows
        n, kw = [x for x in c.calls if x[0] == "create"][-1]
        assert kw["deduplicate"] is False and len(kw["memories"]) == 1
        payload = kw["memories"][0]
        assert payload["id"] == rep.successor_host_id and payload["text"] == "We migrated to SQLite."
        assert payload["memory_type"] == "semantic" and payload["discrete_memory_extracted"] == "t"
        assert payload["namespace"] == "app" and payload["user_id"] == "alice" and "session_id" not in payload
        assert "invalidate:successor" in payload["topics"] and "invalidate:source:slack" in payload["topics"]
        assert f"invalidate:supersedes:{old}" in payload["topics"]
        assert f"invalidate:event_id:{rep.report.event.id}" in payload["topics"]
        assert not any("," in t for t in payload["topics"])
        assert gov.mem.get(gov.our_id(old)).superseded_by == gov.our_id(rep.successor_host_id)
        assert gov.sync().unchanged == 2  # stable across syncs

    def test_successor_survives_background_indexing_lag(self, fake):
        c = FakeMemoryAPIClient(lag=True)
        c.seed("user prefers postgres")
        fake.script("prefers postgres", SUPERSEDE)
        adapter = _adapter(c)
        gov = _gov(adapter, fake, successors=True)
        gov.sync()
        rep = gov.observe("we migrated to sqlite", source="slack")
        succ = rep.successor_host_id
        assert succ not in c.rows and succ in adapter._pending
        s = gov.sync()  # the host does not list it yet; the adapter serves it locally so it is not "gone"
        assert s.removed == 0 and s.unchanged == 2 and gov.status_of(succ) is Status.ACTIVE
        c.flush()
        s = gov.sync()
        assert s.unchanged == 2 and succ not in adapter._pending and c.rows[succ].text == "we migrated to sqlite"

    def test_insert_without_sdk_uses_the_duck_record(self):
        c = FakeMemoryAPIClient(default_namespace="cfg")
        hid = RedisMemoryAdapter(c).insert("hello", "slack", {"invalidate_supersedes": ["a", "b"], "empty": ""})
        assert hid and c.rows[hid].text == "hello" and c.rows[hid].namespace == "cfg"
        assert {"invalidate:supersedes:a", "invalidate:supersedes:b"} <= set(c.rows[hid].topics)
        assert not any(t.startswith("invalidate:empty") for t in c.rows[hid].topics)


# =============================================================================================
# governed search
# =============================================================================================
class TestGovernedSearch:
    def _setup(self, fake, lag: bool = False):
        c = FakeMemoryAPIClient(lag=lag)
        dead = c.seed("user prefers postgres")
        live = c.seed("postgres tips from the wiki")
        fake.script("prefers postgres", SUPERSEDE)
        gov = _gov(_adapter(c), fake)
        gov.sync()
        gov.observe("we migrated to sqlite", source="slack")
        return c, gov, dead, live

    def test_drops_dead_injects_scope_and_keeps_shape(self, fake):
        c, gov, dead, live = self._setup(fake)
        res = governed_search(c, gov, "postgres")
        assert [m.id for m in res.memories] == [live]
        assert isinstance(res, FakeMemoryRecordResults) and res.total == 2  # host fields untouched
        n, kw = c.calls[-1]
        assert n == "search" and kw["text"] == "postgres"
        assert kw["namespace"] == {"eq": "app"} and kw["user_id"] == {"eq": "alice"}

    def test_explicit_scope_is_not_overridden(self, fake):
        c, gov, dead, live = self._setup(fake)
        res = governed_search(c, gov, "postgres", user_id={"eq": "nobody"}, limit=5)
        assert res.memories == [] and c.calls[-1][1]["user_id"] == {"eq": "nobody"}
        assert c.calls[-1][1]["namespace"] is None and c.calls[-1][1]["limit"] == 5

    def test_host_side_prefilter_with_dead_markers(self, fake):
        c, gov, dead, live = self._setup(fake)
        res = governed_search(c, gov, "postgres", topics={"none": dead_markers()})
        assert [m.id for m in res.memories] == [live]
        assert dead not in {m.id for m in c.rows.values() if _tags({"none": dead_markers()}, m.topics)}

    def test_async_twin(self, fake):
        c, gov, dead, live = self._setup(fake)

        async def go():
            return await agoverned_search(c, gov, "postgres")

        assert [m.id for m in asyncio.run(go()).memories] == [live]

    def test_review_is_hidden_by_default(self, fake):
        c = FakeMemoryAPIClient()
        mid = c.seed("user prefers postgres")
        fake.script("postgres", UNCERTAIN)
        gov = _gov(_adapter(c), fake)
        gov.sync()
        gov.observe("postgres might be going away", source="slack")
        assert gov.status_of(mid) is Status.NEEDS_REVIEW
        assert governed_search(c, gov, "postgres").memories == []
        assert c.rows[mid].topics == [f"{MARKER_PREFIX}needs_review"] + [t for t in c.rows[mid].topics if ":" in t[len(MARKER_PREFIX):]]
        gov.keep(mid)
        assert [m.id for m in governed_search(c, gov, "postgres").memories] == [mid]
