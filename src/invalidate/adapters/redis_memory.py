"""Redis Agent Memory Server adapter: put invalidate in front of `agent_memory_client.MemoryAPIClient`.

    from agent_memory_client import MemoryAPIClient, MemoryClientConfig
    from invalidate.adapters import Governor
    from invalidate.adapters.redis_memory import RedisMemoryAdapter, governed_search

    client = MemoryAPIClient(MemoryClientConfig(base_url="http://localhost:8000"))
    gov = Governor(RedisMemoryAdapter(client, namespace="app", user_id="alice"), "ledger.db", mode="flag")
    gov.sync()
    gov.observe("we migrated to SQLite", source="slack")
    hits = governed_search(client, gov, "which database?")   # dead memories dropped

The client is async; this adapter is sync (the `Adapter` protocol is) and runs each coroutine with `_run`:
`asyncio.run` when no loop is running in this thread, otherwise a fresh loop on a helper thread. Nothing from
agent_memory_client is imported at module import time; `MemoryRecord` is imported inside `insert` (with a
duck-typed fallback) so the package imports without the SDK. Signatures mirrored (agent-memory-client 0.14.0,
byte-identical to the `client/v0.14.0` tag; server behaviour checked against server/v0.14.0, server/v0.15.2
and main):

  client.py:1033  await client.search_long_term_memory(text, session_id=None, namespace=None, topics=None,
                  entities=None, created_at=None, last_accessed=None, user_id=None, distance_threshold=None,
                  memory_type=None, recency=None, limit=10, offset=0, optimize_query=False)
                  -> MemoryRecordResults(memories: list[MemoryRecordResult], total, next_offset)   (models.py:388)
                  Filters take the filter objects or plain dicts ({"eq": ...}); filters.py:18 SessionId,
                  :27 Namespace, :36 UserId, :45 Topics(any/all/none). An EMPTY `text` is a filter-only listing
                  on the server (long_term_memory.py: "If no query text is provided, perform a filter-only
                  listing"); the server caps `limit` at 100 (SearchRequest.limit le=100).
  client.py:753   await client.get_long_term_memory(memory_id) -> MemoryRecord ; 404 -> MemoryNotFoundError
  client.py:773   await client.edit_long_term_memory(memory_id, updates: dict) -> MemoryRecord
                  PATCH /v1/long-term-memory/{id}; the server accepts ONLY text, topics, entities, memory_type,
                  namespace, user_id, session_id, event_date, pinned (long_term_memory.py updatable_fields) and
                  REPLACES each given field. There is no free-form metadata field on a memory: MemoryRecord
                  (models.py:168) has id, text, session_id, user_id, namespace, last_accessed, created_at,
                  updated_at, topics, entities, memory_hash, discrete_memory_extracted, memory_type, persisted_at,
                  extracted_from, event_date. (main adds a `metadata` field, but it is not PATCH-able either.)
                  Tag values (topics/entities) must not contain commas (utils/tag_codec.py).
  client.py:731   await client.delete_long_term_memories(memory_ids: Sequence[str]) -> AckResponse
  client.py:661   await client.create_long_term_memory(memories: Sequence[ClientMemoryRecord | MemoryRecord],
                  deduplicate=True) -> AckResponse ; the client fills config.default_namespace and a ULID id
                  when missing; the server indexes in a BACKGROUND task (api.py create_long_term_memory), and
                  deduplicate=True may LLM-merge the text with an existing memory.
  models.py:51    MemoryTypeEnum: "episodic" | "semantic" | "message" ; models.py:244 ClientMemoryRecord
  client.py:126   MemoryClientConfig(base_url, timeout=30.0, default_namespace=None, ...)
"""
from __future__ import annotations

import asyncio
import re
import threading
import uuid
from collections.abc import Awaitable, Iterable
from datetime import datetime
from typing import Any, TypeVar

from .base import DEAD, Governor, HostMemory, Reason

T = TypeVar("T")

MARKER_PREFIX = "invalidate:"
_SCOPE_KEYS = ("namespace", "user_id", "session_id")
_MAX_PAGE = 100  # server-side SearchRequest.limit le=100


def _run(coro: Awaitable[T]) -> T:
    """Run a coroutine to completion from sync code, whether or not an event loop is already running here."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = asyncio.run(coro)  # type: ignore[arg-type]
        except BaseException as e:  # noqa: BLE001 - re-raised on the caller's thread
            box["error"] = e

    t = threading.Thread(target=target, name="invalidate-redis-memory", daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def _tag(value: Any, limit: int = 160) -> str:
    """A topic-safe value: no commas (the server's TAG delimiter), collapsed whitespace, bounded length."""
    s = re.sub(r"\s+", " ", str(value).replace(",", " ")).strip()
    return s[:limit]


def receipt_topics(reason: Reason, prefix: str = MARKER_PREFIX) -> list[str]:
    """The Reason as marker topics: `invalidate:<status>` plus `invalidate:<key>:<value>` for the rest of
    `Reason.as_metadata()`. Empty values are skipped."""
    out = [f"{prefix}{reason.status.value}"]
    for key, value in reason.as_metadata("").items():
        if key == "status" or value in ("", None):
            continue
        out.append(f"{prefix}{key}:{_tag(value)}")
    return out


def dead_markers(prefix: str = MARKER_PREFIX) -> list[str]:
    """Marker topics of dead statuses, for a host-side `topics={"none": [...]}` pre-filter."""
    return [f"{prefix}{s.value}" for s in sorted(DEAD, key=lambda s: s.value)]


def _new_id() -> str:
    try:
        from ulid import ULID  # dependency of agent-memory-client; the server only requires a non-empty id
    except ImportError:
        return uuid.uuid4().hex.upper()
    return str(ULID())


class _Record:
    """Duck-typed stand-in for `agent_memory_client.models.MemoryRecord`: what `create_long_term_memory`
    touches is `.id`, `.namespace` and `.model_dump(exclude_none=True, mode="json")`."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)

    def model_dump(self, *, exclude_none: bool = False, mode: str = "python") -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not (exclude_none and v is None)}


def _make_record(**fields: Any) -> Any:
    try:
        from agent_memory_client.models import MemoryRecord
    except ImportError:
        return _Record(**fields)
    return MemoryRecord(**fields)


class RedisMemoryAdapter:
    """Governs the long-term memories of one scope (namespace / user_id / session_id, any subset) in a Redis
    Agent Memory Server. The server keeps its memories; invalidate keeps the ledger.

    pull   -> `search_long_term_memory(text="", <scope filters>, limit=page_size, offset=...)` paged until a
              short page. Marker topics are stripped from the pulled `metadata["topics"]`.
    flag   -> `edit_long_term_memory(id, {"topics": [...existing minus old markers, *receipt_topics(reason)]})`;
              the text is never touched. The receipt lives in topics because that is the only writable per-memory
              free-form field: `invalidate:superseded`, `invalidate:event:...`, `invalidate:event_source:...`, ...
    delete -> `delete_long_term_memories([id])`.
    insert -> `create_long_term_memory([MemoryRecord(text=<verbatim>, memory_type="semantic", topics=[markers],
              <scope>)], deduplicate=False)`; the id is minted here (ULID) and returned. The server indexes in the
              background, so the record is also remembered locally and served by `pull` until the host lists it.
    """

    name = "redis_memory"

    def __init__(
        self,
        client: Any,
        *,
        namespace: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        page_size: int = _MAX_PAGE,
        marker_prefix: str = MARKER_PREFIX,
        memory_types: Iterable[str] | None = None,
    ) -> None:
        self.client = client
        self.scope: dict[str, str] = {k: v for k, v in zip(_SCOPE_KEYS, (namespace, user_id, session_id)) if v}
        self.page_size = max(1, min(int(page_size), _MAX_PAGE))
        self.prefix = marker_prefix
        if "," in self.prefix:
            raise ValueError("marker_prefix must not contain a comma (the server's tag delimiter)")
        self.memory_types = tuple(memory_types) if memory_types else None
        self._pending: dict[str, HostMemory] = {}

    # -- helpers ----------------------------------------------------------------
    def filters(self) -> dict[str, dict[str, Any]]:
        """The scope as `search_long_term_memory` keyword filters (`{"eq": ...}` dicts)."""
        return {k: {"eq": v} for k, v in self.scope.items()}

    def _search_kw(self) -> dict[str, Any]:
        kw: dict[str, Any] = self.filters()
        if self.memory_types:
            kw["memory_type"] = {"in_": list(self.memory_types)}
        return kw

    def _host_memory(self, m: Any) -> HostMemory | None:
        hid, text = getattr(m, "id", None), getattr(m, "text", None)
        if not hid or not isinstance(text, str) or not text:
            return None
        topics = [str(t) for t in (getattr(m, "topics", None) or [])]
        markers = [t for t in topics if t.startswith(self.prefix)]
        mt = getattr(m, "memory_type", None)
        meta: dict[str, Any] = {
            "topics": [t for t in topics if t not in markers],
            "entities": [str(e) for e in (getattr(m, "entities", None) or [])],
            "memory_type": str(getattr(mt, "value", mt) or ""),
        }
        for k in _SCOPE_KEYS:
            v = getattr(m, k, None)
            if v:
                meta[k] = str(v)
        created = getattr(m, "created_at", None)
        if isinstance(created, datetime):
            meta["created_at"] = created.isoformat()
        if markers:
            meta["invalidate_markers"] = markers
        return HostMemory(id=str(hid), text=text, source="redis_memory", metadata=meta)

    def _current_topics(self, host_id: str) -> list[str]:
        rec = _run(self.client.get_long_term_memory(host_id))
        return [str(t) for t in (getattr(rec, "topics", None) or [])]

    # -- Adapter ----------------------------------------------------------------
    def pull(self) -> Iterable[HostMemory]:
        out: list[HostMemory] = []
        seen: set[str] = set()
        offset = 0
        while True:
            res = _run(self.client.search_long_term_memory(text="", limit=self.page_size, offset=offset, **self._search_kw()))
            page = list(getattr(res, "memories", None) or [])
            for m in page:
                hm = self._host_memory(m)
                if hm is not None and hm.id not in seen:
                    seen.add(hm.id)
                    out.append(hm)
            if len(page) < self.page_size:
                break
            nxt = getattr(res, "next_offset", None)
            offset = nxt if isinstance(nxt, int) and nxt > offset else offset + len(page)
        for hid, hm in list(self._pending.items()):
            if hid in seen:
                del self._pending[hid]  # the host has indexed it; nothing to remember locally
            else:
                out.append(hm)
        return out

    def flag(self, host_id: str, reason: Reason) -> None:
        keep = [t for t in self._current_topics(host_id) if not t.startswith(self.prefix)]
        _run(self.client.edit_long_term_memory(host_id, {"topics": keep + receipt_topics(reason, self.prefix)}))

    def delete(self, host_id: str, reason: Reason) -> None:
        _run(self.client.delete_long_term_memories([host_id]))
        self._pending.pop(host_id, None)

    def insert(self, text: str, source: str, metadata: dict[str, Any]) -> str | None:
        hid = _new_id()
        topics = [f"{self.prefix}successor", f"{self.prefix}source:{_tag(source)}"]
        for k, v in metadata.items():
            key = k[len("invalidate_"):] if k.startswith("invalidate_") else k
            for item in (v if isinstance(v, (list, tuple, set)) else [v]):
                if item not in ("", None):
                    topics.append(f"{self.prefix}{_tag(key)}:{_tag(item)}")
        record = _make_record(
            id=hid, text=text, memory_type="semantic", topics=topics, discrete_memory_extracted="t", **self.scope,
        )
        _run(self.client.create_long_term_memory([record], deduplicate=False))
        self._pending[hid] = HostMemory(
            id=hid, text=text, source="redis_memory",
            metadata={"topics": [], "entities": [], "memory_type": "semantic", **self.scope, "invalidate_markers": topics},
        )
        return hid


# -- read side --------------------------------------------------------------------
def _scope_given(kw: dict[str, Any]) -> bool:
    return any(k in kw for k in _SCOPE_KEYS)


def governed_search(client: Any, gov: Governor, query: str, **kw: Any) -> Any:
    """`client.search_long_term_memory(query, **kw)` run synchronously, with dead (and under-review) memories
    removed. Returns the `MemoryRecordResults` the client returned with `memories` filtered and the other fields
    (`total`, `next_offset`) untouched. When `kw` names no scope filter and the governor's adapter is a
    RedisMemoryAdapter, its scope filters are applied."""
    adapter = gov.adapter
    if not _scope_given(kw) and isinstance(adapter, RedisMemoryAdapter):
        kw = {**adapter.filters(), **kw}
    res = _run(client.search_long_term_memory(query, **kw))
    keep = gov.filter(list(getattr(res, "memories", None) or []), id_of=lambda m: getattr(m, "id", "") or "")
    copy = getattr(res, "model_copy", None)
    if callable(copy):
        return copy(update={"memories": keep})
    res.memories = keep
    return res


async def agoverned_search(client: Any, gov: Governor, query: str, **kw: Any) -> Any:
    """Async twin of `governed_search` for code already inside an event loop."""
    adapter = gov.adapter
    if not _scope_given(kw) and isinstance(adapter, RedisMemoryAdapter):
        kw = {**adapter.filters(), **kw}
    res = await client.search_long_term_memory(query, **kw)
    keep = gov.filter(list(getattr(res, "memories", None) or []), id_of=lambda m: getattr(m, "id", "") or "")
    copy = getattr(res, "model_copy", None)
    if callable(copy):
        return copy(update={"memories": keep})
    res.memories = keep
    return res


__all__ = ["RedisMemoryAdapter", "governed_search", "agoverned_search", "receipt_topics", "dead_markers", "MARKER_PREFIX"]
