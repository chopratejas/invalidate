"""mem0 adapter: put invalidate in front of `mem0.Memory` (OSS) or `mem0.MemoryClient` (platform).

    from mem0 import Memory
    from invalidate.adapters import Governor
    from invalidate.adapters.mem0 import Mem0Adapter, governed_search, guard_add

    memory = Memory()
    gov = Governor(Mem0Adapter(memory, user_id="alice"), "ledger.db", mode="flag", successors=True)
    gov.sync()
    gov.observe("we migrated to SQLite", source="slack")
    hits = governed_search(memory, gov, "which database?")   # dead memories dropped
    add = guard_add(memory, gov)                              # judge-before-write add()

Nothing from the mem0 SDK is imported here; the adapter relies on the object you pass in and detects the
calling convention from its signatures. Signatures mirrored (mem0ai 2.1.0):

  OSS   mem0/memory/main.py:1255  Memory.get_all(*, filters=None, top_k=20, show_expired=False, **kwargs)
                                  -> {"results": [{"id", "memory", "hash", "metadata"?, "created_at", ...}]}
                                  (top-level user_id= raises ValueError, main.py:165; older versions took user_id=
                                  and could return a plain list)
  OSS   mem0/memory/main.py:1815  Memory.update(memory_id, text=None, metadata=None, expiration_date=_UNSET, data=None)
                                  metadata-only updates MERGE into the existing payload (main.py:2059) and keep the text
  OSS   mem0/memory/main.py:1869  Memory.delete(memory_id)
  OSS   mem0/memory/main.py:760   Memory.add(messages, *, user_id=None, agent_id=None, run_id=None, metadata=None,
                                  ..., infer=True, ...) ; infer=False stores each message verbatim (main.py:879) and
                                  returns {"results": [{"id", "memory", "event": "ADD", "actor_id", "role"}]}
  OSS   mem0/memory/main.py:1379  Memory.search(query, *, top_k=20, filters=None, ...) -> {"results": [...]}
  Cloud mem0/client/main.py:330   MemoryClient.get_all(options=None, **kwargs) with filters= ; paginated
                                  {"count", "next", "previous", "results": [...]} (top-level user_id= raises)
  Cloud mem0/client/main.py:425   MemoryClient.update(memory_id, options=None, **kwargs)  (text=, metadata=)
  Cloud mem0/client/main.py:464   MemoryClient.delete(memory_id, delete_linked=False)
  Cloud mem0/client/main.py:265   MemoryClient.add(messages, options=None, **kwargs) ; identity ids go inside
                                  filters={...} (client/types.py:18), plus metadata=, infer=
  Cloud mem0/client/main.py:378   MemoryClient.search(query, options=None, **kwargs) -> {"results": [...]}
"""
from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from typing import Any

from .base import Governor, HostMemory, Reason

_SCOPE_KEYS = ("user_id", "agent_id", "run_id")


def _params(fn: Callable[..., Any]) -> tuple[set[str], bool]:
    """(named parameters, accepts **kwargs). Empty/False when the callable cannot be introspected."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return set(), False
    named = {p.name for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    var_kw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    return named, var_kw


def _named(fn: Callable[..., Any], param: str) -> bool:
    return param in _params(fn)[0]


def _accepts(fn: Callable[..., Any], param: str) -> bool:
    named, var_kw = _params(fn)
    return param in named or var_kw


def results_of(res: Any) -> list[dict[str, Any]]:
    """Normalise a mem0 return value: {"results": [...]} (v1.1+, platform) or a bare list (older OSS)."""
    if res is None:
        return []
    if isinstance(res, dict):
        out = res.get("results")
        return list(out) if isinstance(out, list) else []
    if isinstance(res, list):
        return list(res)
    return []


def _hostable(value: Any) -> Any:
    """Vector stores behind mem0 (chroma, qdrant, pgvector...) differ in what payload types they accept;
    keep receipts to scalars so the write never fails on a list."""
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(v) for v in value)
    if isinstance(value, dict):
        return str(value)
    return value


class Mem0Adapter:
    """Governs one mem0 scope (user_id / agent_id / run_id). mem0 keeps its memories; invalidate keeps the ledger.

    flag   -> `update(memory_id, metadata={...existing, **reason.as_metadata()})`, text untouched.
              On mem0 versions whose `update` has no `metadata` parameter the flag stays in the ledger only
              (the text is never rewritten to smuggle a receipt in). `flagged_in_host` tells you which.
    delete -> `delete(memory_id)`.
    insert -> `add(text, <scope>, infer=False, metadata={...})` so the successor is stored VERBATIM;
              returns results[0]["id"].
    """

    name = "mem0"

    def __init__(
        self,
        memory: Any,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata_prefix: str = "invalidate_",
        *,
        page_size: int = 5000,
    ) -> None:
        self.memory = memory
        self.scope: dict[str, str] = {k: v for k, v in zip(_SCOPE_KEYS, (user_id, agent_id, run_id)) if v}
        if not self.scope:
            raise ValueError("Mem0Adapter needs at least one of user_id, agent_id, run_id")
        self.prefix = metadata_prefix
        self.page_size = page_size
        self.is_platform = hasattr(memory, "users") or hasattr(memory, "project") or hasattr(memory, "batch_update")
        update = getattr(memory, "update", None)
        self.flagged_in_host = bool(update) and _accepts(update, "metadata")

    # -- calling conventions ---------------------------------------------------
    def _scoped(self, fn: Callable[..., Any], *args: Any, **kw: Any) -> Any:
        """Call `fn` with our scope, in the convention its signature advertises; fall back to the other
        convention on TypeError so older/newer mem0 releases both work."""
        top_level = _named(fn, "user_id")  # 0.1.x OSS style; 2.x OSS and the platform want filters={...}
        first = dict(self.scope) if top_level else {"filters": dict(self.scope)}
        second = {"filters": dict(self.scope)} if top_level else dict(self.scope)
        try:
            return fn(*args, **first, **kw)
        except TypeError:
            return fn(*args, **second, **kw)

    def _size_kw(self, fn: Callable[..., Any]) -> dict[str, Any]:
        for p in ("top_k", "limit", "page_size"):
            if _named(fn, p):
                return {p: self.page_size}
        return {}

    # -- Adapter ----------------------------------------------------------------
    def pull(self) -> Iterable[HostMemory]:
        get_all = self.memory.get_all
        res = self._scoped(get_all, **self._size_kw(get_all))
        items = results_of(res)
        page = 1
        # platform pagination: {"count", "next", "previous", "results"}; follow `next` by page number
        while isinstance(res, dict) and res.get("next") and items:
            page += 1
            res = self._scoped(get_all, page=page, **self._size_kw(get_all))
            more = results_of(res)
            if not more:
                break
            items.extend(more)
        out: list[HostMemory] = []
        for it in items:
            if not isinstance(it, dict) or not it.get("id"):
                continue
            text = it.get("memory")
            if not isinstance(text, str):
                continue
            meta = it.get("metadata") or {}
            out.append(HostMemory(id=str(it["id"]), text=text, source="mem0", metadata=dict(meta) if isinstance(meta, dict) else {}))
        return out

    def _existing_metadata(self, host_id: str) -> dict[str, Any]:
        get = getattr(self.memory, "get", None)
        if get is None:
            return {}
        try:
            item = get(host_id)
        except Exception:  # noqa: BLE001 - a missing read must not block the flag
            return {}
        meta = (item or {}).get("metadata") if isinstance(item, dict) else None
        return dict(meta) if isinstance(meta, dict) else {}

    def flag(self, host_id: str, reason: Reason) -> None:
        if not self.flagged_in_host:
            return  # documented limit: this mem0 cannot update metadata without rewriting text
        receipt = {k: _hostable(v) for k, v in reason.as_metadata(self.prefix).items()}
        merged = {**self._existing_metadata(host_id), **receipt}
        self.memory.update(host_id, metadata=merged)

    def delete(self, host_id: str, reason: Reason) -> None:
        self.memory.delete(host_id)

    def insert(self, text: str, source: str, metadata: dict[str, Any]) -> str | None:
        meta = {f"{self.prefix}source": source}
        for k, v in metadata.items():
            key = k if k.startswith(self.prefix) else f"{self.prefix}{k}"
            meta[key] = _hostable(v)
        add = self.memory.add
        kw: dict[str, Any] = {"metadata": meta}
        if _accepts(add, "infer"):
            kw["infer"] = False
        res = self._scoped(add, text, **kw)
        for it in results_of(res):
            if isinstance(it, dict) and it.get("id") and it.get("event", "ADD") in ("ADD", "add", None):
                return str(it["id"])
        for it in results_of(res):
            if isinstance(it, dict) and it.get("id"):
                return str(it["id"])
        return None


# -- read side and write guard ---------------------------------------------------
def _scope_given(kw: dict[str, Any]) -> bool:
    return "filters" in kw or any(k in kw for k in _SCOPE_KEYS)


def governed_search(memory: Any, gov: Governor, query: str, *, annotate: bool = False, **kw: Any) -> Any:
    """`memory.search(query, **kw)` with dead memories removed. Returns the same shape mem0 returned
    ({"results": [...]} with the other keys preserved, or a bare list). When `kw` names no scope and the
    governor's adapter is a Mem0Adapter, its scope is applied.

    `annotate=True` keeps every result and adds a top-level `"invalidate_note"` key to each result dict
    (`Governor.annotate`'s label, e.g. "OUTDATED, replaced as of slack: we moved to SQLite"; None when the
    memory is live). Top level rather than `metadata`, which mem0 may return as None."""
    adapter = gov.adapter
    if not _scope_given(kw) and isinstance(adapter, Mem0Adapter):
        res = adapter._scoped(memory.search, query, **kw)
    else:
        res = memory.search(query, **kw)
    rows = results_of(res)

    def id_of(r: Any) -> str:
        return r.get("id", "") if isinstance(r, dict) else ""

    if annotate:
        keep = [{**r, "invalidate_note": n} if isinstance(r, dict) else r for r, n in gov.annotate(rows, id_of=id_of)]
    else:
        keep = gov.filter(rows, id_of=id_of)
    if isinstance(res, dict):
        return {**res, "results": keep}
    return keep


def user_texts(*args: Any, **kwargs: Any) -> list[str]:
    """The event texts inside a mem0 `add(messages, ...)` call: a string, one {role, content} dict, or a
    list of them; only user-role content is judged."""
    messages = args[0] if args else kwargs.get("messages", "")
    if isinstance(messages, str):
        return [messages]
    if isinstance(messages, dict):
        messages = [messages]
    out: list[str] = []
    for msg in messages or []:
        if isinstance(msg, dict) and msg.get("role", "user") == "user" and isinstance(msg.get("content"), str):
            out.append(msg["content"])
    return out


def guard_add(memory: Any, gov: Governor, *, source: str = "user") -> Callable[..., Any]:
    """`memory.add` wrapped so every user message is judged against the ledger (and flags/deletes pushed)
    before mem0 stores it; the host is re-synced afterwards."""
    return gov.guard(memory.add, text_of=user_texts, source=source)


__all__ = ["Mem0Adapter", "governed_search", "guard_add", "user_texts", "results_of"]
