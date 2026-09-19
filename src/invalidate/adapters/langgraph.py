"""Adapter for a LangGraph `BaseStore` namespace (InMemoryStore, PostgresStore, ...).

    from langgraph.store.memory import InMemoryStore
    from invalidate.adapters import Governor
    from invalidate.adapters.langgraph import LangGraphStoreAdapter, filter_items

    store = InMemoryStore()
    adapter = LangGraphStoreAdapter(store, ("user-1", "memories"), text_key="content")
    gov = Governor(adapter, "ledger.db", mode="flag")
    gov.sync()
    gov.observe("we migrated to SQLite last Tuesday", source="slack")
    live = filter_items(store.search(("user-1", "memories"), query="database"), gov)
    live = store.search(("user-1", "memories"), filter=adapter.live_filter())   # same, host-side

Verified against langgraph 1.x `BaseStore`:

* `store.search(namespace_prefix, limit=, offset=)` is a PREFIX search: items in child namespaces
  come back too. pull() keeps only items whose `namespace` equals the adapter's (set
  `include_children=True` to govern the whole subtree; keys are then `"a/b/key"`).
* Items expose `.key`, `.value` (dict) and `.namespace`. Text is `value[text_key]`; items lacking
  a string there are skipped.
* `store.put(namespace, key, value)` REPLACES the value, so flag() writes `{**value, **receipt}`.
* The `filter=` dialect supports `$eq/$ne/$gt/...` but not `$nin`, and `$ne` matches items that lack
  the key. So besides `reason.as_metadata()` the adapter writes one extra boolean,
  `invalidate_live`, and `live_filter()` is `{"invalidate_live": {"$ne": False}}`.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any

from .base import DEAD, Governor, HostMemory, Reason

_SEP = "/"


def live_filter(prefix: str = "invalidate_") -> dict[str, Any]:
    """A `store.search(..., filter=...)` predicate that excludes contradicted/superseded items."""
    return {f"{prefix}live": {"$ne": False}}


def filter_items(items: Iterable[Any], gov: Governor, *, include_review: bool = False) -> list[Any]:
    """Drop search results (Item/SearchItem, or anything with `.key`) whose memory is dead in the ledger."""
    adapter = gov.adapter
    key_of = getattr(adapter, "host_id_of", None) or (lambda it: it.key)
    return gov.filter(items, id_of=key_of, include_review=include_review)


class LangGraphStoreAdapter:
    """Governs one namespace of a LangGraph store. Host ids are item keys."""

    def __init__(
        self,
        store: Any,
        namespace: tuple[str, ...],
        text_key: str = "content",
        *,
        metadata_prefix: str = "invalidate_",
        name: str | None = None,
        page_size: int = 100,
        max_items: int = 5000,
        include_children: bool = False,
        source: str = "langgraph",
    ) -> None:
        self.store = store
        self.namespace = tuple(namespace)
        self.text_key = text_key
        self.prefix = metadata_prefix
        self.name = name or "langgraph:" + _SEP.join(self.namespace)
        self.page_size = page_size
        self.max_items = max_items
        self.include_children = include_children
        self.source = source

    # -- ids ---------------------------------------------------------------------------
    def host_id_of(self, item: Any) -> str:
        """The host id for a store item: its key, or `child/ns/key` when governing a subtree."""
        ns = tuple(getattr(item, "namespace", self.namespace))
        if self.include_children and ns != self.namespace:
            return _SEP.join((*ns[len(self.namespace):], item.key))
        return str(item.key)

    def _locate(self, host_id: str) -> tuple[tuple[str, ...], str]:
        if self.include_children and _SEP in host_id:
            *rest, key = host_id.split(_SEP)
            return (*self.namespace, *rest), key
        return self.namespace, host_id

    # -- Adapter protocol ------------------------------------------------------------
    def _search_all(self) -> list[Any]:
        found: list[Any] = []
        offset = 0
        while len(found) < self.max_items:
            batch = self.store.search(self.namespace, limit=self.page_size, offset=offset)
            if not batch:
                break
            for it in batch:
                if self.include_children or tuple(it.namespace) == self.namespace:
                    found.append(it)
            offset += len(batch)
            if len(batch) < self.page_size:
                break
        return found[: self.max_items]

    def pull(self) -> Iterable[HostMemory]:
        out: list[HostMemory] = []
        for it in self._search_all():
            value = it.value if isinstance(it.value, dict) else {}
            text = value.get(self.text_key)
            if not isinstance(text, str) or not text.strip():
                continue
            meta = {k: v for k, v in value.items() if k != self.text_key and not str(k).startswith(self.prefix)}
            out.append(
                HostMemory(
                    id=self.host_id_of(it),
                    text=text,
                    kind=str(value.get("kind", "fact")),
                    source=str(value.get("source", self.source)),
                    metadata=meta,
                )
            )
        return out

    def _receipt(self, reason: Reason) -> dict[str, Any]:
        return {**reason.as_metadata(self.prefix), f"{self.prefix}live": reason.status not in DEAD}

    def flag(self, host_id: str, reason: Reason) -> None:
        ns, key = self._locate(host_id)
        item = self.store.get(ns, key)
        if item is None:
            raise KeyError(f"{self.name}: no item {host_id!r}")
        self.store.put(ns, key, {**item.value, **self._receipt(reason)})

    def delete(self, host_id: str, reason: Reason) -> None:
        ns, key = self._locate(host_id)
        self.store.delete(ns, key)

    def insert(self, text: str, source: str, metadata: dict[str, Any]) -> str:
        key = uuid.uuid4().hex
        value = {self.text_key: text, "source": source, f"{self.prefix}status": "active",
                 f"{self.prefix}live": True, **metadata}
        self.store.put(self.namespace, key, value)
        return key

    # -- helpers ---------------------------------------------------------------------
    def live_filter(self) -> dict[str, Any]:
        return live_filter(self.prefix)

    def search(self, gov: Governor, **kw: Any) -> list[Any]:
        """`store.search(namespace, **kw)` with the live filter merged in, then pruned against the ledger."""
        kw["filter"] = {**(kw.get("filter") or {}), **self.live_filter()}
        return filter_items(self.store.search(self.namespace, **kw), gov)


__all__ = ["LangGraphStoreAdapter", "filter_items", "live_filter"]
