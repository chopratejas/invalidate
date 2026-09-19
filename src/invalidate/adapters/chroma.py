"""Adapter for a chromadb collection.

    import chromadb
    from invalidate.adapters import Governor
    from invalidate.adapters.chroma import ChromaAdapter, governed_query

    col = chromadb.PersistentClient("./chroma").get_or_create_collection("memories")
    gov = Governor(ChromaAdapter(col), "ledger.db", mode="flag")
    gov.sync()
    gov.observe("we migrated to SQLite last Tuesday", source="slack")
    governed_query(col, gov, query_texts=["which database?"], n_results=5)   # dead rows excluded

Semantics verified against chromadb 1.5.9 (see tests/test_adapters_vector.py):

* `collection.update(ids, metadatas)` MERGES the given keys into the existing metadata. Older
  releases replaced the whole dict, so `flag()` reads the current metadata and writes the merged
  dict either way; the result is the same on both.
* `where={"invalidate_status": {"$nin": [...]}}` DOES match documents that lack the key (and
  documents with no metadata at all), so unflagged rows stay visible without a stamp. If you are on
  a release where that is not the case, call `adapter.stamp_live()` once after the first
  `Governor.sync()`; it writes `invalidate_status="active"` onto every row missing it.
* Metadata values must be scalars; `metadatas=[{}]` is rejected on add. Lists (e.g. the successor's
  `invalidate_supersedes`) are stored comma-joined; empty metadata is passed as None.
* `collection.get(limit, offset)` pages in insertion order; pull() pages in chunks of `page_size`
  and stops at `max_rows` (default 5000) with a warning.
"""
from __future__ import annotations

import json
import uuid
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .base import DEAD, Governor, HostMemory, Reason

DEAD_VALUES = sorted(s.value for s in DEAD)
_SCALARS = (str, int, float, bool)


def _clean_metadata(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    """Coerce to what Chroma stores: scalar values only, and None instead of an empty dict."""
    out: dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, _SCALARS):
            out[k] = v
        elif isinstance(v, (list, tuple, set, frozenset)):
            out[k] = ",".join(str(x) for x in v)
        elif isinstance(v, dict):
            out[k] = json.dumps(v, default=str)
        else:
            out[k] = str(v)
    return out or None


def live_where(prefix: str = "invalidate_") -> dict[str, Any]:
    """A Chroma `where` filter that excludes contradicted/superseded rows."""
    return {f"{prefix}status": {"$nin": list(DEAD_VALUES)}}


def merge_where(where: dict[str, Any] | None, extra: dict[str, Any]) -> dict[str, Any]:
    """AND `extra` into an existing `where` (Chroma wants a single top-level operator)."""
    if not where:
        return extra
    if list(where.keys()) == ["$and"]:
        return {"$and": [*where["$and"], extra]}
    return {"$and": [where, extra]}


class ChromaAdapter:
    """Governs one chromadb collection. Ids are Chroma ids; receipts land in the row's metadata.

    `embed(text) -> vector` is only needed for successor inserts into a collection created with
    `embedding_function=None` (Chroma then refuses `add` without explicit embeddings)."""

    def __init__(
        self,
        collection: Any,
        metadata_prefix: str = "invalidate_",
        *,
        name: str | None = None,
        page_size: int = 500,
        max_rows: int = 5000,
        source: str = "chroma",
        embed: Callable[[str], Sequence[float]] | None = None,
    ) -> None:
        self.collection = collection
        self.embed = embed
        self.prefix = metadata_prefix
        self.name = name or f"chroma:{getattr(collection, 'name', 'collection')}"
        self.page_size = page_size
        self.max_rows = max_rows
        self.source = source

    # -- Adapter protocol ------------------------------------------------------------
    def pull(self) -> Iterable[HostMemory]:
        out: list[HostMemory] = []
        offset = 0
        total = None
        try:
            total = int(self.collection.count())
        except Exception:  # noqa: BLE001 - count is a nicety; paging works without it
            pass
        if total is not None and total > self.max_rows:
            warnings.warn(
                f"{self.name}: collection has {total} rows; pulling only the first {self.max_rows} "
                f"(raise max_rows= on ChromaAdapter to govern more)",
                stacklevel=2,
            )
        while offset < self.max_rows:
            limit = min(self.page_size, self.max_rows - offset)
            res = self.collection.get(include=["documents", "metadatas"], limit=limit, offset=offset)
            ids = res.get("ids") or []
            if not ids:
                break
            docs = res.get("documents") or [None] * len(ids)
            metas = res.get("metadatas") or [None] * len(ids)
            for rid, doc, meta in zip(ids, docs, metas):
                if not isinstance(doc, str) or not doc.strip():
                    continue
                meta = dict(meta or {})
                out.append(
                    HostMemory(
                        id=str(rid),
                        text=doc,
                        kind=str(meta.get("kind", "fact")),
                        source=str(meta.get("source", self.source)),
                        metadata={k: v for k, v in meta.items() if not str(k).startswith(self.prefix)},
                    )
                )
            offset += len(ids)
            if len(ids) < limit:
                break
        return out

    def _current_metadata(self, host_id: str) -> dict[str, Any]:
        res = self.collection.get(ids=[host_id], include=["metadatas"])
        metas = res.get("metadatas") or []
        return dict(metas[0] or {}) if metas else {}

    def flag(self, host_id: str, reason: Reason) -> None:
        merged = {**self._current_metadata(host_id), **reason.as_metadata(self.prefix)}
        self.collection.update(ids=[host_id], metadatas=[_clean_metadata(merged)])

    def delete(self, host_id: str, reason: Reason) -> None:
        self.collection.delete(ids=[host_id])

    def insert(self, text: str, source: str, metadata: dict[str, Any]) -> str:
        new_id = uuid.uuid4().hex
        meta = _clean_metadata({"source": source, f"{self.prefix}status": "active", **metadata})
        kw: dict[str, Any] = {"ids": [new_id], "documents": [text], "metadatas": [meta] if meta else None}
        if self.embed is not None:
            kw["embeddings"] = [list(self.embed(text))]
        self.collection.add(**kw)
        return new_id

    # -- helpers ---------------------------------------------------------------------
    def live_where(self) -> dict[str, Any]:
        return live_where(self.prefix)

    def stamp_live(self, ids: Iterable[str] | None = None) -> list[str]:
        """Write `invalidate_status="active"` onto rows lacking a status. Returns the ids stamped.

        Not needed on chromadb 1.5.x (`$nin` already matches rows without the key). Call once after
        the first `Governor.sync()` on releases where it does not.
        """
        key = f"{self.prefix}status"
        if ids is None:
            ids = [hm.id for hm in self.pull()]
        todo: list[str] = []
        for rid in ids:
            if not self._current_metadata(rid).get(key):
                todo.append(str(rid))
        for rid in todo:
            self.collection.update(ids=[rid], metadatas=[{**self._current_metadata(rid), key: "active"}])
        return todo


def _prune_query_result(result: dict[str, Any], dead: set[str]) -> dict[str, Any]:
    """Drop columns whose id is dead from every per-query list in a Chroma QueryResult."""
    if not dead:
        return result
    ids = result.get("ids") or []
    keep = [[j for j, rid in enumerate(group) if str(rid) not in dead] for group in ids]
    out = dict(result)
    for field, val in result.items():
        if isinstance(val, list) and len(val) == len(ids) and all(isinstance(g, (list, tuple)) or g is None for g in val):
            try:
                out[field] = [None if g is None else [g[j] for j in keep[i]] for i, g in enumerate(val)]
            except (IndexError, TypeError):
                out[field] = val
    return out


def _annotate_query_result(result: dict[str, Any], gov: Governor) -> dict[str, Any]:
    """Add an `invalidate_notes` column (one list per query, aligned with `ids`) to a Chroma QueryResult."""
    out = dict(result)
    out["invalidate_notes"] = [
        None if group is None else [n for _, n in gov.annotate(group, id_of=str)] for group in (result.get("ids") or [])
    ]
    return out


def governed_query(collection: Any, gov: Governor, *, annotate: bool = False, **query_kwargs: Any) -> dict[str, Any]:
    """`collection.query(...)` with the live filter merged into `where`, then pruned against the ledger.

    The `where` clause hides rows the adapter has flagged in the host; the ledger prune also hides rows
    the Governor knows are dead but has not written to the host (mode="ledger", or a push that errored).

    `annotate=True` runs the query without the live filter, keeps every row, and adds an
    `"invalidate_notes"` column to the QueryResult: one list per query aligned with `ids`, holding
    `Governor.annotate`'s label for retired rows and None for live ones. A column rather than a key in
    `metadatas`, which is absent unless `include` asks for it and is None for rows stored without metadata.
    """
    if annotate:
        return _annotate_query_result(collection.query(**query_kwargs), gov)
    prefix = getattr(gov.adapter, "prefix", "invalidate_")
    query_kwargs["where"] = merge_where(query_kwargs.get("where"), live_where(prefix))
    result = collection.query(**query_kwargs)
    return _prune_query_result(result, gov.dead_ids())


__all__ = ["ChromaAdapter", "governed_query", "live_where", "merge_where", "DEAD_VALUES"]
