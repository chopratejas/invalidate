"""Adapter for one Qdrant collection (`qdrant-client`).

    from qdrant_client import QdrantClient
    from invalidate.adapters import Governor
    from invalidate.adapters.qdrant import QdrantAdapter, governed_query

    client = QdrantClient("http://localhost:6333")            # or QdrantClient(":memory:")
    adapter = QdrantAdapter(client, "memories", text_key="text", vector_fn=embed)
    gov = Governor(adapter, "ledger.db", mode="flag", successors=True)
    gov.sync()
    gov.observe("we migrated to SQLite last Tuesday", source="slack")
    hits = governed_query(client, gov, "memories", embed("which database?"), limit=5)   # dead points excluded

Nothing is imported from the SDK at module import time; `qdrant_client.models` is loaded on first use.
Signatures mirrored (qdrant-client 1.19.1, paths under site-packages/qdrant_client/):

  qdrant_client.py:705   QdrantClient.scroll(collection_name, scroll_filter=None, limit=10, order_by=None, offset=None,
                         with_payload=True, with_vectors=False, ...) -> (list[Record], next_offset | None)
                         Pages in ascending id order; pass the returned offset back in until it is None.
  qdrant_client.py:1189  QdrantClient.set_payload(collection_name, payload, points, key=None, wait=True, ...)
                         MERGES top-level keys into the existing payload (local/local_collection.py:3061:
                         `self.payload[idx] = {**self.payload[idx], **jsonable_payload}`). Unrelated keys survive,
                         a same-named key is replaced (no deep merge of nested dicts). Verified by
                         tests/test_adapter_qdrant.py. `overwrite_payload` (qdrant_client.py:1291) is the
                         replacing variant and is never used here. A missing id raises KeyError in :memory:
                         mode and an UnexpectedResponse (404) against a server; the Governor records it as a push error.
  qdrant_client.py:1136  QdrantClient.delete(collection_name, points_selector, wait=True, ...)
                         with `models.PointIdsList(points=[id])` (http/models/models.py:2303). Unknown ids are no-ops.
  qdrant_client.py:867   QdrantClient.upsert(collection_name, points, wait=True, ...)
                         with `models.PointStruct(id, vector, payload)` (http/models/models.py:2320). Ids are
                         unsigned ints or UUID strings (`ExtendedPointId`, models.py:4143); "pg" is rejected.
  qdrant_client.py:269   QdrantClient.query_points(collection_name, query=None, using=None, prefetch=None,
                         query_filter=None, limit=10, offset=None, with_payload=True, ...) -> QueryResponse(points=[ScoredPoint])
                         There is no `QdrantClient.search` in 1.19.1 (removed after 1.12); query_points is the API.
  http/models/models.py:1012  Filter(should, min_should, must, must_not); a Filter is itself a Condition (models.py:4120),
                         so filters compose by nesting.
  http/models/models.py:988   FieldCondition(key, match=MatchValue(value) | MatchAny(any=[...]) | MatchExcept(except=[...]) ...)
  http/models/models.py:1612  IsEmptyCondition(is_empty=PayloadField(key)) matches points with no such key (or an empty
                         list/null); IsNullCondition (models.py:1620) matches only an explicit null.

The live filter is `Filter(must_not=[FieldCondition(key="invalidate_status", match=MatchAny(any=[dead...]))])`.
Verified: `must_not MatchAny` keeps points that lack the key entirely, so unflagged points need no stamp.
`must MatchExcept` does NOT (it matches only points that have the key with another value), so it is not used.
"""
from __future__ import annotations

import uuid
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .base import DEAD, Governor, HostMemory, Reason

DEAD_VALUES = sorted(s.value for s in DEAD)  # ["contradicted", "superseded"]


def _models() -> Any:
    from qdrant_client import models  # lazy: importing invalidate.adapters must not need qdrant-client

    return models


def live_filter(prefix: str = "invalidate_") -> Any:
    """A Qdrant `Filter` that excludes contradicted/superseded points and keeps every unflagged one."""
    m = _models()
    return m.Filter(must_not=[m.FieldCondition(key=f"{prefix}status", match=m.MatchAny(any=list(DEAD_VALUES)))])


def merge_filter(base: Any | None, extra: Any) -> Any:
    """AND two filters. A `Filter` is a valid `Condition`, so this nests rather than rewriting either side."""
    if base is None:
        return extra
    return _models().Filter(must=[base, extra])


def _point_id(host_id: str) -> int | str:
    """Host ids are strings; Qdrant wants the original unsigned int or UUID string back."""
    return int(host_id) if host_id.isdigit() else host_id


class QdrantAdapter:
    """Governs one Qdrant collection. Ids are the point ids (as strings); receipts land in the payload.

    `text_key` is the payload key holding the memory text; points without a non-empty string there are
    skipped. `scope_filter` (a `models.Filter`) limits pull to a slice of the collection, e.g. one user.
    `vector_fn(text) -> list[float]` (or `{name: list[float]}` for named vectors) is needed only for
    successor inserts; without it the adapter exposes no `insert` and the Governor skips successors.
    `scope_payload` is merged into every inserted point so successors stay inside `scope_filter`.
    """

    def __init__(
        self,
        client: Any,
        collection: str,
        *,
        text_key: str = "text",
        scope_filter: Any | None = None,
        vector_fn: Callable[[str], Sequence[float] | dict[str, Sequence[float]]] | None = None,
        vector_name: str | None = None,
        scope_payload: dict[str, Any] | None = None,
        metadata_prefix: str = "invalidate_",
        name: str | None = None,
        page_size: int = 256,
        max_rows: int = 5000,
        source: str = "qdrant",
    ) -> None:
        self.client = client
        self.collection = collection
        self.text_key = text_key
        self.scope_filter = scope_filter
        self.vector_fn = vector_fn
        self.vector_name = vector_name
        self.scope_payload = dict(scope_payload or {})
        self.prefix = metadata_prefix
        self.name = name or f"qdrant:{collection}"
        self.page_size = page_size
        self.max_rows = max_rows
        self.source = source
        if vector_fn is None:
            # Governor probes `getattr(adapter, "insert", None)`; hide it when we cannot embed.
            self.insert = None  # type: ignore[method-assign,assignment]

    # -- Adapter protocol ------------------------------------------------------------
    def pull(self) -> Iterable[HostMemory]:
        out: list[HostMemory] = []
        offset = None
        while len(out) < self.max_rows:
            limit = min(self.page_size, self.max_rows - len(out))
            records, offset = self.client.scroll(
                self.collection, scroll_filter=self.scope_filter, limit=limit, offset=offset,
                with_payload=True, with_vectors=False,
            )
            for rec in records:
                hm = self._host_memory(rec)
                if hm is not None:
                    out.append(hm)
            if offset is None:
                break
        else:
            warnings.warn(
                f"{self.name}: pulled {self.max_rows} points; more may remain "
                f"(raise max_rows= on QdrantAdapter to govern more)",
                stacklevel=2,
            )
        return out

    def _host_memory(self, rec: Any) -> HostMemory | None:
        payload = dict(getattr(rec, "payload", None) or {})
        text = payload.get(self.text_key)
        if not isinstance(text, str) or not text.strip():
            return None
        return HostMemory(
            id=str(rec.id),
            text=text,
            kind=str(payload.get("kind", "fact")),
            source=str(payload.get("source", self.source)),
            metadata={k: v for k, v in payload.items() if k != self.text_key and not str(k).startswith(self.prefix)},
        )

    def flag(self, host_id: str, reason: Reason) -> None:
        # set_payload merges top-level keys (see module docstring); the text and other keys stay put.
        self.client.set_payload(self.collection, payload=reason.as_metadata(self.prefix), points=[_point_id(host_id)])

    def delete(self, host_id: str, reason: Reason) -> None:
        self.client.delete(self.collection, points_selector=_models().PointIdsList(points=[_point_id(host_id)]))

    def insert(self, text: str, source: str, metadata: dict[str, Any]) -> str:
        if self.vector_fn is None:
            raise RuntimeError(f"{self.name}: insert needs vector_fn= to embed the successor text")
        vector: Any = self.vector_fn(text)
        if not isinstance(vector, dict):
            vector = list(vector)
            if self.vector_name:
                vector = {self.vector_name: vector}
        new_id = str(uuid.uuid4())
        payload = {**self.scope_payload, self.text_key: text, "source": source, f"{self.prefix}status": "active", **metadata}
        self.client.upsert(self.collection, points=[_models().PointStruct(id=new_id, vector=vector, payload=payload)])
        return new_id

    # -- helpers ---------------------------------------------------------------------
    def live_filter(self) -> Any:
        return live_filter(self.prefix)

    def scoped_live_filter(self) -> Any:
        """`scope_filter` AND `live_filter()`: what a query inside this adapter's scope should use."""
        return merge_filter(self.scope_filter, self.live_filter())


def governed_query(
    client: Any,
    gov: Governor,
    collection: str,
    query_vector: Any,
    *,
    query_filter: Any | None = None,
    **query_kwargs: Any,
) -> Any:
    """`client.query_points(...)` with the live filter ANDed into `query_filter`, then pruned against the ledger.

    Returns the `QueryResponse` with dead points removed from `.points`. The filter hides points the adapter
    has flagged in the host; the ledger prune also hides points the Governor knows are dead but has not
    written to the host (mode="ledger", or a push that errored). `using=`, `limit=`, `with_payload=`, ... pass through.
    """
    prefix = getattr(gov.adapter, "prefix", "invalidate_")
    response = client.query_points(
        collection, query=query_vector, query_filter=merge_filter(query_filter, live_filter(prefix)), **query_kwargs
    )
    dead = gov.dead_ids()
    if dead:
        response.points = [p for p in response.points if str(p.id) not in dead]
    return response


__all__ = ["QdrantAdapter", "governed_query", "live_filter", "merge_filter", "DEAD_VALUES"]
