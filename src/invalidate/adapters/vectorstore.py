"""A generic adapter for any vector store, driven by four callables you supply.

pgvector, Pinecone, Qdrant, Weaviate, Milvus, a SQL table: if you can list rows, patch a row's
metadata, delete a row and (optionally) insert one, invalidate can govern it.

    from invalidate.adapters import Governor
    from invalidate.adapters.vectorstore import CallableVectorStoreAdapter, live_filter_metadata

    adapter = CallableVectorStoreAdapter(
        "pinecone",
        list_fn=lambda: index.list_all(),                      # -> iterable of dicts / objects
        update_metadata_fn=lambda id, patch: index.update(id=id, set_metadata=patch),  # MERGE semantics
        delete_fn=lambda id: index.delete(ids=[id]),
        insert_fn=lambda text, meta: index.upsert_text(text, metadata=meta),          # -> new id
    )
    gov = Governor(adapter, "ledger.db", mode="flag")
    gov.sync(); adapter.stamp_live()          # once; see stamp_live() for why
    gov.observe("we migrated to SQLite", source="slack")
    index.query(..., filter=live_filter_metadata())   # dead rows excluded at query time

The adapter never reads the host on flag/delete; it only pushes the `invalidate_*` receipt keys.
`update_metadata_fn` must therefore MERGE the patch into the row's existing metadata (most stores'
"set metadata" / "update" does; if yours replaces, wrap it with a read-merge-write).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .base import DEAD, HostMemory, Reason

DEAD_VALUES = sorted(s.value for s in DEAD)  # ["contradicted", "superseded"]

STATUS_KEY = "invalidate_status"


def live_filter_metadata(prefix: str = "invalidate_") -> dict[str, Any]:
    """The metadata predicate that hides dead rows: `{"invalidate_status": {"$nin": [dead...]}}`.

    Chroma, Pinecone, Qdrant (via their Mongo-style filter dialects) and most others accept this
    shape; translate it if yours differs. Caveat: some stores only match `$nin` against rows that
    HAVE the key, so rows that were never flagged (no `invalidate_status` at all) would vanish from
    results. If your store behaves that way, call `adapter.stamp_live()` once after the first
    `Governor.sync()` so every governed row carries `invalidate_status="active"`; new rows you insert
    afterwards should set it too (or call `stamp_live()` again after a sync).
    """
    return {f"{prefix}status": {"$nin": list(DEAD_VALUES)}}


class CallableVectorStoreAdapter:
    """Adapter over four user-supplied callables. See the module docstring.

    `list_fn() -> Iterable[record]` where each record is a dict or an object exposing `id_key`,
    `text_key` and (optionally) `metadata_key` as keys or attributes. Records without a text are
    skipped. `update_metadata_fn(id, patch)` merges `patch` (only `invalidate_*` keys) into the
    row. `delete_fn(id)` removes the row. `insert_fn(text, metadata) -> id` adds a row and
    returns the host id (omit it and the Governor will not write successors).
    """

    def __init__(
        self,
        name: str,
        list_fn: Callable[[], Iterable[Any]],
        update_metadata_fn: Callable[[str, dict[str, Any]], Any],
        delete_fn: Callable[[str], Any],
        insert_fn: Callable[[str, dict[str, Any]], Any] | None = None,
        *,
        text_key: str = "text",
        id_key: str = "id",
        metadata_key: str = "metadata",
        metadata_prefix: str = "invalidate_",
    ) -> None:
        self.name = name
        self._list = list_fn
        self._update = update_metadata_fn
        self._delete = delete_fn
        self._insert = insert_fn
        self.text_key = text_key
        self.id_key = id_key
        self.metadata_key = metadata_key
        self.prefix = metadata_prefix
        if insert_fn is None:
            # Governor probes `getattr(adapter, "insert", None)`; hide it when the host cannot insert.
            self.insert = None  # type: ignore[method-assign,assignment]

    # -- record access ---------------------------------------------------------------
    @staticmethod
    def _field(rec: Any, key: str, default: Any = None) -> Any:
        if isinstance(rec, dict):
            return rec.get(key, default)
        return getattr(rec, key, default)

    def _record(self, rec: Any) -> HostMemory | None:
        rid = self._field(rec, self.id_key)
        text = self._field(rec, self.text_key)
        if rid is None or not isinstance(text, str) or not text.strip():
            return None
        meta = self._field(rec, self.metadata_key) or {}
        if not isinstance(meta, dict):
            meta = dict(meta) if hasattr(meta, "keys") else {}
        return HostMemory(
            id=str(rid),
            text=text,
            kind=str(meta.get("kind", "fact")),
            source=str(meta.get("source", self.name)),
            metadata={k: v for k, v in meta.items() if not str(k).startswith(self.prefix)},
        )

    # -- Adapter protocol ------------------------------------------------------------
    def pull(self) -> Iterable[HostMemory]:
        """Read-only: lists the host and yields verbatim text with the host's ids."""
        out: list[HostMemory] = []
        for rec in self._list():
            hm = self._record(rec)
            if hm is not None:
                out.append(hm)
        return out

    def flag(self, host_id: str, reason: Reason) -> None:
        self._update(host_id, reason.as_metadata(self.prefix))

    def delete(self, host_id: str, reason: Reason) -> None:
        self._delete(host_id)

    def insert(self, text: str, source: str, metadata: dict[str, Any]) -> str | None:
        if self._insert is None:
            return None
        meta = {"source": source, f"{self.prefix}status": "active", **metadata}
        new_id = self._insert(text, meta)
        return None if new_id is None else str(new_id)

    # -- helpers ---------------------------------------------------------------------
    def live_filter(self) -> dict[str, Any]:
        return live_filter_metadata(self.prefix)

    def stamp_live(self, ids: Iterable[str] | None = None) -> list[str]:
        """Write `invalidate_status="active"` onto rows that have no status yet. Returns the ids stamped.

        Not part of sync (pull is read-only). Call it once after the first `Governor.sync()` when your
        store's `$nin` does not match rows lacking the key; see `live_filter_metadata()`. With `ids`
        given, stamps exactly those (no read); otherwise lists the host and stamps the unflagged rows.
        """
        key = f"{self.prefix}status"
        patch = {key: "active"}
        if ids is not None:
            stamped = [str(i) for i in ids]
        else:
            stamped = []
            for rec in self._list():
                rid = self._field(rec, self.id_key)
                meta = self._field(rec, self.metadata_key) or {}
                if rid is not None and not (isinstance(meta, dict) and meta.get(key)):
                    stamped.append(str(rid))
        for rid in stamped:
            self._update(rid, dict(patch))
        return stamped


__all__ = ["CallableVectorStoreAdapter", "live_filter_metadata", "DEAD_VALUES", "STATUS_KEY"]
