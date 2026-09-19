"""Cognee adapter: govern the data items of one Cognee dataset.

    import cognee
    from invalidate.adapters import Governor
    from invalidate.adapters.cognee import CogneeAdapter, governed_search

    adapter = CogneeAdapter(cognee, dataset_name="alice")            # or dataset_id=UUID(...)
    gov = Governor(adapter, "ledger.db", mode="ledger")
    gov.sync()                                                       # every Data row -> ledger, verbatim
    gov.observe("we migrated to SQLite", source="slack")             # verdicts land in the ledger
    hits = governed_search(cognee, gov, "which database?", query_type=cognee.SearchType.CHUNKS)

Cognee is async; this adapter is sync (the `Adapter` protocol is) and runs each coroutine with `_run`:
`asyncio.run` when no loop is running in this thread, otherwise a fresh loop on a helper thread. Nothing
from the cognee package is imported at module import time; you pass the `cognee` module (or any object
with the same attributes) and the adapter detects what it offers from signatures. `DataItem` is imported
lazily inside `insert` (or injected as `data_item_class=`).

The unit of memory is one **Data row** of one dataset: what `cognee.add("...")` stores before `cognify()`
runs. Its id (`Data.id`, a UUID) is stable across cognify and is the id `cognee.delete` takes; its text is
the file at `raw_data_location`, which for raw text is written verbatim (`text_<md5>.txt`). Chunks,
entities and summaries that cognify derives from a row are Cognee's own wording and are NOT governed
individually: they cannot be deleted one by one, but they carry `document_id == str(Data.id)`, which is
how `governed_search` hides them once the row is dead.

Mirrored (cognee 1.6.0, github.com/topoteretes/cognee main @ 663a2dc, 2026-09-19; same signatures in the
PyPI 1.6.0 wheel):

  cognee/api/v1/datasets/datasets.py:127  datasets.list_datasets(user=None) -> list[Dataset]  (.id, .name)
  cognee/api/v1/datasets/datasets.py:138  datasets.list_data(dataset_id, user=None) -> list[Data]
                                          (.id UUID, .name, .raw_data_location, .mime_type, .extension,
                                          .external_metadata dict|None, .label, .dataset_id; a DataDTO with
                                          the same attributes after cognee.serve())
  cognee/api/v1/datasets/datasets.py:217  datasets.delete_data(dataset_id, data_id, user=None, mode="soft",
                                          delete_dataset_if_empty=False) -> {"status": "success"}; drops the
                                          row AND its graph/vector nodes (delete_data_nodes_and_edges)
  cognee/api/v1/delete/__init__.py:9      cognee.delete(data_id, dataset_id, mode="soft", user=None):
                                          @deprecated since 0.3.9, forwards to datasets.delete_data
  cognee/api/v1/add/add.py:35             cognee.add(data, dataset_name="main_dataset", user=None,
                                          node_set=None, ..., dataset_id=None, ...) -> PipelineRunInfo
                                          (.status, .dataset_id, .pipeline_run_id). "add() stages data and
                                          makes no LLM call of its own" (add.py:230); text is stored verbatim
                                          (save_data_to_file.py:78, LocalFileStorage.store writes utf-8)
  cognee/tasks/ingestion/data_item.py:14  DataItem(data, label=None, external_metadata=None,
                                          system_metadata=None, data_id=None); a pinned data_id becomes the
                                          row id (ingest_data.py:277-283, 484-485) and skips content dedup
  cognee/modules/data/models/Data.py      Data.external_metadata: JSON, "the user's free-form field"
  cognee/api/v1/update/update.py:28       cognee.update(data_id, data, dataset_id, ...) replaces CONTENT and
                                          re-cognifies; a DataItem with external_metadata forces the full
                                          rebuild (update.py:364-370). Not a metadata update: never used here.
  cognee/api/v1/search/search.py:41       cognee.search(query_text, query_type=SearchType.HYBRID_COMPLETION,
                                          user=None, datasets=None, dataset_ids=None, top_k=15, ...) ->
                                          with backend access control (the default when the graph/vector
                                          DBs support it): [{"dataset_id", "dataset_name", "dataset_tenant_id",
                                          "search_result": <result>}]; otherwise the bare <result>
                                          (modules/search/methods/search.py:632-700)
  cognee/modules/retrieval/chunks_retriever.py:57   CHUNKS <result> = [{**chunk.payload, "score"}], payload =
                                          DocumentChunk fields incl. "id", "text", "document_id", "document_name"
  cognee/modules/chunking/TextChunker.py:17         document_id = str(self.document.id)
  cognee/tasks/documents/classify_documents.py:169  Document(id=data_item.id, ...)   -> document_id == Data.id
  cognee/modules/retrieval/summaries_retriever.py:114  SUMMARIES payloads carry "id"/"text" but no document id
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import threading
import uuid
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, TypeVar
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from .base import Governor, HostMemory, Reason

T = TypeVar("T")

TEXT_MIME_PREFIXES = ("text/",)
TEXT_EXTENSIONS = frozenset({"txt", "md", "markdown", "csv", "json", "yaml", "yml", "rst", "html", "htm"})


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

    t = threading.Thread(target=target, name="invalidate-cognee", daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def _call(fn: Callable[..., Any], *args: Any, **kw: Any) -> Any:
    """Call `fn`; if it hands back an awaitable (every real cognee entry point does), run it to completion."""
    res = fn(*args, **kw)
    return _run(res) if inspect.isawaitable(res) else res


def _params(fn: Callable[..., Any]) -> tuple[set[str], bool]:
    """(named parameters, accepts **kwargs). Empty/False when the callable cannot be introspected."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return set(), False
    named = {p.name for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    var_kw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    return named, var_kw


def _accepts(fn: Callable[..., Any], param: str) -> bool:
    named, var_kw = _params(fn)
    return param in named or var_kw


def _uuid(value: Any) -> Any:
    """Cognee's ids are `uuid.UUID`; host ids in the ledger are strings. Convert when possible."""
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return value


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Rows come back as SQLAlchemy models locally and as DataDTO after serve(); dicts from a fake."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _local_path(location: str) -> str | None:
    """Filesystem path for a `file://` URI or a plain path; None for other schemes (s3://, http://)."""
    parsed = urlparse(location)
    if parsed.scheme == "file":
        return url2pathname(unquote(parsed.path))
    if parsed.scheme in ("", None) or (len(parsed.scheme) == 1):  # no scheme, or a Windows drive letter
        return location
    return None


def _hostable(value: Any) -> Any:
    """`external_metadata` is a JSON column: keep receipts to JSON scalars/lists."""
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _hostable(v) for k, v in value.items()}
    return value


async def relational_metadata_writer(data_id: Any, metadata: dict[str, Any]) -> None:
    """Merge `metadata` into `Data.external_metadata` for one row, through Cognee's relational engine.

    This is NOT a public Cognee API: it is the same session-merge-commit that
    `cognee/modules/data/methods/publish_updated_data.py:96-123` performs internally. It is offered for
    `CogneeAdapter(flag_in_host=True)` because Cognee has no public per-item metadata update; the receipt
    becomes visible through `datasets.list_data(...)[i].external_metadata` (and on the Document node after
    the next cognify) but does not change search results. Use at your own risk on new Cognee releases."""
    from sqlalchemy import select  # noqa: PLC0415

    from cognee.infrastructure.databases.relational import get_relational_engine  # noqa: PLC0415
    from cognee.modules.data.models import Data  # noqa: PLC0415

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        row = (await session.execute(select(Data).filter(Data.id == _uuid(data_id)))).scalar_one_or_none()
        if row is None:
            raise KeyError(f"cognee Data row {data_id} not found")
        row.external_metadata = {**(row.external_metadata or {}), **metadata}
        await session.merge(row)
        await session.commit()


class CogneeAdapter:
    """Data rows of one dataset, one row (its verbatim file) = one memory. Cognee keeps its rows and its
    graph; invalidate keeps the ledger.

    pull   -> `datasets.list_data(dataset_id)`; text-typed rows only (`mime_type` text/* or a text
              extension) unless `include_non_text=True`; id = `str(Data.id)`, text = the stored file.
    flag   -> ledger-only by default (`flagged_in_host` is False): Cognee has no public per-item metadata
              update, and its `update()` rewrites content. Pass `flag_in_host=True` to merge the receipt
              into `Data.external_metadata` through the internal relational session, or
              `metadata_writer=` to supply your own `(data_id, receipt) -> None|Awaitable`.
    delete -> `datasets.delete_data(dataset_id=..., data_id=...)` (falls back to the deprecated
              `cognee.delete(data_id=..., dataset_id=...)`). Removes the row and everything cognify derived
              from it; `mode="soft"` is Cognee's default and the only safe one.
    insert -> `cognee.add(DataItem(data=text, data_id=<new uuid>, external_metadata={...}),
              dataset_id=...)`: the successor is stored VERBATIM as a new row with an id known up front,
              and NO cognify runs (so no LLM call). It joins the graph on your next `cognee.cognify()`.
              Without a `DataItem` class the row id is recovered by diffing `list_data` before and after.

    The supported end-to-end path is `Governor(..., mode="ledger")` (or `mode="flag"` with successors) plus
    `governed_search(...)` on the read side; `mode="delete"` is the only mode that changes what Cognee's
    own completions see.
    """

    name = "cognee"

    def __init__(
        self,
        client: Any,
        *,
        dataset_id: Any = None,
        dataset_name: str | None = None,
        user: Any = None,
        metadata_prefix: str = "invalidate_",
        flag_in_host: bool = False,
        metadata_writer: Callable[[Any, dict[str, Any]], Any] | None = None,
        data_item_class: Any = None,
        read_text: Callable[[str], str] | None = None,
        include_non_text: bool = False,
        insert_node_set: list[str] | None = None,
        encoding: str = "utf-8",
    ) -> None:
        self.client = client
        self.user = user
        self.prefix = metadata_prefix
        self.include_non_text = include_non_text
        self.insert_node_set = list(insert_node_set) if insert_node_set else None
        self.encoding = encoding
        self._read_text = read_text
        self._data_item_class = data_item_class
        self.datasets = getattr(client, "datasets", None)
        if self.datasets is None or not hasattr(self.datasets, "list_data"):
            raise ValueError("CogneeAdapter needs a client with `datasets.list_data` (pass the `cognee` module)")
        if dataset_id is None and not dataset_name:
            raise ValueError("CogneeAdapter needs dataset_id= or dataset_name=")
        self.dataset_id = _uuid(dataset_id) if dataset_id is not None else self._resolve_dataset(dataset_name)  # type: ignore[arg-type]
        self.dataset_name = dataset_name
        if metadata_writer is not None:
            self._metadata_writer: Callable[[Any, dict[str, Any]], Any] | None = metadata_writer
        elif flag_in_host:
            self._metadata_writer = relational_metadata_writer
        else:
            self._metadata_writer = None
        self._known_ids: set[str] = set()

    @property
    def flagged_in_host(self) -> bool:
        return self._metadata_writer is not None

    # -- calling conventions ---------------------------------------------------------
    def _user_kw(self, fn: Callable[..., Any]) -> dict[str, Any]:
        return {"user": self.user} if self.user is not None and _accepts(fn, "user") else {}

    def _resolve_dataset(self, name: str) -> Any:
        list_datasets = getattr(self.datasets, "list_datasets", None)
        if list_datasets is None:
            raise ValueError("this cognee has no `datasets.list_datasets`; pass dataset_id= instead")
        for ds in _call(list_datasets, **self._user_kw(list_datasets)) or []:
            if _attr(ds, "name") == name:
                return _uuid(_attr(ds, "id"))
        raise ValueError(f"cognee dataset {name!r} not found; `cognee.add(..., dataset_name={name!r})` creates it")

    @property
    def DataItem(self) -> Any:  # noqa: N802 - mirrors the SDK name
        if self._data_item_class is None:
            try:
                from cognee.tasks.ingestion.data_item import DataItem  # noqa: PLC0415
            except ImportError:
                return None
            self._data_item_class = DataItem
        return self._data_item_class

    # -- text ------------------------------------------------------------------------
    def _is_text(self, row: Any) -> bool:
        if self.include_non_text:
            return True
        mime = str(_attr(row, "mime_type", "") or "")
        ext = str(_attr(row, "extension", "") or "").lstrip(".").lower()
        return mime.startswith(TEXT_MIME_PREFIXES) or ext in TEXT_EXTENSIONS

    def read_text(self, location: str) -> str:
        """The stored content at `raw_data_location`. Local files are read directly (verbatim, utf-8);
        other schemes go through cognee's `open_data_file`."""
        if self._read_text is not None:
            return self._read_text(location)
        path = _local_path(location)
        if path is not None:
            with open(path, encoding=self.encoding, errors="replace", newline="") as f:
                return f.read()
        from cognee.infrastructure.files.utils.open_data_file import open_data_file  # noqa: PLC0415

        async def _read() -> str:
            async with open_data_file(location, mode="rb") as f:
                data = f.read()
                if inspect.isawaitable(data):
                    data = await data
            return data.decode(self.encoding, errors="replace") if isinstance(data, bytes) else str(data)

        return _run(_read())

    def _list_rows(self) -> list[Any]:
        list_data = self.datasets.list_data
        return list(_call(list_data, self.dataset_id, **self._user_kw(list_data)) or [])

    # -- Adapter ----------------------------------------------------------------------
    def pull(self) -> Iterable[HostMemory]:
        out: list[HostMemory] = []
        self._known_ids = set()
        for row in self._list_rows():
            rid = _attr(row, "id")
            if rid is None:
                continue
            self._known_ids.add(str(rid))
            if not self._is_text(row):
                continue
            location = _attr(row, "raw_data_location")
            if not location:
                continue
            text = self.read_text(str(location))
            meta: dict[str, Any] = {"name": _attr(row, "name", ""), "mime_type": _attr(row, "mime_type", "")}
            label = _attr(row, "label")
            if label:
                meta["label"] = label
            out.append(HostMemory(id=str(rid), text=text, source="cognee", metadata=meta))
        return out

    def flag(self, host_id: str, reason: Reason) -> None:
        if self._metadata_writer is None:
            return  # documented limit: no public per-item metadata update in Cognee; the ledger holds it
        receipt = {k: _hostable(v) for k, v in reason.as_metadata(self.prefix).items()}
        _call(self._metadata_writer, _uuid(host_id), receipt)

    def delete(self, host_id: str, reason: Reason) -> None:
        delete_data = getattr(self.datasets, "delete_data", None)
        if delete_data is not None:
            _call(delete_data, dataset_id=self.dataset_id, data_id=_uuid(host_id), **self._user_kw(delete_data))
        else:
            legacy = self.client.delete  # cognee.delete(data_id, dataset_id, mode="soft", user=None), deprecated
            _call(legacy, data_id=_uuid(host_id), dataset_id=self.dataset_id, **self._user_kw(legacy))
        self._known_ids.discard(str(host_id))

    def insert(self, text: str, source: str, metadata: dict[str, Any]) -> str | None:
        add = getattr(self.client, "add", None)
        if add is None:
            raise NotImplementedError("this cognee client has no `add`; successors cannot be inserted")
        meta: dict[str, Any] = {f"{self.prefix}source": source}
        for k, v in metadata.items():
            key = k if k.startswith(self.prefix) else f"{self.prefix}{k}"
            meta[key] = _hostable(v)
        kw: dict[str, Any] = {}
        if _accepts(add, "dataset_id"):
            kw["dataset_id"] = self.dataset_id
        elif self.dataset_name and _accepts(add, "dataset_name"):
            kw["dataset_name"] = self.dataset_name
        else:
            raise NotImplementedError("cognee.add accepts neither dataset_id= nor dataset_name=; cannot target the dataset")
        if self.insert_node_set and _accepts(add, "node_set"):
            kw["node_set"] = self.insert_node_set
        kw.update(self._user_kw(add))

        item_cls = self.DataItem
        if item_cls is not None:
            new_id = uuid.uuid4()
            res = _call(add, item_cls(data=text, external_metadata=meta, data_id=new_id), **kw)
            self._check_run(res)
            self._known_ids.add(str(new_id))
            return str(new_id)

        # No DataItem (older cognee): plain add, then recover the row id by diffing the dataset.
        before = {str(_attr(r, "id")) for r in self._list_rows()}
        res = _call(add, text, **kw)
        self._check_run(res)
        after = self._list_rows()
        new = [str(_attr(r, "id")) for r in after if str(_attr(r, "id")) not in before]
        if len(new) == 1:
            self._known_ids.add(new[0])
            return new[0]
        # identical text already in the dataset is a dedup no-op (ingest_data.py:286): find it by content hash
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()  # noqa: S324 - cognee's own naming scheme
        for r in after:
            if _attr(r, "content_hash") == digest or str(_attr(r, "name", "")).startswith(f"text_{digest}"):
                return str(_attr(r, "id"))
        return None

    @staticmethod
    def _check_run(res: Any) -> None:
        status = str(_attr(res, "status", "") or "")
        if status.endswith("Errored"):
            raise RuntimeError(f"cognee.add failed: {status} {_attr(res, 'error_message', '') or _attr(res, 'payload', '')}")


# -- read side -------------------------------------------------------------------------
def result_ids(item: Any) -> list[str]:
    """Host ids a search payload points at: `document_id` for chunk payloads (the Data row), else `id`."""
    if isinstance(item, dict):
        for key in ("document_id", "id"):
            v = item.get(key)
            if v:
                return [str(v)]
        return []
    for key in ("document_id", "id"):
        v = getattr(item, key, None)
        if v:
            return [str(v)]
    return []


def _map_payloads(res: Any, fn: Callable[[Any], Any]) -> Any:
    """Apply `fn` to every id-bearing payload in every shape `cognee.search` returns (`fn` returns the
    replacement, or None to drop it). Strings (completions) and payloads without an id pass through untouched."""
    if isinstance(res, list):
        out = []
        for item in res:
            if isinstance(item, dict) and "search_result" in item and not result_ids(item):
                out.append({**item, "search_result": _map_payloads(item["search_result"], fn)})
            elif isinstance(item, list):
                out.append(_map_payloads(item, fn))
            elif result_ids(item):
                kept = fn(item)
                if kept is not None:
                    out.append(kept)
            else:
                out.append(item)
        return out
    if isinstance(res, dict) and "search_result" in res:
        return {**res, "search_result": _map_payloads(res["search_result"], fn)}
    return res


def _filter_payloads(res: Any, hide: set[str]) -> Any:
    """Drop payloads whose row is hidden, in every shape `cognee.search` returns."""
    return _map_payloads(res, lambda item: None if any(i in hide for i in result_ids(item)) else item)


def _annotate_payloads(res: Any, gov: Governor) -> Any:
    """Label every payload: dict payloads gain `"invalidate_note"`; objects get the attribute when they
    accept one. The note comes from `Governor.annotate` over the distinct ids the result set exposes."""
    ids: list[str] = []

    def collect(item: Any) -> Any:
        ids.extend(result_ids(item))
        return item

    _map_payloads(res, collect)
    notes = dict(gov.annotate(sorted(set(ids)), id_of=str))

    def label(item: Any) -> Any:
        note = next((notes[i] for i in result_ids(item) if notes.get(i)), None)
        if isinstance(item, dict):
            return {**item, "invalidate_note": note}
        try:
            item.invalidate_note = note
        except (AttributeError, TypeError, ValueError):
            pass
        return item

    return _map_payloads(res, label)


def governed_search(client: Any, gov: Governor, query: str, *, include_review: bool = True, annotate: bool = False,
                    **kw: Any) -> Any:
    """`cognee.search(query, **kw)` with results of dead (and, by default, under-review) rows removed.

    Works on payload-shaped results: `SearchType.CHUNKS` / `CHUNKS_LEXICAL` (filtered by `document_id`)
    and anything else that exposes `document_id` or `id` matching a governed row. Completion strings
    (`GRAPH_COMPLETION`, `RAG_COMPLETION`, ...) pass through unchanged: Cognee's LLM has already read the
    stale chunk by then, so use `Governor(mode="delete")` when those search types must not see dead facts.
    `SUMMARIES` payloads carry no document id and pass through. When `kw` names no dataset and the
    governor's adapter is a CogneeAdapter, the search is scoped to its dataset. Runs the coroutine for you;
    from async code call `cognee.search` yourself and pass the result to `filter_results`.

    `annotate=True` keeps every payload and adds an `"invalidate_note"` key to each payload dict
    (`Governor.annotate`'s label for retired rows, None otherwise); completion strings still pass through."""
    adapter = gov.adapter
    if "datasets" not in kw and "dataset_ids" not in kw and isinstance(adapter, CogneeAdapter):
        kw["dataset_ids"] = [adapter.dataset_id]
        kw.update(adapter._user_kw(client.search))
    res = _call(client.search, query, **kw)
    return filter_results(res, gov, include_review=include_review, annotate=annotate)


def filter_results(res: Any, gov: Governor, *, include_review: bool = True, annotate: bool = False) -> Any:
    """The filtering half of `governed_search`, for results you already have (e.g. awaited yourself).
    `annotate=True` labels instead of filtering, as in `governed_search`."""
    if annotate:
        return _annotate_payloads(res, gov)
    hide = set(gov.dead_ids())
    if not include_review:
        hide |= {gov.host_id(m) for m in gov.review()}
    return _filter_payloads(res, hide)


__all__ = ["CogneeAdapter", "governed_search", "filter_results", "result_ids", "relational_metadata_writer", "_run"]
