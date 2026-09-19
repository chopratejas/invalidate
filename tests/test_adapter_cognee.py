"""Cognee adapter against a faithful fake of the `cognee` module surface.

The fake mirrors the method signatures, async-ness and return shapes read from cognee 1.6.0
(github.com/topoteretes/cognee main @ 663a2dc, identical to the PyPI 1.6.0 wheel):
  cognee/api/v1/add/add.py:35            async add(data, dataset_name="main_dataset", user=None, node_set=None,
                                         ..., dataset_id=None, ...) -> PipelineRunInfo
  cognee/api/v1/delete/__init__.py:9     async delete(data_id, dataset_id, mode="soft", user=None)  (deprecated)
  cognee/api/v1/datasets/datasets.py     class datasets: async list_datasets(user=None) (:127),
                                         async list_data(dataset_id, user=None) (:138),
                                         async delete_data(dataset_id, data_id, user=None, mode="soft",
                                                           delete_dataset_if_empty=False) (:217)
  cognee/api/v1/search/search.py:41      async search(query_text, query_type=..., user=None, datasets=None,
                                         dataset_ids=None, ..., top_k=15, ...) -> per-dataset dicts with
                                         "search_result" (access control on) or the bare result
  cognee/tasks/ingestion/data_item.py:14 DataItem(data, label=None, external_metadata=None, system_metadata=None,
                                         data_id=None)
  cognee/modules/data/models/Data.py     Data rows: id, name, raw_data_location, mime_type, extension,
                                         external_metadata, label, dataset_id, content_hash
Text rows are written to a real temp file as `text_<md5>.txt` (save_data_to_file.py:78-79), so pull reads
the same bytes the real adapter would. No network, no cognee import.
"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

import pytest
from conftest import CONTRADICT, FakeJudge, SUPERSEDE, UNCERTAIN

from invalidate.adapters import Governor, Reason
from invalidate.adapters.cognee import CogneeAdapter, _run, filter_results, governed_search, result_ids
from invalidate.types import Status, now


# =============================================================================================
# fakes
# =============================================================================================
@dataclass
class FakeDataItem:
    """cognee/tasks/ingestion/data_item.py:14 DataItem."""

    data: Any
    label: Optional[str] = None
    external_metadata: Optional[dict] = None
    system_metadata: Optional[dict] = None
    data_id: Optional[UUID] = None


@dataclass
class FakeDataset:
    """cognee/modules/data/models/Dataset.py: id, name (+ owner_id ...)."""

    id: UUID
    name: str


@dataclass
class FakeData:
    """cognee/modules/data/models/Data.py columns used by the adapter (+ DataDTO after serve())."""

    id: UUID
    name: str
    raw_data_location: str
    dataset_id: UUID
    mime_type: str = "text/plain"
    extension: str = "txt"
    content_hash: str = ""
    external_metadata: Optional[dict] = None
    label: Optional[str] = None


@dataclass
class FakePipelineRunInfo:
    """cognee/modules/pipelines/models/PipelineRunInfo.py:9."""

    status: str
    pipeline_run_id: UUID
    dataset_id: UUID
    dataset_name: str
    payload: Any = None


class FakeSearchType(str):
    CHUNKS = "CHUNKS"
    CHUNKS_LEXICAL = "CHUNKS_LEXICAL"
    SUMMARIES = "SUMMARIES"
    GRAPH_COMPLETION = "GRAPH_COMPLETION"
    HYBRID_COMPLETION = "HYBRID_COMPLETION"


class FakeCognee:
    """The `cognee` module surface: add, delete, datasets, search, SearchType. `access_control` toggles the
    two result shapes of modules/search/methods/search.py:632-700."""

    SearchType = FakeSearchType

    def __init__(self, root: Path, *, access_control: bool = True, with_delete_data: bool = True) -> None:
        self.root = root
        self.access_control = access_control
        self.rows: dict[UUID, FakeData] = {}
        self.datasets_by_id: dict[UUID, FakeDataset] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.cognified: set[UUID] = set()  # chunks only exist for rows cognify has seen
        self.datasets = (_FakeDatasets if with_delete_data else _FakeOldDatasets)(self)

    # -- seeding helpers (what add()+cognify() would have left behind) --
    def dataset(self, name: str) -> FakeDataset:
        for ds in self.datasets_by_id.values():
            if ds.name == name:
                return ds
        ds = FakeDataset(id=uuid4(), name=name)
        self.datasets_by_id[ds.id] = ds
        return ds

    def _store_text(self, text: str) -> tuple[str, str, str]:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()  # save_data_to_file.py:78
        name = f"text_{digest}.txt"
        path = self.root / digest / name  # _storage_key: <content-md5>/<filename>
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:  # LocalFileStorage.store
            f.write(text)
        return path.as_uri(), digest, name

    def seed(self, text: str, dataset: str = "main_dataset", *, cognified: bool = True, **kw: Any) -> FakeData:
        ds = self.dataset(dataset)
        loc, digest, name = self._store_text(text)
        row = FakeData(id=uuid4(), name=name.rsplit(".", 1)[0], raw_data_location=loc, dataset_id=ds.id,
                       content_hash=digest, external_metadata={}, **kw)
        self.rows[row.id] = row
        if cognified:
            self.cognified.add(row.id)
        return row

    def seed_file(self, path: Path, dataset: str = "main_dataset", mime_type: str = "application/pdf") -> FakeData:
        ds = self.dataset(dataset)
        row = FakeData(id=uuid4(), name=path.stem, raw_data_location=path.as_uri(), dataset_id=ds.id,
                       mime_type=mime_type, extension=path.suffix.lstrip("."))
        self.rows[row.id] = row
        return row

    # -- cognee/api/v1/add/add.py:35 --
    async def add(self, data, dataset_name: str = "main_dataset", user=None, node_set: Optional[list[str]] = None,
                  vector_db_config=None, graph_db_config=None, dataset_id: Optional[UUID] = None,
                  preferred_loaders=None, incremental_loading: bool = True, data_per_batch: Optional[int] = 20,
                  importance_weight: Optional[float] = 0.5, run_in_background: bool = False, llm_config=None,
                  embedding_config=None, data_cache: bool = True, skip_connection_test: bool = False, **kwargs):
        await asyncio.sleep(0)
        self.calls.append(("add", {"data": data, "dataset_name": dataset_name, "dataset_id": dataset_id,
                                   "node_set": node_set, "user": user}))
        ds = self.datasets_by_id[dataset_id] if dataset_id is not None else self.dataset(dataset_name)
        items = data if isinstance(data, list) else [data]
        for item in items:
            pinned = None
            ext_meta: dict[str, Any] = {}
            if isinstance(item, FakeDataItem):
                pinned = item.data_id
                ext_meta = dict(item.external_metadata or {})
                item = item.data
            assert isinstance(item, str), "the fake only ingests text"
            loc, digest, name = self._store_text(item)
            if node_set:
                ext_meta["node_set"] = node_set  # ingest_data.py:418-419
            if pinned is None:
                # content dedup within (dataset, owner): identical text is a no-op (ingest_data.py:286)
                if any(r.content_hash == digest and r.dataset_id == ds.id for r in self.rows.values()):
                    continue
                pinned = uuid4()
            # a pinned id skips dedup and becomes the row id (ingest_data.py:277-283, 484-485)
            self.rows[pinned] = FakeData(id=pinned, name=name.rsplit(".", 1)[0], raw_data_location=loc,
                                         dataset_id=ds.id, content_hash=digest, external_metadata=ext_meta)
        return FakePipelineRunInfo(status="PipelineRunCompleted", pipeline_run_id=uuid4(), dataset_id=ds.id,
                                   dataset_name=ds.name)

    # -- cognee/api/v1/delete/__init__.py:9 (deprecated wrapper) --
    async def delete(self, data_id: UUID, dataset_id: UUID, mode: str = "soft", user=None):
        self.calls.append(("delete", {"data_id": data_id, "dataset_id": dataset_id, "mode": mode}))
        return await _FakeDatasets.delete_data_impl(self, dataset_id=dataset_id, data_id=data_id, user=user, mode=mode)

    # -- cognee/api/v1/search/search.py:41 --
    async def search(self, query_text: str, query_type=FakeSearchType.HYBRID_COMPLETION, user=None, datasets=None,
                     dataset_ids=None, system_prompt_path: str = "answer_simple_question.txt",
                     system_prompt: Optional[str] = None, top_k: int = 15, node_type=None, node_name=None,
                     node_name_filter_operator: str = "OR", only_context: bool = False, session_id=None, **kwargs):
        await asyncio.sleep(0)
        self.calls.append(("search", {"query_text": query_text, "query_type": query_type, "datasets": datasets,
                                      "dataset_ids": dataset_ids, "top_k": top_k, "user": user}))
        if isinstance(dataset_ids, UUID):
            dataset_ids = [dataset_ids]
        targets = [self.datasets_by_id[i] for i in dataset_ids] if dataset_ids else list(self.datasets_by_id.values())
        per_dataset = []
        for ds in targets:
            rows = [r for r in self.rows.values() if r.dataset_id == ds.id and r.id in self.cognified]
            if query_type in (FakeSearchType.CHUNKS, FakeSearchType.CHUNKS_LEXICAL):
                # chunks_retriever.py:57-60: [{**payload, "score"}]; payload = DocumentChunk fields
                result = []
                for i, r in enumerate(rows):
                    text = Path(r.raw_data_location[len("file://"):]).read_text(encoding="utf-8")
                    if query_text.lower() in text.lower():
                        result.append({"id": str(uuid4()), "text": text, "chunk_index": 0, "chunk_size": len(text),
                                       "cut_type": "paragraph_end", "document_id": str(r.id),  # TextChunker.py:17
                                       "document_name": r.name, "score": 0.1 * i})
            elif query_type == FakeSearchType.SUMMARIES:
                result = [{"id": str(uuid4()), "text": f"summary of {r.name}", "score": 0.2} for r in rows]
            else:
                result = [f"completion about {query_text}"]
            per_dataset.append((ds, result))
        if self.access_control:
            return [{"dataset_id": ds.id, "dataset_name": ds.name, "dataset_tenant_id": None, "search_result": result}
                    for ds, result in per_dataset]
        flat = [result for _, result in per_dataset]
        return flat[0] if len(flat) == 1 and isinstance(flat[0], list) else flat


class _FakeDatasets:
    """cognee/api/v1/datasets/datasets.py `class datasets` (staticmethods on the real thing)."""

    def __init__(self, cog: FakeCognee) -> None:
        self._cog = cog

    async def list_datasets(self, user=None):
        await asyncio.sleep(0)
        return list(self._cog.datasets_by_id.values())

    async def list_data(self, dataset_id: UUID, user=None):
        await asyncio.sleep(0)
        self._cog.calls.append(("list_data", {"dataset_id": dataset_id, "user": user}))
        if dataset_id not in self._cog.datasets_by_id:
            raise LookupError(f"Dataset {dataset_id} not accessible.")
        return [r for r in self._cog.rows.values() if r.dataset_id == dataset_id]

    async def delete_data(self, dataset_id: UUID, data_id: UUID, user=None, mode: str = "soft",
                          delete_dataset_if_empty: bool = False):
        self._cog.calls.append(("delete_data", {"dataset_id": dataset_id, "data_id": data_id, "mode": mode, "user": user}))
        return await self.delete_data_impl(self._cog, dataset_id=dataset_id, data_id=data_id, user=user, mode=mode)

    @staticmethod
    async def delete_data_impl(cog: FakeCognee, *, dataset_id: UUID, data_id: UUID, user=None, mode: str = "soft"):
        await asyncio.sleep(0)
        assert isinstance(data_id, UUID) and isinstance(dataset_id, UUID)
        row = cog.rows.get(data_id)
        if row is not None and row.dataset_id != dataset_id:
            raise PermissionError(f"Data {data_id} not accessible.")
        cog.rows.pop(data_id, None)
        cog.cognified.discard(data_id)
        return {"status": "success"}


class _FakeOldDatasets(_FakeDatasets):
    """Older cognee: only the top-level cognee.delete exists."""

    delete_data = None  # type: ignore[assignment]


# =============================================================================================
# fixtures
# =============================================================================================
@pytest.fixture
def cog(tmp_path: Path) -> FakeCognee:
    return FakeCognee(tmp_path)


def _adapter(cog: FakeCognee, **kw: Any) -> CogneeAdapter:
    kw.setdefault("data_item_class", FakeDataItem)
    if "dataset_id" not in kw:
        kw.setdefault("dataset_name", "main_dataset")
    return CogneeAdapter(cog, **kw)


def _gov(adapter: CogneeAdapter, fake: FakeJudge, **kw: Any) -> Governor:
    return Governor(adapter, ":memory:", judge=fake, **kw)


def _reason(status: Status) -> Reason:
    return Reason(status, status.value, "event text", "slack", "ev1", 0.05, now())


# =============================================================================================
# tests
# =============================================================================================
class TestConstruction:
    def test_needs_datasets_namespace(self):
        class Bare:
            pass

        with pytest.raises(ValueError):
            CogneeAdapter(Bare(), dataset_name="x")

    def test_needs_a_dataset(self, cog):
        with pytest.raises(ValueError):
            CogneeAdapter(cog)

    def test_resolves_dataset_name_via_list_datasets(self, cog):
        ds = cog.dataset("alice")
        cog.dataset("bob")
        a = _adapter(cog, dataset_name="alice")
        assert a.dataset_id == ds.id and not a.flagged_in_host

    def test_unknown_dataset_name_raises(self, cog):
        with pytest.raises(ValueError, match="not found"):
            _adapter(cog, dataset_name="nope")

    def test_dataset_id_accepts_str_or_uuid(self, cog):
        ds = cog.dataset("alice")
        assert _adapter(cog, dataset_id=str(ds.id)).dataset_id == ds.id
        assert _adapter(cog, dataset_id=ds.id).dataset_id == ds.id


class TestPull:
    def test_pull_reads_rows_verbatim_with_stable_ids(self, cog):
        r1 = cog.seed("Alice prefers Postgres\n\n  with trailing spaces  ")
        r2 = cog.seed("Alice lives in Berlin", label="profile")
        cog.seed("Bob's fact", dataset="bob")
        got = {m.id: m for m in _adapter(cog).pull()}
        assert set(got) == {str(r1.id), str(r2.id)}
        assert got[str(r1.id)].text == "Alice prefers Postgres\n\n  with trailing spaces  "
        assert got[str(r2.id)].metadata == {"name": r2.name, "mime_type": "text/plain", "label": "profile"}
        assert got[str(r2.id)].source == "cognee"
        assert [m.id for m in _adapter(cog).pull()] == [m.id for m in _adapter(cog).pull()]

    def test_pull_skips_non_text_rows_unless_asked(self, cog, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4 binary")
        cog.seed("a text row")
        cog.seed_file(pdf)
        assert len(list(_adapter(cog).pull())) == 1
        assert len(list(_adapter(cog, include_non_text=True).pull())) == 2

    def test_pull_passes_user_through(self, cog):
        cog.seed("x")
        _adapter(cog, user="u1").pull()
        assert cog.calls[-1] == ("list_data", {"dataset_id": cog.dataset("main_dataset").id, "user": "u1"})

    def test_read_text_can_be_injected(self, cog):
        r = cog.seed("stored")
        a = _adapter(cog, read_text=lambda loc: f"injected:{loc.endswith('.txt')}")
        assert [m.text for m in a.pull()] == ["injected:True"]
        assert r.id in cog.rows


class TestFlag:
    def test_flag_is_ledger_only_by_default(self, cog, fake):
        r = cog.seed("Alice prefers Postgres")
        fake.script("Postgres", CONTRADICT)
        gov = _gov(_adapter(cog), fake, mode="flag")
        gov.sync()
        rep = gov.observe("Alice dropped Postgres", source="slack")
        assert not rep.errors and rep.pushes[0].action == "flag"
        assert gov.status_of(str(r.id)) is Status.CONTRADICTED
        assert cog.rows[r.id].external_metadata == {}  # nothing written into cognee
        assert not any(c[0] in ("delete_data", "delete", "add") for c in cog.calls)

    def test_metadata_writer_merges_receipt_into_external_metadata(self, cog, fake):
        r = cog.seed("Alice prefers Postgres")
        cog.rows[r.id].external_metadata = {"node_set": ["profile"], "origin": "test"}
        writes: list[tuple[UUID, dict]] = []

        async def writer(data_id: UUID, receipt: dict) -> None:  # same contract as relational_metadata_writer
            await asyncio.sleep(0)
            writes.append((data_id, receipt))
            row = cog.rows[data_id]
            row.external_metadata = {**(row.external_metadata or {}), **receipt}

        fake.script("Postgres", SUPERSEDE)
        a = _adapter(cog, metadata_writer=writer)
        assert a.flagged_in_host
        gov = _gov(a, fake, mode="flag")
        gov.sync()
        rep = gov.observe("Alice migrated to SQLite", source="slack")
        assert not rep.errors
        assert writes[0][0] == r.id and isinstance(writes[0][0], UUID)
        meta = cog.rows[r.id].external_metadata
        assert meta["node_set"] == ["profile"] and meta["origin"] == "test"  # merged, not replaced
        assert meta["invalidate_status"] == "superseded" and meta["invalidate_event"] == "Alice migrated to SQLite"
        assert meta["invalidate_event_source"] == "slack"
        # text is never rewritten, and the row is still there
        assert Path(cog.rows[r.id].raw_data_location[len("file://"):]).read_text() == "Alice prefers Postgres"

    def test_keep_restores_in_ledger_and_writes_active_receipt_when_writable(self, cog, fake):
        r = cog.seed("Alice prefers Postgres")

        def writer(data_id: UUID, receipt: dict) -> None:  # sync writers are fine too
            row = cog.rows[data_id]
            row.external_metadata = {**(row.external_metadata or {}), **receipt}

        fake.script("Postgres", CONTRADICT)
        gov = _gov(_adapter(cog, metadata_writer=writer), fake, mode="flag")
        gov.sync()
        gov.observe("Alice dropped Postgres", source="slack")
        assert cog.rows[r.id].external_metadata["invalidate_status"] == "contradicted"
        gov.keep(str(r.id))
        assert gov.status_of(str(r.id)) is Status.ACTIVE
        assert cog.rows[r.id].external_metadata["invalidate_status"] == "active"

    def test_needs_review_stays_in_ledger(self, cog, fake):
        r = cog.seed("Alice prefers Postgres")
        fake.script("Postgres", UNCERTAIN)
        gov = _gov(_adapter(cog), fake)
        gov.sync()
        gov.observe("Alice might switch databases", source="slack")
        assert gov.status_of(str(r.id)) is Status.NEEDS_REVIEW and [m.id for m in gov.review()] == [gov.our_id(str(r.id))]


class TestDelete:
    def test_delete_mode_calls_datasets_delete_data(self, cog, fake):
        r = cog.seed("Alice prefers Postgres")
        keep = cog.seed("Alice lives in Berlin")
        fake.script("Postgres", CONTRADICT)
        gov = _gov(_adapter(cog), fake, mode="delete")
        gov.sync()
        rep = gov.observe("Alice dropped Postgres", source="slack")
        assert rep.pushes[0].action == "delete" and not rep.errors
        call = [c for c in cog.calls if c[0] == "delete_data"][0][1]
        assert call["data_id"] == r.id and call["dataset_id"] == cog.dataset("main_dataset").id and call["mode"] == "soft"
        assert r.id not in cog.rows and keep.id in cog.rows
        assert gov.status_of(str(r.id)) is Status.CONTRADICTED
        s = gov.sync()
        assert s.total == 1 and s.removed == 0  # a dead row vanishing from the host is not a "removal"

    def test_delete_falls_back_to_deprecated_cognee_delete(self, tmp_path):
        cog = FakeCognee(tmp_path, with_delete_data=False)
        r = cog.seed("x")
        _adapter(cog).delete(str(r.id), _reason(Status.CONTRADICTED))
        assert cog.calls[-1] == ("delete", {"data_id": r.id, "dataset_id": cog.dataset("main_dataset").id, "mode": "soft"})
        assert r.id not in cog.rows

    def test_forget_by_human_deletes_in_host(self, cog, fake):
        r = cog.seed("x")
        gov = _gov(_adapter(cog), fake, mode="flag")
        gov.sync()
        gov.forget(str(r.id))
        assert r.id not in cog.rows and gov.status_of(str(r.id)) is Status.DELETED


class TestInsert:
    def test_insert_pins_a_data_item_verbatim_without_cognify(self, cog):
        cog.dataset("main_dataset")
        a = _adapter(cog, insert_node_set=["invalidate"])
        hid = a.insert("Alice migrated to SQLite", "slack", {"invalidate_supersedes": ["abc"], "event_id": "ev1"})
        call = [c for c in cog.calls if c[0] == "add"][-1][1]
        item = call["data"]
        assert isinstance(item, FakeDataItem) and item.data == "Alice migrated to SQLite"
        assert str(item.data_id) == hid and call["dataset_id"] == cog.dataset("main_dataset").id
        assert call["node_set"] == ["invalidate"]
        assert item.external_metadata == {"invalidate_source": "slack", "invalidate_supersedes": ["abc"],
                                          "invalidate_event_id": "ev1"}
        row = cog.rows[UUID(hid)]
        assert Path(row.raw_data_location[len("file://"):]).read_text(encoding="utf-8") == "Alice migrated to SQLite"
        assert row.external_metadata["node_set"] == ["invalidate"]
        assert UUID(hid) not in cog.cognified  # nothing derived until the user runs cognify()
        assert [m.text for m in a.pull()] == ["Alice migrated to SQLite"]  # but it is pulled as a memory

    def test_insert_without_data_item_diffs_the_dataset(self, cog):
        cog.seed("existing")
        a = _NoDataItemAdapter(cog, dataset_name="main_dataset")  # older cognee: no DataItem importable
        hid = a.insert("brand new", "slack", {})
        assert [c for c in cog.calls if c[0] == "add"][-1][1]["data"] == "brand new" and UUID(hid) in cog.rows
        # identical text is a dedup no-op in cognee: the existing row id comes back
        again = a.insert("brand new", "slack", {})
        assert again == hid

    def test_insert_raises_on_errored_run(self, cog):
        async def failing_add(data, dataset_name="main_dataset", user=None, dataset_id=None, **kw):
            return FakePipelineRunInfo(status="PipelineRunErrored", pipeline_run_id=uuid4(), dataset_id=dataset_id,
                                       dataset_name="main_dataset", payload="boom")

        cog.dataset("main_dataset")
        cog.add = failing_add
        with pytest.raises(RuntimeError, match="PipelineRunErrored"):
            _adapter(cog).insert("x", "slack", {})

    def test_insert_without_add_is_not_implemented(self, cog):
        cog.dataset("main_dataset")
        a = _adapter(cog)
        a.client = _NoAdd(cog)
        with pytest.raises(NotImplementedError):
            a.insert("x", "slack", {})


class _NoDataItemAdapter(CogneeAdapter):
    @property
    def DataItem(self):  # noqa: N802
        return None


class _NoAdd:
    def __init__(self, cog: FakeCognee) -> None:
        self.datasets = cog.datasets


class TestGovernedSearch:
    def _dead_setup(self, cog: FakeCognee, fake: FakeJudge, **gov_kw: Any) -> tuple[Governor, FakeData, FakeData]:
        dead = cog.seed("Alice prefers Postgres")
        live = cog.seed("Alice lives in Berlin")
        fake.script("Postgres", CONTRADICT)
        gov = _gov(_adapter(cog), fake, **gov_kw)
        gov.sync()
        gov.observe("Alice dropped Postgres", source="slack")
        return gov, dead, live

    def test_filters_chunks_by_document_id_with_access_control_shape(self, cog, fake):
        gov, dead, live = self._dead_setup(cog, fake, mode="ledger")
        res = governed_search(cog, gov, "Alice", query_type=FakeSearchType.CHUNKS)
        assert isinstance(res, list) and len(res) == 1 and set(res[0]) == {"dataset_id", "dataset_name", "dataset_tenant_id", "search_result"}
        docs = [p["document_id"] for p in res[0]["search_result"]]
        assert docs == [str(live.id)]
        # scope + user were applied for us
        call = cog.calls[-1][1]
        assert call["dataset_ids"] == [cog.dataset("main_dataset").id] and call["query_type"] == "CHUNKS"

    def test_filters_bare_shape_without_access_control(self, tmp_path, fake):
        cog = FakeCognee(tmp_path, access_control=False)
        gov, dead, live = self._dead_setup(cog, fake, mode="ledger")
        res = governed_search(cog, gov, "Alice", query_type=FakeSearchType.CHUNKS)
        assert [p["document_id"] for p in res] == [str(live.id)]

    def test_completions_and_summaries_pass_through(self, cog, fake):
        gov, dead, live = self._dead_setup(cog, fake, mode="ledger")
        res = governed_search(cog, gov, "Alice")  # default HYBRID_COMPLETION -> strings
        assert res[0]["search_result"] == ["completion about Alice"]
        res = governed_search(cog, gov, "Alice", query_type=FakeSearchType.SUMMARIES)
        assert len(res[0]["search_result"]) == 2  # no document id on summaries: nothing to filter on

    def test_review_hidden_by_default_and_explicit_scope_respected(self, cog, fake):
        r = cog.seed("Alice prefers Postgres")
        other = cog.dataset("other")
        fake.script("Postgres", UNCERTAIN)
        gov = _gov(_adapter(cog), fake, mode="ledger")
        gov.sync()
        gov.observe("Alice might switch databases", source="slack")
        assert governed_search(cog, gov, "Alice", query_type=FakeSearchType.CHUNKS)[0]["search_result"] == []
        hits = governed_search(cog, gov, "Alice", query_type=FakeSearchType.CHUNKS, include_review=True)
        assert [p["document_id"] for p in hits[0]["search_result"]] == [str(r.id)]
        res = governed_search(cog, gov, "Alice", query_type=FakeSearchType.CHUNKS, dataset_ids=[other.id])
        assert cog.calls[-1][1]["dataset_ids"] == [other.id] and res[0]["search_result"] == []

    def test_filter_results_for_results_you_awaited_yourself(self, cog, fake):
        gov, dead, live = self._dead_setup(cog, fake, mode="ledger")

        async def inner():
            raw = await cog.search("Alice", query_type=FakeSearchType.CHUNKS, dataset_ids=[gov.adapter.dataset_id])
            return filter_results(raw, gov)

        res = asyncio.run(inner())
        assert [p["document_id"] for p in res[0]["search_result"]] == [str(live.id)]

    def test_result_ids_prefers_document_id(self):
        assert result_ids({"id": "chunk", "document_id": "doc"}) == ["doc"]
        assert result_ids({"id": "chunk"}) == ["chunk"]
        assert result_ids("a completion") == [] and result_ids({}) == []


class TestAsync:
    def test_run_inside_a_running_loop_uses_a_thread(self, cog):
        r = cog.seed("x")

        async def inner():
            return [m.id for m in _adapter(cog).pull()]

        assert asyncio.run(inner()) == [str(r.id)]

    def test_run_propagates_errors(self):
        async def boom():
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            _run(boom())

        async def outer():
            with pytest.raises(RuntimeError):
                _run(boom())

        asyncio.run(outer())


class TestGovernorEndToEnd:
    def test_supersede_inserts_a_verbatim_successor_and_survives_resync(self, cog, fake):
        old = cog.seed("Alice prefers Postgres")
        cog.seed("Alice lives in Berlin")
        fake.script("prefers Postgres", SUPERSEDE)
        gov = _gov(_adapter(cog), fake, mode="flag", successors=True)
        assert gov.sync().added == 2
        rep = gov.observe("Alice migrated to SQLite", source="slack")
        assert not rep.errors and rep.successor_host_id is not None
        succ_id = UUID(rep.successor_host_id)
        assert succ_id in cog.rows and old.id in cog.rows  # flag mode: the old row is kept
        assert gov.mem.get(gov.our_id(str(old.id))).superseded_by == gov.our_id(rep.successor_host_id)
        assert gov.status_of(str(old.id)) is Status.SUPERSEDED
        s = gov.sync()  # the successor row is pulled under the same id: no duplicate memory
        assert s.added == 0 and s.removed == 0 and s.total == 3
        succ = gov.mem.get(gov.our_id(rep.successor_host_id))
        assert succ.fact == "Alice migrated to SQLite" and succ.status is Status.ACTIVE
        # read side: the dead row's chunks are hidden, the live ones are not
        cog.cognified.add(succ_id)
        hits = governed_search(cog, gov, "Alice", query_type=FakeSearchType.CHUNKS)[0]["search_result"]
        assert {p["document_id"] for p in hits} == {str(r.id) for r in cog.rows.values()} - {str(old.id)}
        # judge again: the successor itself can be contradicted, and that lives in the ledger too
        fake.script("migrated to SQLite", CONTRADICT)
        rep2 = gov.observe("Alice never migrated", source="slack")
        assert not rep2.errors and gov.status_of(rep.successor_host_id) is Status.CONTRADICTED
        hits2 = governed_search(cog, gov, "Alice", query_type=FakeSearchType.CHUNKS)[0]["search_result"]
        assert {p["document_id"] for p in hits2} == {p["document_id"] for p in hits} - {str(succ_id)}
