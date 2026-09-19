"""Chroma, LangGraph store and the callable vector-store adapter, driven by the fake judge. No network."""
from __future__ import annotations

import uuid
import warnings

import pytest
from conftest import CONFIRM, CONTRADICT, SUPERSEDE, FakeJudge

from invalidate import Status
from invalidate.adapters import Governor, HostMemory
from invalidate.adapters.chroma import ChromaAdapter, _prune_query_result, governed_query, live_where, merge_where
from invalidate.adapters.langgraph import LangGraphStoreAdapter, filter_items
from invalidate.adapters.vectorstore import CallableVectorStoreAdapter, live_filter_metadata

chromadb = pytest.importorskip("chromadb")
InMemoryStore = pytest.importorskip("langgraph.store.memory").InMemoryStore

FACTS = {
    "pg": "user prefers Postgres",
    "deploy": "deploys run at 2pm UTC",
    "alice": "Alice owns the billing service",
    "replica": "prod reads go through the Postgres replica",
}
DEAD_STATUSES = {"contradicted", "superseded"}


def make_judge() -> FakeJudge:
    # "we migrated to SQLite": supersedes the preference, contradicts the replica config.
    return FakeJudge().script("prefers Postgres", SUPERSEDE).script("Postgres replica", CONTRADICT)


# --- hosts ---------------------------------------------------------------------------------

def _vec(i: int) -> list[float]:
    v = [0.0] * 8
    v[i % 8] = 1.0
    return v


@pytest.fixture
def collection():
    client = chromadb.EphemeralClient()
    col = client.create_collection(f"t-{uuid.uuid4().hex[:8]}", embedding_function=None)
    ids = list(FACTS)
    col.add(ids=ids, documents=[FACTS[i] for i in ids], embeddings=[_vec(n) for n in range(len(ids))],
            metadatas=[{"kind": "preference"}, {"n": 1}, None, {"kind": "config"}])
    yield col
    client.delete_collection(col.name)


@pytest.fixture
def lg_store():
    store = InMemoryStore()
    ns = ("user-1", "memories")
    for k, text in FACTS.items():
        store.put(ns, k, {"content": text, "n": len(text)})
    store.put(ns, "no-text", {"other": 1})                   # skipped: no content
    store.put((*ns, "child"), "nested", {"content": "nested fact"})  # skipped: child namespace
    return store, ns


class DictHost:
    """A stand-in for pgvector/Pinecone/etc. behind CallableVectorStoreAdapter."""

    def __init__(self) -> None:
        self.rows = {k: {"id": k, "text": v, "metadata": {"kind": "fact"}} for k, v in FACTS.items()}
        self.calls: list[tuple] = []

    def list(self):
        return list(self.rows.values())

    def update(self, rid, patch):
        self.calls.append(("update", rid, dict(patch)))
        self.rows[rid]["metadata"].update(patch)   # merge semantics

    def delete(self, rid):
        self.calls.append(("delete", rid))
        self.rows.pop(rid)

    def insert(self, text, meta):
        rid = f"new-{len(self.rows)}"
        self.rows[rid] = {"id": rid, "text": text, "metadata": dict(meta)}
        return rid


# --- Chroma --------------------------------------------------------------------------------

def _chroma_meta(col, rid: str) -> dict:
    return col.get(ids=[rid], include=["metadatas"])["metadatas"][0] or {}


def test_chroma_sync_pulls_verbatim(collection):
    gov = Governor(ChromaAdapter(collection), ":memory:", judge=make_judge())
    rep = gov.sync()
    assert (rep.total, rep.added) == (4, 4)
    m = gov.mem.get(gov.our_id("pg"))
    assert m.fact == FACTS["pg"] and m.kind == "preference" and m.metadata["host_id"] == "pg"
    assert gov.sync().unchanged == 4


def test_chroma_flag_merges_metadata_and_where_hides_dead(collection):
    gov = Governor(ChromaAdapter(collection), ":memory:", judge=make_judge(), mode="flag")
    gov.sync()
    rep = gov.observe("we migrated to SQLite last Tuesday", source="slack")
    assert not rep.errors and {p.host_id for p in rep.pushes} == {"pg", "replica"}

    meta = _chroma_meta(collection, "pg")
    assert meta["invalidate_status"] == "superseded" and meta["kind"] == "preference"   # existing key kept
    assert meta["invalidate_event"] == "we migrated to SQLite last Tuesday" and meta["invalidate_event_source"] == "slack"
    assert _chroma_meta(collection, "replica")["invalidate_status"] == "contradicted"
    assert _chroma_meta(collection, "replica")["kind"] == "config"
    assert "invalidate_status" not in _chroma_meta(collection, "deploy")

    # $nin must keep rows that were never flagged (deploy, alice with metadata=None) and drop the dead ones.
    live = collection.get(where=live_where())["ids"]
    assert set(live) == {"deploy", "alice"}
    assert collection.count() == 4

    # resync is idempotent: flags do not change the text, so nothing resets to active.
    rep2 = gov.sync()
    assert rep2.unchanged == 4 and rep2.updated == 0 and rep2.removed == 0
    assert gov.status_of("pg") is Status.SUPERSEDED
    # the invalidate_* receipt never leaks back into the ledger's own metadata
    assert not any(k.startswith("invalidate_") for k in gov.mem.get(gov.our_id("pg")).metadata)


def test_chroma_governed_query_excludes_dead(collection):
    gov = Governor(ChromaAdapter(collection), ":memory:", judge=make_judge())
    gov.sync()
    gov.observe("we migrated to SQLite last Tuesday", source="slack")
    res = governed_query(collection, gov, query_embeddings=[_vec(0)], n_results=4, include=["documents", "metadatas", "distances"])
    assert set(res["ids"][0]) == {"deploy", "alice"}
    assert len(res["documents"][0]) == 2 and len(res["distances"][0]) == 2
    # user-supplied where is AND-ed, not clobbered
    res2 = governed_query(collection, gov, query_embeddings=[_vec(0)], n_results=4, where={"n": {"$eq": 1}})
    assert res2["ids"][0] == ["deploy"]


def test_chroma_governed_query_prunes_from_ledger_in_ledger_mode(collection):
    gov = Governor(ChromaAdapter(collection), ":memory:", judge=make_judge(), mode="ledger")
    gov.sync()
    rep = gov.observe("we migrated to SQLite last Tuesday", source="slack")
    assert rep.pushes == []
    assert "invalidate_status" not in _chroma_meta(collection, "pg")   # host untouched
    res = governed_query(collection, gov, query_embeddings=[_vec(0)], n_results=4)
    assert set(res["ids"][0]) == {"deploy", "alice"}                  # still hidden, via the ledger


def test_chroma_delete_mode(collection):
    gov = Governor(ChromaAdapter(collection), ":memory:", judge=make_judge(), mode="delete")
    gov.sync()
    rep = gov.observe("we migrated to SQLite last Tuesday", source="slack")
    assert {(p.host_id, p.action) for p in rep.pushes} == {("pg", "delete"), ("replica", "delete")}
    assert set(collection.get()["ids"]) == {"deploy", "alice"}
    rep2 = gov.sync()
    assert rep2.total == 2 and rep2.removed == 0                      # dead rows are not "gone", they were pushed
    assert gov.status_of("pg") is Status.SUPERSEDED


def test_chroma_successor_insert(collection):
    gov = Governor(ChromaAdapter(collection, embed=lambda t: _vec(7)), ":memory:", judge=make_judge(), successors=True)
    gov.sync()
    rep = gov.observe("we migrated to SQLite last Tuesday", source="slack")
    sid = rep.successor_host_id
    assert sid and collection.count() == 5
    got = collection.get(ids=[sid], include=["documents", "metadatas"])
    assert got["documents"][0] == "we migrated to SQLite last Tuesday"
    meta = got["metadatas"][0]
    assert meta["invalidate_supersedes"] == "pg" and meta["invalidate_status"] == "active" and meta["source"] == "slack"
    assert gov.mem.get(gov.our_id("pg")).superseded_by == gov.our_id(sid)
    assert sid in collection.get(where=live_where())["ids"]
    assert gov.sync().unchanged == 5                                  # successor is already in the ledger


def test_chroma_successor_insert_without_embed_is_reported(collection):
    gov = Governor(ChromaAdapter(collection), ":memory:", judge=make_judge(), successors=True)
    gov.sync()
    rep = gov.observe("we migrated to SQLite last Tuesday", source="slack")
    assert rep.successor_host_id is None and [p.action for p in rep.errors] == ["insert"]
    assert gov.status_of("pg") is Status.SUPERSEDED and collection.count() == 4


def test_chroma_flag_error_is_reported_and_ledger_still_moves(collection):
    class Broken(ChromaAdapter):
        def flag(self, host_id, reason):
            if host_id == "pg":
                raise RuntimeError("chroma down")
            super().flag(host_id, reason)

    gov = Governor(Broken(collection), ":memory:", judge=make_judge())
    gov.sync()
    rep = gov.observe("we migrated to SQLite last Tuesday", source="slack")
    assert [p.host_id for p in rep.errors] == ["pg"] and "chroma down" in rep.errors[0].error
    assert gov.status_of("pg") is Status.SUPERSEDED
    assert _chroma_meta(collection, "replica")["invalidate_status"] == "contradicted"
    assert "invalidate_status" not in _chroma_meta(collection, "pg")
    # governed_query still hides pg through the ledger prune
    res = governed_query(collection, gov, query_embeddings=[_vec(0)], n_results=4)
    assert "pg" not in res["ids"][0]


def test_chroma_keep_and_forget_mirror(collection):
    gov = Governor(ChromaAdapter(collection), ":memory:", judge=make_judge())
    gov.sync()
    gov.observe("we migrated to SQLite last Tuesday", source="slack")
    gov.keep("pg")
    assert _chroma_meta(collection, "pg")["invalidate_status"] == "active"
    assert "pg" in collection.get(where=live_where())["ids"]
    gov.forget("deploy")
    assert "deploy" not in collection.get()["ids"]


def test_chroma_stamp_live_and_paging(collection):
    adapter = ChromaAdapter(collection, page_size=3)
    assert {hm.id for hm in adapter.pull()} == set(FACTS)
    stamped = adapter.stamp_live()
    assert set(stamped) == set(FACTS)
    assert all(_chroma_meta(collection, i)["invalidate_status"] == "active" for i in FACTS)
    assert adapter.stamp_live() == []
    small = ChromaAdapter(collection, max_rows=2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert len(list(small.pull())) == 2
    assert any("pulling only the first 2" in str(x.message) for x in w)


def test_chroma_helpers():
    assert live_where() == {"invalidate_status": {"$nin": ["contradicted", "superseded"]}}
    assert merge_where(None, {"a": 1}) == {"a": 1}
    assert merge_where({"b": 2}, {"a": 1}) == {"$and": [{"b": 2}, {"a": 1}]}
    assert merge_where({"$and": [{"b": 2}]}, {"a": 1}) == {"$and": [{"b": 2}, {"a": 1}]}
    res = {"ids": [["a", "b"]], "documents": [["x", "y"]], "distances": [[0.1, 0.2]], "embeddings": None, "included": ["documents"]}
    out = _prune_query_result(res, {"a"})
    assert out["ids"] == [["b"]] and out["documents"] == [["y"]] and out["distances"] == [[0.2]] and out["embeddings"] is None


# --- LangGraph -----------------------------------------------------------------------------

def test_langgraph_sync_skips_textless_and_children(lg_store):
    store, ns = lg_store
    gov = Governor(LangGraphStoreAdapter(store, ns), ":memory:", judge=make_judge())
    rep = gov.sync()
    assert rep.total == 4 and {gov.host_id(m) for m in gov.mem.list()} == set(FACTS)
    m = gov.mem.get(gov.our_id("pg"))
    assert m.fact == FACTS["pg"] and m.metadata["n"] == len(FACTS["pg"])
    assert gov.sync().unchanged == 4


def test_langgraph_flag_merges_and_filters(lg_store):
    store, ns = lg_store
    adapter = LangGraphStoreAdapter(store, ns)
    gov = Governor(adapter, ":memory:", judge=make_judge())
    gov.sync()
    rep = gov.observe("we migrated to SQLite last Tuesday", source="slack")
    assert not rep.errors and {p.host_id for p in rep.pushes} == {"pg", "replica"}
    v = store.get(ns, "pg").value
    assert v["content"] == FACTS["pg"] and v["n"] == len(FACTS["pg"])       # put replaces; we merged
    assert v["invalidate_status"] == "superseded" and v["invalidate_live"] is False
    assert store.get(ns, "replica").value["invalidate_status"] == "contradicted"
    assert "invalidate_status" not in store.get(ns, "deploy").value

    items = store.search(ns, limit=50)
    assert {i.key for i in filter_items(items, gov)} == {"deploy", "alice", "no-text", "nested"}
    assert {i.key for i in store.search(ns, filter=adapter.live_filter(), limit=50)} == {"deploy", "alice", "no-text", "nested"}
    assert {i.key for i in adapter.search(gov, limit=50)} == {"deploy", "alice", "no-text", "nested"}

    rep2 = gov.sync()
    assert rep2.unchanged == 4 and rep2.updated == 0
    assert gov.status_of("pg") is Status.SUPERSEDED


def test_langgraph_delete_mode_and_successor(lg_store):
    store, ns = lg_store
    gov = Governor(LangGraphStoreAdapter(store, ns), ":memory:", judge=make_judge(), mode="delete", successors=True)
    gov.sync()
    rep = gov.observe("we migrated to SQLite last Tuesday", source="slack")
    assert store.get(ns, "pg") is None and store.get(ns, "replica") is None
    sid = rep.successor_host_id
    assert sid and store.get(ns, sid).value["content"] == "we migrated to SQLite last Tuesday"
    assert store.get(ns, sid).value["invalidate_supersedes"] == ["pg"]
    assert gov.mem.get(gov.our_id("pg")).superseded_by == gov.our_id(sid)
    rep2 = gov.sync()
    assert rep2.total == 3 and rep2.removed == 0 and rep2.added == 0


def test_langgraph_flag_error_captured(lg_store):
    store, ns = lg_store
    store.delete(ns, "pg")                      # vanish behind the ledger's back: flag() raises KeyError
    gov = Governor(LangGraphStoreAdapter(store, ns), ":memory:", judge=make_judge())
    gov.sync()
    store.put(ns, "pg", {"content": FACTS["pg"]})
    gov.sync()
    store.delete(ns, "pg")
    rep = gov.observe("we migrated to SQLite last Tuesday", source="slack")
    assert [p.host_id for p in rep.errors] == ["pg"] and "KeyError" in rep.errors[0].error
    assert gov.status_of("pg") is Status.SUPERSEDED
    assert store.get(ns, "replica").value["invalidate_status"] == "contradicted"


def test_langgraph_children_and_paging():
    store = InMemoryStore()
    ns = ("org",)
    for i in range(7):
        store.put((*ns, f"team{i % 2}"), f"k{i}", {"content": f"fact {i}"})
    adapter = LangGraphStoreAdapter(store, ns, include_children=True, page_size=3)
    ids = {hm.id for hm in adapter.pull()}
    assert ids == {f"team{i % 2}/k{i}" for i in range(7)}
    gov = Governor(adapter, ":memory:", judge=FakeJudge().script("fact 3", CONTRADICT))
    gov.sync()
    gov.observe("fact 3 is wrong", source="test")
    assert store.get(("org", "team1"), "k3").value["invalidate_status"] == "contradicted"
    assert LangGraphStoreAdapter(store, ns, page_size=3).pull() == []   # exact namespace holds nothing


# --- CallableVectorStoreAdapter --------------------------------------------------------------

def test_callable_adapter_round_trip():
    host = DictHost()
    adapter = CallableVectorStoreAdapter("pgvector", host.list, host.update, host.delete, host.insert)
    gov = Governor(adapter, ":memory:", judge=make_judge(), successors=True)
    assert gov.sync().added == 4
    rep = gov.observe("we migrated to SQLite last Tuesday", source="slack")
    assert not rep.errors
    patch = next(c for c in host.calls if c[0] == "update" and c[1] == "pg")[2]
    assert set(patch) == {"invalidate_status", "invalidate_disposition", "invalidate_event", "invalidate_event_source",
                          "invalidate_event_id", "invalidate_still_true", "invalidate_at"}   # only receipt keys
    assert host.rows["pg"]["metadata"]["invalidate_status"] == "superseded" and host.rows["pg"]["metadata"]["kind"] == "fact"
    assert host.rows["replica"]["metadata"]["invalidate_status"] == "contradicted"
    sid = rep.successor_host_id
    assert sid and host.rows[sid]["metadata"]["invalidate_supersedes"] == ["pg"] and host.rows[sid]["metadata"]["invalidate_status"] == "active"
    assert gov.sync().unchanged == 5
    live = [r for r in host.list() if r["metadata"].get("invalidate_status") not in DEAD_STATUSES]
    assert {r["id"] for r in live} == {"deploy", "alice", sid}
    assert live_filter_metadata() == {"invalidate_status": {"$nin": ["contradicted", "superseded"]}}
    assert adapter.live_filter() == live_filter_metadata()


def test_callable_adapter_objects_delete_mode_and_no_insert():
    class Row:
        def __init__(self, rid, text):
            self.pk, self.body, self.meta = rid, text, {}

    rows = {k: Row(k, v) for k, v in FACTS.items()}
    rows["blank"] = Row("blank", "   ")
    deleted = []
    adapter = CallableVectorStoreAdapter(
        "qdrant", lambda: rows.values(), lambda i, p: rows[i].meta.update(p), lambda i: deleted.append(rows.pop(i)),
        id_key="pk", text_key="body", metadata_key="meta",
    )
    assert getattr(adapter, "insert", None) is None
    gov = Governor(adapter, ":memory:", judge=make_judge(), mode="delete", successors=True)
    assert gov.sync().total == 4                                       # blank text skipped
    rep = gov.observe("we migrated to SQLite last Tuesday", source="slack")
    assert rep.successor_host_id is None and not rep.errors
    assert {r.pk for r in deleted} == {"pg", "replica"} and set(rows) == {"deploy", "alice", "blank"}


def test_callable_adapter_stamp_live_and_errors():
    host = DictHost()
    adapter = CallableVectorStoreAdapter("pg", host.list, host.update, host.delete)
    assert set(adapter.stamp_live()) == set(FACTS)
    assert all(r["metadata"]["invalidate_status"] == "active" for r in host.rows.values())
    assert adapter.stamp_live() == []
    assert adapter.stamp_live(["deploy"]) == ["deploy"]                # explicit ids: no read, always written

    def boom(rid, patch):
        raise ConnectionError("pool exhausted")

    gov = Governor(CallableVectorStoreAdapter("pg", host.list, boom, host.delete), ":memory:", judge=make_judge())
    gov.sync()
    rep = gov.observe("we migrated to SQLite last Tuesday", source="slack")
    assert {p.host_id for p in rep.errors} == {"pg", "replica"} and "pool exhausted" in rep.errors[0].error
    assert gov.status_of("pg") is Status.SUPERSEDED and gov.status_of("replica") is Status.CONTRADICTED
    assert gov.dead_ids() == {"pg", "replica"}
    assert host.rows["pg"]["metadata"]["invalidate_status"] == "active"   # host unchanged, ledger is truth


def test_confirm_does_not_touch_host(collection):
    gov = Governor(ChromaAdapter(collection), ":memory:", judge=FakeJudge().script("prefers Postgres", CONFIRM))
    gov.sync()
    rep = gov.observe("yep, still on Postgres", source="slack")
    assert rep.pushes == [] and "invalidate_status" not in _chroma_meta(collection, "pg")
    assert isinstance(next(iter(ChromaAdapter(collection).pull())), HostMemory)


def test_chroma_governed_query_annotate_keeps_dead_with_notes_column(collection):
    gov = Governor(ChromaAdapter(collection), ":memory:", judge=make_judge())
    gov.sync()
    gov.observe("we migrated to SQLite last Tuesday", source="slack")
    res = governed_query(collection, gov, annotate=True, query_embeddings=[_vec(0)], n_results=4)
    assert set(res["ids"][0]) == set(FACTS)                       # no live filter: flagged rows come back
    assert len(res["documents"][0]) == 4 and len(res["invalidate_notes"]) == 1
    notes = dict(zip(res["ids"][0], res["invalidate_notes"][0]))
    assert notes["pg"] == "OUTDATED, replaced as of slack: we migrated to SQLite last Tuesday"
    assert notes["replica"] == "OUTDATED, no longer true as of slack: we migrated to SQLite last Tuesday"
    assert notes["deploy"] is None and notes["alice"] is None
    assert "invalidate_notes" not in governed_query(collection, gov, query_embeddings=[_vec(0)], n_results=4)


def test_langgraph_filter_items_annotate_labels_value_copies(lg_store):
    store, ns = lg_store
    adapter = LangGraphStoreAdapter(store, ns)
    gov = Governor(adapter, ":memory:", judge=make_judge())
    gov.sync()
    gov.observe("we migrated to SQLite last Tuesday", source="slack")
    items = filter_items(store.search(ns, limit=50), gov, annotate=True)
    notes = {i.key: i.value.get("invalidate_note") for i in items}
    assert set(notes) == {"pg", "deploy", "alice", "replica", "no-text", "nested"}
    assert notes["pg"] == "OUTDATED, replaced as of slack: we migrated to SQLite last Tuesday"
    assert notes["replica"] == "OUTDATED, no longer true as of slack: we migrated to SQLite last Tuesday"
    assert notes["deploy"] is None and notes["nested"] is None
    assert next(i for i in items if i.key == "pg").value["content"] == FACTS["pg"]
    assert "invalidate_note" not in store.get(ns, "pg").value        # a copy was annotated, not the store
    assert {i.key for i in adapter.search(gov, annotate=True, limit=50)} == set(notes)   # no live filter either
    assert "pg" not in {i.key for i in adapter.search(gov, limit=50)}
