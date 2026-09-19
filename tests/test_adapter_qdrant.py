"""QdrantAdapter against a real `QdrantClient(":memory:")` with 4-dim one-hot vectors, driven by the fake judge. No network."""
from __future__ import annotations

import uuid

import pytest
from conftest import CONFIRM, CONTRADICT, SUPERSEDE, FakeJudge

from invalidate import Status
from invalidate.adapters import Governor, HostMemory, Reason
from invalidate.adapters.qdrant import DEAD_VALUES, QdrantAdapter, governed_query, live_filter, merge_filter

qdrant_client = pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient, models  # noqa: E402
from qdrant_client.http.models import QueryResponse  # noqa: E402  (models.QueryResponse is the fastembed one)

DIM = 4
FACTS = {
    "pg": "user prefers Postgres",
    "deploy": "deploys run at 2pm UTC",
    "alice": "Alice owns the billing service",
    "replica": "prod reads go through the Postgres replica",
}
NS = uuid.UUID("12345678-1234-5678-1234-567812345678")
IDS = {k: str(uuid.uuid5(NS, k)) for k in FACTS}          # Qdrant string ids must be UUIDs
BY_ID = {v: k for k, v in IDS.items()}
INT_ID = 42                                              # an integer-id point, pulled as "42"
DEAD_STATUSES = set(DEAD_VALUES)


def _vec(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i % DIM] = 1.0
    return v


def embed(text: str) -> list[float]:
    """Deterministic fake embedding: one-hot on the text length."""
    return _vec(len(text))


def make_judge() -> FakeJudge:
    # "we migrated to SQLite": supersedes the preference, contradicts the replica config.
    return FakeJudge().script("prefers Postgres", SUPERSEDE).script("Postgres replica", CONTRADICT)


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture
def collection(client: QdrantClient) -> str:
    name = "memories"
    client.create_collection(name, vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))
    points = [
        models.PointStruct(id=IDS["pg"], vector=_vec(0), payload={"text": FACTS["pg"], "kind": "preference", "user": "alice"}),
        models.PointStruct(id=IDS["deploy"], vector=_vec(1), payload={"text": FACTS["deploy"], "n": 1, "user": "alice"}),
        models.PointStruct(id=IDS["alice"], vector=_vec(2), payload={"text": FACTS["alice"], "user": "bob"}),
        models.PointStruct(id=IDS["replica"], vector=_vec(3), payload={"text": FACTS["replica"], "kind": "config", "user": "alice"}),
        models.PointStruct(id=INT_ID, vector=_vec(0), payload={"text": "int-id fact", "user": "alice"}),
        models.PointStruct(id=str(uuid.uuid5(NS, "no-text")), vector=_vec(1), payload={"other": 1}),   # skipped
        models.PointStruct(id=str(uuid.uuid5(NS, "blank")), vector=_vec(2), payload={"text": "   "}),   # skipped
        models.PointStruct(id=str(uuid.uuid5(NS, "no-payload")), vector=_vec(3), payload=None),         # skipped
    ]
    client.upsert(name, points=points)
    return name


def payload_of(client: QdrantClient, collection: str, pid: str | int) -> dict:
    recs = client.retrieve(collection, ids=[pid], with_payload=True)
    return dict(recs[0].payload or {}) if recs else {}


def reason(status: Status = Status.SUPERSEDED) -> Reason:
    return Reason(status, status.value, "we migrated to SQLite", "slack", "ev1", 0.05, 1.0)


# --- pull ------------------------------------------------------------------------------------

def test_pull_scrolls_every_page_and_skips_textless_points(client, collection):
    adapter = QdrantAdapter(client, collection, page_size=2)       # 8 points -> 4 scroll pages
    pulled = {hm.id: hm for hm in adapter.pull()}
    assert set(pulled) == {*IDS.values(), str(INT_ID)}             # 3 textless points skipped, int id stringified
    pg = pulled[IDS["pg"]]
    assert isinstance(pg, HostMemory) and pg.text == FACTS["pg"] and pg.kind == "preference" and pg.source == "qdrant"
    assert pg.metadata == {"kind": "preference", "user": "alice"}   # text key itself is not echoed into metadata
    assert pulled[IDS["deploy"]].kind == "fact" and pulled[IDS["deploy"]].metadata == {"n": 1, "user": "alice"}
    assert adapter.name == f"qdrant:{collection}"


def test_pull_respects_scope_filter_and_max_rows(client, collection):
    scope = models.Filter(must=[models.FieldCondition(key="user", match=models.MatchValue(value="alice"))])
    adapter = QdrantAdapter(client, collection, scope_filter=scope)
    assert {hm.id for hm in adapter.pull()} == {IDS["pg"], IDS["deploy"], IDS["replica"], str(INT_ID)}

    with pytest.warns(UserWarning, match="max_rows"):
        few = QdrantAdapter(client, collection, page_size=2, max_rows=3).pull()
    assert len(few) == 3

    client.create_collection("empty", vectors_config=models.VectorParams(size=DIM, distance=models.Distance.DOT))
    assert QdrantAdapter(client, "empty").pull() == []


# --- flag: set_payload merges ------------------------------------------------------------------

def test_flag_merges_receipt_into_payload(client, collection):
    adapter = QdrantAdapter(client, collection)
    adapter.flag(IDS["pg"], reason())
    p = payload_of(client, collection, IDS["pg"])
    assert p["text"] == FACTS["pg"] and p["kind"] == "preference" and p["user"] == "alice"   # existing keys survive
    assert p["invalidate_status"] == "superseded" and p["invalidate_event"] == "we migrated to SQLite"
    assert p["invalidate_event_source"] == "slack" and p["invalidate_still_true"] == 0.05
    assert set(p) == {"text", "kind", "user", *reason().as_metadata()}

    # a second flag overwrites the same receipt keys and nothing else
    adapter.flag(IDS["pg"], reason(Status.ACTIVE))
    p2 = payload_of(client, collection, IDS["pg"])
    assert p2["invalidate_status"] == "active" and p2["kind"] == "preference" and set(p2) == set(p)

    adapter.flag(str(INT_ID), reason(Status.CONTRADICTED))          # "42" -> 42
    assert payload_of(client, collection, INT_ID)["invalidate_status"] == "contradicted"
    assert payload_of(client, collection, INT_ID)["text"] == "int-id fact"


def test_set_payload_merges_where_overwrite_payload_replaces(client, collection):
    """The semantic the adapter depends on: set_payload is a top-level merge, not a replace."""
    receipt = reason().as_metadata()
    client.set_payload(collection, payload=receipt, points=[IDS["deploy"]])
    merged = payload_of(client, collection, IDS["deploy"])
    assert merged["text"] == FACTS["deploy"] and merged["n"] == 1 and merged["invalidate_status"] == "superseded"

    client.overwrite_payload(collection, payload=receipt, points=[IDS["alice"]])
    replaced = payload_of(client, collection, IDS["alice"])
    assert "text" not in replaced and replaced["invalidate_status"] == "superseded"   # the control: this one loses text


def test_flag_unknown_id_raises_so_governor_records_a_push_error(client, collection):
    adapter = QdrantAdapter(client, collection)
    with pytest.raises(Exception):
        adapter.flag(str(uuid.uuid4()), reason())


# --- delete -----------------------------------------------------------------------------------

def test_delete_removes_the_point(client, collection):
    adapter = QdrantAdapter(client, collection)
    before = client.count(collection).count
    adapter.delete(IDS["replica"], reason(Status.CONTRADICTED))
    adapter.delete(str(INT_ID), reason(Status.CONTRADICTED))
    assert client.count(collection).count == before - 2
    assert client.retrieve(collection, ids=[IDS["replica"], INT_ID]) == []
    adapter.delete(str(uuid.uuid4()), reason())                      # unknown id: no-op, no error
    assert client.count(collection).count == before - 2


# --- insert -----------------------------------------------------------------------------------

def test_insert_requires_vector_fn_and_stores_text_verbatim(client, collection):
    assert getattr(QdrantAdapter(client, collection), "insert", None) is None    # no vector_fn: no successors

    adapter = QdrantAdapter(client, collection, vector_fn=embed, scope_payload={"user": "alice"})
    text = "we migrated to SQLite  (verbatim, with   odd spacing)"
    new_id = adapter.insert(text, "slack", {"invalidate_supersedes": [IDS["pg"]], "invalidate_event_id": "ev1"})
    assert isinstance(new_id, str) and uuid.UUID(new_id)
    p = payload_of(client, collection, new_id)
    assert p["text"] == text and p["source"] == "slack" and p["user"] == "alice"
    assert p["invalidate_status"] == "active" and p["invalidate_supersedes"] == [IDS["pg"]] and p["invalidate_event_id"] == "ev1"
    rec = client.retrieve(collection, ids=[new_id], with_vectors=True)[0]
    assert rec.vector == embed(text)
    assert new_id in {hm.id for hm in adapter.pull()}


def test_insert_named_vectors(client):
    client.create_collection("nv", vectors_config={"dense": models.VectorParams(size=DIM, distance=models.Distance.DOT)})
    by_name = QdrantAdapter(client, "nv", vector_fn=embed, vector_name="dense")
    a = by_name.insert("alpha", "test", {})
    by_dict = QdrantAdapter(client, "nv", vector_fn=lambda t: {"dense": embed(t)})
    b = by_dict.insert("beta", "test", {})
    recs = {str(r.id): r for r in client.retrieve("nv", ids=[a, b], with_vectors=True)}
    assert recs[a].vector == {"dense": embed("alpha")} and recs[b].vector == {"dense": embed("beta")}
    assert client.query_points("nv", query=embed("alpha"), using="dense", limit=1).points[0].id == a


# --- live_filter / merge_filter ---------------------------------------------------------------

def test_live_filter_hides_dead_and_keeps_points_without_the_key(client, collection):
    adapter = QdrantAdapter(client, collection)
    f = adapter.live_filter()
    assert isinstance(f, models.Filter) and f.must_not[0].key == "invalidate_status"
    assert f.must_not[0].match.any == ["contradicted", "superseded"]
    assert live_filter("x_").must_not[0].key == "x_status"

    all_ids = {str(r.id) for r in client.scroll(collection, limit=100)[0]}
    assert {str(r.id) for r in client.scroll(collection, scroll_filter=f, limit=100)[0]} == all_ids   # nothing flagged yet

    adapter.flag(IDS["pg"], reason(Status.SUPERSEDED))
    adapter.flag(IDS["replica"], reason(Status.CONTRADICTED))
    adapter.flag(IDS["deploy"], reason(Status.NEEDS_REVIEW))         # not dead: stays visible
    live = {str(r.id) for r in client.scroll(collection, scroll_filter=f, limit=100)[0]}
    assert live == all_ids - {IDS["pg"], IDS["replica"]}

    scope = models.Filter(must=[models.FieldCondition(key="user", match=models.MatchValue(value="alice"))])
    assert merge_filter(None, f) is f
    both = merge_filter(scope, f)
    assert {str(r.id) for r in client.scroll(collection, scroll_filter=both, limit=100)[0]} == {IDS["deploy"], str(INT_ID)}
    scoped = QdrantAdapter(client, collection, scope_filter=scope).scoped_live_filter()
    assert {str(r.id) for r in client.scroll(collection, scroll_filter=scoped, limit=100)[0]} == {IDS["deploy"], str(INT_ID)}


# --- governed_query ----------------------------------------------------------------------------

def test_governed_query_filters_host_flags_and_ledger(client, collection):
    gov = Governor(QdrantAdapter(client, collection), ":memory:", judge=make_judge(), mode="ledger")
    gov.sync()
    gov.observe("we migrated to SQLite last Tuesday", source="slack")
    assert gov.dead_ids() == {IDS["pg"], IDS["replica"]} and gov.mem  # ledger knows; host untouched
    assert "invalidate_status" not in payload_of(client, collection, IDS["pg"])

    raw = client.query_points(collection, query=_vec(0), limit=10)
    assert {str(p.id) for p in raw.points} >= {IDS["pg"], IDS["replica"]}

    res = governed_query(client, gov, collection, _vec(0), limit=10)
    assert isinstance(res, QueryResponse)
    got = {str(p.id) for p in res.points}
    assert IDS["pg"] not in got and IDS["replica"] not in got and {IDS["deploy"], IDS["alice"], str(INT_ID)} <= got
    assert res.points[0].payload["text"] in {FACTS["pg"], "int-id fact"} or res.points[0].score == 1.0

    # an existing query_filter is preserved (ANDed), kwargs pass through
    scope = models.Filter(must=[models.FieldCondition(key="user", match=models.MatchValue(value="bob"))])
    res = governed_query(client, gov, collection, _vec(2), query_filter=scope, limit=3, with_payload=False)
    assert [str(p.id) for p in res.points] == [IDS["alice"]] and res.points[0].payload is None

    # flag mode: the host filter alone hides them (ledger prune is a no-op on top)
    gov2 = Governor(QdrantAdapter(client, collection), ":memory:", judge=make_judge(), mode="flag")
    gov2.sync()
    gov2.observe("we migrated to SQLite last Tuesday", source="slack")
    filtered = client.query_points(collection, query=_vec(0), query_filter=live_filter(), limit=10)
    assert {str(p.id) for p in filtered.points}.isdisjoint({IDS["pg"], IDS["replica"]})
    assert {str(p.id) for p in governed_query(client, gov2, collection, _vec(0), limit=10).points} == {str(p.id) for p in filtered.points}


# --- Governor end to end ----------------------------------------------------------------------

def test_governor_end_to_end_flag_mode_with_successor(client, collection):
    adapter = QdrantAdapter(client, collection, vector_fn=embed)
    gov = Governor(adapter, ":memory:", judge=make_judge(), mode="flag", successors=True)
    rep = gov.sync()
    assert (rep.total, rep.added) == (5, 5)
    m = gov.mem.get(gov.our_id(IDS["pg"]))
    assert m.fact == FACTS["pg"] and m.kind == "preference" and m.metadata["host_id"] == IDS["pg"]
    assert gov.sync().unchanged == 5

    rep = gov.observe("we migrated to SQLite last Tuesday", source="slack")
    assert not rep.errors
    assert {(p.host_id, p.action) for p in rep.pushes if p.action != "insert"} == {(IDS["pg"], "flag"), (IDS["replica"], "flag")}
    assert gov.status_of(IDS["pg"]) is Status.SUPERSEDED and gov.status_of(IDS["replica"]) is Status.CONTRADICTED
    assert gov.status_of(IDS["deploy"]) is Status.ACTIVE

    pg = payload_of(client, collection, IDS["pg"])
    assert pg["text"] == FACTS["pg"] and pg["kind"] == "preference"                  # text never rewritten
    assert pg["invalidate_status"] == "superseded" and pg["invalidate_event"] == "we migrated to SQLite last Tuesday"
    assert pg["invalidate_event_source"] == "slack" and pg["invalidate_disposition"] == "superseded"
    assert payload_of(client, collection, IDS["replica"])["invalidate_status"] == "contradicted"
    assert "invalidate_status" not in payload_of(client, collection, IDS["deploy"])

    sid = rep.successor_host_id
    assert sid and uuid.UUID(sid)
    succ = payload_of(client, collection, sid)
    assert succ["text"] == "we migrated to SQLite last Tuesday" and succ["source"] == "slack"
    assert succ["invalidate_status"] == "active" and succ["invalidate_supersedes"] == [IDS["pg"]]
    assert gov.mem.get(gov.our_id(IDS["pg"])).superseded_by == gov.our_id(sid)

    # resync: receipts are not text changes; the successor is already known; no invalidate_* leaks into the ledger
    rep2 = gov.sync()
    assert (rep2.total, rep2.unchanged, rep2.updated, rep2.removed) == (6, 6, 0, 0)
    assert not any(k.startswith("invalidate_") for k in gov.mem.get(gov.our_id(IDS["pg"])).metadata)

    live = {str(r.id) for r in client.scroll(collection, scroll_filter=adapter.live_filter(), limit=100)[0]}
    assert {IDS["pg"], IDS["replica"]}.isdisjoint(live) and {IDS["deploy"], IDS["alice"], sid} <= live
    assert client.count(collection).count == 9                                       # flag mode deletes nothing

    # human controls mirror into the host
    gov.keep(IDS["replica"])
    assert payload_of(client, collection, IDS["replica"])["invalidate_status"] == "active"
    gov.forget(IDS["deploy"])
    assert client.retrieve(collection, ids=[IDS["deploy"]]) == []
    gov.close()


def test_governor_delete_mode_and_confirm_leaves_host_alone(client, collection):
    gov = Governor(QdrantAdapter(client, collection), ":memory:", judge=make_judge(), mode="delete", successors=True)
    gov.sync()
    rep = gov.observe("we migrated to SQLite last Tuesday", source="slack")
    assert rep.successor_host_id is None and not rep.errors                          # no vector_fn: no successor
    assert {(p.host_id, p.action) for p in rep.pushes} == {(IDS["pg"], "delete"), (IDS["replica"], "delete")}
    assert client.retrieve(collection, ids=[IDS["pg"], IDS["replica"]]) == []
    assert client.count(collection).count == 6
    assert gov.sync().removed == 0 and gov.dead_ids() == {IDS["pg"], IDS["replica"]}

    gov2 = Governor(QdrantAdapter(client, collection), ":memory:", judge=FakeJudge().script("2pm UTC", CONFIRM))
    gov2.sync()
    rep = gov2.observe("yep, deploys still at 2pm", source="slack")
    assert rep.pushes == [] and "invalidate_status" not in payload_of(client, collection, IDS["deploy"])


def test_governor_push_error_keeps_ledger_truth(client, collection):
    gov = Governor(QdrantAdapter(client, collection), ":memory:", judge=make_judge(), mode="flag")
    gov.sync()
    client.delete(collection, points_selector=models.PointIdsList(points=[IDS["pg"]]))   # host loses the point under us
    rep = gov.observe("we migrated to SQLite last Tuesday", source="slack")
    assert [p.host_id for p in rep.errors] == [IDS["pg"]] and "KeyError" in rep.errors[0].error
    assert gov.status_of(IDS["pg"]) is Status.SUPERSEDED
    assert payload_of(client, collection, IDS["replica"])["invalidate_status"] == "contradicted"
    assert IDS["pg"] not in {str(p.id) for p in governed_query(client, gov, collection, _vec(0), limit=10).points}
