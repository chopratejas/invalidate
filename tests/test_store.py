"""SQLiteStore: round-trips, filtering, ordering, persistence, concurrency."""
from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from invalidate import Disposition, Event, Memory, SQLiteStore, Status, Verdict, Votes

S = Status
D = Disposition


@pytest.fixture
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


def _mem(**kw) -> Memory:
    base = dict(fact="user prefers Postgres")
    base.update(kw)
    return Memory(**base)


def _verdict(memory_id="mem_a", event_id="evt_a", created_at=None, **kw) -> Verdict:
    base = dict(
        event_id=event_id, memory_id=memory_id,
        votes=Votes(bears=0.9, still_true=0.1, replaces=0.8, hypothetical=0.05),
        disposition=D.SUPERSEDED, from_status=S.ACTIVE, to_status=S.SUPERSEDED, applied=True,
    )
    if created_at is not None:
        base["created_at"] = created_at
    base.update(kw)
    return Verdict(**base)


# --------------------------------------------------------------------------- memories


def test_memory_round_trip_all_fields(store):
    m = Memory(
        fact="user prefers Postgres", id="mem_x1", namespace="team-a", kind="preference", source="chat",
        status=S.NEEDS_REVIEW, p_true=0.42, created_at=1000.5, updated_at=1001.5, last_checked=1002.5,
        expires_at=2000.0, superseded_by="mem_y2",
        metadata={"tags": ["db", "pref"], "nested": {"n": 1, "f": 1.5, "b": True, "none": None}},
    )
    store.add_memory(m)
    got = store.get_memory("mem_x1")
    assert got == m
    assert got is not m
    assert got.status is S.NEEDS_REVIEW
    assert isinstance(got.metadata, dict)
    assert got.metadata["nested"]["none"] is None


def test_memory_round_trip_none_values_and_defaults(store):
    m = _mem()
    assert m.last_checked is None and m.expires_at is None and m.superseded_by is None and m.metadata == {}
    store.add_memory(m)
    got = store.get_memory(m.id)
    assert got == m
    assert got.last_checked is None
    assert got.expires_at is None
    assert got.superseded_by is None
    assert got.metadata == {}
    assert got.status is S.ACTIVE
    assert got.p_true == 1.0


def test_get_memory_missing_returns_none(store):
    assert store.get_memory("nope") is None


def test_add_memory_duplicate_id_raises(store):
    store.add_memory(_mem(id="dup"))
    with pytest.raises(sqlite3.IntegrityError):
        store.add_memory(_mem(id="dup"))


def test_update_memory_persists_every_mutable_field(store):
    m = _mem(id="m1")
    store.add_memory(m)
    m.namespace = "other"
    m.fact = "user prefers SQLite"
    m.kind = "pref"
    m.source = "slack"
    m.status = S.SUPERSEDED
    m.p_true = 0.05
    m.updated_at = 5.0
    m.last_checked = 6.0
    m.expires_at = 7.0
    m.superseded_by = "m2"
    m.metadata = {"k": "v"}
    store.update_memory(m)
    got = store.get_memory("m1")
    assert got == m
    assert got.fact == "user prefers SQLite"
    assert got.status is S.SUPERSEDED


def test_update_memory_unknown_id_is_noop(store):
    store.update_memory(_mem(id="ghost"))
    assert store.get_memory("ghost") is None
    assert store.list_memories() == []


def test_list_memories_orders_by_created_at_then_rowid(store):
    store.add_memory(_mem(id="c", created_at=30))
    store.add_memory(_mem(id="a", created_at=10))
    store.add_memory(_mem(id="b", created_at=20))
    store.add_memory(_mem(id="b2", created_at=20))  # tie broken by insertion order
    assert [m.id for m in store.list_memories()] == ["a", "b", "b2", "c"]


def test_list_memories_filters_by_namespace(store):
    store.add_memory(_mem(id="a", namespace="x"))
    store.add_memory(_mem(id="b", namespace="y"))
    store.add_memory(_mem(id="c", namespace="x"))
    assert [m.id for m in store.list_memories("x")] == ["a", "c"]
    assert [m.id for m in store.list_memories("y")] == ["b"]
    assert store.list_memories("z") == []
    assert len(store.list_memories(None)) == 3


def test_list_memories_filters_by_statuses(store):
    store.add_memory(_mem(id="a", status=S.ACTIVE))
    store.add_memory(_mem(id="b", status=S.FROZEN))
    store.add_memory(_mem(id="c", status=S.CONTRADICTED))
    store.add_memory(_mem(id="d", status=S.DELETED))
    assert [m.id for m in store.list_memories(statuses={S.ACTIVE})] == ["a"]
    assert [m.id for m in store.list_memories(statuses=[S.FROZEN, S.CONTRADICTED])] == ["b", "c"]
    # any iterable works, including a generator
    assert [m.id for m in store.list_memories(statuses=(s for s in (S.DELETED,)))] == ["d"]


def test_list_memories_empty_statuses_returns_empty_list(store):
    store.add_memory(_mem(id="a"))
    assert store.list_memories(statuses=set()) == []
    assert store.list_memories(statuses=[]) == []
    assert store.list_memories("default", statuses=frozenset()) == []


def test_list_memories_statuses_none_means_all(store):
    for s in S:
        store.add_memory(_mem(id=s.value, status=s))
    assert len(store.list_memories(statuses=None)) == len(S)


def test_list_memories_combined_namespace_status_and_limit(store):
    for i in range(5):
        store.add_memory(_mem(id=f"x{i}", namespace="x", status=S.ACTIVE, created_at=i))
    store.add_memory(_mem(id="y0", namespace="y", status=S.ACTIVE, created_at=0))
    store.add_memory(_mem(id="x9", namespace="x", status=S.FROZEN, created_at=9))
    got = store.list_memories("x", {S.ACTIVE}, limit=3)
    assert [m.id for m in got] == ["x0", "x1", "x2"]


def test_list_memories_limit_zero(store):
    store.add_memory(_mem(id="a"))
    assert store.list_memories(limit=0) == []


# --------------------------------------------------------------------------- events


def test_event_round_trip_all_fields(store):
    e = Event(text="we migrated to SQLite", id="evt_1", namespace="team", source="slack", created_at=123.25,
              metadata={"channel": "#eng", "n": [1, 2]})
    store.add_event(e)
    got = store.get_event("evt_1")
    assert got == e
    assert got.metadata == {"channel": "#eng", "n": [1, 2]}


def test_event_round_trip_defaults(store):
    e = Event(text="hi")
    store.add_event(e)
    got = store.get_event(e.id)
    assert got == e
    assert got.namespace == "default" and got.source == "unknown" and got.metadata == {}


def test_get_event_missing_returns_none(store):
    assert store.get_event("nope") is None


def test_list_events_orders_newest_first_and_filters_and_limits(store):
    store.add_event(Event(text="a", id="a", created_at=10))
    store.add_event(Event(text="b", id="b", created_at=30, namespace="other"))
    store.add_event(Event(text="c", id="c", created_at=20))
    store.add_event(Event(text="c2", id="c2", created_at=20))  # tie: later insert first
    assert [e.id for e in store.list_events()] == ["b", "c2", "c", "a"]
    assert [e.id for e in store.list_events("default")] == ["c2", "c", "a"]
    assert [e.id for e in store.list_events("other")] == ["b"]
    assert [e.id for e in store.list_events(limit=2)] == ["b", "c2"]
    assert [e.id for e in store.list_events("default", limit=1)] == ["c2"]


# --------------------------------------------------------------------------- verdicts


def test_verdict_round_trip_and_autoincrement_id(store):
    v = _verdict(created_at=42.0)
    assert v.id is None
    store.add_verdicts([v])
    got = store.list_verdicts()
    assert len(got) == 1
    g = got[0]
    assert isinstance(g.id, int) and g.id >= 1
    assert g.event_id == "evt_a" and g.memory_id == "mem_a"
    assert g.votes == Votes(bears=0.9, still_true=0.1, replaces=0.8, hypothetical=0.05)
    assert g.disposition is D.SUPERSEDED
    assert g.from_status is S.ACTIVE and g.to_status is S.SUPERSEDED
    assert g.applied is True and isinstance(g.applied, bool)
    assert g.created_at == 42.0
    assert g.changed is True


def test_verdict_applied_false_round_trips_as_bool(store):
    store.add_verdicts([_verdict(applied=False, to_status=S.ACTIVE, disposition=D.UNRELATED)])
    g = store.list_verdicts()[0]
    assert g.applied is False
    assert g.changed is False


def test_add_verdicts_empty_is_noop_and_accepts_generators(store):
    store.add_verdicts([])
    store.add_verdicts(iter(()))
    assert store.list_verdicts() == []
    store.add_verdicts(_verdict(memory_id=f"m{i}") for i in range(3))
    assert len(store.list_verdicts()) == 3


def test_list_verdicts_by_memory_id_and_event_id(store):
    store.add_verdicts([
        _verdict(memory_id="m1", event_id="e1", created_at=1),
        _verdict(memory_id="m2", event_id="e1", created_at=2),
        _verdict(memory_id="m1", event_id="e2", created_at=3),
    ])
    assert [(v.memory_id, v.event_id) for v in store.list_verdicts(memory_id="m1")] == [("m1", "e1"), ("m1", "e2")]
    assert [(v.memory_id, v.event_id) for v in store.list_verdicts(event_id="e1")] == [("m1", "e1"), ("m2", "e1")]
    assert [(v.memory_id, v.event_id) for v in store.list_verdicts(memory_id="m2", event_id="e2")] == []
    assert [(v.memory_id, v.event_id) for v in store.list_verdicts(memory_id="m1", event_id="e2")] == [("m1", "e2")]
    assert store.list_verdicts(memory_id="ghost") == []


def test_list_verdicts_ordered_by_created_at_then_id(store):
    store.add_verdicts([
        _verdict(memory_id="late", created_at=100),
        _verdict(memory_id="early", created_at=1),
        _verdict(memory_id="tie1", created_at=50),
        _verdict(memory_id="tie2", created_at=50),
    ])
    assert [v.memory_id for v in store.list_verdicts()] == ["early", "tie1", "tie2", "late"]
    tie = [v for v in store.list_verdicts() if v.created_at == 50]
    assert tie[0].id < tie[1].id


# --------------------------------------------------------------------------- file-backed persistence


def test_file_store_persists_across_reopen(tmp_path):
    path = str(tmp_path / "inv.db")
    s1 = SQLiteStore(path)
    s1.add_memory(_mem(id="m1", metadata={"a": 1}))
    s1.add_event(Event(text="e", id="e1"))
    s1.add_verdicts([_verdict(memory_id="m1", event_id="e1")])
    s1.close()

    s2 = SQLiteStore(path)
    try:
        assert s2.get_memory("m1").metadata == {"a": 1}
        assert s2.get_event("e1").text == "e"
        assert len(s2.list_verdicts(memory_id="m1")) == 1
        # schema creation is idempotent: reopening did not wipe or duplicate anything
        assert len(s2.list_memories()) == 1
    finally:
        s2.close()


def test_file_store_uses_wal_journal_mode(tmp_path):
    s = SQLiteStore(str(tmp_path / "wal.db"))
    try:
        assert s._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        s.close()


def test_memory_store_does_not_use_wal(store):
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "memory"


def test_store_path_attribute(tmp_path, store):
    assert store.path == ":memory:"
    p = str(tmp_path / "x.db")
    s = SQLiteStore(p)
    try:
        assert s.path == p
    finally:
        s.close()


def test_close_then_use_raises():
    s = SQLiteStore(":memory:")
    s.close()
    with pytest.raises(sqlite3.ProgrammingError):
        s.list_memories()


def test_two_stores_on_same_file_see_each_others_writes(tmp_path):
    path = str(tmp_path / "shared.db")
    a, b = SQLiteStore(path), SQLiteStore(path)
    try:
        a.add_memory(_mem(id="from_a"))
        assert b.get_memory("from_a") is not None
    finally:
        a.close()
        b.close()


# --------------------------------------------------------------------------- thread-safety smoke


def test_concurrent_adds_from_threads(store):
    n_threads, per = 8, 40
    errors: list[BaseException] = []

    def worker(t: int) -> None:
        try:
            for i in range(per):
                store.add_memory(_mem(id=f"t{t}-{i}", namespace=f"ns{t % 2}"))
                store.add_event(Event(text="e", id=f"e{t}-{i}"))
                store.add_verdicts([_verdict(memory_id=f"t{t}-{i}", event_id=f"e{t}-{i}")])
                store.list_memories(limit=5)  # interleave reads
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []
    assert len(store.list_memories()) == n_threads * per
    assert len(store.list_events()) == n_threads * per
    assert len(store.list_verdicts()) == n_threads * per
    assert len(store.list_memories("ns0")) == (n_threads // 2) * per


def test_concurrent_updates_do_not_corrupt(store):
    store.add_memory(_mem(id="shared", p_true=1.0))

    def bump(val: float) -> None:
        m = store.get_memory("shared")
        m.p_true = val
        m.status = S.NEEDS_REVIEW
        store.update_memory(m)

    threads = [threading.Thread(target=bump, args=(i / 10,)) for i in range(10)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    got = store.get_memory("shared")
    assert got.status is S.NEEDS_REVIEW
    assert 0.0 <= got.p_true <= 0.9
