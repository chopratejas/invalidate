"""Lazy mode, validate(), observe_many(), cursors, budgets, and the pair screen."""
from __future__ import annotations

import sqlite3

import pytest
from conftest import CONFIRM, CONTRADICT, SUPERSEDE, UNRELATED, FakeJudge, NoPairScreenJudge

from invalidate import Invalidate, Policy, Status, Votes
from invalidate.store import SQLiteStore


def make(fake, **kw):
    kw.setdefault("second_opinion", False)  # call-count tests; the pass has its own file
    return Invalidate(":memory:", judge=fake, policy=Policy(**kw))


# ---------------------------------------------------------------- cursors in eager mode
def test_eager_observe_advances_checked_seq_for_every_judgeable_memory(fake):
    mem = make(fake)
    a = mem.remember("we use Postgres")
    b = mem.remember("deploys at 2pm")
    rep = mem.observe("anything", source="slack")
    assert rep.event.seq == 1
    assert mem.get(a.id).checked_seq == 1 and mem.get(b.id).checked_seq == 1
    assert mem.pending() == 0
    mem.observe("more", source="slack")
    assert mem.get(a.id).checked_seq == 2


def test_event_seq_is_per_namespace_and_monotonic(fake):
    mem = make(fake)
    e1 = mem.observe("a", namespace="x").event
    e2 = mem.observe("b", namespace="x").event
    e3 = mem.observe("c", namespace="y").event
    assert (e1.seq, e2.seq, e3.seq) == (1, 2, 1)
    assert mem.store.max_seq("x") == 2 and mem.store.max_seq("y") == 1


def test_dry_run_does_not_advance_cursor(fake):
    mem = make(fake)
    a = mem.remember("fact")
    mem.observe("e", dry_run=True)
    assert mem.get(a.id).checked_seq == 0 and mem.store.max_seq("default") == 0


# ---------------------------------------------------------------- lazy mode basics
def test_lazy_observe_appends_only(fake):
    fake.script("Postgres", SUPERSEDE)
    mem = make(fake, lazy=True)
    a = mem.remember("we use Postgres")
    rep = mem.observe("we moved to SQLite", source="slack")
    assert rep.judged == 0 and rep.requests == 0 and rep.pending == 1
    assert rep.event.seq == 1
    assert fake.observe_calls == []
    assert mem.get(a.id).status is Status.ACTIVE  # nothing judged yet
    assert mem.pending() == 1
    assert len(mem.events()) == 1


def test_validate_judges_pending_and_applies(fake):
    fake.script("Postgres", SUPERSEDE)
    mem = make(fake, lazy=True)
    a = mem.remember("we use Postgres")
    b = mem.remember("lunch at noon")
    mem.observe("we moved to SQLite", source="slack")
    rep = mem.validate()
    assert rep.memories == 2 and rep.events == 1 and rep.pairs == 2
    assert rep.bearing == 1  # only the Postgres pair passes the pair screen (bears 0.95 vs 0.0)
    assert [v.memory_id for v in rep.changed] == [a.id]
    assert mem.get(a.id).status is Status.SUPERSEDED
    assert mem.get(b.id).status is Status.ACTIVE
    assert mem.get(a.id).checked_seq == 1 and mem.get(b.id).checked_seq == 1
    assert mem.pending() == 0
    assert len(mem.history(a.id)) == 1 and mem.history(b.id) == []
    # Idempotent: nothing left to do.
    rep2 = mem.validate()
    assert rep2.memories == 0 and rep2.requests == 0


def test_validate_applies_events_in_order_per_memory(fake):
    """superseded in event 1, re-confirmed in event 2 -> the later event wins. Reverse order -> superseded."""
    mem = make(fake, lazy=True)
    a = mem.remember("standup at 9")

    # Script per event: FakeJudge is event-independent, so use a judge that keys on event text.
    class ByEvent(FakeJudge):
        def observe(self, event, memories):
            self.observe_calls.append((event, list(memories)))
            votes = SUPERSEDE if "moved" in event.text else CONFIRM
            from invalidate.judge import JudgeResult, ObserveBatch
            return ObserveBatch([votes] * len(memories), JudgeResult(1, "fake"))

    mem._judge = ByEvent(default_votes=CONFIRM)
    mem.observe("standup moved to 10", source="slack")
    mem.observe("standup is at 9 as always", source="slack")
    rep = mem.validate()
    vs = [v for v in rep.verdicts if v.memory_id == a.id]
    assert [v.to_status for v in vs] == [Status.SUPERSEDED, Status.SUPERSEDED]  # confirm cannot resurrect a dead memory
    # Now the opposite order on a fresh memory (born current: the two standup events are not pending for it).
    b = mem.remember("retro on Fridays")
    mem.observe("retro is on Fridays, confirmed", source="slack")
    mem.observe("retro moved to Thursdays", source="slack")
    rep = mem.validate([b])
    vs = [v for v in rep.verdicts if v.memory_id == b.id]
    assert [v.from_status for v in vs] == [Status.ACTIVE, Status.ACTIVE]
    assert [v.to_status for v in vs] == [Status.ACTIVE, Status.SUPERSEDED]
    assert mem.get(b.id).status is Status.SUPERSEDED


def test_validate_subset_leaves_others_pending(fake):
    fake.script("Postgres", CONTRADICT)
    mem = make(fake, lazy=True)
    a = mem.remember("we use Postgres")
    b = mem.remember("lunch at noon")
    mem.observe("Postgres is gone")
    rep = mem.validate([a])
    assert rep.memories == 1 and mem.get(a.id).status is Status.CONTRADICTED
    assert mem.get(b.id).checked_seq == 0 and mem.pending() == 1


def test_validate_budget_reaches_most_stale_first(fake):
    mem = make(fake, lazy=True, pair_memories=1)
    a = mem.remember("a")
    mem.observe("e1")
    b = mem.remember("b")  # born current at seq 1
    mem.observe("e2")
    assert (mem.get(a.id).checked_seq, mem.get(b.id).checked_seq) == (0, 1)
    rep = mem.validate(budget_requests=1)
    assert rep.memories == 1 and rep.pending == 1 and rep.requests <= 1
    assert mem.get(a.id).checked_seq == 2 and mem.get(b.id).checked_seq == 1  # a was more stale, a went first
    rep2 = mem.validate()
    assert rep2.memories == 1 and mem.pending() == 0


def test_remembered_memory_is_born_current(fake):
    fake.script("Postgres", SUPERSEDE)
    mem = make(fake, lazy=True)
    mem.observe("we moved to SQLite")  # last week
    a = mem.remember("we use Postgres")  # stated today: newer than the event
    assert a.checked_seq == 1 and mem.pending() == 0
    rep = mem.validate()
    assert rep.memories == 0 and mem.get(a.id).status is Status.ACTIVE


def test_validate_without_pair_screen_sends_every_pair_to_full_judge():
    fake = NoPairScreenJudge()
    fake.script("Postgres", SUPERSEDE)
    mem = make(fake, lazy=True)
    a = mem.remember("we use Postgres")
    mem.remember("lunch at noon")
    mem.observe("we moved to SQLite")
    rep = mem.validate()
    assert rep.pairs == 2 and rep.bearing == 2 and fake.pair_calls == []
    assert mem.get(a.id).status is Status.SUPERSEDED


def test_pair_screen_batches_respect_policy_shape(fake):
    mem = make(fake, lazy=True, pair_events=2, pair_memories=3)
    for i in range(7):
        mem.remember(f"fact {i}")
    for i in range(5):
        mem.observe(f"event {i}")
    rep = mem.validate()
    assert rep.pairs == 35
    shapes = sorted((len(e), len(m)) for e, m, _ in fake.pair_calls)
    assert max(len(e) for e, _, _ in fake.pair_calls) <= 2 and max(len(m) for _, m, _ in fake.pair_calls) <= 3
    assert len(shapes) == 3 * 3  # ceil(5/2)=3 event chunks x ceil(7/3)=3 memory chunks


def test_pair_screen_only_asks_about_events_the_memory_has_not_seen(fake):
    mem = make(fake, lazy=True)
    a = mem.remember("a")
    b = mem.remember("b")
    mem.observe("e1")
    mem.validate([a])  # a is at seq 1, b still at 0
    mem.observe("e2")
    rep = mem.validate()
    # a needs only e2 (1 pair); b needs e1 and e2 (2 pairs)
    assert rep.pairs == 3
    events, mems, pairs = fake.pair_calls[-1]
    asked = {(events[j].text, mems[i].id) for j, i in pairs}
    assert ("e1", a.id) not in asked and ("e1", b.id) in asked and ("e2", a.id) in asked


def test_lazy_successor_is_created_at_validate_time(fake):
    fake.script("Postgres", SUPERSEDE)
    mem = make(fake, lazy=True)
    a = mem.remember("we use Postgres", kind="config")
    mem.observe("we moved to SQLite", source="slack", remember_successor=True)
    mem.validate()
    a2 = mem.get(a.id)
    assert a2.status is Status.SUPERSEDED and a2.superseded_by is not None
    succ = mem.get(a2.superseded_by)
    assert succ.fact == "we moved to SQLite" and succ.kind == "config" and succ.checked_seq == 1
    assert mem.pending() == 0  # the successor is born current


def test_successor_is_shared_across_memories_of_one_event(fake):
    fake.script("Postgres", SUPERSEDE)
    mem = make(fake, lazy=True, pair_memories=1)  # two memory chunks, one event
    a = mem.remember("we use Postgres for A")
    b = mem.remember("we use Postgres for B")
    mem.observe("we moved to SQLite", remember_successor=True)
    mem.validate()
    sa, sb = mem.get(a.id).superseded_by, mem.get(b.id).superseded_by
    assert sa == sb and sorted(mem.get(sa).metadata["supersedes"]) == sorted([a.id, b.id])
    assert len([m for m in mem.list() if m.fact == "we moved to SQLite"]) == 1


# ---------------------------------------------------------------- recall validates
def test_recall_validates_candidates_in_lazy_mode(fake):
    fake.script("Postgres", SUPERSEDE)
    fake.relevance = {"Postgres": 0.9, "lunch": 0.8}
    mem = make(fake, lazy=True)
    a = mem.remember("we use Postgres")
    b = mem.remember("lunch at noon")
    mem.observe("we moved to SQLite")
    rep = mem.recall("database?")
    assert [r.memory.id for r in rep.results] == [b.id]
    assert rep.validated is not None and rep.validated.memories == 2
    assert mem.get(a.id).status is Status.SUPERSEDED
    assert rep.requests >= 2  # relevance + validation


def test_recall_validate_false_returns_stale_in_lazy_mode(fake):
    fake.script("Postgres", SUPERSEDE)
    fake.relevance = {"Postgres": 0.9}
    mem = make(fake, lazy=True)
    a = mem.remember("we use Postgres")
    mem.observe("we moved to SQLite")
    rep = mem.recall("database?", validate=False)
    assert [r.memory.id for r in rep.results] == [a.id] and rep.validated is None


def test_recall_candidates_restrict_pool(fake):
    fake.relevance = {"a": 0.9, "b": 0.9}
    mem = make(fake)
    a = mem.remember("fact a")
    mem.remember("fact b")
    rep = mem.recall("?", candidates=[a])
    assert rep.considered == 1 and [r.memory.id for r in rep.results] == [a.id]


# ---------------------------------------------------------------- observe_many
def test_observe_many_eager_judges_pool_once_against_all_events(fake):
    fake.script("Postgres", SUPERSEDE)
    mem = make(fake)
    a = mem.remember("we use Postgres")
    mem.remember("lunch at noon")
    events, rep = mem.observe_many(["we moved to SQLite", "", "lunch unchanged"], source="slack")
    assert [e.seq for e in events] == [1, 2]
    assert rep is not None and rep.events == 2 and rep.pairs == 4
    assert mem.get(a.id).status is Status.SUPERSEDED and mem.pending() == 0


def test_observe_many_lazy_only_appends(fake):
    mem = make(fake, lazy=True)
    mem.remember("x")
    events, rep = mem.observe_many(["a", "b"])
    assert rep is None and len(events) == 2 and mem.pending() == 1


# ---------------------------------------------------------------- eager + behind
def test_eager_observe_does_not_advance_a_memory_that_is_behind(fake):
    """Switching lazy -> eager must not skip the events a memory missed."""
    mem = make(fake, lazy=True)
    a = mem.remember("a")
    mem.observe("e1")  # a is now behind (seq 1, a at 0)
    mem.policy.lazy = False
    fake.script("a", UNRELATED)
    mem.observe("e2")  # eager: judged against e2, but still owes e1
    assert mem.get(a.id).checked_seq == 0 and mem.pending() == 1
    rep = mem.validate()
    assert rep.pairs == 2  # e1 and e2 (e2 again: cheap, and keeps the rule simple)


# ---------------------------------------------------------------- store migration
def test_store_migration_numbers_existing_events(tmp_path):
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE memories (id TEXT PRIMARY KEY, namespace TEXT NOT NULL DEFAULT 'default', fact TEXT NOT NULL,
          kind TEXT NOT NULL DEFAULT 'fact', source TEXT NOT NULL DEFAULT 'unknown', status TEXT NOT NULL DEFAULT 'active',
          p_true REAL NOT NULL DEFAULT 1.0, created_at REAL NOT NULL, updated_at REAL NOT NULL, last_checked REAL,
          expires_at REAL, superseded_by TEXT, metadata TEXT NOT NULL DEFAULT '{}');
        CREATE TABLE events (id TEXT PRIMARY KEY, namespace TEXT NOT NULL DEFAULT 'default', text TEXT NOT NULL,
          source TEXT NOT NULL DEFAULT 'unknown', created_at REAL NOT NULL, metadata TEXT NOT NULL DEFAULT '{}');
        CREATE TABLE verdicts (id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL, memory_id TEXT NOT NULL,
          bears REAL NOT NULL, still_true REAL NOT NULL, replaces REAL NOT NULL, hypothetical REAL NOT NULL,
          disposition TEXT NOT NULL, from_status TEXT NOT NULL, to_status TEXT NOT NULL, applied INTEGER NOT NULL,
          created_at REAL NOT NULL);
        INSERT INTO memories (id, fact, created_at, updated_at) VALUES ('m1', 'f', 1, 1);
        INSERT INTO events (id, text, created_at) VALUES ('e1', 'a', 10), ('e2', 'b', 20), ('e3', 'c', 5);
        """
    )
    conn.commit()
    conn.close()
    store = SQLiteStore(path)
    evs = {e.id: e.seq for e in store.list_events_after("default", 0)}
    assert evs == {"e3": 1, "e1": 2, "e2": 3}
    assert store.get_memory("m1").checked_seq == 0
    assert store.max_seq("default") == 3
    store.close()
