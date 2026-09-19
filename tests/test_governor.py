"""Governor over InMemoryAdapter with the fake judge. No network."""
from __future__ import annotations

import pytest

from conftest import CONFIRM, CONTRADICT, SUPERSEDE, UNCERTAIN, FakeJudge
from invalidate.adapters import DEAD, LIVE, Governor, GovernorReport, InMemoryAdapter, Push, Reason
from invalidate.adapters.base import _h
from invalidate.types import Status

ITEMS = {"a": "user prefers Postgres", "b": "deploys run at 2pm UTC", "c": "Alice owns the billing service"}


@pytest.fixture
def adapter() -> InMemoryAdapter:
    return InMemoryAdapter(dict(ITEMS))


@pytest.fixture
def make_gov(fake: FakeJudge, adapter: InMemoryAdapter):
    made: list[Governor] = []

    def _make(**kw) -> Governor:
        g = Governor(kw.pop("adapter", adapter), ":memory:", judge=fake, **kw)
        made.append(g)
        return g

    yield _make
    for g in made:
        g.close()


@pytest.fixture
def gov(make_gov) -> Governor:
    return make_gov()


class FailingAdapter(InMemoryAdapter):
    """flag/delete/insert raise so push errors can be observed."""

    def flag(self, host_id, reason):
        raise RuntimeError("host down")

    def delete(self, host_id, reason):
        raise RuntimeError("host down")

    def insert(self, text, source, metadata):
        raise RuntimeError("host down")


# -- construction ------------------------------------------------------------------------
def test_invalid_mode_raises(fake, adapter):
    with pytest.raises(ValueError):
        Governor(adapter, ":memory:", judge=fake, mode="yolo")


def test_namespace_and_id_mapping(gov, adapter):
    assert gov.namespace == "host:memory"
    assert gov.our_id("a") == "memory:a"
    gov.sync()
    m = gov.mem.get("memory:a")
    assert Governor.host_id(m) == "a"
    assert m.namespace == "host:memory"
    assert m.metadata["host"] == "memory"
    assert m.metadata["host_hash"] == _h(ITEMS["a"])


def test_custom_namespace(make_gov):
    g = make_gov(namespace="custom")
    assert g.namespace == "custom"
    g.sync()
    assert all(m.namespace == "custom" for m in g.mem.list())


def test_context_manager(fake, adapter):
    with Governor(adapter, ":memory:", judge=fake) as g:
        assert g.sync().added == 3


# -- sync ----------------------------------------------------------------------------------
def test_sync_adds_everything_first_time(gov):
    rep = gov.sync()
    assert (rep.added, rep.updated, rep.removed, rep.unchanged, rep.total) == (3, 0, 0, 0, 3)
    assert {m.fact for m in gov.mem.list()} == set(ITEMS.values())
    assert "3 host memories: 3 added" in str(rep)


def test_sync_is_idempotent(gov):
    gov.sync()
    rep = gov.sync()
    assert (rep.added, rep.updated, rep.removed, rep.unchanged) == (0, 0, 0, 3)


def test_sync_adds_new_host_rows(gov, adapter):
    gov.sync()
    adapter.items["d"] = {"text": "lunch is at noon on Fridays", "meta": {"team": "eng"}}
    rep = gov.sync()
    assert (rep.added, rep.unchanged) == (1, 3)
    assert gov.mem.get("memory:d").metadata["team"] == "eng"


def test_sync_text_change_resets_row(make_gov, adapter, fake):
    fake.script("memory:a", SUPERSEDE)
    gov = make_gov(successors=True)
    gov.sync()
    gov.observe("we migrated to SQLite", source="slack")
    m = gov.mem.get("memory:a")
    assert m.status is Status.SUPERSEDED and m.superseded_by == "memory:new1" and m.p_true < 0.5
    adapter.items["a"]["text"] = "user prefers SQLite"
    rep = gov.sync()
    assert rep.updated == 1
    m = gov.mem.get("memory:a")
    assert m.fact == "user prefers SQLite"
    assert m.status is Status.ACTIVE
    assert m.superseded_by is None
    assert m.p_true == 1.0
    assert m.metadata["host_hash"] == _h("user prefers SQLite")


def test_sync_removes_only_live_rows(gov, adapter):
    gov.sync()
    del adapter.items["b"]
    rep = gov.sync()
    assert rep.removed == 1
    assert gov.status_of("b") is Status.DELETED
    # Gone rows are not counted again.
    rep = gov.sync()
    assert rep.removed == 0 and rep.unchanged == 2


def test_sync_does_not_count_row_we_deleted_in_delete_mode(make_gov, adapter, fake):
    fake.script("memory:a", CONTRADICT)
    g = make_gov(mode="delete")
    g.sync()
    g.observe("nobody prefers Postgres anymore", source="slack")
    assert "a" not in adapter.items
    assert g.status_of("a") is Status.CONTRADICTED
    rep = g.sync()
    assert rep.removed == 0
    assert g.status_of("a") is Status.CONTRADICTED


def test_sync_contradicted_row_that_vanishes_stays_contradicted(gov, adapter, fake):
    fake.script("memory:a", CONTRADICT)
    gov.sync()
    gov.observe("nobody prefers Postgres anymore", source="slack")
    assert gov.status_of("a") is Status.CONTRADICTED
    del adapter.items["a"]
    rep = gov.sync()
    assert rep.removed == 0
    assert gov.status_of("a") is Status.CONTRADICTED


def test_sync_forgotten_row_not_counted_removed(gov, adapter):
    gov.sync()
    gov.forget("c")
    assert "c" not in adapter.items
    rep = gov.sync()
    assert rep.removed == 0 and rep.total == 2


def test_sync_removes_needs_review_and_frozen_rows_too(gov, adapter, fake):
    fake.script("memory:a", UNCERTAIN)
    gov.sync()
    gov.observe("something about Postgres", source="slack")
    gov.mem.freeze("memory:b")
    assert gov.status_of("a") is Status.NEEDS_REVIEW
    adapter.items.clear()
    rep = gov.sync()
    assert rep.removed == 3
    assert LIVE == {Status.ACTIVE, Status.NEEDS_REVIEW, Status.FROZEN}
    assert DEAD == {Status.CONTRADICTED, Status.SUPERSEDED}


def test_sync_ignores_ledger_rows_without_host_id(gov, adapter):
    gov.sync()
    gov.mem.remember("a local-only fact", source="test")
    rep = gov.sync()
    assert rep.unchanged == 3 and rep.removed == 0


def test_sync_flag_metadata_does_not_look_like_a_text_change(gov, adapter, fake):
    fake.script("memory:a", SUPERSEDE)
    gov.sync()
    gov.observe("we migrated to SQLite", source="slack")
    assert adapter.items["a"]["meta"]["invalidate_status"] == "superseded"
    rep = gov.sync()
    assert rep.updated == 0 and rep.unchanged == 3
    assert gov.status_of("a") is Status.SUPERSEDED


# -- observe: modes ------------------------------------------------------------------------
def test_observe_flag_mode_writes_reason_metadata(gov, adapter, fake):
    fake.script("memory:a", SUPERSEDE)
    gov.sync()
    rep = gov.observe("we migrated to SQLite", source="slack")
    meta = adapter.items["a"]["meta"]
    assert meta["invalidate_status"] == "superseded"
    assert meta["invalidate_disposition"] == "superseded"
    assert meta["invalidate_event"] == "we migrated to SQLite"
    assert meta["invalidate_event_source"] == "slack"
    assert meta["invalidate_event_id"] == rep.report.event.id
    assert meta["invalidate_still_true"] == pytest.approx(0.05)
    assert isinstance(meta["invalidate_at"], float)
    assert [p.action for p in rep.pushes] == ["flag"]
    assert rep.pushes[0] == Push("a", "flag", Status.SUPERSEDED)
    assert "b" not in {p.host_id for p in rep.pushes}
    assert "invalidate_status" not in adapter.items["b"]["meta"]


def test_observe_flag_mode_contradicted(gov, adapter, fake):
    fake.script("memory:c", CONTRADICT)
    gov.sync()
    rep = gov.observe("Alice left the company", source="hr")
    assert adapter.items["c"]["meta"]["invalidate_status"] == "contradicted"
    assert rep.pushes == [Push("c", "flag", Status.CONTRADICTED)]
    assert gov.status_of("c") is Status.CONTRADICTED


def test_observe_delete_mode_deletes_dead_but_flags_review(make_gov, adapter, fake):
    fake.script("memory:a", SUPERSEDE).script("memory:b", UNCERTAIN)
    g = make_gov(mode="delete")
    g.sync()
    rep = g.observe("we migrated to SQLite and deploys moved", source="slack")
    assert "a" not in adapter.items
    assert adapter.items["b"]["meta"]["invalidate_status"] == "needs_review"
    assert {(p.host_id, p.action, p.status) for p in rep.pushes} == {
        ("a", "delete", Status.SUPERSEDED), ("b", "flag", Status.NEEDS_REVIEW)}
    assert g.status_of("a") is Status.SUPERSEDED
    assert g.status_of("b") is Status.NEEDS_REVIEW


def test_observe_ledger_mode_pushes_nothing(make_gov, adapter, fake):
    fake.script("memory:a", SUPERSEDE)
    g = make_gov(mode="ledger")
    g.sync()
    rep = g.observe("we migrated to SQLite", source="slack")
    assert rep.pushes == []
    assert adapter.items["a"]["meta"] == {}
    assert adapter.log == []
    assert g.status_of("a") is Status.SUPERSEDED


def test_observe_confirm_from_review_flags_active(gov, adapter, fake):
    fake.script("memory:a", UNCERTAIN)
    gov.sync()
    gov.observe("Postgres, maybe?", source="slack")
    assert gov.status_of("a") is Status.NEEDS_REVIEW
    fake.script("memory:a", CONFIRM)
    rep = gov.observe("Postgres it is", source="slack")
    assert gov.status_of("a") is Status.ACTIVE
    assert rep.pushes == [Push("a", "flag", Status.ACTIVE)]
    assert adapter.items["a"]["meta"]["invalidate_status"] == "active"


def test_observe_unrelated_pushes_nothing(gov, adapter):
    gov.sync()
    rep = gov.observe("the weather is nice", source="slack")
    assert rep.pushes == []
    assert rep.report.judged == 3
    assert all(item["meta"] == {} for item in adapter.items.values())


def test_observe_dry_run_pushes_nothing(gov, adapter, fake):
    fake.script("memory:a", SUPERSEDE)
    gov.sync()
    rep = gov.observe("we migrated to SQLite", source="slack", dry_run=True)
    assert rep.pushes == []
    assert rep.successor_host_id is None
    assert gov.status_of("a") is Status.ACTIVE
    assert adapter.items["a"]["meta"] == {}
    assert len(rep.report.changed) == 1  # the verdict is reported, just not applied


def test_observe_dry_run_with_successors_inserts_nothing(make_gov, adapter, fake):
    fake.script("memory:a", SUPERSEDE)
    g = make_gov(successors=True)
    g.sync()
    rep = g.observe("we migrated to SQLite", source="slack", dry_run=True)
    assert rep.successor_host_id is None and len(adapter.items) == 3


def test_observe_passes_kwargs_through(gov, fake):
    gov.sync()
    rep = gov.observe("hello world", source="slack", metadata={"k": 1})
    assert rep.report.event.metadata == {"k": 1}
    assert fake.observe_calls[0][0].source == "slack"


# -- successors ----------------------------------------------------------------------------
def test_successors_insert_verbatim_and_link(make_gov, adapter, fake):
    fake.script("memory:a", SUPERSEDE)
    g = make_gov(successors=True)
    g.sync()
    rep = g.observe("we migrated to SQLite last Tuesday", source="slack")
    assert rep.successor_host_id == "new1"
    assert adapter.items["new1"]["text"] == "we migrated to SQLite last Tuesday"
    assert adapter.items["new1"]["meta"]["source"] == "slack"
    assert adapter.items["new1"]["meta"]["invalidate_supersedes"] == ["a"]
    assert adapter.items["new1"]["meta"]["invalidate_event_id"] == rep.report.event.id
    a = g.mem.get("memory:a")
    assert a.status is Status.SUPERSEDED
    assert a.superseded_by == "memory:new1"
    succ = g.mem.get("memory:new1")
    assert succ.fact == "we migrated to SQLite last Tuesday"
    assert succ.status is Status.ACTIVE
    assert Governor.host_id(succ) == "new1"
    assert succ.metadata["event_id"] == rep.report.event.id
    assert Push("new1", "insert", Status.ACTIVE) in rep.pushes
    assert Push("a", "flag", Status.SUPERSEDED) in rep.pushes


def test_successor_is_synced_as_unchanged_next_time(make_gov, adapter, fake):
    fake.script("memory:a", SUPERSEDE)
    g = make_gov(successors=True)
    g.sync()
    g.observe("we migrated to SQLite", source="slack")
    rep = g.sync()
    assert rep.added == 0 and rep.unchanged == 4


def test_successors_not_inserted_when_nothing_superseded(make_gov, adapter, fake):
    fake.script("memory:a", CONTRADICT)
    g = make_gov(successors=True)
    g.sync()
    rep = g.observe("Postgres is gone", source="slack")
    assert rep.successor_host_id is None
    assert len(adapter.items) == 3
    assert [p.action for p in rep.pushes] == ["flag"]


def test_successors_off_by_default(gov, adapter, fake):
    fake.script("memory:a", SUPERSEDE)
    gov.sync()
    rep = gov.observe("we migrated to SQLite", source="slack")
    assert rep.successor_host_id is None and len(adapter.items) == 3
    assert gov.mem.get("memory:a").superseded_by is None


def test_successors_ignored_in_ledger_mode(make_gov, adapter, fake):
    fake.script("memory:a", SUPERSEDE)
    g = make_gov(successors=True, mode="ledger")
    g.sync()
    rep = g.observe("we migrated to SQLite", source="slack")
    assert rep.successor_host_id is None and len(adapter.items) == 3


def test_successors_skipped_when_adapter_cannot_insert(make_gov, fake):
    class NoInsert:
        name = "noinsert"

        def __init__(self):
            self.inner = InMemoryAdapter(dict(ITEMS))

        def pull(self):
            return self.inner.pull()

        def flag(self, hid, reason):
            self.inner.flag(hid, reason)

        def delete(self, hid, reason):
            self.inner.delete(hid, reason)

    ad = NoInsert()
    fake.script("noinsert:a", SUPERSEDE)
    g = make_gov(adapter=ad, successors=True)
    g.sync()
    rep = g.observe("we migrated to SQLite", source="slack")
    assert rep.successor_host_id is None
    assert g.status_of("a") is Status.SUPERSEDED
    assert len(ad.inner.items) == 3


def test_successor_id_reused_when_host_dedupes(make_gov, fake):
    class Dedup(InMemoryAdapter):
        def insert(self, text, source, metadata):
            for k, v in self.items.items():
                if v["text"] == text:
                    return k
            return super().insert(text, source, metadata)

    ad = Dedup({"a": "user prefers Postgres"})
    fake.script("memory:a", SUPERSEDE).script("memory:b", SUPERSEDE)
    g = make_gov(adapter=ad, successors=True)
    g.sync()
    first = g.observe("we migrated to SQLite", source="slack")
    ad.items["b"] = {"text": "prod runs on Postgres", "meta": {}}
    g.sync()
    second = g.observe("we migrated to SQLite", source="slack")
    assert first.successor_host_id == second.successor_host_id == "new1"
    assert g.mem.get("memory:b").superseded_by == "memory:new1"
    assert len([m for m in g.mem.list() if m.fact == "we migrated to SQLite"]) == 1


# -- push errors ---------------------------------------------------------------------------
def test_flag_error_is_captured_and_ledger_still_updates(make_gov, fake):
    ad = FailingAdapter(dict(ITEMS))
    fake.script("memory:a", SUPERSEDE)
    g = make_gov(adapter=ad)
    g.sync()
    rep = g.observe("we migrated to SQLite", source="slack")
    assert len(rep.pushes) == 1
    p = rep.pushes[0]
    assert p.host_id == "a" and p.action == "flag" and p.status is Status.SUPERSEDED
    assert p.error and "host down" in p.error
    assert rep.errors == [p]
    assert g.status_of("a") is Status.SUPERSEDED
    assert g.mem.get("memory:a").p_true == pytest.approx(0.05)


def test_delete_error_is_captured(make_gov, fake):
    ad = FailingAdapter(dict(ITEMS))
    fake.script("memory:a", CONTRADICT)
    g = make_gov(adapter=ad, mode="delete")
    g.sync()
    rep = g.observe("Postgres is gone", source="slack")
    assert rep.pushes[0].action == "delete" and rep.pushes[0].error
    assert g.status_of("a") is Status.CONTRADICTED


def test_insert_error_is_captured(make_gov, fake):
    ad = FailingAdapter(dict(ITEMS))
    fake.script("memory:a", SUPERSEDE)
    g = make_gov(adapter=ad, successors=True)
    g.sync()
    rep = g.observe("we migrated to SQLite", source="slack")
    ins = [p for p in rep.pushes if p.action == "insert"]
    assert len(ins) == 1 and ins[0].error and ins[0].host_id == ""
    assert rep.successor_host_id is None
    assert g.status_of("a") is Status.SUPERSEDED
    assert g.mem.get("memory:a").superseded_by is None
    assert len(rep.errors) == 2  # flag + insert


def test_pushes_reset_between_observes(gov, adapter, fake):
    fake.script("memory:a", SUPERSEDE)
    gov.sync()
    first = gov.observe("we migrated to SQLite", source="slack")
    second = gov.observe("we migrated to SQLite again", source="slack")
    assert len(first.pushes) == 1 and second.pushes == []


# -- report --------------------------------------------------------------------------------
def test_report_summary_and_errors(gov, adapter, fake):
    fake.script("memory:a", SUPERSEDE)
    gov.sync()
    rep = gov.observe("we migrated to SQLite", source="slack")
    assert isinstance(rep, GovernorReport)
    assert rep.summary().startswith(rep.report.summary())
    assert rep.summary().endswith("; 1 pushed to host")
    assert rep.errors == []


def test_report_summary_counts_errors(make_gov, fake):
    ad = FailingAdapter(dict(ITEMS))
    fake.script("memory:a", SUPERSEDE).script("memory:b", CONTRADICT)
    g = make_gov(adapter=ad)
    g.sync()
    rep = g.observe("everything changed", source="slack")
    assert rep.summary().endswith("; 0 pushed to host, 2 push errors")
    assert len(rep.errors) == 2


# -- read side -----------------------------------------------------------------------------
def test_status_of_dead_ids_review(gov, fake):
    fake.script("memory:a", SUPERSEDE).script("memory:b", CONTRADICT).script("memory:c", UNCERTAIN)
    gov.sync()
    assert gov.status_of("a") is Status.ACTIVE
    assert gov.status_of("zzz") is None
    gov.observe("everything changed", source="slack")
    assert gov.dead_ids() == {"a", "b"}
    assert [Governor.host_id(m) for m in gov.review()] == ["c"]
    assert gov.status_of("c") is Status.NEEDS_REVIEW


def test_filter_hides_dead_and_optionally_review(gov, fake):
    fake.script("memory:a", SUPERSEDE).script("memory:c", UNCERTAIN)
    gov.sync()
    gov.observe("everything changed", source="slack")
    results = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "unknown"}]
    kept = gov.filter(results, id_of=lambda r: r["id"])
    assert [r["id"] for r in kept] == ["b", "c", "unknown"]
    kept = gov.filter(results, id_of=lambda r: r["id"], include_review=False)
    assert [r["id"] for r in kept] == ["b", "unknown"]


def test_filter_coerces_ids_to_str(gov, fake):
    ad = InMemoryAdapter({"1": "user prefers Postgres", "2": "deploys run at 2pm UTC"})
    g = Governor(ad, ":memory:", judge=fake.script("memory:1", CONTRADICT))
    try:
        g.sync()
        g.observe("Postgres is gone", source="slack")
        assert g.filter([1, 2, 3], id_of=lambda x: x) == [2, 3]
    finally:
        g.close()


def test_recall_delegates_to_engine(gov, fake):
    fake.relevance["memory:a"] = 0.9
    gov.sync()
    rep = gov.recall("which database?")
    assert [m.id for m in rep.memories] == ["memory:a"]


# -- guard ---------------------------------------------------------------------------------
def test_guard_observes_string_then_calls_through_and_syncs(gov, adapter, fake):
    fake.script("memory:a", SUPERSEDE)
    gov.sync()
    calls = []

    def add(text, **kw):
        calls.append((text, kw))
        adapter.items["d"] = {"text": text, "meta": {}}
        return "d"

    guarded = gov.guard(add)
    assert guarded.__name__ == "add"
    out = guarded("we migrated to SQLite", user="u1")
    assert out == "d"
    assert calls == [("we migrated to SQLite", {"user": "u1"})]
    assert gov.status_of("a") is Status.SUPERSEDED
    assert fake.observe_calls[0][0].source == "user"
    assert gov.status_of("d") is Status.ACTIVE  # synced after the add


def test_guard_observes_only_user_messages(gov, fake):
    gov.sync()
    add = lambda msgs: len(msgs)  # noqa: E731
    guarded = gov.guard(add)
    n = guarded([
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "we migrated to SQLite"},
        {"role": "assistant", "content": "noted"},
        {"role": "user", "content": "   "},
        {"content": "no role means user"},
    ])
    assert n == 5
    texts = [e.text for e, _ in fake.observe_calls]
    assert texts == ["we migrated to SQLite", "no role means user"]


def test_guard_custom_text_of_and_source(gov, fake):
    gov.sync()
    guarded = gov.guard(lambda **kw: "ok", text_of=lambda **kw: [kw["note"]], source="crm")
    assert guarded(note="Alice left") == "ok"
    assert fake.observe_calls[0][0].text == "Alice left"
    assert fake.observe_calls[0][0].source == "crm"


def test_guard_keyword_messages_default(gov, fake):
    gov.sync()
    guarded = gov.guard(lambda **kw: None)
    guarded(messages=[{"role": "user", "content": "hello there"}])
    assert [e.text for e, _ in fake.observe_calls] == ["hello there"]


# -- human controls ------------------------------------------------------------------------
def test_keep_restores_and_mirrors(gov, adapter, fake):
    fake.script("memory:a", SUPERSEDE)
    gov.sync()
    gov.observe("we migrated to SQLite", source="slack")
    gov.mem.supersede("memory:a", by="memory:b")
    m = gov.keep("a")
    assert m.status is Status.ACTIVE and m.superseded_by is None
    assert gov.status_of("a") is Status.ACTIVE
    meta = adapter.items["a"]["meta"]
    assert meta["invalidate_status"] == "active"
    assert meta["invalidate_disposition"] == "restored"
    assert meta["invalidate_event_source"] == "human"
    assert adapter.log[-1][0] == "flag"


def test_forget_deletes_and_mirrors(gov, adapter):
    gov.sync()
    m = gov.forget("b")
    assert m.status is Status.DELETED
    assert gov.status_of("b") is Status.DELETED
    assert "b" not in adapter.items
    assert adapter.log[-1][:2] == ("delete", "b")
    assert adapter.log[-1][2]["invalidate_status"] == "deleted"


def test_keep_forget_ledger_mode_touch_nothing_in_host(make_gov, adapter):
    g = make_gov(mode="ledger")
    g.sync()
    g.forget("b")
    g.keep("b")
    assert "b" in adapter.items and adapter.log == []
    assert g.status_of("b") is Status.ACTIVE


def test_keep_unknown_raises(gov):
    gov.sync()
    with pytest.raises(KeyError):
        gov.keep("nope")


def test_reason_line_and_metadata():
    r = Reason(Status.SUPERSEDED, "superseded", "we migrated to SQLite", "slack", "evt_1", 0.12, 1.0)
    assert r.line() == "invalidate: superseded by “we migrated to SQLite” (slack, still true 12%)"
    assert Reason(Status.CONTRADICTED, "c", "x", "s", "e", 0.0, 1.0).line().startswith("invalidate: contradicted by")
    assert Reason(Status.NEEDS_REVIEW, "c", "x", "s", "e", 0.5, 1.0).line().startswith("invalidate: unclear after")
    assert Reason(Status.ACTIVE, "c", "x", "s", "e", 1.0, 1.0).line().startswith("invalidate: restored after")
    md = r.as_metadata(prefix="x_")
    assert md["x_status"] == "superseded" and md["x_still_true"] == 0.12 and md["x_event_id"] == "evt_1"
