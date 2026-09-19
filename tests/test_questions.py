"""questions.py: the exact shape of what Jev is asked, and what it is never shown."""
from __future__ import annotations

import json

import pytest
from typesafe_sdk import Noul

from invalidate import questions as Q
from invalidate.types import Event, Memory


def _dump(qs: dict[str, Noul]) -> dict:
    return {k: v.model_dump() for k, v in qs.items()}


def _text(n: Noul) -> str:
    return json.dumps(n.model_dump())


# --------------------------------------------------------------------------- observe_questions


@pytest.mark.parametrize("n", [0, 1, 2, 5, 20])
def test_observe_questions_count_is_3n_plus_2(n: int):
    qs = Q.observe_questions(n)
    assert len(qs) == 3 * n + 2
    assert all(isinstance(q, Noul) for q in qs.values())


def test_observe_questions_ids_are_exactly_as_expected():
    qs = Q.observe_questions(3)
    expected = {Q.HYPOTHETICAL, Q.DIRECTIVE}
    for i in range(3):
        expected |= {f"{Q.BEARS}_{i}", f"{Q.STILL_TRUE}_{i}", f"{Q.REPLACES}_{i}"}
    assert set(qs) == expected


def test_observe_questions_hypothetical_present_exactly_once():
    qs = Q.observe_questions(4)
    assert sum(1 for k in qs if k.startswith(Q.HYPOTHETICAL)) == 1
    assert Q.HYPOTHETICAL in qs


def test_observe_questions_zero_memories_only_event_level():
    assert sorted(Q.observe_questions(0)) == sorted([Q.DIRECTIVE, Q.HYPOTHETICAL])


@pytest.mark.parametrize("prefix", [Q.BEARS, Q.STILL_TRUE, Q.REPLACES])
def test_each_memory_question_references_its_own_fact_path_with_backticks(prefix: str):
    n = 4
    qs = Q.observe_questions(n)
    for i in range(n):
        text = _text(qs[f"{prefix}_{i}"])
        assert f"`memories[{i}].fact`" in text
        # and not any other index
        for j in range(n):
            if j != i:
                assert f"memories[{j}]" not in text
        assert "`event.text`" in text


def test_memory_questions_compare_event_text_with_fact():
    qs = Q.observe_questions(1)
    for prefix in (Q.BEARS, Q.STILL_TRUE, Q.REPLACES):
        instr = qs[f"{prefix}_0"].model_dump()["instructions"]
        assert instr["compare"] == ["`event.text`", "`memories[0].fact`"]


def test_hypothetical_question_is_about_event_form_only():
    q = Q.observe_questions(1)[Q.HYPOTHETICAL].model_dump()
    assert "`event.text`" in q["instructions"]["question"]
    assert "memories" not in json.dumps(q)


def test_every_observe_question_has_true_and_false_criteria_with_examples():
    for k, q in Q.observe_questions(2).items():
        d = q.model_dump()
        assert d["type"] == "noul", k
        crit = d["criteria"]
        assert set(crit) >= {"true", "false"}, k
        for side in ("true", "false"):
            assert crit[side]["what"], k
            assert len(crit[side]["examples"]) >= 2, k


def test_observe_questions_never_mention_dates_or_arithmetic():
    text = json.dumps(_dump(Q.observe_questions(2))).lower()
    for banned in ("created_at", "updated_at", "expires_at", "timestamp", "p_true"):
        assert banned not in text


# --------------------------------------------------------------------------- recall_questions


@pytest.mark.parametrize("n", [0, 1, 3, 20])
def test_recall_questions_count_is_n(n: int):
    qs = Q.recall_questions(n)
    assert len(qs) == n
    assert set(qs) == {f"{Q.RELEVANT}_{i}" for i in range(n)}


def test_recall_questions_reference_query_and_fact():
    qs = Q.recall_questions(3)
    for i in range(3):
        d = qs[f"{Q.RELEVANT}_{i}"].model_dump()
        assert f"`memories[{i}].fact`" in d["instructions"]["question"]
        assert d["instructions"]["compare"] == ["`query`", f"`memories[{i}].fact`"]
        assert d["criteria"]["true"]["what"] and d["criteria"]["false"]["what"]


# --------------------------------------------------------------------------- state views


def test_memory_view_contains_only_fact_kind_source():
    m = Memory(fact="user prefers Postgres", kind="preference", source="chat", metadata={"secret": "x"})
    assert Q.memory_view(m) == {"fact": "user prefers Postgres", "kind": "preference", "source": "chat"}


def test_event_view_contains_only_text_and_source():
    e = Event(text="we migrated", source="slack", metadata={"ticket": 1})
    assert Q.event_view(e) == {"text": "we migrated", "source": "slack"}


def test_observe_state_shape():
    e = Event(text="evt")
    ms = [Memory(fact="a"), Memory(fact="b")]
    s = Q.observe_state(e, ms)
    assert set(s) == {"event", "memories"}
    assert s["event"] == Q.event_view(e)
    assert s["memories"] == [Q.memory_view(m) for m in ms]
    assert [m["fact"] for m in s["memories"]] == ["a", "b"]


def test_recall_state_shape():
    ms = [Memory(fact="a")]
    s = Q.recall_state("which db?", ms)
    assert s == {"query": "which db?", "memories": [Q.memory_view(ms[0])]}


def test_state_never_leaks_ids_timestamps_status_or_metadata():
    m = Memory(
        fact="user prefers Postgres", id="mem_LEAKME", kind="preference", source="chat",
        created_at=1234567890.5, updated_at=1234567891.5, last_checked=1234567892.5,
        expires_at=1234567893.5, superseded_by="mem_OTHER", p_true=0.42,
        metadata={"leak": "METADATA_LEAK"},
    )
    e = Event(text="we migrated", id="evt_LEAKME", source="slack", created_at=1234567894.5,
              metadata={"leak": "EVENT_META_LEAK"})
    for state in (Q.observe_state(e, [m]), Q.recall_state("q", [m])):
        text = json.dumps(state)
        for leak in ("mem_LEAKME", "evt_LEAKME", "mem_OTHER", "1234567", "METADATA_LEAK", "EVENT_META_LEAK",
                     "0.42", "active", "status", "created_at", "expires_at"):
            assert leak not in text, leak


# --------------------------------------------------------------------------- serialisation & budget


def test_observe_questions_are_json_serialisable():
    payload = json.dumps(_dump(Q.observe_questions(5)))
    back = json.loads(payload)
    assert len(back) == 17
    assert back[Q.HYPOTHETICAL]["type"] == "noul"


def test_recall_questions_are_json_serialisable():
    back = json.loads(json.dumps(_dump(Q.recall_questions(5))))
    assert len(back) == 5


def test_observe_state_is_json_serialisable():
    s = Q.observe_state(Event(text="e"), [Memory(fact="f", metadata={"x": object()})])
    json.dumps(s)  # metadata is excluded from the view, so this must not raise


def test_observe_questions_budget_for_default_batch_of_20_under_90k_chars():
    size = len(json.dumps(_dump(Q.observe_questions(20))))
    assert size < 90_000, size


def test_recall_questions_budget_for_default_batch_of_20_under_90k_chars():
    size = len(json.dumps(_dump(Q.recall_questions(20))))
    assert size < 90_000, size


def test_observe_questions_size_grows_linearly():
    s1 = len(json.dumps(_dump(Q.observe_questions(1))))
    s0 = len(json.dumps(_dump(Q.observe_questions(0))))
    s10 = len(json.dumps(_dump(Q.observe_questions(10))))
    per = s1 - s0
    assert abs((s10 - s0) - 10 * per) < 200  # index digits vary a little


def test_questions_are_fresh_objects_each_call():
    a = Q.observe_questions(1)
    b = Q.observe_questions(1)
    assert a is not b
    assert a[Q.HYPOTHETICAL] is not b[Q.HYPOTHETICAL]
