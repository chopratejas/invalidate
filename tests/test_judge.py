"""JevJudge with a stub client; MissingAPIKey; run_batches."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

import typesafe_sdk
from invalidate import JevJudge, MissingAPIKey, Votes
from invalidate import questions as Q
from invalidate.judge import JudgeResult, ObserveBatch, RecallBatch, run_batches
from invalidate.types import Event, Memory


class StubClient:
    """Records system_one(state, questions) calls and answers from a script."""

    def __init__(self, answers: dict[str, float], input_tokens=123, model="jev-1.13", with_usage=True):
        self.answers = answers
        self.input_tokens = input_tokens
        self.model = model
        self.with_usage = with_usage
        self.calls: list[tuple[dict, dict]] = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        ans = {k: SimpleNamespace(noul=self.answers[k]) for k in questions}
        resp = SimpleNamespace(answers=ans, model=self.model)
        if self.with_usage:
            resp.usage = SimpleNamespace(input_tokens=self.input_tokens)
        return resp


def _mems(n: int) -> list[Memory]:
    return [Memory(fact=f"fact {i}", id=f"mem_{i}") for i in range(n)]


# --------------------------------------------------------------------------- observe


def test_observe_maps_votes_in_order_and_copies_hypothetical():
    answers = {Q.HYPOTHETICAL: 0.33, Q.DIRECTIVE: 0.11}
    for i in range(3):
        answers[f"{Q.BEARS}_{i}"] = 0.1 * (i + 1)
        answers[f"{Q.STILL_TRUE}_{i}"] = 0.2 * (i + 1)
        answers[f"{Q.REPLACES}_{i}"] = 0.3 * (i + 1)
        answers[f"{Q.PARTIAL}_{i}"] = 0.05 * (i + 1)
    client = StubClient(answers)
    judge = JevJudge(client=client, staged=False)
    batch = judge.observe(Event(text="evt"), _mems(3))

    assert isinstance(batch, ObserveBatch)
    assert len(batch.votes) == 3
    for i, v in enumerate(batch.votes):
        assert isinstance(v, Votes)
        assert v.bears == pytest.approx(0.1 * (i + 1))
        assert v.still_true == pytest.approx(0.2 * (i + 1))
        assert v.replaces == pytest.approx(0.3 * (i + 1))
        assert v.hypothetical == pytest.approx(0.33)
        assert v.directive == pytest.approx(0.11)
        assert v.partial == pytest.approx(0.05 * (i + 1))


def test_observe_sends_state_and_3n_plus_2_questions_to_client():
    answers = {Q.HYPOTHETICAL: 0.0, Q.DIRECTIVE: 0.0}
    for i in range(2):
        answers.update({f"{Q.BEARS}_{i}": 0.5, f"{Q.STILL_TRUE}_{i}": 0.5, f"{Q.REPLACES}_{i}": 0.5, f"{Q.PARTIAL}_{i}": 0.5})
    client = StubClient(answers)
    judge = JevJudge(client=client, staged=False)
    e = Event(text="evt", source="slack")
    ms = _mems(2)
    judge.observe(e, ms)
    assert len(client.calls) == 1
    state, questions = client.calls[0]
    assert state == Q.observe_state(e, ms)
    assert set(questions) == set(Q.observe_questions(2))
    assert len(questions) == 10


def test_observe_propagates_usage_and_model():
    answers = {Q.HYPOTHETICAL: 0.0, Q.DIRECTIVE: 0.0, f"{Q.BEARS}_0": 1.0, f"{Q.STILL_TRUE}_0": 1.0, f"{Q.REPLACES}_0": 0.0, f"{Q.PARTIAL}_0": 0.0}
    judge = JevJudge(client=StubClient(answers, input_tokens=456, model="jev-9"))
    batch = judge.observe(Event(text="e"), _mems(1))
    assert batch.usage == JudgeResult(input_tokens=456, model="jev-9")
    assert isinstance(batch.usage.input_tokens, int)


def test_observe_missing_usage_and_model_default_to_zero_and_none():
    answers = {Q.HYPOTHETICAL: 0.0, Q.DIRECTIVE: 0.0, f"{Q.BEARS}_0": 1.0, f"{Q.STILL_TRUE}_0": 1.0, f"{Q.REPLACES}_0": 0.0, f"{Q.PARTIAL}_0": 0.0}
    judge = JevJudge(client=StubClient(answers, model=None, with_usage=False))
    batch = judge.observe(Event(text="e"), _mems(1))
    assert batch.usage == JudgeResult(0, None)


def test_observe_usage_none_tokens_becomes_zero():
    answers = {Q.HYPOTHETICAL: 0.0, Q.DIRECTIVE: 0.0, f"{Q.BEARS}_0": 1.0, f"{Q.STILL_TRUE}_0": 1.0, f"{Q.REPLACES}_0": 0.0, f"{Q.PARTIAL}_0": 0.0}
    judge = JevJudge(client=StubClient(answers, input_tokens=None))
    assert judge.observe(Event(text="e"), _mems(1)).usage.input_tokens == 0


def test_observe_empty_memories_short_circuits_without_calling_client():
    client = StubClient({})
    judge = JevJudge(client=client)
    batch = judge.observe(Event(text="e"), [])
    assert batch == ObserveBatch([], JudgeResult(0, None))
    assert client.calls == []


def test_observe_noul_is_coerced_to_float():
    answers = {Q.HYPOTHETICAL: "0.25", Q.DIRECTIVE: "0.1", f"{Q.BEARS}_0": 1, f"{Q.STILL_TRUE}_0": "0.5", f"{Q.REPLACES}_0": 0, f"{Q.PARTIAL}_0": "0.3"}
    v = JevJudge(client=StubClient(answers), staged=False).observe(Event(text="e"), _mems(1)).votes[0]
    assert v == Votes(bears=1.0, still_true=0.5, replaces=0.0, hypothetical=0.25, directive=0.1, partial=0.3)
    assert all(isinstance(x, float) for x in (v.bears, v.still_true, v.replaces, v.hypothetical, v.directive))


def test_staged_observe_asks_replaces_and_partial_only_for_low_still_true():
    answers = {Q.HYPOTHETICAL: 0.0, Q.DIRECTIVE: 0.0}
    still = [0.9, 0.2, 0.35, 0.36]  # 1 and 2 are at or below stage_below=0.35
    for i, st in enumerate(still):
        answers.update({f"{Q.BEARS}_{i}": 0.9, f"{Q.STILL_TRUE}_{i}": st, f"{Q.REPLACES}_{i}": 0.7, f"{Q.PARTIAL}_{i}": 0.4})
    client = StubClient(answers, input_tokens=100)
    judge = JevJudge(client=client)  # staged by default
    batch = judge.observe(Event(text="e"), _mems(4))
    assert len(client.calls) == 2
    q1, q2 = client.calls[0][1], client.calls[1][1]
    assert set(q1) == {Q.HYPOTHETICAL, Q.DIRECTIVE} | {f"{Q.BEARS}_{i}" for i in range(4)} | {f"{Q.STILL_TRUE}_{i}" for i in range(4)}
    assert set(q2) == {f"{Q.REPLACES}_1", f"{Q.PARTIAL}_1", f"{Q.REPLACES}_2", f"{Q.PARTIAL}_2"}
    assert client.calls[0][0] is client.calls[1][0] or client.calls[0][0] == client.calls[1][0]  # same state
    assert [v.replaces for v in batch.votes] == [0.0, 0.7, 0.7, 0.0]
    assert [v.partial for v in batch.votes] == [0.0, 0.4, 0.4, 0.0]
    assert [v.still_true for v in batch.votes] == still
    assert batch.usage.input_tokens == 200  # both requests counted


def test_staged_observe_skips_stage_two_when_nothing_is_low():
    answers = {Q.HYPOTHETICAL: 0.0, Q.DIRECTIVE: 0.0, f"{Q.BEARS}_0": 0.9, f"{Q.STILL_TRUE}_0": 0.8,
               f"{Q.REPLACES}_0": 0.7, f"{Q.PARTIAL}_0": 0.4}
    client = StubClient(answers, input_tokens=100)
    batch = JevJudge(client=client).observe(Event(text="e"), _mems(1))
    assert len(client.calls) == 1 and batch.usage.input_tokens == 100
    assert batch.votes[0].replaces == 0.0 and batch.votes[0].partial == 0.0


def test_engine_passes_policy_stage_threshold():
    from invalidate import Invalidate, Policy

    mem = Invalidate(":memory:", policy=Policy(contradict_max=0.5, margin=0.1), api_key="x")
    j = mem.judge
    assert isinstance(j, JevJudge) and j.staged and j.stage_below == pytest.approx(0.4)


# --------------------------------------------------------------------------- recall


def test_recall_maps_relevance_in_order_and_propagates_usage():
    answers = {f"{Q.RELEVANT}_{i}": 0.25 * i for i in range(4)}
    client = StubClient(answers, input_tokens=99, model="jev-r")
    judge = JevJudge(client=client)
    ms = _mems(4)
    batch = judge.recall("which db?", ms)
    assert isinstance(batch, RecallBatch)
    assert batch.relevance == pytest.approx([0.0, 0.25, 0.5, 0.75])
    assert batch.usage == JudgeResult(99, "jev-r")
    state, questions = client.calls[0]
    assert state == Q.recall_state("which db?", ms)
    assert set(questions) == {f"{Q.RELEVANT}_{i}" for i in range(4)}


def test_recall_empty_memories_short_circuits_without_calling_client():
    client = StubClient({})
    batch = JevJudge(client=client).recall("q", [])
    assert batch == RecallBatch([], JudgeResult(0, None))
    assert client.calls == []


# --------------------------------------------------------------------------- construction / API key


def test_missing_api_key_raises_when_no_key_and_no_client(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(MissingAPIKey) as ei:
        JevJudge()
    assert "TYPESAFE_API_KEY" in str(ei.value)
    assert isinstance(ei.value, RuntimeError)


def test_empty_api_key_string_is_treated_as_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(MissingAPIKey):
        JevJudge(api_key="")
    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    with pytest.raises(MissingAPIKey):
        JevJudge()


def test_client_argument_skips_key_lookup(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    client = StubClient({})
    judge = JevJudge(client=client, model="m")
    assert judge.client is client
    assert judge.model == "m"


def test_explicit_api_key_constructs_sdk_client(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    made = {}

    class FakeSDKClient:
        def __init__(self, api_key, model, timeout):
            made.update(api_key=api_key, model=model, timeout=timeout)

    monkeypatch.setattr(typesafe_sdk, "TypeSafeClient", FakeSDKClient)
    judge = JevJudge(api_key="sk-test", model="jev-x", timeout=5.0)
    assert isinstance(judge.client, FakeSDKClient)
    assert made == {"api_key": "sk-test", "model": "jev-x", "timeout": 5.0}
    assert judge.model == "jev-x"


def test_env_api_key_is_used_when_no_explicit_key(monkeypatch):
    made = {}

    class FakeSDKClient:
        def __init__(self, api_key, model, timeout):
            made["api_key"] = api_key

    monkeypatch.setattr(typesafe_sdk, "TypeSafeClient", FakeSDKClient)
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-from-env")
    JevJudge()
    assert made["api_key"] == "sk-from-env"


def test_explicit_api_key_wins_over_env(monkeypatch):
    made = {}

    class FakeSDKClient:
        def __init__(self, api_key, model, timeout):
            made["api_key"] = api_key

    monkeypatch.setattr(typesafe_sdk, "TypeSafeClient", FakeSDKClient)
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-from-env")
    JevJudge(api_key="sk-explicit")
    assert made["api_key"] == "sk-explicit"


# --------------------------------------------------------------------------- run_batches


def test_run_batches_empty_returns_empty():
    assert run_batches(lambda b: b, [], max_workers=8) == []


def test_run_batches_preserves_order_under_concurrency():
    def slow(b):
        time.sleep(0.02 * (5 - b))  # earlier batches finish later
        return b * 10

    assert run_batches(slow, [1, 2, 3, 4, 5], max_workers=5) == [10, 20, 30, 40, 50]


def test_run_batches_sequential_when_single_worker_runs_in_caller_thread():
    seen = []

    def fn(b):
        seen.append(threading.get_ident())
        return b

    assert run_batches(fn, [1, 2, 3], max_workers=1) == [1, 2, 3]
    assert set(seen) == {threading.get_ident()}
    seen.clear()
    assert run_batches(fn, [1, 2], max_workers=0) == [1, 2]
    assert set(seen) == {threading.get_ident()}


def test_run_batches_single_batch_runs_in_caller_thread_even_with_many_workers():
    seen = []
    run_batches(lambda b: seen.append(threading.get_ident()), ["only"], max_workers=8)
    assert seen == [threading.get_ident()]


def test_run_batches_uses_worker_threads_for_multiple_batches():
    seen = set()
    run_batches(lambda b: seen.add(threading.get_ident()), [1, 2, 3], max_workers=3)
    assert threading.get_ident() not in seen


def test_run_batches_propagates_exceptions_sequential():
    def boom(b):
        if b == 2:
            raise ValueError("bad batch")
        return b

    with pytest.raises(ValueError, match="bad batch"):
        run_batches(boom, [1, 2, 3], max_workers=1)


def test_run_batches_propagates_exceptions_concurrent():
    def boom(b):
        if b == 2:
            raise KeyError("bad batch")
        return b

    with pytest.raises(KeyError):
        run_batches(boom, [1, 2, 3], max_workers=4)


def test_run_batches_results_match_input_length_and_content():
    out = run_batches(lambda b: sum(b), [[1, 2], [3], [4, 5, 6], []], max_workers=2)
    assert out == [3, 3, 15, 0]
