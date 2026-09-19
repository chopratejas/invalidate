"""Shared fixtures: a deterministic, scriptable fake Judge. Never talks to the network."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pytest

from invalidate import Invalidate, Policy, Votes
from invalidate.judge import JudgeResult, ObserveBatch, PairBatch, RecallBatch
from invalidate.types import Event, Memory

UNRELATED = Votes(bears=0.0, still_true=1.0, replaces=0.0, hypothetical=0.0)
CONFIRM = Votes(bears=0.95, still_true=0.95, replaces=0.05, hypothetical=0.0)
CONTRADICT = Votes(bears=0.95, still_true=0.05, replaces=0.1, hypothetical=0.0)
SUPERSEDE = Votes(bears=0.95, still_true=0.05, replaces=0.9, hypothetical=0.0)
UNCERTAIN = Votes(bears=0.9, still_true=0.5, replaces=0.5, hypothetical=0.0)
HYPOTHETICAL = Votes(bears=0.95, still_true=0.05, replaces=0.9, hypothetical=0.95)


@dataclass
class FakeJudge:
    """Scriptable judge.

    `votes` maps a memory id *or* a substring of the fact to the Votes to return.
    `relevance` does the same for recall(). Unscripted memories get `default_votes`
    / `default_relevance`. Every call is recorded (thread-safely) in `observe_calls`
    and `recall_calls` as (event|query, [memories]).
    """

    votes: dict[str, Votes] = field(default_factory=dict)
    relevance: dict[str, float] = field(default_factory=dict)
    default_votes: Votes = UNRELATED
    default_relevance: float = 0.0
    tokens_per_call: int = 7
    model: str | None = "fake-jev"
    observe_calls: list[tuple[Event, list[Memory]]] = field(default_factory=list)
    recall_calls: list[tuple[str, list[Memory]]] = field(default_factory=list)
    pair_calls: list[tuple[list[Event], list[Memory], list[tuple[int, int]]]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def script(self, key: str, votes: Votes) -> "FakeJudge":
        self.votes[key] = votes
        return self

    def _lookup(self, table: dict, m: Memory, default):
        if m.id in table:
            return table[m.id]
        for key, val in table.items():
            if key in m.fact:
                return val
        return default

    def observe(self, event: Event, memories: list[Memory]) -> ObserveBatch:
        with self._lock:
            self.observe_calls.append((event, list(memories)))
        if not memories:
            return ObserveBatch([], JudgeResult(0, None))
        votes = [self._lookup(self.votes, m, self.default_votes) for m in memories]
        return ObserveBatch(votes, JudgeResult(self.tokens_per_call, self.model))

    def screen_pairs(self, events: list[Event], memories: list[Memory], pairs: list[tuple[int, int]]) -> PairBatch:
        """Bears vote of the scripted Votes for each pair (event-independent, like the scripted observe())."""
        with self._lock:
            self.pair_calls.append((list(events), list(memories), list(pairs)))
        scores = {(j, i): self._lookup(self.votes, memories[i], self.default_votes).bears for j, i in pairs}
        return PairBatch(scores, JudgeResult(self.tokens_per_call, self.model))

    def recall(self, query: str, memories: list[Memory]) -> RecallBatch:
        with self._lock:
            self.recall_calls.append((query, list(memories)))
        if not memories:
            return RecallBatch([], JudgeResult(0, None))
        rel = [self._lookup(self.relevance, m, self.default_relevance) for m in memories]
        return RecallBatch(rel, JudgeResult(self.tokens_per_call, self.model))

    # convenience for tests
    @property
    def observed_ids(self) -> list[list[str]]:
        return [[m.id for m in ms] for _, ms in self.observe_calls]


class NoPairScreenJudge(FakeJudge):
    """A judge without the optional pair screen: validate() sends every pair to the full judgment."""

    screen_pairs = None  # type: ignore[assignment]


@pytest.fixture
def fake() -> FakeJudge:
    return FakeJudge()


@pytest.fixture
def mem(fake: FakeJudge) -> Invalidate:
    # second_opinion off: these tests count judge calls exactly. tests/test_second_opinion.py covers the pass.
    inv = Invalidate(":memory:", judge=fake, policy=Policy(second_opinion=False))
    yield inv
    inv.close()


@pytest.fixture
def make_mem(fake: FakeJudge):
    """Factory for an Invalidate with a custom Policy (and optional callback), sharing the fake judge."""
    created: list[Invalidate] = []

    def _make(policy: Policy | None = None, **kw) -> Invalidate:
        if policy is None:
            policy = Policy(second_opinion=False)
        inv = Invalidate(":memory:", judge=fake, policy=policy, **kw)
        created.append(inv)
        return inv

    yield _make
    for inv in created:
        inv.close()
