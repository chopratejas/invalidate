"""Judge: the only place that talks to Jev.

`Judge` is a Protocol so tests and other stores can substitute a deterministic judge.
`JevJudge` wraps the TypeSafe SDK. Batches are independent and run in a thread pool.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from . import questions as Q
from .types import Event, Memory, Votes


@dataclass
class JudgeResult:
    input_tokens: int
    model: str | None


@dataclass
class ObserveBatch:
    votes: list[Votes]
    usage: JudgeResult


@dataclass
class RecallBatch:
    relevance: list[float]
    usage: JudgeResult


class Judge(Protocol):
    def observe(self, event: Event, memories: list[Memory]) -> ObserveBatch: ...
    def recall(self, query: str, memories: list[Memory]) -> RecallBatch: ...


class JevJudge:
    """Talks to TypeSafe's System One endpoint with the official SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from typesafe_sdk import TypeSafeClient

            key = api_key or os.environ.get("TYPESAFE_API_KEY")
            if not key:
                from .env import load_dotenv

                key = load_dotenv().get("TYPESAFE_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
            if not key:
                raise MissingAPIKey(
                    "TYPESAFE_API_KEY is not set. Create one at https://console.typesafe.ai/, then export it or put "
                    "TYPESAFE_API_KEY=... in a .env file in the working directory, or pass api_key=."
                )
            client = TypeSafeClient(api_key=key, model=model, timeout=timeout)
        self.client = client
        self.model = model

    def observe(self, event: Event, memories: list[Memory]) -> ObserveBatch:
        if not memories:
            return ObserveBatch([], JudgeResult(0, None))
        resp = self.client.system_one(Q.observe_state(event, memories), Q.observe_questions(len(memories)))
        a = resp.answers
        hyp = _p(a[Q.HYPOTHETICAL])
        directive = _p(a[Q.DIRECTIVE])
        votes = [
            Votes(
                bears=_p(a[f"{Q.BEARS}_{i}"]),
                still_true=_p(a[f"{Q.STILL_TRUE}_{i}"]),
                replaces=_p(a[f"{Q.REPLACES}_{i}"]),
                hypothetical=hyp,
                directive=directive,
            )
            for i in range(len(memories))
        ]
        return ObserveBatch(votes, _usage(resp))

    def recall(self, query: str, memories: list[Memory]) -> RecallBatch:
        if not memories:
            return RecallBatch([], JudgeResult(0, None))
        resp = self.client.system_one(Q.recall_state(query, memories), Q.recall_questions(len(memories)))
        rel = [_p(resp.answers[f"{Q.RELEVANT}_{i}"]) for i in range(len(memories))]
        return RecallBatch(rel, _usage(resp))


class MissingAPIKey(RuntimeError):
    pass


class JudgeMisaligned(RuntimeError):
    """The judge returned a different number of answers than memories sent."""


def _p(answer: Any) -> float:
    return float(answer.noul)


def _usage(resp: Any) -> JudgeResult:
    u = getattr(resp, "usage", None)
    toks = getattr(u, "input_tokens", None) if u is not None else None
    return JudgeResult(int(toks or 0), getattr(resp, "model", None))


def run_batches(fn, batches: list[Any], max_workers: int) -> list[Any]:
    """Run `fn(batch)` over batches concurrently, preserving order. Exceptions propagate."""
    if not batches:
        return []
    if len(batches) == 1 or max_workers <= 1:
        return [fn(b) for b in batches]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as pool:
        return list(pool.map(fn, batches))
