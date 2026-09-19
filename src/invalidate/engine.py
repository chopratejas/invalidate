"""The memory governor. remember / observe / recall, plus the human controls.

Code owns every write. Jev only votes.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any

from .judge import Judge, JevJudge, run_batches
from .policy import Policy
from .store import SQLiteStore, Store
from .types import (
    Disposition,
    Event,
    Memory,
    ObserveReport,
    Recalled,
    RecallReport,
    Status,
    Verdict,
    now,
)

Transition = Callable[[Memory, Verdict], None]


class Invalidate:
    """Drop this in front of any memory store.

        mem = Invalidate("memories.db")
        mem.remember("user prefers Postgres", source="chat", kind="preference")
        report = mem.observe("we migrated to SQLite last Tuesday", source="slack")
        mem.recall("which database?")
    """

    def __init__(
        self,
        db: str | Store = "invalidate.db",
        *,
        judge: Judge | None = None,
        policy: Policy | None = None,
        namespace: str = "default",
        api_key: str | None = None,
        model: str | None = None,
        on_transition: Transition | None = None,
    ) -> None:
        self.store: Store = SQLiteStore(db) if isinstance(db, str) else db
        self.policy = policy or Policy()
        self.namespace = namespace
        self._judge = judge
        self._api_key = api_key
        self._model = model
        self.on_transition = on_transition

    # Lazy so that remember()/list()/freeze() work without an API key.
    @property
    def judge(self) -> Judge:
        if self._judge is None:
            self._judge = JevJudge(api_key=self._api_key, model=self._model)
        return self._judge

    # -- write side -------------------------------------------------------------
    def remember(
        self,
        fact: str,
        *,
        source: str = "unknown",
        kind: str = "fact",
        ttl: float | None = None,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        id: str | None = None,
    ) -> Memory:
        """Store a fact verbatim. `ttl` is a hard lease in seconds (code-enforced), on top of the semantic one."""
        fact = fact.strip()
        if not fact:
            raise ValueError("fact must be non-empty")
        m = Memory(
            fact=fact,
            source=source,
            kind=kind,
            namespace=namespace or self.namespace,
            metadata=metadata or {},
            expires_at=(now() + ttl) if ttl is not None else None,
        )
        if id:
            m.id = id
        self.store.add_memory(m)
        return m

    def observe(
        self,
        text: str,
        *,
        source: str = "unknown",
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        candidates: Iterable[Memory] | None = None,
        dry_run: bool = False,
    ) -> ObserveReport:
        """Judge new evidence against every judgeable memory and apply the policy.

        `candidates` lets you pre-filter (tags, namespace, your own index). Default: every memory
        in the namespace whose status is in `policy.judge_statuses` and which has not expired.
        `dry_run=True` judges and returns verdicts but writes nothing.
        """
        text = text.strip()
        if not text:
            raise ValueError("event text must be non-empty")
        ns = namespace or self.namespace
        event = Event(text=text, source=source, namespace=ns, metadata=metadata or {})
        t0 = time.perf_counter()

        pool = list(candidates) if candidates is not None else self.store.list_memories(ns)
        judgeable: list[Memory] = []
        skipped = 0
        t_now = now()
        for m in pool:
            if m.status in self.policy.judge_statuses and not m.is_expired(t_now):
                judgeable.append(m)
            else:
                skipped += 1

        batches = _chunk(judgeable, self.policy.batch_size)
        results = run_batches(lambda b: self.judge.observe(event, b), batches, self.policy.max_workers)

        verdicts: list[Verdict] = []
        tokens = 0
        model = None
        for batch, res in zip(batches, results):
            tokens += res.usage.input_tokens
            model = res.usage.model or model
            for m, votes in zip(batch, res.votes):
                d = self.policy.dispose(votes)
                to = self.policy.transition(m.status, d)
                v = Verdict(
                    event_id=event.id, memory_id=m.id, votes=votes, disposition=d,
                    from_status=m.status, to_status=to, applied=(to != m.status),
                )
                verdicts.append(v)
                if not dry_run:
                    self._apply(m, v)

        if not dry_run:
            self.store.add_event(event)
            self.store.add_verdicts(verdicts)

        return ObserveReport(
            event=event, verdicts=verdicts, judged=len(judgeable), skipped=skipped, requests=len(batches),
            input_tokens=tokens, latency_ms=(time.perf_counter() - t0) * 1000, model=model,
        )

    def _apply(self, m: Memory, v: Verdict) -> None:
        m.last_checked = v.created_at
        if v.disposition is not Disposition.UNRELATED:
            m.p_true = v.votes.still_true
        if v.to_status != m.status:
            m.status = v.to_status
            m.updated_at = v.created_at
        self.store.update_memory(m)
        if v.changed and self.on_transition is not None:
            self.on_transition(m, v)

    # -- read side --------------------------------------------------------------
    def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        namespace: str | None = None,
        include_review: bool = False,
        min_relevance: float | None = None,
    ) -> RecallReport:
        """Return live memories relevant to `query`, ranked by Jev's relevance vote. No embeddings."""
        ns = namespace or self.namespace
        t0 = time.perf_counter()
        statuses = {Status.ACTIVE, Status.FROZEN} | ({Status.NEEDS_REVIEW} if include_review else set())
        t_now = now()
        pool = [m for m in self.store.list_memories(ns, statuses) if not m.is_expired(t_now)]
        batches = _chunk(pool, self.policy.batch_size)
        results = run_batches(lambda b: self.judge.recall(query, b), batches, self.policy.max_workers)
        threshold = self.policy.relevance_min if min_relevance is None else min_relevance
        scored: list[Recalled] = []
        tokens = 0
        for batch, res in zip(batches, results):
            tokens += res.usage.input_tokens
            for m, r in zip(batch, res.relevance):
                if r >= threshold:
                    scored.append(Recalled(memory=m, relevance=r))
        scored.sort(key=lambda r: r.relevance, reverse=True)
        return RecallReport(
            query=query, results=scored[:limit], considered=len(pool), requests=len(batches),
            input_tokens=tokens, latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def list(self, *, statuses: Iterable[Status] | None = None, namespace: str | None = None) -> list[Memory]:
        return self.store.list_memories(namespace or self.namespace, statuses)

    def get(self, memory_id: str) -> Memory:
        m = self.store.get_memory(memory_id)
        if m is None:
            raise KeyError(memory_id)
        return m

    def history(self, memory_id: str) -> list[Verdict]:
        return self.store.list_verdicts(memory_id=memory_id)

    def events(self, *, namespace: str | None = None, limit: int | None = None) -> list[Event]:
        return self.store.list_events(namespace or self.namespace, limit)

    # -- human controls -----------------------------------------------------------
    def freeze(self, memory_id: str) -> Memory:
        """Pin a memory. It is still judged on observe() for the audit log, but never auto-flipped."""
        return self._set_status(memory_id, Status.FROZEN)

    def unfreeze(self, memory_id: str) -> Memory:
        return self._set_status(memory_id, Status.ACTIVE)

    def restore(self, memory_id: str) -> Memory:
        """Human override: put a contradicted/superseded/review memory back to active."""
        m = self._set_status(memory_id, Status.ACTIVE)
        m.superseded_by = None
        self.store.update_memory(m)
        return m

    def forget(self, memory_id: str) -> Memory:
        return self._set_status(memory_id, Status.DELETED)

    def supersede(self, memory_id: str, *, by: str) -> Memory:
        """Link a superseded memory to its verbatim successor (a memory you `remember`ed)."""
        m = self.get(memory_id)
        m.superseded_by = by
        m.status = Status.SUPERSEDED
        m.updated_at = now()
        self.store.update_memory(m)
        return m

    def sweep(self, *, namespace: str | None = None) -> list[Memory]:
        """Mark hard-TTL-expired memories as expired. Pure code; no model call."""
        t_now = now()
        expired = []
        for m in self.store.list_memories(namespace or self.namespace, {Status.ACTIVE, Status.NEEDS_REVIEW, Status.FROZEN}):
            if m.is_expired(t_now):
                m.status = Status.EXPIRED
                m.updated_at = t_now
                self.store.update_memory(m)
                expired.append(m)
        return expired

    def _set_status(self, memory_id: str, status: Status) -> Memory:
        m = self.get(memory_id)
        m.status = status
        m.updated_at = now()
        self.store.update_memory(m)
        return m

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "Invalidate":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _chunk(items: list[Memory], size: int) -> list[list[Memory]]:
    size = max(1, size)
    return [items[i : i + size] for i in range(0, len(items), size)]
