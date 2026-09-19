"""The memory governor. remember / observe / recall, plus the human controls.

Code owns every write. Jev only votes.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any

from .judge import Judge, JevJudge, JudgeMisaligned, run_batches
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

_CONTENT_VERDICTS = frozenset(
    {Disposition.CONFIRMED, Disposition.CONTRADICTED, Disposition.SUPERSEDED, Disposition.UNCERTAIN, Disposition.PARTIAL}
)


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

    def check(self) -> str:
        """Fail fast: build the judge (raises MissingAPIKey) and confirm the API answers. Returns the model id."""
        judge = self.judge
        client = getattr(judge, "client", None)
        models = getattr(client, "models", None)
        if models is not None and hasattr(models, "list"):
            listed = models.list()
            names = [getattr(m, "name", str(m)) for m in getattr(listed, "models", listed)]
            return getattr(judge, "model", None) or (names[0] if names else "unknown")
        return getattr(judge, "model", None) or "unknown"

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
        remember_successor: bool = False,
        successor_kind: str | None = None,
    ) -> ObserveReport:
        """Judge new evidence against every judgeable memory and apply the policy.

        `candidates` lets you pre-filter (tags, namespace, your own index). Default: every memory
        in the namespace whose status is in `policy.judge_statuses` and which has not expired.
        `dry_run=True` judges and returns verdicts but writes nothing.
        `remember_successor=True` stores the event text verbatim as a new memory when it supersedes
        at least one existing memory, and links each superseded row to it (`superseded_by`).
        The successor takes the kind of the first superseded memory unless `successor_kind` is given.
        """
        text = text.strip()
        if not text:
            raise ValueError("event text must be non-empty")
        ns = namespace or self.namespace
        event = Event(text=text, source=source, namespace=ns, metadata=metadata or {})
        t0 = time.perf_counter()

        if candidates is None:
            pool = self.store.list_memories(ns)
        else:
            # Reload by id so stale objects cannot clobber concurrent freeze()/forget(), and stay in-namespace.
            pool = []
            for c in candidates:
                fresh = self.store.get_memory(c.id)
                if fresh is not None and fresh.namespace == ns:
                    pool.append(fresh)
        judgeable: list[Memory] = []
        skipped = 0
        t_now = now()
        for m in pool:
            if m.status in self.policy.judge_statuses and (m.status is Status.FROZEN or not m.is_expired(t_now)):
                judgeable.append(m)
            else:
                skipped += 1

        tokens = 0
        model = None
        screened_out = 0
        n_screen_requests = 0
        screen = getattr(self.judge, "screen", None)
        if screen is not None and len(judgeable) > self.policy.screen_above:
            # Stage 1: one short bears-only question per memory, big batches. Drops only clear non-matches.
            sbatches = _chunk(judgeable, self.policy.screen_batch_size)
            sresults = run_batches(lambda b: screen(event, b), sbatches, self.policy.max_workers)
            kept: list[Memory] = []
            for sb, sr in zip(sbatches, sresults):
                if len(sr.relevance) != len(sb):
                    raise JudgeMisaligned(f"screen returned {len(sr.relevance)} scores for {len(sb)} memories")
                tokens += sr.usage.input_tokens
                model = sr.usage.model or model
                for m, b in zip(sb, sr.relevance):
                    if b >= self.policy.screen_min:
                        kept.append(m)
            n_screen_requests = len(sbatches)
            screened_out = len(judgeable) - len(kept)
            judgeable = kept

        batches = _chunk(judgeable, self.policy.batch_size)
        results = run_batches(lambda b: self.judge.observe(event, b), batches, self.policy.max_workers)

        verdicts: list[Verdict] = []
        pairs: list[tuple[Memory, Verdict]] = []
        for batch, res in zip(batches, results):
            if len(res.votes) != len(batch):
                raise JudgeMisaligned(f"judge returned {len(res.votes)} votes for {len(batch)} memories")
            tokens += res.usage.input_tokens
            model = res.usage.model or model
            for m, votes in zip(batch, res.votes):
                d = self.policy.dispose(votes)
                to = self.policy.transition(m.status, d, source=event.source)
                v = Verdict(
                    event_id=event.id, memory_id=m.id, votes=votes, disposition=d,
                    from_status=m.status, to_status=to, applied=(to != m.status),
                )
                verdicts.append(v)
                pairs.append((m, v))

        successor: Memory | None = None
        if not dry_run:
            # Audit trail first: if a callback or the process dies mid-apply, the event and votes survive.
            self.store.add_event(event)
            self.store.add_verdicts(verdicts)
            for m, v in pairs:
                self._apply(m, v)
            superseded = [v for v in verdicts if v.applied and v.to_status is Status.SUPERSEDED]
            if remember_successor and superseded:
                first = self.get(superseded[0].memory_id)
                successor = self.remember(
                    event.text, source=event.source, kind=successor_kind or first.kind, namespace=ns,
                    metadata={"supersedes": [v.memory_id for v in superseded], "event_id": event.id},
                )
                for v in superseded:
                    m = self.get(v.memory_id)
                    m.superseded_by = successor.id
                    self.store.update_memory(m)

        return ObserveReport(
            successor=successor,
            event=event, verdicts=verdicts, judged=len(judgeable), skipped=skipped, screened_out=screened_out,
            requests=len(batches) + n_screen_requests, input_tokens=tokens,
            latency_ms=(time.perf_counter() - t0) * 1000, model=model,
        )

    def _apply(self, m: Memory, v: Verdict) -> None:
        m.last_checked = v.created_at
        if v.disposition in _CONTENT_VERDICTS:
            # Only a report about the world moves belief. Questions and commands are logged in the
            # verdict history but leave p_true alone.
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
            if len(res.relevance) != len(batch):
                raise JudgeMisaligned(f"judge returned {len(res.relevance)} relevances for {len(batch)} memories")
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
        """Pin a memory. It is still judged on observe() (votes and p_true are recorded for the audit log),
        but its status is never changed by observe() or sweep(). Only unfreeze()/forget() move it."""
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
        if by == memory_id:
            raise ValueError("a memory cannot supersede itself")
        self.get(by)  # KeyError if the successor does not exist
        m.superseded_by = by
        m.status = Status.SUPERSEDED
        m.updated_at = now()
        self.store.update_memory(m)
        return m

    def sweep(self, *, namespace: str | None = None) -> list[Memory]:
        """Mark hard-TTL-expired memories as expired. Pure code; no model call. Frozen memories are left alone."""
        t_now = now()
        expired = []
        for m in self.store.list_memories(namespace or self.namespace, {Status.ACTIVE, Status.NEEDS_REVIEW}):
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
