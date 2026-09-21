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
    ValidateReport,
    Verdict,
    now,
)

Transition = Callable[[Memory, Verdict], None]

_CONTENT_VERDICTS = frozenset(
    {Disposition.CONFIRMED, Disposition.CONTRADICTED, Disposition.SUPERSEDED, Disposition.UNCERTAIN, Disposition.PARTIAL}
)
_DEAD = frozenset({Status.CONTRADICTED, Status.SUPERSEDED})


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
        lazy: bool | None = None,
    ) -> None:
        self.store: Store = SQLiteStore(db) if isinstance(db, str) else db
        self.policy = policy or Policy()
        if lazy is not None:
            self.policy.lazy = lazy
        self.namespace = namespace
        self._judge = judge
        self._api_key = api_key
        self._model = model
        self.on_transition = on_transition

    # Lazy so that remember()/list()/freeze() work without an API key.
    @property
    def judge(self) -> Judge:
        if self._judge is None:
            self._judge = JevJudge(api_key=self._api_key, model=self._model, staged=self.policy.staged,
                                   stage_below=self.policy.contradict_max - self.policy.margin)
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
        # Born current: a fact stored now postdates every event already in the log, so those events are not
        # evidence against it. Only events that arrive after it are pending for it.
        m.checked_seq = self.store.max_seq(m.namespace)
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
        defer: bool | None = None,
    ) -> ObserveReport:
        """Judge new evidence against every judgeable memory and apply the policy.

        `candidates` lets you pre-filter (tags, namespace, your own index). Default: every memory
        in the namespace whose status is in `policy.judge_statuses` and which has not expired.
        `dry_run=True` judges and returns verdicts but writes nothing.
        `remember_successor=True` stores the event text verbatim as a new memory when it supersedes
        at least one existing memory, and links each superseded row to it (`superseded_by`).
        The successor takes the kind of the first superseded memory unless `successor_kind` is given.
        `defer=True` (the default when `policy.lazy`) appends the event to the log and judges nothing now;
        memories are judged against it when next read (`recall(validate=True)`, `Governor.filter`) or by
        `validate()`. The report then has `judged == 0` and `pending` set.
        """
        text = text.strip()
        if not text:
            raise ValueError("event text must be non-empty")
        ns = namespace or self.namespace
        event = Event(text=text, source=source, namespace=ns, metadata=metadata or {})
        t0 = time.perf_counter()
        if defer is None:
            defer = self.policy.lazy
        if defer and not dry_run:
            if remember_successor:
                event.metadata["_remember_successor"] = True
                if successor_kind:
                    event.metadata["_successor_kind"] = successor_kind
            self.store.add_event(event)
            pending = self.store.count_pending(ns, self.policy.judge_statuses)
            return ObserveReport(event=event, verdicts=[], judged=0, skipped=0, requests=0, input_tokens=0,
                                 latency_ms=(time.perf_counter() - t0) * 1000, pending=pending)
        prev_seq = self.store.max_seq(ns)

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

        all_judgeable = list(judgeable)
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
        n_observe_requests = 0
        for batch, res in zip(batches, results):
            if len(res.votes) != len(batch):
                raise JudgeMisaligned(f"judge returned {len(res.votes)} votes for {len(batch)} memories")
            tokens += res.usage.input_tokens
            n_observe_requests += res.requests
            model = res.usage.model or model
            for m, votes in zip(batch, res.votes):
                d = self.policy.dispose(votes)
                to = self.policy.transition(m.status, d, source=event.source, still_true=votes.still_true)
                v = Verdict(
                    event_id=event.id, memory_id=m.id, votes=votes, disposition=d,
                    from_status=m.status, to_status=to, applied=(to != m.status),
                )
                verdicts.append(v)
                pairs.append((m, v))

        n_second = 0
        if self.policy.second_opinion:
            kill_idx = [k for k, (m, v) in enumerate(pairs) if v.applied and v.to_status in _DEAD]
            second, n_second, t2 = self._second_opinions([(event, pairs[k][0]) for k in kill_idx])
            tokens += t2
            for k, v2 in zip(kill_idx, second):
                m, v = pairs[k]
                if not self._agrees(v2, m.status, event.source):
                    to = self.policy.transition(m.status, Disposition.UNCERTAIN, source=event.source,
                                                still_true=v2.still_true)
                    v = Verdict(event_id=event.id, memory_id=m.id, votes=v2, disposition=Disposition.UNCERTAIN,
                                from_status=m.status, to_status=to, applied=(to != m.status))
                    pairs[k] = (m, v)
                    verdicts[k] = v

        successor: Memory | None = None
        if not dry_run:
            # Audit trail first: if a callback or the process dies mid-apply, the event and votes survive.
            self.store.add_event(event)
            self.store.add_verdicts(verdicts)
            # A memory that was current before this event is current after it. One that was already behind
            # the log stays behind (validate() will judge the events it missed, and this one, in order).
            current = {m.id for m in all_judgeable if m.checked_seq >= prev_seq}
            for m, v in pairs:
                if m.id in current:
                    m.checked_seq = event.seq
                self._apply(m, v)
            self.store.set_checked_seq([mid for mid in current if mid not in {m.id for m, _ in pairs}], event.seq)
            superseded = [v for v in verdicts if v.applied and v.to_status is Status.SUPERSEDED]
            if remember_successor and superseded:
                successor = self._successor_for(event, superseded, successor_kind)

        return ObserveReport(
            successor=successor,
            event=event, verdicts=verdicts, judged=len(judgeable), skipped=skipped, screened_out=screened_out,
            requests=n_observe_requests + n_screen_requests + n_second, input_tokens=tokens,
            latency_ms=(time.perf_counter() - t0) * 1000, model=model,
        )

    def _second_opinions(self, kills: list[tuple[Event, Memory]]) -> tuple[list[Any], int, int]:
        """Re-judge each (event, memory) with the memory alone in the state. Returns (votes, requests, tokens)."""
        if not kills:
            return [], 0, 0
        results = run_batches(lambda em: self.judge.observe(em[0], [em[1]]), kills, self.policy.max_workers)
        votes = []
        tokens = 0
        requests = 0
        for (e, m), res in zip(kills, results):
            if len(res.votes) != 1:
                raise JudgeMisaligned(f"second opinion returned {len(res.votes)} votes for 1 memory")
            votes.append(res.votes[0])
            tokens += res.usage.input_tokens
            # Each second opinion is itself an observe(), so it can be two calls when staged.
            requests += res.requests
        return votes, requests, tokens

    def _agrees(self, votes: Any, status: Status, source: str) -> bool:
        d = self.policy.dispose(votes)
        return self.policy.transition(status, d, source=source, still_true=votes.still_true) in _DEAD

    def _successor_for(self, event: Event, superseded: list[Verdict], successor_kind: str | None) -> Memory:
        """Store the event verbatim as the successor of every memory it superseded (once per event)."""
        ns = event.namespace
        existing = [m for m in self.store.list_memories(ns) if m.metadata.get("event_id") == event.id and "supersedes" in m.metadata]
        if existing:
            successor = existing[0]
            successor.metadata["supersedes"] = sorted(set(successor.metadata["supersedes"]) | {v.memory_id for v in superseded})
            self.store.update_memory(successor)
        else:
            first = self.get(superseded[0].memory_id)
            successor = self.remember(
                event.text, source=event.source, kind=successor_kind or first.kind, namespace=ns,
                metadata={"supersedes": [v.memory_id for v in superseded], "event_id": event.id},
            )
            # The successor is born current: it postdates every event in the log so far.
            successor.checked_seq = self.store.max_seq(ns)
            self.store.update_memory(successor)
        for v in superseded:
            m = self.get(v.memory_id)
            m.superseded_by = successor.id
            self.store.update_memory(m)
        return successor

    def observe_many(
        self, texts: Iterable[str], *, source: str = "unknown", namespace: str | None = None,
        remember_successor: bool = False, successor_kind: str | None = None,
    ) -> tuple[list[Event], ValidateReport | None]:
        """Batch ingest. Appends every event to the log, then (unless lazy) judges the whole pool against all of
        them in one pass using the pair screen (many events x many memories per request), which costs about half
        the tokens of one observe() per event. Returns the events and the validation report."""
        ns = namespace or self.namespace
        events = [
            self.observe(t, source=source, namespace=ns, defer=True, remember_successor=remember_successor,
                         successor_kind=successor_kind).event
            for t in texts if t and t.strip()
        ]
        if self.policy.lazy or not events:
            return events, None
        return events, self.validate(namespace=ns)

    def pending(self, namespace: str | None = None) -> int:
        """How many judgeable memories are behind the event log (have events they have not been judged against)."""
        return self.store.count_pending(namespace or self.namespace, self.policy.judge_statuses)

    def validate(
        self,
        memories: Iterable[Memory] | None = None,
        *,
        namespace: str | None = None,
        budget_requests: int | None = None,
    ) -> ValidateReport:
        """Judge memories against every event they have not seen yet, oldest first, and apply the policy.

        This is the read-side and background half of lazy mode, and the engine behind observe_many().
        Stage 1 screens (event, memory) pairs many-to-many per request (`policy.pair_events` x
        `policy.pair_memories`, one short question per pair). Stage 2 runs the full six-question judgment only
        on pairs that pass `policy.pair_min`. Verdicts for one memory are applied in event order, so a fact
        superseded in March and re-confirmed in June ends up where June left it.

        `memories=None` means every judgeable memory in the namespace that is behind the log.
        `budget_requests` caps Jev requests for this call; memories not reached stay pending (`report.pending`).
        """
        ns = namespace or self.namespace
        t0 = time.perf_counter()
        max_seq = self.store.max_seq(ns)
        t_now = now()
        if memories is None:
            pool = self.store.list_memories(ns, self.policy.judge_statuses)
        else:
            pool = []
            for c in memories:
                fresh = self.store.get_memory(c.id)
                if fresh is not None and fresh.namespace == ns and fresh.status in self.policy.judge_statuses:
                    pool.append(fresh)
        behind = [m for m in pool if m.checked_seq < max_seq and (m.status is Status.FROZEN or not m.is_expired(t_now))]
        behind.sort(key=lambda m: m.checked_seq)  # the most stale first: they are the ones a budget should reach
        if not behind:
            return ValidateReport(0, 0, 0, 0, [], 0, 0, (time.perf_counter() - t0) * 1000)

        screen_pairs = getattr(self.judge, "screen_pairs", None)
        requests = 0
        tokens = 0
        model = None
        # ---- stage 1: pair screen. Jobs are (events chunk, memories chunk, pairs) ------------------------------
        jobs: list[tuple[list[Event], list[Memory], list[tuple[int, int]]]] = []
        reached: list[Memory] = []
        event_cache: dict[int, list[Event]] = {}
        seen_events: dict[str, Event] = {}
        for mchunk in _chunk(behind, self.policy.pair_memories):
            lo = min(m.checked_seq for m in mchunk)
            if lo not in event_cache:
                event_cache[lo] = self.store.list_events_after(ns, lo)
            evs = event_cache[lo]
            chunk_jobs = []
            for echunk in _chunk_any(evs, self.policy.pair_events):
                pairs = [(j, i) for j, e in enumerate(echunk) for i, m in enumerate(mchunk) if e.seq > m.checked_seq]
                if pairs:
                    chunk_jobs.append((echunk, mchunk, pairs))
            if budget_requests is not None and requests + len(chunk_jobs) > budget_requests:
                break
            requests += len(chunk_jobs)
            jobs.extend(chunk_jobs)
            reached.extend(mchunk)
            for e in evs:
                seen_events[e.id] = e
        pending_after = len(behind) - len(reached)

        bearing: dict[str, dict[str, Memory]] = {}  # event id -> memory id -> memory
        npairs = 0
        if screen_pairs is None:
            # Judge without a pair screen (small pools, or a judge that lacks it): every pair goes to stage 2.
            for echunk, mchunk, pairs in jobs:
                npairs += len(pairs)
                for j, i in pairs:
                    bearing.setdefault(echunk[j].id, {})[mchunk[i].id] = mchunk[i]
        else:
            results = run_batches(lambda job: screen_pairs(job[0], job[1], job[2]), jobs, self.policy.max_workers)
            for (echunk, mchunk, pairs), res in zip(jobs, results):
                if len(res.scores) != len(pairs):
                    raise JudgeMisaligned(f"pair screen returned {len(res.scores)} scores for {len(pairs)} pairs")
                npairs += len(pairs)
                tokens += res.usage.input_tokens
                model = res.usage.model or model
                for j, i in pairs:
                    if res.scores[(j, i)] >= self.policy.pair_min:
                        bearing.setdefault(echunk[j].id, {})[mchunk[i].id] = mchunk[i]

        # ---- stage 2: full judgment for bearing pairs, grouped by event ----------------------------------------
        full_jobs: list[tuple[Event, list[Memory]]] = []
        for eid, mems in bearing.items():
            for b in _chunk(list(mems.values()), self.policy.batch_size):
                full_jobs.append((seen_events[eid], b))
        full_results = run_batches(lambda job: self.judge.observe(job[0], job[1]), full_jobs, self.policy.max_workers)
        raw: dict[str, list[tuple[Event, Any]]] = {}  # memory id -> [(event, votes)]
        for (e, b), res in zip(full_jobs, full_results):
            if len(res.votes) != len(b):
                raise JudgeMisaligned(f"judge returned {len(res.votes)} votes for {len(b)} memories")
            tokens += res.usage.input_tokens
            model = res.usage.model or model
            requests += res.requests
            for m, votes in zip(b, res.votes):
                raw.setdefault(m.id, []).append((e, votes))

        # ---- apply, in event order per memory ----------------------------------------------------------------
        def chain(m: Memory, overrides: dict[str, Any]) -> list[Verdict]:
            out: list[Verdict] = []
            status = m.status
            for e, votes in sorted(raw.get(m.id, []), key=lambda ev: ev[0].seq):
                if e.id in overrides:
                    votes, d = overrides[e.id], Disposition.UNCERTAIN
                else:
                    d = self.policy.dispose(votes)
                to = self.policy.transition(status, d, source=e.source, still_true=votes.still_true)
                out.append(Verdict(event_id=e.id, memory_id=m.id, votes=votes, disposition=d,
                                   from_status=status, to_status=to, applied=(to != status)))
                status = to
            return out

        plan: list[tuple[Memory, list[Verdict]]] = [(m, chain(m, {})) for m in reached]
        if self.policy.second_opinion:
            overrides: dict[str, dict[str, Any]] = {}
            checked: set[tuple[str, str]] = set()
            # An override can expose a later kill in the same chain (review -> superseded by the next event),
            # so loop until every kill in the final chain has had its second opinion.
            while True:
                kills = [(seen_events[v.event_id], m, v) for m, vs in plan for v in vs
                         if v.applied and v.to_status in _DEAD and (v.event_id, m.id) not in checked]
                if not kills:
                    break
                second, n2, t2 = self._second_opinions([(e, m) for e, m, _ in kills])
                requests += n2
                tokens += t2
                for (e, m, v), v2 in zip(kills, second):
                    checked.add((e.id, m.id))
                    if not self._agrees(v2, v.from_status, e.source):
                        overrides.setdefault(m.id, {})[e.id] = v2
                plan = [(m, chain(m, overrides.get(m.id, {}))) for m, _ in plan]
        verdicts: list[Verdict] = [v for _, vs in plan for v in vs]
        self.store.add_verdicts(verdicts)
        by_event_superseded: dict[str, list[Verdict]] = {}
        for m, vs in plan:
            for v in vs:
                self._apply(m, v)
                if v.applied and v.to_status is Status.SUPERSEDED:
                    by_event_superseded.setdefault(v.event_id, []).append(v)
            m.checked_seq = max_seq
            self.store.update_memory(m)
        for eid, sup in by_event_superseded.items():
            e = seen_events[eid]
            if e.metadata.get("_remember_successor"):
                self._successor_for(e, sup, e.metadata.get("_successor_kind"))

        return ValidateReport(
            memories=len(reached), events=len(seen_events), pairs=npairs,
            bearing=sum(len(v) for v in bearing.values()), verdicts=verdicts, requests=requests,
            input_tokens=tokens, latency_ms=(time.perf_counter() - t0) * 1000, pending=pending_after, model=model,
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
        candidates: Iterable[Memory] | None = None,
        validate: bool | None = None,
        annotate: bool = False,
    ) -> RecallReport:
        """Return live memories relevant to `query`, ranked by Jev's relevance vote. No embeddings.

        `candidates` restricts the pool (for example the top-k from your vector store); relevance is still judged.
        `validate=True` (the default in lazy mode) first judges the top candidates against every event they have
        not seen yet, so nothing stale is returned even when observe() only appended to the log.
        `annotate=True` is the add-only serving mode: superseded and contradicted memories stay in the pool and
        each `Recalled.note` says what retired them ("OUTDATED, replaced as of <source>: <event text>"), the
        same label `Governor.annotate` attaches. Live results carry `note=None`.
        """
        ns = namespace or self.namespace
        t0 = time.perf_counter()
        if validate is None:
            validate = self.policy.lazy
        statuses = {Status.ACTIVE, Status.FROZEN} | ({Status.NEEDS_REVIEW} if include_review else set())
        if annotate:
            statuses |= {Status.SUPERSEDED, Status.CONTRADICTED}
        t_now = now()
        if candidates is None:
            pool = [m for m in self.store.list_memories(ns, statuses) if not m.is_expired(t_now)]
        else:
            pool = []
            for c in candidates:
                fresh = self.store.get_memory(c.id)
                if fresh is not None and fresh.namespace == ns and fresh.status in statuses and not fresh.is_expired(t_now):
                    pool.append(fresh)
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
        validated: ValidateReport | None = None
        requests = len(batches)
        if validate and scored:
            # Check a little more than `limit` so a candidate that dies can be replaced from the next ranks.
            head = scored[: max(limit * 2, limit + 5)]
            validated = self.validate([r.memory for r in head], namespace=ns)
            requests += validated.requests
            tokens += validated.input_tokens
            fresh = {m.id: m for m in (self.store.get_memory(r.memory.id) for r in head) if m is not None}
            kept: list[Recalled] = []
            for r in head:
                m = fresh.get(r.memory.id)
                if m is not None and m.status in statuses:
                    kept.append(Recalled(memory=m, relevance=r.relevance))
            scored = kept
        results = scored[:limit]
        if annotate:
            for r in results:
                r.note = self.note_for(r.memory)
        return RecallReport(
            query=query, results=results, considered=len(pool), requests=requests,
            input_tokens=tokens, latency_ms=(time.perf_counter() - t0) * 1000, validated=validated,
        )

    def note_for(self, m: Memory) -> str | None:
        """The serving note for a retired memory, built from the last applied verdict that moved it to its
        current status: "OUTDATED, replaced as of <event source>: <event text>" (superseded) or
        "OUTDATED, no longer true as of ...: ..." (contradicted). None for every other status."""
        if m.status not in (Status.SUPERSEDED, Status.CONTRADICTED):
            return None
        word = "replaced" if m.status is Status.SUPERSEDED else "no longer true"
        last = [v for v in self.history(m.id) if v.applied and v.to_status is m.status]
        if last:
            e = self.store.get_event(last[-1].event_id)
            if e is not None:
                return f"OUTDATED, {word} as of {e.source}: {e.text}"
        return f"OUTDATED, {word}"

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


def _chunk_any(items: list[Any], size: int) -> list[list[Any]]:
    size = max(1, size)
    return [items[i:i + size] for i in range(0, len(items), size)]


def _chunk(items: list[Memory], size: int) -> list[list[Memory]]:
    size = max(1, size)
    return [items[i : i + size] for i in range(0, len(items), size)]
