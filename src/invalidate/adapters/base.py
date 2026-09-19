"""The invalidation layer for any memory system.

An `Adapter` exposes a host store as verbatim text with the host's own ids, and knows how to mirror a
status change back (flag, delete, insert). A `Governor` puts invalidate's engine on top: it keeps the
verdict ledger in its own SQLite, judges events against every host memory, and pushes the outcome
into the host under a mode you choose. Hosts keep their data; invalidate keeps the truth about it.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..engine import Invalidate
from ..judge import Judge
from ..policy import Policy
from ..store import Store
from ..types import Disposition, Event, Memory, ObserveReport, RecallReport, Status, ValidateReport, Verdict, now

LIVE = frozenset({Status.ACTIVE, Status.NEEDS_REVIEW, Status.FROZEN})
DEAD = frozenset({Status.CONTRADICTED, Status.SUPERSEDED})


@dataclass
class HostMemory:
    """One memory as the host holds it. `text` is verbatim; `id` is the host's id."""

    id: str
    text: str
    kind: str = "fact"
    source: str = "host"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Reason:
    """Why a status changed. This is what adapters write into the host as the receipt."""

    status: Status
    disposition: str
    event_text: str
    event_source: str
    event_id: str
    still_true: float
    at: float

    def as_metadata(self, prefix: str = "invalidate_") -> dict[str, Any]:
        return {
            f"{prefix}status": self.status.value,
            f"{prefix}disposition": self.disposition,
            f"{prefix}event": self.event_text[:500],
            f"{prefix}event_source": self.event_source,
            f"{prefix}event_id": self.event_id,
            f"{prefix}still_true": self.still_true,
            f"{prefix}at": self.at,
        }

    def line(self) -> str:
        verb = {"superseded": "superseded by", "contradicted": "contradicted by", "needs_review": "unclear after",
                "active": "restored after"}.get(self.status.value, self.status.value)
        return f"invalidate: {verb} “{self.event_text[:160]}” ({self.event_source}, still true {self.still_true:.0%})"


@runtime_checkable
class Adapter(Protocol):
    name: str

    def pull(self) -> Iterable[HostMemory]: ...
    def flag(self, host_id: str, reason: Reason) -> None: ...
    def delete(self, host_id: str, reason: Reason) -> None: ...


class SupportsInsert(Protocol):
    def insert(self, text: str, source: str, metadata: dict[str, Any]) -> str | None: ...


@dataclass
class SyncReport:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    total: int = 0

    def __str__(self) -> str:
        return f"{self.total} host memories: {self.added} added, {self.updated} updated, {self.removed} gone, {self.unchanged} unchanged"


@dataclass
class Push:
    host_id: str
    action: str  # flag | delete | insert | clear
    status: Status
    error: str | None = None


@dataclass
class GovernorReport:
    report: ObserveReport
    pushes: list[Push]
    successor_host_id: str | None = None

    @property
    def errors(self) -> list[Push]:
        return [p for p in self.pushes if p.error]

    def summary(self) -> str:
        n_ok = sum(1 for p in self.pushes if not p.error)
        s = f"{self.report.summary()}; {n_ok} pushed to host"
        if self.errors:
            s += f", {len(self.errors)} push errors"
        return s


def _h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


class Governor:
    """invalidate as a layer over a host memory system.

        gov = Governor(ChromaAdapter(collection), "ledger.db", mode="flag")
        gov.sync()                                   # pull host memories into the ledger (idempotent)
        gov.observe("we migrated to SQLite", source="slack")   # judge all, push flags/deletes to the host
        gov.filter(results, id_of=lambda r: r["id"])           # hide dead memories at query time
    """

    MODES = ("flag", "delete", "ledger")

    def __init__(
        self,
        adapter: Adapter,
        db: str | Store = "invalidate.db",
        *,
        mode: str = "flag",
        successors: bool = False,
        policy: Policy | None = None,
        judge: Judge | None = None,
        namespace: str | None = None,
        api_key: str | None = None,
        lazy: bool | None = None,
    ) -> None:
        """`lazy=True`: observe() only appends to the event log; the memories your host returns are judged
        against their pending events inside filter() before they reach the prompt (and by validate() in the
        background). Cost then scales with what is read, not with what is stored. See SCALING.md."""
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.adapter = adapter
        self.mode = mode
        self.successors = successors
        self.namespace = namespace or f"host:{adapter.name}"
        self.mem = Invalidate(db, judge=judge, policy=policy, namespace=self.namespace, api_key=api_key, lazy=lazy)
        self._pushes: list[Push] = []
        self.mem.on_transition = self._on_transition

    @property
    def lazy(self) -> bool:
        return self.mem.policy.lazy

    # -- ids ----------------------------------------------------------------------
    def our_id(self, host_id: str) -> str:
        return f"{self.adapter.name}:{host_id}"

    @staticmethod
    def host_id(m: Memory) -> str:
        return str(m.metadata.get("host_id", ""))

    # -- sync ---------------------------------------------------------------------
    def sync(self) -> SyncReport:
        """Pull the host and reconcile. New ids are remembered verbatim; changed text resets a row to
        active (the host rewrote it, so it is a new claim); ids that vanished while live are marked deleted."""
        rep = SyncReport()
        pulled = {hm.id: hm for hm in self.adapter.pull()}
        rep.total = len(pulled)
        existing = {self.host_id(m): m for m in self.mem.list() if self.host_id(m)}
        for hid, hm in pulled.items():
            m = existing.get(hid)
            if m is None:
                self.mem.remember(
                    hm.text, kind=hm.kind, source=hm.source, id=self.our_id(hid),
                    metadata={"host": self.adapter.name, "host_id": hid, "host_hash": _h(hm.text), **hm.metadata},
                )
                rep.added += 1
            elif m.metadata.get("host_hash") != _h(hm.text):
                m.fact = hm.text
                m.status = Status.ACTIVE
                m.p_true = 1.0
                m.superseded_by = None
                m.updated_at = now()
                m.metadata = {**m.metadata, "host_hash": _h(hm.text), **hm.metadata}
                self.mem.store.update_memory(m)
                rep.updated += 1
            else:
                rep.unchanged += 1
        for hid, m in existing.items():
            if hid not in pulled and m.status in LIVE:
                self.mem.forget(m.id)
                rep.removed += 1
        return rep

    # -- judge --------------------------------------------------------------------
    def observe(self, text: str, *, source: str = "unknown", dry_run: bool = False, **kw: Any) -> GovernorReport:
        self._pushes = []
        report = self.mem.observe(text, source=source, dry_run=dry_run, **kw)
        successor_host_id = None
        if not dry_run and self.successors and self.mode != "ledger":
            superseded = [v for v in report.verdicts if v.changed and v.to_status is Status.SUPERSEDED]
            insert = getattr(self.adapter, "insert", None)
            if superseded and insert is not None:
                try:
                    successor_host_id = insert(
                        report.event.text, report.event.source,
                        {"invalidate_supersedes": [self.host_id(self.mem.get(v.memory_id)) for v in superseded],
                         "invalidate_event_id": report.event.id},
                    )
                    self._pushes.append(Push(successor_host_id or "", "insert", Status.ACTIVE))
                except Exception as e:  # noqa: BLE001 - host errors must not lose the ledger
                    self._pushes.append(Push("", "insert", Status.ACTIVE, error=repr(e)))
                if successor_host_id:
                    # Content-addressed hosts (markdown) hand back the same id for the same text: reuse the row.
                    succ = self.mem.store.get_memory(self.our_id(successor_host_id))
                    if succ is None:
                        succ = self.mem.remember(
                            report.event.text, source=report.event.source, id=self.our_id(successor_host_id),
                            metadata={"host": self.adapter.name, "host_id": successor_host_id,
                                      "host_hash": _h(report.event.text), "event_id": report.event.id},
                        )
                    for v in superseded:
                        self.mem.supersede(v.memory_id, by=succ.id)
        return GovernorReport(report=report, pushes=list(self._pushes), successor_host_id=successor_host_id)

    def observe_many(self, texts: Iterable[str], *, source: str = "unknown") -> GovernorReport:
        """Batch ingest: append every event, then (unless lazy) judge the pool against all of them in one pass
        (many events x many memories per Jev request). Pushes flags/deletes/successors like observe()."""
        self._pushes = []
        events, rep = self.mem.observe_many(texts, source=source, remember_successor=False)
        if rep is not None:
            self._push_successors(rep.verdicts)
        first = events[0] if events else Event(text="", namespace=self.namespace)
        report = ObserveReport(event=first, verdicts=rep.verdicts if rep else [], judged=rep.memories if rep else 0,
                               skipped=0, requests=rep.requests if rep else 0, input_tokens=rep.input_tokens if rep else 0,
                               latency_ms=rep.latency_ms if rep else 0.0, pending=0 if rep else self.mem.pending())
        return GovernorReport(report=report, pushes=list(self._pushes), successor_host_id=None)

    def validate(self, host_ids: Iterable[str] | None = None, *, budget_requests: int | None = None) -> ValidateReport:
        """Judge memories against the events they have not seen yet and push the outcomes to the host.
        `host_ids=None` drains every memory that is behind the log (use `budget_requests` from a background job)."""
        self._pushes = []
        mems = None
        if host_ids is not None:
            mems = [m for m in (self.mem.store.get_memory(self.our_id(h)) for h in host_ids) if m is not None]
        rep = self.mem.validate(mems, budget_requests=budget_requests)
        self._push_successors(rep.verdicts)
        return rep

    def _push_successors(self, verdicts: list[Verdict]) -> None:
        """After validate()/observe_many(): insert one verbatim successor per superseding event into the host."""
        if not self.successors or self.mode == "ledger":
            return
        insert = getattr(self.adapter, "insert", None)
        if insert is None:
            return
        by_event: dict[str, list[Verdict]] = {}
        for v in verdicts:
            if v.changed and v.to_status is Status.SUPERSEDED:
                by_event.setdefault(v.event_id, []).append(v)
        for eid, sup in by_event.items():
            e = self.mem.store.get_event(eid)
            if e is None:
                continue
            try:
                hid = insert(e.text, e.source, {"invalidate_supersedes": [self.host_id(self.mem.get(v.memory_id)) for v in sup],
                                                "invalidate_event_id": e.id})
                self._pushes.append(Push(hid or "", "insert", Status.ACTIVE))
            except Exception as ex:  # noqa: BLE001
                self._pushes.append(Push("", "insert", Status.ACTIVE, error=repr(ex)))
                continue
            if hid:
                succ = self.mem.store.get_memory(self.our_id(hid))
                if succ is None:
                    succ = self.mem.remember(
                        e.text, source=e.source, id=self.our_id(hid),
                        metadata={"host": self.adapter.name, "host_id": hid, "host_hash": _h(e.text), "event_id": e.id},
                    )
                for v in sup:
                    self.mem.supersede(v.memory_id, by=succ.id)

    def _on_transition(self, m: Memory, v: Verdict) -> None:
        if self.mode == "ledger":
            return
        hid = self.host_id(m)
        if not hid:
            return
        reason = Reason(
            status=v.to_status, disposition=v.disposition.value, event_text=self._event_text(v.event_id),
            event_source=self._event_source(v.event_id), event_id=v.event_id, still_true=v.votes.still_true, at=v.created_at,
        )
        action = "flag"
        try:
            if v.to_status in DEAD and self.mode == "delete":
                action = "delete"
                self.adapter.delete(hid, reason)
            else:
                self.adapter.flag(hid, reason)
            self._pushes.append(Push(hid, action, v.to_status))
        except Exception as e:  # noqa: BLE001
            self._pushes.append(Push(hid, action, v.to_status, error=repr(e)))

    # observe() persists the event before applying, so these lookups succeed inside the callback.
    def _event_text(self, event_id: str) -> str:
        e = self.mem.store.get_event(event_id)
        return e.text if e else ""

    def _event_source(self, event_id: str) -> str:
        e = self.mem.store.get_event(event_id)
        return e.source if e else "unknown"

    # -- read side ----------------------------------------------------------------
    def recall(self, query: str, **kw: Any) -> RecallReport:
        return self.mem.recall(query, **kw)

    def status_of(self, host_id: str) -> Status | None:
        m = self.mem.store.get_memory(self.our_id(host_id))
        return m.status if m else None

    def dead_ids(self) -> set[str]:
        return {self.host_id(m) for m in self.mem.list(statuses=DEAD) if self.host_id(m)}

    def review(self) -> list[Memory]:
        return self.mem.list(statuses=[Status.NEEDS_REVIEW])

    def filter(
        self, results: Iterable[Any], *, id_of: Callable[[Any], str], include_review: bool = True,
        validate: bool | None = None,
    ) -> list[Any]:
        """Drop host results whose memory is dead (contradicted or superseded). Unknown ids pass through.
        Memories under review are served by default: an uncertain vote is not a known-false fact, and on
        LongMemEval hiding them cost more correct answers than it saved (evals/longmemeval/README.md).
        They still sit in the review queue for a human. Pass include_review=False to hide them too.
        `validate=True` (default in lazy mode) first judges these results against the events they have not
        seen yet, so a stale memory is caught on the way to the prompt even if observe() never judged it."""
        results = list(results)
        if validate is None:
            validate = self.lazy
        if validate and results:
            self.validate([str(id_of(r)) for r in results])
        hide = set(self.dead_ids())
        if not include_review:
            hide |= {self.host_id(m) for m in self.review()}
        return [r for r in results if str(id_of(r)) not in hide]

    def annotate(
        self, results: Iterable[Any], *, id_of: Callable[[Any], str], validate: bool | None = None,
    ) -> list[tuple[Any, str | None]]:
        """Serving mode for prompts: keep every result and attach a note to the retired ones saying what
        retired them ("OUTDATED, replaced as of <event source>: <event text>"). The model sees the change
        chain explicitly instead of inferring it from dates, and questions about the previous value
        ("where did I keep them before?") stay answerable. On LongMemEval knowledge-update questions this
        beat hiding (evals/longmemeval/README.md). Returns [(result, note_or_None), ...]."""
        results = list(results)
        if validate is None:
            validate = self.lazy
        if validate and results:
            self.validate([str(id_of(r)) for r in results])
        out: list[tuple[Any, str | None]] = []
        for r in results:
            m = self.mem.store.get_memory(self.our_id(str(id_of(r))))
            note = None
            if m is not None and m.status in DEAD:
                last = [v for v in self.mem.history(m.id) if v.applied and v.to_status is m.status]
                word = "replaced" if m.status is Status.SUPERSEDED else "no longer true"
                if last:
                    e = self.mem.store.get_event(last[-1].event_id)
                    note = f"OUTDATED, {word} as of {e.source}: {e.text}" if e else f"OUTDATED, {word}"
                else:
                    note = f"OUTDATED, {word}"
            out.append((r, note))
        return out

    def guard(self, add: Callable[..., Any], *, text_of: Callable[..., Iterable[str]] | None = None, source: str = "user") -> Callable[..., Any]:
        """Wrap a host `add(...)`: judge the incoming text against every memory first, then call through.
        `text_of(*args, **kwargs)` yields the event texts; default takes the first positional arg
        (a string, or a list of {role, content} messages with role == user)."""

        def _default(*a: Any, **k: Any) -> Iterable[str]:
            x = a[0] if a else k.get("messages", k.get("text", ""))
            if isinstance(x, str):
                return [x]
            out = []
            for msg in x or []:
                if isinstance(msg, dict) and msg.get("role", "user") == "user" and msg.get("content"):
                    out.append(str(msg["content"]))
            return out

        extract = text_of or _default

        def wrapped(*a: Any, **k: Any) -> Any:
            for t in extract(*a, **k):
                if t.strip():
                    self.observe(t, source=source)
            result = add(*a, **k)
            self.sync()
            return result

        wrapped.__name__ = getattr(add, "__name__", "add")
        return wrapped

    # -- human controls, mirrored -------------------------------------------------
    def keep(self, host_id: str) -> Memory:
        m = self.mem.restore(self.our_id(host_id))
        if self.mode != "ledger":
            self.adapter.flag(host_id, Reason(Status.ACTIVE, "restored", "kept by a human", "human", "", 1.0, now()))
        return m

    def forget(self, host_id: str) -> Memory:
        m = self.mem.forget(self.our_id(host_id))
        if self.mode != "ledger":
            self.adapter.delete(host_id, Reason(Status.DELETED, "forgotten", "forgotten by a human", "human", "", 0.0, now()))
        return m

    def close(self) -> None:
        self.mem.close()

    def __enter__(self) -> "Governor":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class InMemoryAdapter:
    """A reference adapter over a dict. Used in tests and as the template for new adapters."""

    name = "memory"

    def __init__(self, items: dict[str, str] | None = None) -> None:
        self.items: dict[str, dict[str, Any]] = {k: {"text": v, "meta": {}} for k, v in (items or {}).items()}
        self.log: list[tuple[str, str, dict[str, Any]]] = []
        self._n = 0

    def pull(self) -> Iterable[HostMemory]:
        return [HostMemory(id=k, text=v["text"], metadata=dict(v["meta"])) for k, v in self.items.items()]

    def flag(self, host_id: str, reason: Reason) -> None:
        self.items[host_id]["meta"].update(reason.as_metadata())
        self.log.append(("flag", host_id, reason.as_metadata()))

    def delete(self, host_id: str, reason: Reason) -> None:
        self.items.pop(host_id, None)
        self.log.append(("delete", host_id, reason.as_metadata()))

    def insert(self, text: str, source: str, metadata: dict[str, Any]) -> str:
        self._n += 1
        hid = f"new{self._n}"
        self.items[hid] = {"text": text, "meta": {"source": source, **metadata}}
        self.log.append(("insert", hid, metadata))
        return hid


__all__ = ["Adapter", "HostMemory", "Reason", "Governor", "GovernorReport", "SyncReport", "Push", "InMemoryAdapter",
           "SupportsInsert", "LIVE", "DEAD"]
