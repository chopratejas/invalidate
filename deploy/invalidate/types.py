"""Core data types for invalidate.

Everything here is plain data. No model calls, no I/O.
"""
from __future__ import annotations

import enum
import secrets
import time
from dataclasses import dataclass, field
from typing import Any


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(6)}"


def now() -> float:
    return time.time()


class Status(str, enum.Enum):
    """Lifecycle status of a memory. Only code moves a memory between these."""

    ACTIVE = "active"            # believed true; returned by recall
    NEEDS_REVIEW = "needs_review"  # evidence bears on it but the verdict is uncertain
    CONTRADICTED = "contradicted"  # evidence says it no longer holds; no replacement given
    SUPERSEDED = "superseded"    # evidence says it no longer holds and states the new value
    FROZEN = "frozen"            # human-pinned; judged and logged, never auto-flipped
    EXPIRED = "expired"          # hard TTL elapsed
    DELETED = "deleted"          # soft-deleted by the caller

    @property
    def live(self) -> bool:
        """Statuses that recall may return."""
        return self in (Status.ACTIVE, Status.FROZEN)


class Disposition(str, enum.Enum):
    """What Jev's votes, composed by policy, say an event means for one memory."""

    UNRELATED = "unrelated"
    CONFIRMED = "confirmed"
    CONTRADICTED = "contradicted"
    SUPERSEDED = "superseded"
    UNCERTAIN = "uncertain"
    HYPOTHETICAL = "hypothetical"  # event bears on the memory but is a question/plan/proposal: logged, never written
    DIRECTIVE = "directive"        # event is an instruction to the system about what to record: logged, never written
    PARTIAL = "partial"            # event changes only part of a compound fact: needs a rewrite, goes to review


@dataclass
class Memory:
    fact: str
    id: str = field(default_factory=lambda: new_id("mem"))
    namespace: str = "default"
    kind: str = "fact"
    source: str = "unknown"
    status: Status = Status.ACTIVE
    p_true: float = 1.0
    created_at: float = field(default_factory=now)
    updated_at: float = field(default_factory=now)
    last_checked: float | None = None
    expires_at: float | None = None
    superseded_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, at: float | None = None) -> bool:
        return self.expires_at is not None and (at if at is not None else now()) >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["status"] = self.status.value
        return d


@dataclass
class Event:
    text: str
    id: str = field(default_factory=lambda: new_id("evt"))
    namespace: str = "default"
    source: str = "unknown"
    created_at: float = field(default_factory=now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class Votes:
    """Raw probabilities from Jev for one (event, memory) pair. Never rewritten."""

    bears: float
    still_true: float
    replaces: float
    hypothetical: float  # event-level; repeated on every pair for auditability
    directive: float = 0.0  # event-level: the event commands a system/assistant about what to store or believe
    partial: float = 0.0  # the event changes only part of what the fact asserts


@dataclass
class Verdict:
    """One judged (event, memory) pair and what code did about it."""

    event_id: str
    memory_id: str
    votes: Votes
    disposition: Disposition
    from_status: Status
    to_status: Status
    applied: bool  # False when the status did not change (unrelated, frozen, etc.)
    created_at: float = field(default_factory=now)
    id: int | None = None

    @property
    def changed(self) -> bool:
        return self.from_status != self.to_status


@dataclass
class ObserveReport:
    event: Event
    verdicts: list[Verdict]
    judged: int
    skipped: int  # memories not sent to the judge (wrong status, expired, ...)
    requests: int
    input_tokens: int
    latency_ms: float
    model: str | None = None
    screened_out: int = 0  # memories dropped by the cheap bears-only screen (large pools only)
    successor: Memory | None = None  # set when remember_successor=True stored the event as a new memory

    @property
    def changed(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.changed]

    def count(self, disposition: Disposition) -> int:
        return sum(1 for v in self.verdicts if v.disposition is disposition)

    @property
    def cost_usd(self) -> float:
        # jev-1.13: $0.042 per million input tokens; output tokens are free.
        return self.input_tokens * 0.042 / 1_000_000

    def summary(self) -> str:
        parts = [f"{self.judged} judged"]
        for d in (Disposition.CONTRADICTED, Disposition.SUPERSEDED, Disposition.PARTIAL, Disposition.UNCERTAIN,
                  Disposition.HYPOTHETICAL, Disposition.DIRECTIVE, Disposition.CONFIRMED):
            n = self.count(d)
            if n:
                parts.append(f"{n} {d.value}")
        if self.screened_out:
            parts.append(f"{self.screened_out} screened out")
        parts.append(f"{self.requests} req")
        parts.append(f"{self.latency_ms:.0f} ms")
        parts.append(f"${self.cost_usd:.5f}")
        return ", ".join(parts)


@dataclass
class Recalled:
    memory: Memory
    relevance: float


@dataclass
class RecallReport:
    query: str
    results: list[Recalled]
    considered: int
    requests: int
    input_tokens: int
    latency_ms: float

    @property
    def memories(self) -> list[Memory]:
        return [r.memory for r in self.results]

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * 0.042 / 1_000_000
