"""Policy: how code turns Jev's votes into dispositions and status transitions.

Jev only votes. Every threshold that decides a write lives here, in plain code,
so it can be tuned against your own data without touching the questions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .types import Disposition, Status, Votes


@dataclass
class Policy:
    # --- vote thresholds --------------------------------------------------
    # Defaults were chosen by sweeping evals/cases.py (157 labeled cases) for the highest strict
    # accuracy that does not add a single false invalidation. Re-run `evals/run_eval.py` on your data.
    bears_min: float = 0.6
    """Below this, the event is treated as unrelated to the memory: no write."""

    confirm_min: float = 0.6
    """still_true at or above this (and bearing) confirms the memory."""

    contradict_max: float = 0.4
    """still_true at or below this (and bearing) contradicts the memory."""

    replace_min: float = 0.6
    """If contradicted and replaces >= this, the memory is superseded, not just contradicted."""

    hypothetical_max: float = 0.7
    """If the event is judged a question/proposal/plan at or above this, any non-confirming
    vote becomes HYPOTHETICAL: logged for the audit trail, never written to status."""

    partial_min: float = 0.85
    """If a memory would be contradicted/superseded but `partial` is at or above this, the event changed
    only a secondary detail of a compound fact: the memory goes to needs_review for a rewrite instead of
    dying. High on purpose: Jev reads most facts as having *some* second part, so only a strong vote counts."""

    margin: float = 0.05
    """Dead band around confirm_min and contradict_max. A still_true vote within it is uncertain by rule,
    so a vote that wobbles across the line run to run lands in needs_review consistently."""

    directive_max: float = 0.7
    """If the event is judged to be a command to the system/assistant about what to record, override, or
    believe (rather than a report of something in the world) at or above this, any non-unrelated vote
    becomes DIRECTIVE: logged, never written. Prompt-injection defense; source trust stays with the caller."""

    relevance_min: float = 0.5
    """recall(): minimum relevance probability to return a memory."""

    # --- transition rules --------------------------------------------------
    judge_statuses: frozenset[Status] = field(
        default_factory=lambda: frozenset({Status.ACTIVE, Status.NEEDS_REVIEW, Status.FROZEN})
    )
    """Which memories are sent to the judge on observe()."""

    review_resolves: bool = True
    """A confirmed verdict moves needs_review back to active."""

    review_only_sources: frozenset[str] = field(default_factory=frozenset)
    """Events from these sources can never flip a memory to contradicted/superseded; the worst they can
    do is send it to needs_review. Put untrusted channels here (customer email, public webhooks)."""

    # --- batching ----------------------------------------------------------
    batch_size: int = 20
    """Memories per full-judgment Jev request: 4 questions each, plus two event-level questions."""

    max_workers: int = 8
    """Concurrent Jev requests."""

    screen_above: int = 200
    """When more than this many memories are judgeable, run a cheap bears-only screen first and send only
    the memories that pass it to the full judgment. Set to 0 to always screen, or a huge number to never."""

    screen_batch_size: int = 100
    """Memories per screening request (one short question each)."""

    screen_min: float = 0.3
    """Screen threshold; deliberately looser than bears_min so the screen only drops clear non-matches."""

    def dispose(self, v: Votes) -> Disposition:
        if v.bears < self.bears_min:
            return Disposition.UNRELATED
        # Form checks come before content checks: a command or a question never writes, not even a confirmation.
        if v.directive >= self.directive_max:
            return Disposition.DIRECTIVE
        if v.hypothetical >= self.hypothetical_max:
            return Disposition.HYPOTHETICAL
        if v.still_true >= self.confirm_min + self.margin:
            return Disposition.CONFIRMED
        if v.still_true <= self.contradict_max - self.margin:
            if v.partial >= self.partial_min:
                return Disposition.PARTIAL
            if v.replaces >= self.replace_min:
                return Disposition.SUPERSEDED
            return Disposition.CONTRADICTED
        return Disposition.UNCERTAIN

    def transition(self, status: Status, d: Disposition, *, source: str | None = None) -> Status:
        """Next status for a memory given its current status, a disposition, and the event's source."""
        if source is not None and source in self.review_only_sources and d in (
            Disposition.CONTRADICTED, Disposition.SUPERSEDED, Disposition.PARTIAL,
        ):
            d = Disposition.UNCERTAIN
        if status is Status.FROZEN:
            return status  # judged for the audit log, never flipped
        if status not in (Status.ACTIVE, Status.NEEDS_REVIEW):
            return status
        if d in (Disposition.UNRELATED, Disposition.HYPOTHETICAL, Disposition.DIRECTIVE):
            return status
        if d is Disposition.CONFIRMED:
            return Status.ACTIVE if (status is Status.ACTIVE or self.review_resolves) else status
        if d is Disposition.CONTRADICTED:
            return Status.CONTRADICTED
        if d is Disposition.SUPERSEDED:
            return Status.SUPERSEDED
        if d in (Disposition.UNCERTAIN, Disposition.PARTIAL):
            return Status.NEEDS_REVIEW
        return status
