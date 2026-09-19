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

    review_below: float = 0.5
    """An uncertain verdict moves a memory to needs_review only when still_true is below this. An uncertain
    vote that leans true is logged and leaves the status alone. On the labelled set every uncertain vote at
    or above 0.5 was on a fact that was in fact still true; on LongMemEval, sending them to review flooded
    the queue with true facts (evals/longmemeval/README.md)."""

    staged: bool = True
    """Ask the full judgment in two stages: bears + still_true (+ the two event questions) for every memory, then
    replaces + partial only for memories whose still_true is at or below contradict_max - margin, which is the only
    place the policy reads those two votes. Answers are independent inside a request, so this changes cost, not
    votes. Skipped votes are recorded as 0."""

    second_opinion: bool = True
    """A contradicted/superseded verdict is re-judged with the memory alone in the state before it is written.
    If the clean-context vote disagrees, the memory goes to needs_review instead of dying. Batches of
    near-identical memories (twenty facts about the same service) are the one place Jev's distractor weakness
    showed up in scale tests (scripts/scale_lazy.py); kills are rare, so this costs one request per kill."""

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

    # --- pair screening (many events x many memories in one request) ------------
    pair_events: int = 10
    """Events per pair-screen request. Used by validate() and observe_many(). evals/screen_matrix.py:
    10 x 25 keeps 134/136 labelled bearing pairs at pair_min 0.15 for 52 tokens per pair;
    5 x 100 drops to 129/136, so keep the memory side small and the event side wide."""

    pair_memories: int = 25
    """Memories per pair-screen request."""

    pair_min: float = 0.15
    """Pair-screen threshold. Lower than screen_min because the pair question is stricter (fewer
    unrelated pairs pass) and the full judgment runs after it anyway. Measured: 134/136 bearing pairs kept."""

    lazy: bool = False
    """When True, observe() only appends the event to the log and judges nothing; memories are judged
    against their pending events when they are next read (recall(validate=True), Governor.filter) or by
    validate()/sweep in the background. Cost then scales with what is used, not with what is stored."""

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

    def transition(
        self, status: Status, d: Disposition, *, source: str | None = None, still_true: float | None = None,
    ) -> Status:
        """Next status for a memory given its current status, a disposition, the event's source, and (optionally)
        the still_true vote, which decides whether an uncertain verdict is worth a human's time."""
        if source is not None and source in self.review_only_sources and d in (
            Disposition.CONTRADICTED, Disposition.SUPERSEDED, Disposition.PARTIAL,
        ):
            d = Disposition.UNCERTAIN
        if d is Disposition.UNCERTAIN and still_true is not None and still_true >= self.review_below:
            return status  # leans true: logged, not queued
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
