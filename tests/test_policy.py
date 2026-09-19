"""Policy.dispose and Policy.transition: every branch and every boundary, as coded."""
from __future__ import annotations

import pytest

from invalidate import Disposition, Policy, Status, Votes

D = Disposition
S = Status


def v(bears=0.9, still_true=0.9, replaces=0.0, hypothetical=0.0) -> Votes:
    return Votes(bears=bears, still_true=still_true, replaces=replaces, hypothetical=hypothetical)


# --------------------------------------------------------------------------- defaults


def test_default_thresholds_and_batching():
    p = Policy()
    assert (p.bears_min, p.confirm_min, p.contradict_max, p.replace_min, p.hypothetical_max) == (
        0.5, 0.7, 0.3, 0.6, 0.7,
    )
    assert p.relevance_min == 0.5
    assert p.judge_statuses == frozenset({S.ACTIVE, S.NEEDS_REVIEW, S.FROZEN})
    assert isinstance(p.judge_statuses, frozenset)
    assert p.review_resolves is True
    assert p.batch_size == 20
    assert p.max_workers == 8


def test_judge_statuses_override_does_not_leak_between_instances():
    a = Policy(judge_statuses=frozenset({S.ACTIVE}))
    b = Policy()
    assert a.judge_statuses == frozenset({S.ACTIVE})
    assert b.judge_statuses == frozenset({S.ACTIVE, S.NEEDS_REVIEW, S.FROZEN})


# --------------------------------------------------------------------------- dispose: branches


def test_dispose_unrelated_when_bears_below_threshold_even_if_contradicting():
    assert Policy().dispose(v(bears=0.49, still_true=0.0, replaces=1.0)) is D.UNRELATED


def test_dispose_unrelated_ignores_hypothetical():
    assert Policy().dispose(v(bears=0.1, still_true=0.0, hypothetical=1.0)) is D.UNRELATED


def test_dispose_confirmed_when_bearing_and_still_true_high():
    assert Policy().dispose(v(bears=0.9, still_true=0.95)) is D.CONFIRMED


def test_dispose_confirmed_is_not_downgraded_by_hypothetical():
    assert Policy().dispose(v(bears=0.9, still_true=0.95, hypothetical=1.0)) is D.CONFIRMED


def test_dispose_confirmed_ignores_replaces():
    assert Policy().dispose(v(bears=0.9, still_true=0.95, replaces=1.0)) is D.CONFIRMED


def test_dispose_contradicted_without_replacement():
    assert Policy().dispose(v(bears=0.9, still_true=0.1, replaces=0.2)) is D.CONTRADICTED


def test_dispose_superseded_when_replaces_high():
    assert Policy().dispose(v(bears=0.9, still_true=0.1, replaces=0.9)) is D.SUPERSEDED


def test_dispose_uncertain_when_still_true_in_the_middle():
    assert Policy().dispose(v(bears=0.9, still_true=0.5)) is D.UNCERTAIN


def test_dispose_uncertain_in_middle_regardless_of_replaces():
    assert Policy().dispose(v(bears=0.9, still_true=0.5, replaces=1.0)) is D.UNCERTAIN


def test_dispose_hypothetical_makes_contradiction_hypothetical():
    assert Policy().dispose(v(bears=0.9, still_true=0.1, replaces=0.2, hypothetical=0.9)) is D.HYPOTHETICAL


def test_dispose_hypothetical_makes_supersession_hypothetical():
    assert Policy().dispose(v(bears=0.9, still_true=0.1, replaces=0.95, hypothetical=0.9)) is D.HYPOTHETICAL


def test_dispose_hypothetical_makes_uncertain_hypothetical():
    # A question that merely unsettles still_true must not push a memory into review.
    assert Policy().dispose(v(bears=0.9, still_true=0.5, replaces=0.2, hypothetical=0.9)) is D.HYPOTHETICAL


# --------------------------------------------------------------------------- dispose: boundaries


def test_boundary_bears_min_is_inclusive():
    p = Policy()
    assert p.dispose(v(bears=0.5, still_true=0.95)) is D.CONFIRMED
    assert p.dispose(v(bears=0.4999, still_true=0.95)) is D.UNRELATED


def test_boundary_confirm_min_is_inclusive():
    p = Policy()
    assert p.dispose(v(still_true=0.7)) is D.CONFIRMED
    assert p.dispose(v(still_true=0.6999)) is D.UNCERTAIN


def test_boundary_contradict_max_is_inclusive():
    p = Policy()
    assert p.dispose(v(still_true=0.3, replaces=0.0)) is D.CONTRADICTED
    assert p.dispose(v(still_true=0.3001, replaces=0.0)) is D.UNCERTAIN


def test_boundary_replace_min_is_inclusive():
    p = Policy()
    assert p.dispose(v(still_true=0.1, replaces=0.6)) is D.SUPERSEDED
    assert p.dispose(v(still_true=0.1, replaces=0.5999)) is D.CONTRADICTED


def test_boundary_hypothetical_max_is_inclusive():
    p = Policy()
    assert p.dispose(v(still_true=0.1, replaces=0.9, hypothetical=0.7)) is D.HYPOTHETICAL
    assert p.dispose(v(still_true=0.1, replaces=0.9, hypothetical=0.6999)) is D.SUPERSEDED


def test_boundary_extreme_probabilities():
    p = Policy()
    assert p.dispose(v(bears=1.0, still_true=1.0, replaces=1.0, hypothetical=1.0)) is D.CONFIRMED
    assert p.dispose(v(bears=1.0, still_true=0.0, replaces=1.0, hypothetical=0.0)) is D.SUPERSEDED
    assert p.dispose(v(bears=1.0, still_true=0.0, replaces=0.0, hypothetical=0.0)) is D.CONTRADICTED
    assert p.dispose(v(bears=0.0, still_true=0.0, replaces=0.0, hypothetical=0.0)) is D.UNRELATED


# --------------------------------------------------------------------------- dispose: custom thresholds


def test_custom_thresholds_are_honoured():
    p = Policy(bears_min=0.2, confirm_min=0.9, contradict_max=0.5, replace_min=0.95, hypothetical_max=0.5)
    # bears 0.3 is now enough to bear
    assert p.dispose(v(bears=0.3, still_true=0.95)) is D.CONFIRMED
    # 0.85 is no longer confirmed under confirm_min=0.9
    assert p.dispose(v(bears=0.9, still_true=0.85)) is D.UNCERTAIN
    # 0.45 is now a contradiction under contradict_max=0.5
    assert p.dispose(v(bears=0.9, still_true=0.45, replaces=0.9)) is D.CONTRADICTED  # replaces < 0.95
    assert p.dispose(v(bears=0.9, still_true=0.45, replaces=0.96)) is D.SUPERSEDED
    # hypothetical 0.5 now makes it hypothetical
    assert p.dispose(v(bears=0.9, still_true=0.45, replaces=0.96, hypothetical=0.5)) is D.HYPOTHETICAL


def test_custom_thresholds_can_disable_hypothetical_downgrade():
    p = Policy(hypothetical_max=1.01)
    assert p.dispose(v(still_true=0.0, replaces=0.9, hypothetical=1.0)) is D.SUPERSEDED


def test_custom_thresholds_can_make_everything_bear():
    p = Policy(bears_min=0.0)
    assert p.dispose(v(bears=0.0, still_true=0.0)) is D.CONTRADICTED


# --------------------------------------------------------------------------- transition matrix


def _expected(status: S, d: D, review_resolves: bool) -> S:
    if status is S.FROZEN or status in (S.CONTRADICTED, S.SUPERSEDED, S.EXPIRED, S.DELETED):
        return status
    if d in (D.UNRELATED, D.HYPOTHETICAL):
        return status
    if d is D.CONFIRMED:
        if status is S.ACTIVE:
            return S.ACTIVE
        return S.ACTIVE if review_resolves else S.NEEDS_REVIEW
    if d is D.CONTRADICTED:
        return S.CONTRADICTED
    if d is D.SUPERSEDED:
        return S.SUPERSEDED
    return S.NEEDS_REVIEW


@pytest.mark.parametrize("status", list(S))
@pytest.mark.parametrize("d", list(D))
@pytest.mark.parametrize("review_resolves", [True, False])
def test_transition_full_matrix(status: S, d: D, review_resolves: bool):
    p = Policy(review_resolves=review_resolves)
    assert p.transition(status, d) is _expected(status, d, review_resolves)


@pytest.mark.parametrize("d", list(D))
def test_frozen_never_changes(d: D):
    assert Policy().transition(S.FROZEN, d) is S.FROZEN


@pytest.mark.parametrize("status", [S.CONTRADICTED, S.SUPERSEDED, S.EXPIRED, S.DELETED])
@pytest.mark.parametrize("d", list(D))
def test_terminal_statuses_never_change(status: S, d: D):
    assert Policy().transition(status, d) is status


def test_active_transitions():
    p = Policy()
    assert p.transition(S.ACTIVE, D.UNRELATED) is S.ACTIVE
    assert p.transition(S.ACTIVE, D.CONFIRMED) is S.ACTIVE
    assert p.transition(S.ACTIVE, D.CONTRADICTED) is S.CONTRADICTED
    assert p.transition(S.ACTIVE, D.SUPERSEDED) is S.SUPERSEDED
    assert p.transition(S.ACTIVE, D.UNCERTAIN) is S.NEEDS_REVIEW


def test_needs_review_confirmed_resolves_to_active_by_default():
    assert Policy().transition(S.NEEDS_REVIEW, D.CONFIRMED) is S.ACTIVE


def test_needs_review_confirmed_stays_when_review_does_not_resolve():
    assert Policy(review_resolves=False).transition(S.NEEDS_REVIEW, D.CONFIRMED) is S.NEEDS_REVIEW


def test_needs_review_other_dispositions():
    p = Policy()
    assert p.transition(S.NEEDS_REVIEW, D.UNRELATED) is S.NEEDS_REVIEW
    assert p.transition(S.NEEDS_REVIEW, D.UNCERTAIN) is S.NEEDS_REVIEW
    assert p.transition(S.NEEDS_REVIEW, D.CONTRADICTED) is S.CONTRADICTED
    assert p.transition(S.NEEDS_REVIEW, D.SUPERSEDED) is S.SUPERSEDED


def test_review_resolves_false_does_not_affect_active_confirmed():
    assert Policy(review_resolves=False).transition(S.ACTIVE, D.CONFIRMED) is S.ACTIVE


# --------------------------------------------------------------------------- Status helpers


def test_status_live_only_active_and_frozen():
    assert {s for s in S if s.live} == {S.ACTIVE, S.FROZEN}


def test_status_and_disposition_values_are_stable_strings():
    assert S.NEEDS_REVIEW.value == "needs_review"
    assert S("superseded") is S.SUPERSEDED
    assert D("unrelated") is D.UNRELATED
