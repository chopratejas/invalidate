"""A kill needs two independent votes: the batch vote and a clean-context re-judgment of the memory alone."""
from __future__ import annotations

from dataclasses import dataclass, field

from conftest import CONFIRM, SUPERSEDE, UNRELATED, FakeJudge

from invalidate import Invalidate, Policy, Status
from invalidate.judge import JudgeResult, ObserveBatch, PairBatch
from invalidate.types import Disposition


@dataclass
class ContextJudge(FakeJudge):
    """Votes SUPERSEDE for `victim` only when it shares a batch with other memories (the distractor effect);
    alone in the state it votes `alone_votes`."""

    victim: str = "deploys"
    alone_votes: object = CONFIRM
    solo_calls: list = field(default_factory=list)

    def screen_pairs(self, events, memories, pairs):
        scores = {(j, i): (1.0 if self.victim in memories[i].fact else self._lookup(self.votes, memories[i], self.default_votes).bears)
                  for j, i in pairs}
        return PairBatch(scores, JudgeResult(1, "fake"))

    def observe(self, event, memories):
        self.observe_calls.append((event, list(memories)))
        if len(memories) == 1:
            self.solo_calls.append(memories[0].fact)
            m = memories[0]
            v = self.alone_votes if self.victim in m.fact else self._lookup(self.votes, m, self.default_votes)
            return ObserveBatch([v], JudgeResult(1, "fake"))
        batch_victim = CONFIRM if "still" in event.text else SUPERSEDE
        votes = [batch_victim if self.victim in m.fact else self._lookup(self.votes, m, self.default_votes) for m in memories]
        return ObserveBatch(votes, JudgeResult(1, "fake"))


def test_disagreement_cancels_the_kill():
    j = ContextJudge().script("owned by Dana", SUPERSEDE)
    mem = Invalidate(":memory:", judge=j)  # second_opinion defaults on
    a = mem.remember("svc-159 is owned by Dana")
    b = mem.remember("svc-159 deploys at noon UTC")
    c = mem.remember("lunch at noon")
    rep = mem.observe("svc-159 is now owned by Jae", source="slack")
    assert mem.get(a.id).status is Status.SUPERSEDED  # both votes agree
    # batch said kill, alone said confirm at 0.95: the clean vote leans true, so the memory stays active and
    # the disagreement is logged as an uncertain verdict (a clean vote below review_below would queue it).
    assert mem.get(b.id).status is Status.ACTIVE
    assert mem.get(c.id).status is Status.ACTIVE
    assert sorted(j.solo_calls) == sorted(["svc-159 is owned by Dana", "svc-159 deploys at noon UTC"])
    vb = [v for v in rep.verdicts if v.memory_id == b.id][0]
    assert vb.disposition is Disposition.UNCERTAIN and vb.votes == CONFIRM and not vb.applied  # the deciding vote is recorded
    assert rep.requests == 1 + 2  # one batch + two second opinions
    assert len(mem.history(b.id)) == 1


def test_disagreement_with_a_leaning_false_clean_vote_queues_review():
    from conftest import UNCERTAIN

    j = ContextJudge(alone_votes=UNCERTAIN)  # still_true 0.45
    mem = Invalidate(":memory:", judge=j)
    b = mem.remember("svc-159 deploys at noon UTC")
    mem.remember("lunch at noon")
    mem.observe("svc-159 is now owned by Jae")
    assert mem.get(b.id).status is Status.NEEDS_REVIEW


def test_second_opinion_off_writes_the_batch_vote():
    j = ContextJudge()
    mem = Invalidate(":memory:", judge=j, policy=Policy(second_opinion=False))
    b = mem.remember("svc-159 deploys at noon UTC")
    mem.remember("lunch at noon")
    mem.observe("svc-159 is now owned by Jae")
    assert mem.get(b.id).status is Status.SUPERSEDED and j.solo_calls == []


def test_no_second_opinion_when_nothing_dies(fake):
    fake.script("x", CONFIRM)
    mem = Invalidate(":memory:", judge=fake)
    mem.remember("x")
    mem.remember("y")
    rep = mem.observe("e")
    assert rep.requests == 1 and len(fake.observe_calls) == 1


def test_unrelated_second_opinion_also_cancels_the_kill():
    j = ContextJudge(alone_votes=UNRELATED)  # still_true 1.0: nothing to review
    mem = Invalidate(":memory:", judge=j)
    b = mem.remember("svc-159 deploys at noon UTC")
    mem.remember("lunch at noon")
    mem.observe("svc-159 is now owned by Jae")
    assert mem.get(b.id).status is Status.ACTIVE


def test_dry_run_still_takes_second_opinion_but_writes_nothing():
    j = ContextJudge()
    mem = Invalidate(":memory:", judge=j)
    b = mem.remember("svc-159 deploys at noon UTC")
    mem.remember("lunch at noon")
    rep = mem.observe("svc-159 is now owned by Jae", dry_run=True)
    assert [(v.disposition, v.to_status) for v in rep.verdicts if v.memory_id == b.id] == [(Disposition.UNCERTAIN, Status.ACTIVE)]
    assert mem.get(b.id).status is Status.ACTIVE and mem.history(b.id) == []


def test_validate_path_takes_second_opinion_per_event():
    j = ContextJudge().script("lunch", CONFIRM)  # lunch bears too, so the victim shares its batch
    mem = Invalidate(":memory:", judge=j, lazy=True)
    b = mem.remember("svc-159 deploys at noon UTC")
    mem.remember("lunch at noon")
    mem.observe("svc-159 is now owned by Jae")
    mem.observe("nothing to see here")
    rep = mem.validate()
    assert mem.get(b.id).status is Status.ACTIVE
    vb = [v for v in rep.verdicts if v.memory_id == b.id]
    assert vb[0].disposition is Disposition.UNCERTAIN and vb[0].to_status is Status.ACTIVE
    assert j.solo_calls == ["svc-159 deploys at noon UTC", "svc-159 deploys at noon UTC"]  # once per killing event


def test_validate_chain_is_recomputed_after_override():
    """Event 1 kills (overridden by a clean confirm), event 2 confirms: active throughout."""
    j = ContextJudge().script("lunch", CONFIRM)
    mem = Invalidate(":memory:", judge=j, lazy=True)
    b = mem.remember("svc-159 deploys at noon UTC")
    mem.remember("lunch at noon")
    mem.observe("svc-159 is now owned by Jae")
    mem.observe("svc-159 still deploys at noon")
    rep = mem.validate()
    vb = [v for v in rep.verdicts if v.memory_id == b.id]
    assert [v.to_status for v in vb] == [Status.ACTIVE, Status.ACTIVE]
    assert mem.get(b.id).status is Status.ACTIVE
