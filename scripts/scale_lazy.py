#!/usr/bin/env python
"""Scale check: 20,000 memories, three paths, real Jev calls. Prints the numbers SCALING.md quotes.

  A  eager observe(): one event against the whole pool (screen + full judgment)
  B  lazy: 200 events appended for free, then one recall over the host's top-10 candidates, validated
  C  batch ingest: observe_many() with 10 events against the whole pool (pair screen + full judgment)

    python scripts/scale_lazy.py --memories 20000 --events 200 --ingest 10
"""
from __future__ import annotations

import argparse
import random
import sys
import time

from invalidate import Invalidate, Policy, Status

PEOPLE = ["Alice", "Bob", "Chen", "Dana", "Eli", "Fatima", "Gus", "Hana", "Ivan", "Jae"]
TIMES = ["2pm UTC", "6pm UTC", "9am UTC", "midnight UTC", "noon UTC"]
REGIONS = ["us-east-1", "eu-west-1", "ap-south-1", "us-west-2"]
PLANS = ["Starter", "Team", "Enterprise"]
DBS = ["Postgres", "MySQL", "SQLite", "DynamoDB", "Mongo"]
LANGS = ["Go", "Python", "TypeScript", "Rust", "Java"]

TEMPLATES = [
    lambda i, r: f"svc-{i} is owned by {r.choice(PEOPLE)}",
    lambda i, r: f"svc-{i} deploys at {r.choice(TIMES)} on weekdays",
    lambda i, r: f"svc-{i} runs in {r.choice(REGIONS)}",
    lambda i, r: f"svc-{i} stores its data in {r.choice(DBS)}",
    lambda i, r: f"svc-{i} is written in {r.choice(LANGS)}",
    lambda i, r: f"customer acct-{i} is on the {r.choice(PLANS)} plan",
    lambda i, r: f"the rate limit for svc-{i} is {r.choice([100, 500, 1000])} requests per minute",
    lambda i, r: f"the on-call rotation for svc-{i} is {r.choice(PEOPLE)} then {r.choice(PEOPLE)}",
    lambda i, r: f"svc-{i} has a p99 latency budget of {r.choice([100, 200, 400])} ms",
    lambda i, r: f"the runbook for svc-{i} lives in the wiki under Platform/{r.choice(['A', 'B', 'C'])}",
]

CHATTER = [
    "reminder: all-hands is Thursday at 4pm", "the office kitchen is out of oat milk again",
    "PR review queue is long today, please pick one up", "welcome our new intern on the data team",
    "the Q3 planning doc is open for comments", "friday lunch is tacos", "vpn will be flaky during the upgrade window",
    "someone left a laptop charger in room 4B", "design review moved to the big room", "great launch yesterday, team",
]


def build(mem: Invalidate, n: int, seed: int) -> list[tuple[str, str]]:
    r = random.Random(seed)
    targets: list[tuple[str, str]] = []
    facts = []
    for i in range(n):
        t = TEMPLATES[i % len(TEMPLATES)]
        facts.append(t(i // len(TEMPLATES), r))
    for f in facts:
        mem.remember(f, source="wiki", kind="fact")
    # Ten owner facts become targets: an event will name a new owner.
    owners = [f for f in facts if " is owned by " in f]
    for f in r.sample(owners, 10):
        svc, old = f.split(" is owned by ")
        new = r.choice([p for p in PEOPLE if p != old])
        targets.append((f, f"{svc} is now owned by {new}; {old} moved to the platform team"))
    return targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memories", type=int, default=20000)
    ap.add_argument("--events", type=int, default=200)
    ap.add_argument("--ingest", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip", default="", help="comma list of paths to skip: A,B,C")
    a = ap.parse_args()
    skip = set(a.skip.split(",")) if a.skip else set()
    r = random.Random(a.seed)

    mem = Invalidate(":memory:", policy=Policy(max_workers=8))
    t0 = time.perf_counter()
    targets = build(mem, a.memories, a.seed)
    print(f"pool: {a.memories:,} memories built in {time.perf_counter() - t0:.1f}s; {len(targets)} targets")
    by_fact = {m.fact: m for m in mem.list()}

    rows = []
    if "A" not in skip:
        fact, ev = targets[0]
        rep = mem.observe(ev, source="slack")
        st = mem.get(by_fact[fact].id).status
        print(f"\nA eager observe, 1 event x {a.memories:,}: {rep.summary()}  screened_out={rep.screened_out}  target->{st.value}")
        rows.append(("A eager observe, 1 event", a.memories, 1, rep.judged + rep.screened_out, rep.requests, rep.latency_ms, rep.cost_usd))
        others = [v for v in rep.changed if v.memory_id != by_fact[fact].id]
        print(f"   event: {ev}")
        for v in others:
            print(f"   also {v.to_status.value:12s} {mem.get(v.memory_id).fact}  (still true {v.votes.still_true:.2f})")
        targets = targets[1:]

    if "B" not in skip:
        mem.policy.lazy = True
        events = [t[1] for t in targets[:9]]
        while len(events) < a.events:
            events.append(r.choice(CHATTER) + f" ({len(events)})")
        r.shuffle(events)
        t0 = time.perf_counter()
        mem.observe_many(events, source="slack")
        append_ms = (time.perf_counter() - t0) * 1000
        print(f"\nB lazy: {a.events} events appended in {append_ms:.0f} ms, 0 requests; pending={mem.pending():,}")
        # The host's vector search hands back its top-k; Governor.filter() validates exactly those. Simulate:
        # the 9 targets + 1 unrelated fact, validated directly.
        unrelated = by_fact[next(f for f in by_fact if "deploys at" in f)]
        cands = [by_fact[f] for f, _ in targets[:9]] + [unrelated]
        v = mem.validate(cands)
        print(f"   validate host top-{len(cands)}: {v.summary()}")
        hit = sum(1 for f, _ in targets[:9] if mem.get(by_fact[f].id).status is not Status.ACTIVE)
        print(f"   targets retired: {hit}/9   unrelated survived: {mem.get(unrelated.id).status.value}")
        # Second read of the same candidates: cursors are current, nothing to do.
        v2 = mem.validate(cands)
        print(f"   second read of the same candidates: {v2.requests} requests, {v2.latency_ms:.0f} ms")
        rows.append(("B lazy recall, top-10 x 200 events", len(cands), a.events, v.pairs, v.requests, v.latency_ms, v.cost_usd))
        mem.policy.lazy = False

    if "C" not in skip:
        # Fresh cursors: mark everything current, then ingest a batch eagerly.
        mem.store.set_checked_seq([m.id for m in mem.list()], mem.store.max_seq(mem.namespace))
        batch = [r.choice(CHATTER) + f" [{i}]" for i in range(a.ingest)]
        t0 = time.perf_counter()
        evs, v = mem.observe_many(batch, source="slack")
        print(f"\nC batch ingest, {a.ingest} events x {a.memories:,}: {v.summary()}")
        rows.append(("C batch ingest, 10 events", a.memories, a.ingest, v.pairs, v.requests, v.latency_ms, v.cost_usd))

    print("\n| path | memories | events | pairs | requests | latency | cost |")
    print("|---|---|---|---|---|---|---|")
    for name, m, e, pairs, req, ms, usd in rows:
        print(f"| {name} | {m:,} | {e} | {pairs:,} | {req} | {ms / 1000:.1f} s | ${usd:.4f} |")


if __name__ == "__main__":
    main()
