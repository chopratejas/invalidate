"""Two live checks for the two-stage observe.

1. Screen quality: over the 157 eval pairs, does the cheap bears-only screen ever drop a pair that the
   full `bears` vote says is bearing (>= bears_min)?
2. Scale: 500 memories, one event. Requests / tokens / wall time with screening vs. without.
"""
from __future__ import annotations

import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evals"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from invalidate import Invalidate, Policy, load_dotenv  # noqa: E402
from invalidate.judge import JevJudge  # noqa: E402
from invalidate.types import Event, Memory  # noqa: E402

load_dotenv()
from cases import CASES  # noqa: E402

judge = JevJudge()
policy = Policy()

# ---- 1. screen quality -------------------------------------------------------------------
print("1. screen vs full bears on the eval pairs")
by_event: dict[tuple[str, str], list[dict]] = {}
for c in CASES:
    by_event.setdefault((c["event"], c["event_source"]), []).append(c)
dropped_bearing = 0
kept_unrelated = 0
total = 0
bearing = 0
t0 = time.perf_counter()
tok = 0
for (text, src), cs in by_event.items():
    ev = Event(text=text, source=src)
    ms = [Memory(fact=c["memory"], kind=c["kind"]) for c in cs]
    sc = judge.screen(ev, ms)
    full = judge.observe(ev, ms)
    tok += sc.usage.input_tokens
    for s, v in zip(sc.relevance, full.votes):
        total += 1
        if v.bears >= policy.bears_min:
            bearing += 1
            if s < policy.screen_min:
                dropped_bearing += 1
        elif s >= policy.screen_min:
            kept_unrelated += 1
print(f"   pairs={total} bearing(full>={policy.bears_min})={bearing} "
      f"screen<{policy.screen_min} on a bearing pair: {dropped_bearing}   "
      f"screen kept non-bearing pairs: {kept_unrelated}   screen tokens/pair≈{tok/total:.0f}")

# ---- 2. scale -----------------------------------------------------------------------------
print("2. 500 memories, one event")
random.seed(7)
subjects = ["the marketing site", "lunch", "the on-call rotation", "the Jest config", "the office wifi",
            "the Redis cache", "the CI runner", "expense reports", "the standup time", "the logo"]
facts = [f"{random.choice(subjects)} note #{i}: {random.choice(['is unchanged', 'was reviewed', 'is documented in the wiki', 'is owned by the platform team'])}" for i in range(490)]
facts += [f"service {i} reads from the Postgres replica" for i in range(10)]
event = "we migrated everything off Postgres to SQLite last Tuesday"

for label, pol in (("no screen", Policy(screen_above=10**9)), ("screen", Policy(screen_above=200))):
    mem = Invalidate(":memory:", judge=judge, policy=pol)
    for f in facts:
        mem.remember(f, kind="fact")
    t0 = time.perf_counter()
    r = mem.observe(event, source="slack")
    wall = time.perf_counter() - t0
    flipped = [mem.get(v.memory_id).fact for v in r.changed]
    print(f"   {label:9s}: {r.requests} requests, {r.input_tokens} tokens, ${r.cost_usd:.4f}, {wall:.1f}s wall, "
          f"judged={r.judged} screened_out={r.screened_out} flipped={len(flipped)} "
          f"(all Postgres facts: {all('Postgres' in f for f in flipped) and len(flipped)==10})")
