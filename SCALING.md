# Scaling

How invalidate stays exhaustive without doing memories × events work on every event.

All numbers below were measured against Jev (jev-1.13.0) on 2026-09-19 with `scripts/scale_lazy.py`
and `evals/screen_matrix.py`. Rerun them; they cost about a dollar.

## The shape of the problem

One event against one memory costs about 70 tokens to screen and about 1,400 to judge in full. The
product of memories and events is what grows. At 20,000 memories:

| path | memories | events | pairs | Jev requests | latency | cost |
|---|---|---|---|---|---|---|
| eager `observe()`, one event | 20,000 | 1 | 20,000 | 211 | 8.9 s | $0.110 |
| batch `observe_many()`, ten events | 20,000 | 10 | 199,870 | 801 | 94.5 s | $0.586 |
| lazy read: host top-10, 200 pending events | 10 | 200 | 2,000 | 42 | 1.5 s | $0.009 |
| the same read again | 10 | 0 | 0 | 0 | 0 ms | $0 |

Eager checking of a 20,000-memory pool is $0.11 per event and 9 seconds. At 500 events a day that is
$55 a day and a permanently busy key. Beyond that it stops being sensible, and the answer is not to
check fewer memories per event by similarity. The answer is to change when the check runs.

## Four mechanisms, in the order they matter

### 1. Lazy mode: check what you read, against what happened since you last read it

```python
mem = Invalidate("memories.db", lazy=True)
mem.observe("we moved to SQLite last Tuesday", source="slack")   # appends to the log; 0 requests
mem.recall("which database?")                                     # judges the top candidates first
```

Every event gets a sequence number. Every memory carries a cursor: the last event it has been judged
against. `observe()` in lazy mode appends the event and returns in under a millisecond. When a memory is
about to be read, `validate()` judges it against exactly the events between its cursor and the head of
the log, applies them in order, and moves the cursor. The next read of the same memory costs nothing
until a new event arrives.

Cost is now proportional to what is read, not to what is stored. A memory nobody asks about is never
checked, and it is never served either, so nothing stale reaches a prompt. `Governor.filter()` does the
same for the results your host's vector search returns: the host retrieves by similarity, which it is
good at, and invalidate judges the retrieved set against the events since, which similarity cannot do.

Reads that touch a cold memory pay for its backlog once. `validate(budget_requests=N)` from a background
job drains backlogs during idle time, most stale first. `pending()` tells you how far behind the pool is.

A newly remembered fact is born current: events already in the log are older than it and are not
evidence against it. A successor stored by a superseding event is born current too.

### 2. Pair screening: many events × many memories per request

State holds a list of events and a list of memories. One short question per pair, all answered in
parallel inside one Jev request. Memory text is paid for once per event batch instead of once per event.

Measured over the full 157 × 157 matrix of the labelled eval (24,649 pairs, 136 labelled bearing pairs):

| form | shape per request | bearing pairs kept @0.2 | tokens per pair | latency p50 |
|---|---|---|---|---|
| single event, one question per memory (the eager screen) | 1 × 100 | 133 / 136 | 115 | 197 ms |
| pairs | 5 × 100 | 129 / 136 | 72 | 378 ms |
| pairs | 20 × 25 | 134 / 136 | 68 | 442 ms |
| **pairs (default)** | **10 × 25** | **134 / 136** | **70** | **263 ms** |
| pairs | 40 × 12 | worse | 69 | 426 ms |

Many memories in one state hurts more than many events. Defaults: `pair_events=10`, `pair_memories=25`,
`pair_min=0.2` (0.15 kept 136 of 136 on this set at 26% of unrelated pairs passing instead of 19%).
The 64k-token request cap allows about 1,000 pair questions; 250 is the latency sweet spot.

`validate()` and `observe_many()` use the pair screen. The eager single-event `observe()` keeps the
one-question-per-memory screen at 100 memories per request because with one event there is nothing to
amortize and fewer requests matter more.

### 3. Batch ingest: `observe_many()`

For high event rates in eager mode, ingest in batches. Ten events against 20,000 memories cost $0.059
per event with the pair screen against $0.110 with one `observe()` per event, and a quarter of the
requests. Throughput is bounded by the Jev limits (1,200 requests and 15M tokens a minute): about
2,000 pairs a second per key, or 170M pairs a day.

### 4. Scope in code, and leases

Memories belong to a user, a project, a team. An event from user A's chat has no business with user B's
memories, and `namespace` keeps them apart. A 100,000-memory pool is usually a few hundred memories per
scope, back in the cheap regime for eager mode. Memories also have hard TTLs; expired ones are not checked.
`observe(candidates=...)` and `recall(candidates=...)` accept any subset if you have a better index.

## A kill takes two votes

Scale testing found the one place Jev's known distractor weakness shows up: a batch of twenty
near-identical memories (twenty facts about the same service and the same person). In that context a
vote that is 0.90 for "still true" when the memory is judged alone came back 0.03. Two facts were wrongly
superseded out of 143 judged.

So a contradicted or superseded verdict is not written until the memory has been re-judged alone in
the state. If the clean vote disagrees, the memory goes to `needs_review` instead. Kills are rare, so this
adds a request only when something is about to die. With it on, the same 20,000-memory run produced one
true kill and four reviews, and no false invalidations. `Policy.second_opinion` turns it off.

## What "exhaustive" means after all this

Every memory that reaches a prompt has been judged against every event that arrived after it was
stored, in order. That is the guarantee. Whether that judging happened at event time (eager), at read
time (lazy), or in a background drain is a cost decision, and all three are the same code path with the
same questions and the same policy.

## Not done yet

- **A measured recall number for an embedding prefilter.** If your host returns the top 500 by
  similarity and invalidate screens those, what fraction of bearing pairs does the prefilter drop? The
  matrix harness can answer it; it has not been run with an embedding model yet.
- **Hierarchical screening.** Group memories by kind or topic, ask once per group whether the event bears
  on it, expand only groups that pass. Same questions, staged.
- **Async client.** The thread pool is capped at 8; an async client would let the rate limit be the only
  ceiling.
