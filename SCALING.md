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
| lazy read: host top-10, 200 pending events | 10 | 200 | 2,000 | 40 | 2.2 s | $0.007 |
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

| form | shape per request | bearing pairs kept | unrelated pairs passing | tokens per pair | latency p50 |
|---|---|---|---|---|---|
| single event, one question per memory (the eager screen) | 1 × 100 | 133 / 136 @0.2 | 63% | 115 | 197 ms |
| pairs, restated criteria | 5 × 100 | 129 / 136 @0.2 | 59% | 72 | 378 ms |
| pairs, restated criteria | 20 × 25 | 134 / 136 @0.2 | 24% | 68 | 442 ms |
| pairs, restated criteria | 10 × 25 | 134 / 136 @0.2 | 19% | 70 | 263 ms |
| **pairs, minimal question (default)** | **10 × 25** | **134 / 136 @0.15** | **15%** | **52** | **248 ms** |
| pairs, minimal question, fact-only state | 10 × 25 | 130 / 136 @0.15 | 23% | 51 | 232 ms |

Many memories in one state hurts more than many events. The minimal question ("Is `events[3].text` about
`memories[7].fact`?", criteria "same subject" / "different subject") beats the longer wording on every axis;
Jev already knows what "about" means and restating it only costs tokens. Dropping `kind` and `source` from
the state saves one token per pair and loses four bearing pairs, so they stay. Defaults: `pair_events=10`,
`pair_memories=25`, `pair_min=0.15`. The 64k-token request cap allows about 1,000 pair questions; 250 is
the latency sweet spot. The unrelated-pass column is inflated here because the eval reuses subjects on
purpose; in a real pool most pairs share nothing.

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

## Where the tokens go, and what was cut

A lazy read of 10 memories against 200 pending events is 2,000 screened pairs and about 15 full
judgments. The screen is 75% of the tokens. Inside the screen, the question text repeated per pair is
about 80% of the tokens, because state is read once per request and questions are billed each.

| change | measured effect |
|---|---|
| minimal pair question | 70 to 52 tokens per pair, same recall, fewer unrelated pairs pass |
| staged full judgment: `replaces` and `partial` asked only when `still_true` ≤ contradict_max − margin, the only place the policy reads them | the same votes (answers are independent inside a request); the lazy read fell from $0.0094 to $0.0069; the labelled eval from 399k to 355k tokens with accuracy unchanged (89.2% strict, 0 false invalidations); +90 ms mean latency when a second stage runs |
| second opinion only on kills | one extra request per kill, none otherwise |

What is left on the table, in order of size:

1. **Fewer pairs, not cheaper pairs.** Everything above is per pair. The floor per pair question is about
   35 tokens of JSON structure, so the screen cannot get more than another third cheaper. Reducing the
   number of pairs is the only large lever left, which is the prefilter question below.
2. **Criteria in state.** The four full-judgment questions carry long criteria and examples repeated
   per memory. Moving them into `state` once and asking short questions that refer to them would cut
   the full stage several-fold, but it is indirection, which the Jev docs say to avoid. Untested;
   the eval would say in one run.
3. **Deduplicate.** Identical memory text across users, or the same event ingested twice, is judged
   once per (event text, memory text). Not implemented; matters only where duplicates are common.
4. **Event-level questions once per event.** `hypothetical` and `directive` are asked once per request,
   so a 20-batch event asks them 8 times. Two questions each; small.

## Can a similarity top-k do the prefiltering?

Only as a wide net, and only with its recall measured. On the labelled set, for each event, rank all
157 memories by cosine similarity (text-embedding-3-small) and find where the memory it bears on lands:

| net | whole event | best sentence of the event |
|---|---|---|
| top 8 (5% of the pool) | 73% found | 74% |
| top 16 (10%) | 92% | 93% |
| top 32 (20%) | 98.5% | **100%** |
| top 64 (40%) | 100% | 100% |

So a Zep-style top 10 with the LLM as judge misses about a quarter of the invalidations on this set,
silently. A net of 20% of the pool, with long events split into sentences and the union taken, finds
every one, and Jev then judges only that fifth. That is a 5× cut in pairs for the eager and batch
paths at an embedding cost that rounds to zero, and it composes with everything above. The misses in
narrow nets are exactly the interesting cases: a long retro post that mentions the migration in one line,
"we're no longer on Postgres", "Dana moved teams" against a fact that names Dana and a different service.

Caveats that keep this off by default: the eval pool is 157 memories dense with shared subjects, and
the fraction may not transfer to a 100,000-memory pool; the number to publish is recall on your own
events, and `evals/embedding_recall.py` measures it. When it ships, it will keep a random 1% of events
exhaustive so the miss rate of the net stays measured in production rather than assumed. Nothing changes
about who judges: similarity chooses what to look at, Jev decides what is true.

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

- **The wide-net prefilter as a shipped option.** Measured above on the labelled set; not yet a
  `Policy` switch with the 1% exhaustive audit sample.
- **Hierarchical screening.** Group memories by kind or topic, ask once per group whether the event bears
  on it, expand only groups that pass. Same questions, staged.
- **Async client.** The thread pool is capped at 8; an async client would let the rate limit be the only
  ceiling.
