# Why Mem0, Letta and Zep do not already do this

Checked against their current docs and source on 2026-09-18. Links are to the
things quoted.

## What each one actually does when a fact changes

**Mem0** (`docs.mem0.ai/core-concepts/memory-operations/add`, `mem0/memory/main.py`).
The add pipeline is now additive: one LLM call extracts facts from messages, a
vector search fetches the top 10 similar existing memories, and the extracted
facts are inserted. The docs state it plainly: "New memories are added without
overwriting or deleting existing memories." Staleness is handled by ranking at
retrieval time. Nothing re-judges an old memory when a new one arrives, and an
event that is not itself a memory (a PR description, a schema migration, an
outage notice) is never compared to anything.

**Zep / Graphiti** (`graphiti_core/utils/maintenance/edge_operations.py`,
`graphiti_core/prompts/dedupe_edges.py`). The closest thing to invalidation
that exists. Each new episode is run through an LLM to extract facts as graph
edges. For every extracted edge, a hybrid search (default limit 10) retrieves
candidate existing edges, and a generative LLM is asked for `contradicted_facts`
among those candidates; the losers get `invalid_at` set. Old facts are kept, not
deleted, which is the right instinct. But the check is scoped to what similarity
search returned for the *new* fact, it only runs for information that the
extractor first turned into a fact, and it is a generative LLM call per new edge.

**Letta** (`docs.letta.com/guides/agents/memory`). Memory blocks are "editable
by agents via memory tools (and directly by the developer via the API)".
Archival memory is retrieval. There is no mechanism that re-checks stored
memory against new information; the agent has to notice, in context, that
something it remembers is now wrong and choose to edit it. That is the
"every loop introduces another opportunity to go off the rails" architecture
the TypeSafe docs contrast with software.

## The five gaps, and why they are structural rather than accidental

1. **Retrieval-scoped, not exhaustive.** Both Mem0 and Graphiti compare the new
   thing against the top-10 by similarity. "Our database of choice changed; we
   are all-in on SQLite" does not embed near "user prefers Postgres". The miss is
   silent, which is the worst kind. They cannot widen the scope because each
   comparison is a generative LLM call at cents and seconds; top-10 is a budget,
   not a design choice. Jev at $0.042 per million tokens and ~150 ms makes
   "judge everything" the cheap default (invalidate: 500 memories vs one event,
   8 requests, $0.006, 0.8 s).

2. **Only memories judge memories.** Their invalidation runs on facts an LLM
   extracted from conversation. A Slack message, a merged migration, a schema
   diff, a support email are not conversation turns and never enter the check.
   invalidate's `observe()` takes any text as an event and judges it against
   the store; the eval set has schema diffs and 200-word PR descriptions in it.

3. **They rewrite; invalidate does not.** Extraction is generation: the memory
   that gets stored is the LLM's paraphrase, and an update is another paraphrase.
   Every rewrite is a chance to drift. invalidate stores the fact verbatim and
   only ever changes its *status*; a superseding event is stored verbatim as the
   successor. This is not a style preference; it is what makes the verdict log
   auditable and the store safe to put in front of someone else's data.

4. **The decision is inside the model.** A generative "is this contradicted?"
   returns a yes. There is no probability to threshold, no way to route the
   0.45 case to a human, no way to say "a question never writes" or "a customer
   email may flag but not flip". invalidate returns six calibrated votes per
   pair and the policy that turns them into writes is a dataclass of floats you
   own and can sweep against labeled cases (`evals/`).

5. **Coupled to their store.** Mem0's logic lives in Mem0's store; Graphiti's in
   the graph. Neither can govern a memory that lives in a Redis cache, a RAG
   chunk index, or a competitor's product. invalidate is a layer: `Store` and
   `Judge` are protocols, and `on_transition` mirrors status changes into
   whatever you actually keep memories in.

## Why they will not just add it next quarter

They could call Jev tomorrow; nothing here is secret. But each of them would
have to change something load-bearing:

- Mem0 just moved *away* from update-on-add to additive storage, presumably
  because LLM-judged updates were slow, expensive and flaky at scale. Bringing
  invalidation back means reversing a product decision, and doing it with a
  generative model reintroduces the cost that drove them off it.
- Graphiti's invalidation is entangled with extraction and dedupe in one prompt
  over one search result set. Making it exhaustive and event-driven is a
  rewrite of the ingestion path, not a flag.
- Letta's philosophy is agent agency over memory. A governor that flips memory
  underneath the agent contradicts the design, and their users chose Letta for
  that design.
- All three sell a store. The value of a standalone governor is that it works in
  front of *any* store, including each other's. That is a feature for a layer and
  a strategic problem for a platform.

The durable moat for invalidate is not the API call. It is the question design
(each vote is a literal, atomic, threshold-able judgment tuned against Jev's
documented jaggedness), the labeled eval set with hard negatives, the policy
that refuses to guess (form checks before content checks, margins, review-only
sources), and being store-neutral by construction.
