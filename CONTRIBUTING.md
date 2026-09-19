# Contributing

## Adding an adapter

1. Read `src/invalidate/adapters/base.py`. `InMemoryAdapter` is the template; the `Adapter`
   protocol is three methods (`pull`, `flag`, `delete`) plus optional `insert`.
2. Install the host SDK into `.venv` only to read its source. Mirror the exact signatures and
   return shapes, and cite the version and file:line in the module docstring. Import the SDK
   lazily or not at all: `import invalidate.adapters` must stay dependency-free.
3. `pull()` is read-only and returns verbatim text with the host's own ids. If the host has no
   stable ids, content-address them (`sha1(normalized text)[:10]`) as the Markdown adapter does.
4. `flag()` writes `reason.as_metadata()` if the host has metadata, or `reason.line()` if it only
   holds text (Letta writes a sidecar note; Markdown writes an HTML comment). `Status.ACTIVE`
   means undo. Never rewrite the stored text.
5. `delete()` may be a soft delete if the host keeps history (Graphiti sets `invalid_at`).
6. `insert()` stores the event text verbatim (Mem0: `add(..., infer=False)`). If the host can only
   store through its own extraction, return the id anyway and document that the stored form is not
   verbatim. Make sure a subsequent `pull()` returns the successor, or the next `sync()` will
   forget it.
7. Tests: `tests/test_adapters_<host>.py` against a faithful fake of the client. Cover sync,
   flag, delete, insert, filter, error paths, and every documented limit. Run the shared
   `FakeJudge` from `tests/conftest.py`. If the host runs in-process (Chroma, LangGraph), test
   against the real thing.
8. A live script under `scripts/live_<host>.py` and a page `docs/adapters/<host>.md`: install,
   15-line usage, what flag/delete/insert do in that host, real limits.
9. Add a row to the table in `ADAPTERS.md` and an extra in `pyproject.toml`.

## Adding eval cases

`evals/cases.py`. Every case needs a category, a memory, an event, an expected label, and a
`lenient` list when more than one label is defensible. Include hard negatives whose surface form
matches the category but whose label is the opposite. Never copy a case string into the question
criteria in `src/invalidate/questions.py`; paraphrase with different entities.

## Changing the questions or the policy

Run `python evals/run_eval.py --out evals/results/<name>.json` before and after, and paste both
summaries in the PR. A change that raises strict accuracy but adds a false invalidation is a
regression.

## Growth

The layer wins by being in front of every store. The order that matters: the stores with the
most agents (Mem0, LangGraph, plain vector DBs), then the ones with the strongest opinions
(Letta, Graphiti), then event sources that no memory product ingests (Slack, GitHub, Linear
webhooks), then the surfaces where developers already keep memory in files (Claude Code,
Cursor rules, AGENTS.md). Each adapter should ship with a live-verified script and a one-line
demo that runs on the contributor's own data.
