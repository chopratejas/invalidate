# LongMemEval, with and without invalidate

The question this answers: given a memory system a team already has, does adding invalidate in front
of it change what the agent gets right? The host, the retrieval, the answer model and the grader are
held constant. The only difference between the two arms is the layer.

## Setup

- **Benchmark.** [LongMemEval](https://github.com/xiaowu0162/LongMemEval), 500 questions over
  timestamped multi-session chat histories. Question types: knowledge-update (78), temporal-reasoning
  (133), multi-session (133), single-session-user (70), single-session-assistant (56),
  single-session-preference (30). The `oracle` split gives only the evidence sessions; the `s` split
  gives about 50 sessions and 244 user turns per question.
- **Host.** Mem0/Zep-style: an LLM (Claude Haiku 4.5) extracts atomic facts from each user turn.
  Each fact is one memory with its session date. Retrieval is top-k by cosine similarity
  (text-embedding-3-small). Extractions are cached on disk and shared by both arms.
- **Baseline arm.** Retrieve the top-k facts, answer.
- **invalidate arm.** The same host with the layer in lazy mode. Every extracted fact is also an event.
  At question time the host's top-2k candidates are validated against the events they have not seen,
  facts judged superseded, contradicted or uncertain are hidden, and the first k live facts are
  answered from. The host is never modified (ledger mode).
- **Answer model.** Claude Sonnet 5, the same prompt in both arms: memories with dates, the question
  with its date, "say you do not know if the memories do not contain it".
- **Grader.** gpt-4o-2024-08-06 with the benchmark's own `get_anscheck_prompt` templates, verbatim.
  Note the knowledge-update template is lenient: a response that mentions the old value alongside the
  updated one is still correct.

```
python evals/longmemeval/run.py --split oracle --types knowledge-update --arms base,inv
python evals/longmemeval/run.py --split oracle --arms base,inv
python evals/longmemeval/run.py --split s --types knowledge-update --arms base,inv
```

Results are cached per question under `results/` and the report is recomputed from the cache.

## What to look at

- **Knowledge-update accuracy**, the category the layer is for.
- **Every other category**, which must not drop. A drop there means a still-true fact was hidden,
  which is the false-kill rate showing up as lost answers.
- **Hidden evidence.** The harness knows which turns carry the answer (`has_answer`). For
  knowledge-update questions the old value is evidence too, so hiding it is the point. For every other
  type, a hidden evidence fact is a false kill.
- **Cost.** Jev tokens and requests per question.

## Why whole chat turns are the wrong unit

The first version of this harness stored user turns verbatim. Almost nothing was ever invalidated,
and the votes showed why: nearly every LongMemEval user turn is a request for advice, so the
event-level "is this a question or a plan" gate fired and, by design, a question never writes.
That gate is right for a chat turn and wrong as the unit of memory. Memory products extract facts
from turns for the same reason, so the facts host is the fair comparison. The turn-level host is still
available with `--host turns`.
