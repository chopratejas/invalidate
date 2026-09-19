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

## Results so far (2026-09-19)

Same host, same answer model, same grader in both arms. `base` is the host alone; `inv` is the host with
the layer in front, serving in annotate mode unless stated.

**Oracle split, knowledge-update, 78 questions, Sonnet 5 answering**

| serving mode of the layer | inv | base |
|---|---|---|
| hide dead and reviewed facts (the first, wrong default) | 78.2% | 97.4% |
| hide dead only | 91.0% | 97.4% |
| annotate: keep every fact, label retired ones with their replacement | 94.9% | 97.4% |
| annotate, Haiku 4.5 answering | 93.6% | 93.6% |

The two remaining losses in annotate mode: one response contained the right answer in parentheses and the
grader said no; one uncertain fact was unlabelled and the model hedged. The oracle split gives only the
evidence sessions with dates, so a strong answer model resolves updates itself and the layer has nothing to add.

**Oracle split, the other 422 questions (regression check), annotate mode**

| type | n | base | inv |
|---|---|---|---|
| temporal-reasoning | 133 | 84.2% | 82.0% |
| multi-session | 133 | 72.2% | 72.9% |
| single-session-user | 70 | 95.7% | 97.1% |
| single-session-assistant | 56 | 3.6% | 5.4% |
| single-session-preference | 30 | 80.0% | 76.7% |
| all | 422 | 71.3% | 71.1% |

8 gained, 9 lost: noise. (single-session-assistant is near zero in both arms because this host stores
user turns only.)

**S split, knowledge-update, 78 questions, ~244 user turns and ~1,000 facts per question**

| k served | base | inv | questions where only the old value was retrieved | base on those |
|---|---|---|---|---|
| 10 | 94.9% | 93.6% | few | |
| 1 | 47.4% | pending (TypeSafe credits ran out) | 30 of 78 | 4 of 30 |

At k=10 both old and new values usually rank into the top ten, because they are near-identical text, and the
answer model resolves them by date. At k=1 retrieval returns only the old value on 30 of 78 questions and the
baseline answers 4 of those. That slice is where the layer's annotation carries the replacement into the prompt;
the treatment number for it is the one still to run, with k = 3 and 5.

**What the numbers say so far.** As a layer the product cannot make the host worse when it serves in annotate
mode, and the regression set confirms that. It has not yet shown a gain on a public benchmark; the gain, if it
exists, lives where retrieval serves a stale fact without its update, which the k sweep isolates.

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
