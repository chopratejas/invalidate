# invalidate evals

A labeled dataset and a runner for measuring how well `Policy.dispose(votes)` turns
Jev's five probabilities (`bears`, `still_true`, `replaces`, `hypothetical`, `directive`) into the
right disposition for one (event, memory) pair.

Files:

- `cases.py` — 157 labeled cases across 16 categories (`python evals/cases.py` prints the counts).
- `run_eval.py` — the CLI runner: judges, reports, sweeps thresholds, saves raw votes.
- `results/` — saved runs (`<timestamp>.json`). Rerun a report or sweep offline with `--from`.

## Running

```sh
. .venv/bin/activate
export TYPESAFE_API_KEY=...          # or put TYPESAFE_API_KEY=... in ./.env

python evals/run_eval.py                          # batched (default), all cases
python evals/run_eval.py --per-case               # one memory per request
python evals/run_eval.py --category transient --category hypothetical
python evals/run_eval.py --ids tr_04 tr_05        # specific cases
python evals/run_eval.py --limit 20
python evals/run_eval.py --policy '{"bears_min": 0.6, "contradict_max": 0.25}'
python evals/run_eval.py --workers 4 --batch-size 10
python evals/run_eval.py --out evals/results/baseline.json
python evals/run_eval.py --from evals/results/baseline.json --policy '{"confirm_min": 0.8}'
python evals/run_eval.py --dry-run                # no API key: keyword FakeJudge
```

Batching modes:

- `--batched` (default) groups cases that share the same event text and source, and sends
  up to `--batch-size` (20) memories per request, exactly as `Invalidate.observe()` would.
  It calls `JevJudge.observe(Event, [Memory, ...])` directly, so the votes are the same ones
  the engine would see.
- `--per-case` sends one memory per request. Compare the two to see whether packing
  memories into one state (Jev's "large state full of irrelevant detail" edge) moves the votes.

`--dry-run` swaps in a deterministic keyword judge (`FakeJudge` in `run_eval.py`). Its
accuracy is meaningless; it exists so the whole pipeline, report and sweep can be exercised
without a key.

`--from FILE` skips the API entirely and re-reports from saved votes. Combine with
`--policy` to try a policy, or `--category`/`--ids`/`--limit` to zoom in.

## What the report shows

- **strict accuracy**: predicted disposition equals `expected`.
- **lenient accuracy**: predicted disposition is `expected` or one of the case's `lenient`
  alternatives. Lenient labels exist where two careful humans could disagree (transient
  vs uncertain, partial change vs superseded, and so on).
- **false invalidations**: predicted `contradicted` or `superseded` when neither `expected`
  nor any lenient alternative allows it. This is the failure that matters most: a memory
  that was still true gets flipped off. Watch this number before accuracy.
- per-category table, confusion matrix (rows expected, columns predicted), and every
  strict failure with its probabilities and the acceptable labels.
- usage: request count, input tokens, cost at $0.042 per million input tokens, mean/p95
  latency per request, and the versioned model id the API returned (`jev-1.13.x`).

## Categories

Every category contains hard negatives: cases whose surface form matches the category
but whose correct label is the opposite of what the category name suggests. The
`note` field on each case says why.

| category | what it tests | typical expected |
| --- | --- | --- |
| `direct_supersede` | event states the new value for the same thing | superseded |
| `contradict_no_replacement` | fact no longer holds, no new value given | contradicted |
| `confirm` | event restates or relies on the fact | confirmed |
| `unrelated` | different subject, including same-word traps (Postgres the cat, slack in the schedule, Jenkins the person) | unrelated |
| `transient` | outage, PTO, one-off skip, temporary exception; must NOT invalidate | confirmed (lenient: uncertain) |
| `hypothetical` | question, proposal, wish, undecided plan | uncertain (lenient: unrelated, confirmed) |
| `partial` | event changes one part of a compound fact | uncertain (lenient: contradicted, superseded) |
| `negation_and_correction` | "correction: I meant X", retractions, double negation | varies |
| `reversal` | "we moved back to X" after the memory says Y | superseded |
| `schema_or_code_change` | migration or diff snippet as the event vs a schema/config note | varies |
| `preference_vs_fact` | a default or policy changed; the user's preference did not | confirmed (lenient: unrelated) |
| `adversarial` | injected instructions, text arguing for its own classification | unrelated/confirmed, never contradicted |
| `temporal_phrasing` | "as of Q3", "since the reorg", "until last month": time words must not confuse the semantics | by meaning |
| `multi_entity` | event mentions two things, only one matches the memory | by which one matched |
| `paraphrase_confirm` | same fact in different words, with near-miss negatives (different number/timezone) | confirmed |
| `long_event` | 100–200 word Slack post / PR description with the relevant sentence buried | varies |

Kinds (`fact`, `preference`, `decision`, `config`, `relationship`, `schedule`) are passed
to Jev as `memories[i].kind`, the same as in production.

## Reading the sweep

After the report, the runner grids the five policy thresholds over the cached votes
(3600 policies, no API calls):

| threshold | values |
| --- | --- |
| `bears_min` | 0.3, 0.4, 0.5, 0.6, 0.7 |
| `confirm_min` | 0.6, 0.65, 0.7, 0.75, 0.8, 0.85 |
| `contradict_max` | 0.15, 0.2, 0.25, 0.3, 0.35, 0.4 |
| `replace_min` | 0.4, 0.5, 0.6, 0.7 |
| `hypothetical_max` | 0.5, 0.6, 0.7, 0.8, 0.9 |

It prints the current policy's row, the top 5 by strict accuracy, and the policy with the
fewest false invalidations (ties broken by accuracy). How to read it:

- If the top-5 rows differ from the current policy by only one threshold and a few cases,
  the defaults are fine; do not chase the grid.
- If lowering `contradict_max` or raising `bears_min` removes false invalidations at a
  small accuracy cost, take that trade. A memory wrongly kept alive is a stale answer; a
  memory wrongly killed is a lost fact.
- `hypothetical_max` only matters for cases where `still_true` is already low. If the
  sweep is flat along that axis, Jev is answering `still_true` correctly for questions
  and plans and the guard is not doing work.
- Sweep results on 157 cases are noisy at the single-case level. Treat a threshold as
  meaningfully better only when it wins by several cases, and rerun `--per-case` to check
  the win survives a different batching.

Everything the sweep needs is in the saved JSON, so you can re-sweep with a different grid
by editing `SWEEP_GRID` in `run_eval.py` and running `--from`.

## Adding cases

Append a `_c(...)` call in `cases.py`. Keep both strings under 60 words (long events
excepted), make it answerable from the two strings alone, and if reasonable humans could
disagree, list the acceptable alternatives in `lenient`. `python evals/cases.py` validates
the file.

## Scoring note

The policy has two "form" dispositions that never write, `hypothetical` and `directive`. The
label vocabulary predates them, so the runner scores both as `uncertain` (the closest "do not
flip" label). The sweep does not vary `directive_max`; it is a safety knob, not an accuracy knob.

## Results so far

| run | strict | lenient | false invalidations | notes |
|---|---|---|---|---|
| `results/baseline.json` | 75.2% | 93.0% | 3 | original 3+1 questions, thresholds 0.5/0.7/0.3 |
| `results/v2.json` | 82.8% | 94.9% | 1 | + `directive` vote, form checks before confirm, better criteria examples |
| `results/v2.json` with shipped defaults | 86.6% | 96.2% | 1 | thresholds 0.6/0.6/0.4 chosen by the sweep (tuned on this set) |
| `results/v3.json` | 77.1% | 85.4% | 0 | + `partial` vote (first phrasing) and `margin`: partial fired on plain supersedes |
| `results/v4.json` (shipped) | 89.2% | 97.5% | 0 | `partial` reframed around the central claim, `partial_min` 0.85, `margin` 0.05 |
