#!/usr/bin/env python
"""Evaluation runner for invalidate.

Judges every case in evals/cases.py with Jev (or a fake judge in --dry-run), applies
Policy.dispose, and reports accuracy, a confusion matrix, failures, cost, latency,
and a threshold sweep over the cached probabilities.

    python evals/run_eval.py                       # batched, all cases
    python evals/run_eval.py --per-case            # one memory per request
    python evals/run_eval.py --dry-run             # no API key needed
    python evals/run_eval.py --from evals/results/x.json   # offline re-report + sweep

Standard library + the installed `invalidate` package only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from cases import CASES, CATEGORIES, EXPECTED_LABELS, acceptable, validate  # noqa: E402

from invalidate import Event, Memory, Policy, Votes  # noqa: E402
from invalidate.judge import JudgeResult, ObserveBatch  # noqa: E402

USD_PER_MILLION_INPUT_TOKENS = 0.042
LABELS = list(EXPECTED_LABELS)
INVALIDATING = {"contradicted", "superseded"}

# Threshold grid for the sweep. Pure code over cached votes; no API calls.
SWEEP_GRID = {
    "bears_min": [0.3, 0.4, 0.5, 0.6, 0.7],
    "confirm_min": [0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
    "contradict_max": [0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    "replace_min": [0.4, 0.5, 0.6, 0.7],
    "hypothetical_max": [0.5, 0.6, 0.7, 0.8, 0.9],
}
POLICY_FIELDS = {f.name for f in fields(Policy)}
THRESHOLD_FIELDS = ("bears_min", "confirm_min", "contradict_max", "replace_min", "hypothetical_max")


# ----------------------------------------------------------------------------- setup
def load_api_key() -> str | None:
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if key:
        return key
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "TYPESAFE_API_KEY":
                v = v.strip().strip("'\"")
                if v:
                    os.environ["TYPESAFE_API_KEY"] = v
                    return v
    return None


def build_policy(overrides_json: str | None) -> Policy:
    if not overrides_json:
        return Policy()
    try:
        overrides = json.loads(overrides_json)
    except json.JSONDecodeError as e:
        sys.exit(f"--policy is not valid JSON: {e}")
    if not isinstance(overrides, dict):
        sys.exit("--policy must be a JSON object, e.g. '{\"bears_min\": 0.6}'")
    unknown = set(overrides) - POLICY_FIELDS
    if unknown:
        sys.exit(f"--policy has unknown fields {sorted(unknown)}; valid: {sorted(POLICY_FIELDS)}")
    return Policy(**overrides)


def policy_thresholds(p: Policy) -> dict[str, float]:
    return {k: getattr(p, k) for k in THRESHOLD_FIELDS}


# ----------------------------------------------------------------------------- fake judge
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "for", "on", "in", "at", "to", "of", "and",
    "or", "with", "by", "from", "as", "it", "its", "this", "that", "these", "those", "we", "our", "user",
    "users", "has", "have", "had", "not", "no", "new", "per", "now", "run", "runs", "use", "uses",
    "prefers", "service", "services", "does", "do", "can", "will",
}


def _content_words(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9_]+", text.lower()))
    return {w for w in words if len(w) >= 3 and w not in _STOPWORDS}


class FakeJudge:
    """Deterministic keyword judge so the pipeline can be validated without an API key.

    bears        0.9 if any content word of the memory appears in the event, else 0.1
    still_true   0.1 if the event contains 'migrat' / 'moved' / 'no longer' / 'now', else 0.9
    replaces     1 - still_true
    hypothetical 0.9 if the event ends with '?', else 0.1
    """

    model = "fake-judge-0.0"

    def observe(self, event: Event, memories: list[Memory]) -> ObserveBatch:
        text = event.text.lower()
        event_words = _content_words(text)
        changed = any(k in text for k in ("migrat", "moved", "no longer", "now"))
        still_true = 0.1 if changed else 0.9
        hyp = 0.9 if event.text.rstrip().endswith("?") else 0.1
        votes = []
        for m in memories:
            bears = 0.9 if _content_words(m.fact) & event_words else 0.1
            votes.append(Votes(bears=bears, still_true=still_true, replaces=round(1 - still_true, 3), hypothetical=hyp))
        # ~4 chars per token is close enough for a cost estimate in dry runs.
        approx_tokens = (len(event.text) + sum(len(m.fact) for m in memories)) // 4 + 120 * len(memories) + 200
        time.sleep(0.002)
        return ObserveBatch(votes, JudgeResult(approx_tokens, self.model))

    def recall(self, query: str, memories: list[Memory]):  # pragma: no cover - not used by evals
        raise NotImplementedError


# ----------------------------------------------------------------------------- judging
def select_cases(args: argparse.Namespace) -> list[dict]:
    validate()
    cases = list(CASES)
    if args.category:
        wanted = set(args.category)
        unknown = wanted - set(CATEGORIES)
        if unknown:
            sys.exit(f"unknown category {sorted(unknown)}; valid: {list(CATEGORIES)}")
        cases = [c for c in cases if c["category"] in wanted]
    if args.limit is not None:
        cases = cases[: args.limit]
    if args.ids:
        wanted_ids = set(args.ids)
        cases = [c for c in cases if c["id"] in wanted_ids]
    return cases


def plan_requests(cases: list[dict], per_case: bool, batch_size: int) -> list[list[dict]]:
    """Group cases into request-sized lists. Batched: same event text+source share a request."""
    if per_case:
        return [[c] for c in cases]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for c in cases:
        key = (c["event"], c["event_source"])
        if key not in groups:
            order.append(key)
        groups[key].append(c)
    requests: list[list[dict]] = []
    for key in order:
        members = groups[key]
        for i in range(0, len(members), max(1, batch_size)):
            requests.append(members[i : i + batch_size])
    return requests


def judge_all(judge, requests: list[list[dict]], workers: int) -> tuple[list[dict], list[dict]]:
    """Returns (case_records, request_records). Errors are recorded, not raised."""

    def one(idx_and_group):
        idx, group = idx_and_group
        first = group[0]
        event = Event(text=first["event"], source=first["event_source"])
        memories = [Memory(fact=c["memory"], kind=c["kind"], source="eval") for c in group]
        t0 = time.perf_counter()
        try:
            res = judge.observe(event, memories)
            latency = (time.perf_counter() - t0) * 1000
            return idx, res, latency, None
        except Exception as e:  # noqa: BLE001 - report and continue
            latency = (time.perf_counter() - t0) * 1000
            return idx, None, latency, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"

    indexed = list(enumerate(requests))
    if workers <= 1 or len(indexed) <= 1:
        outcomes = [one(x) for x in indexed]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(indexed))) as pool:
            outcomes = list(pool.map(one, indexed))

    case_records: list[dict] = []
    request_records: list[dict] = []
    for idx, res, latency, err in outcomes:
        group = requests[idx]
        request_records.append({
            "index": idx,
            "n_memories": len(group),
            "case_ids": [c["id"] for c in group],
            "input_tokens": res.usage.input_tokens if res else 0,
            "model": res.usage.model if res else None,
            "latency_ms": latency,
            "error": err,
        })
        for j, c in enumerate(group):
            rec = {**c, "request_index": idx}
            if res is None:
                rec["error"] = err.splitlines()[0] if err else "unknown error"
                rec["votes"] = None
            else:
                rec["votes"] = asdict(res.votes[j])
            case_records.append(rec)
    return case_records, request_records


# ----------------------------------------------------------------------------- scoring
def predict(policy: Policy, rec: dict) -> str:
    if rec.get("votes") is None:
        return "error"
    return policy.dispose(Votes(**rec["votes"])).value


def score(policy: Policy, records: list[dict]) -> dict:
    n = len(records)
    strict = lenient = false_inv = 0
    per_cat: dict[str, Counter] = defaultdict(Counter)
    confusion: Counter = Counter()
    failures: list[dict] = []
    for rec in records:
        pred = predict(policy, rec)
        ok_strict = pred == rec["expected"]
        ok_lenient = pred in acceptable(rec)
        fi = pred in INVALIDATING and not (acceptable(rec) & INVALIDATING)
        strict += ok_strict
        lenient += ok_lenient
        false_inv += fi
        cat = per_cat[rec["category"]]
        cat["n"] += 1
        cat["strict"] += ok_strict
        cat["lenient"] += ok_lenient
        cat["false_inv"] += fi
        confusion[(rec["expected"], pred)] += 1
        if not ok_strict:
            failures.append({**rec, "predicted": pred, "ok_lenient": ok_lenient, "false_invalidation": fi})
    return {
        "n": n,
        "strict": strict,
        "lenient": lenient,
        "false_invalidations": false_inv,
        "strict_acc": strict / n if n else 0.0,
        "lenient_acc": lenient / n if n else 0.0,
        "per_category": per_cat,
        "confusion": confusion,
        "failures": failures,
    }


# ----------------------------------------------------------------------------- reporting
def pct(a: int, b: int) -> str:
    return f"{100 * a / b:5.1f}%" if b else "   n/a"


def print_report(policy: Policy, records: list[dict], requests: list[dict], meta: dict) -> dict:
    s = score(policy, records)
    print()
    print("=" * 78)
    print(f"invalidate eval  |  mode={meta.get('mode')}  judge={meta.get('judge')}  cases={s['n']}")
    print(f"policy thresholds: {json.dumps(policy_thresholds(policy))}")
    print("=" * 78)
    print(f"strict accuracy   : {s['strict']:3d}/{s['n']}  {pct(s['strict'], s['n'])}")
    print(f"lenient accuracy  : {s['lenient']:3d}/{s['n']}  {pct(s['lenient'], s['n'])}")
    print(f"false invalidations (predicted contradicted/superseded where no acceptable label allows it): "
          f"{s['false_invalidations']}")

    # per-category table
    print()
    print(f"{'category':28s} {'n':>3s} {'strict':>8s} {'lenient':>8s} {'false_inv':>9s}")
    print("-" * 60)
    for cat in CATEGORIES:
        c = s["per_category"].get(cat)
        if not c:
            continue
        print(f"{cat:28s} {c['n']:3d} {pct(c['strict'], c['n']):>8s} {pct(c['lenient'], c['n']):>8s} {c['false_inv']:9d}")

    # confusion matrix
    preds = LABELS + (["error"] if any((e, "error") in s["confusion"] for e in LABELS) else [])
    print()
    print("confusion matrix (rows = expected, cols = predicted)")
    head = f"{'':14s}" + "".join(f"{p[:11]:>12s}" for p in preds)
    print(head)
    for e in LABELS:
        row = f"{e:14s}" + "".join(f"{s['confusion'].get((e, p), 0):12d}" for p in preds)
        print(row)

    # failures
    print()
    print(f"failing cases (strict): {len(s['failures'])}")
    for f in s["failures"]:
        v = f.get("votes") or {}
        acc = sorted(acceptable(f))
        flag = "  [FALSE INVALIDATION]" if f["false_invalidation"] else ("  [ok lenient]" if f["ok_lenient"] else "")
        print(f"- {f['id']:6s} {f['category']:26s} expected={f['expected']:12s} predicted={f['predicted']:12s}{flag}")
        if v:
            print(f"    bears={v['bears']:.2f} still_true={v['still_true']:.2f} replaces={v['replaces']:.2f} "
                  f"hypothetical={v['hypothetical']:.2f}  acceptable={acc}")
        else:
            print(f"    error: {f.get('error')}")
        print(f"    memory: {f['memory'][:90]}")
        ev = f["event"].replace("\n", " ")
        print(f"    event : {ev[:90]}{'...' if len(ev) > 90 else ''}")

    # usage
    ok_reqs = [r for r in requests if not r.get("error")]
    errs = [r for r in requests if r.get("error")]
    tokens = sum(r["input_tokens"] for r in ok_reqs)
    lats = sorted(r["latency_ms"] for r in ok_reqs)
    models = Counter(r["model"] for r in ok_reqs if r["model"])
    print()
    print("usage")
    print(f"  requests        : {len(requests)} ({len(errs)} failed)")
    print(f"  input tokens    : {tokens}")
    print(f"  cost            : ${tokens * USD_PER_MILLION_INPUT_TOKENS / 1_000_000:.5f} "
          f"(at ${USD_PER_MILLION_INPUT_TOKENS}/M input tokens)")
    if lats:
        p95 = lats[min(len(lats) - 1, int(round(0.95 * (len(lats) - 1))))]
        print(f"  latency/request : mean {statistics.fmean(lats):.0f} ms, p95 {p95:.0f} ms, max {lats[-1]:.0f} ms")
    print(f"  model           : {', '.join(f'{m} ({n} req)' for m, n in models.items()) or 'n/a'}")
    if errs:
        print("  errors:")
        for r in errs:
            print(f"    request {r['index']} ({len(r['case_ids'])} cases): {r['error'].splitlines()[0]}")
    return s


# ----------------------------------------------------------------------------- sweep
def sweep(records: list[dict], base: Policy) -> dict:
    valid = [r for r in records if r.get("votes")]
    if not valid:
        print("\nsweep skipped: no votes")
        return {}
    votes = [Votes(**r["votes"]) for r in valid]
    expected = [r["expected"] for r in valid]
    accept = [acceptable(r) for r in valid]
    allow_inv = [bool(a & INVALIDATING) for a in accept]

    names = list(SWEEP_GRID)
    results = []
    for combo in itertools.product(*(SWEEP_GRID[n] for n in names)):
        kw = dict(zip(names, combo))
        p = Policy(**kw)
        strict = lenient = fi = 0
        for v, e, a, ai in zip(votes, expected, accept, allow_inv):
            d = p.dispose(v).value
            strict += d == e
            lenient += d in a
            fi += (d in INVALIDATING) and not ai
        results.append({**kw, "strict": strict, "lenient": lenient, "false_inv": fi})

    n = len(valid)
    by_acc = sorted(results, key=lambda r: (-r["strict"], -r["lenient"], r["false_inv"]))
    by_fi = sorted(results, key=lambda r: (r["false_inv"], -r["strict"], -r["lenient"]))
    base_kw = policy_thresholds(base)
    base_row = next((r for r in results if all(abs(r[k] - base_kw[k]) < 1e-9 for k in names)), None)

    def fmt(r: dict) -> str:
        th = " ".join(f"{k}={r[k]:.2f}" for k in names)
        return f"strict {r['strict']:3d}/{n} ({100*r['strict']/n:5.1f}%)  lenient {r['lenient']:3d} ({100*r['lenient']/n:5.1f}%)  false_inv {r['false_inv']:3d}  | {th}"

    print()
    print("=" * 78)
    print(f"threshold sweep over cached votes: {len(results)} policies x {n} cases (no API calls)")
    print("=" * 78)
    if base_row:
        print("current policy:")
        print("  " + fmt(base_row))
    else:
        s = score(base, valid)
        print(f"current policy (not on grid): strict {s['strict']}/{n} lenient {s['lenient']} false_inv {s['false_invalidations']}")
    print("top 5 by strict accuracy:")
    for r in by_acc[:5]:
        print("  " + fmt(r))
    print("fewest false invalidations (tie-break: strict accuracy):")
    print("  " + fmt(by_fi[0]))
    best_fi = by_fi[0]["false_inv"]
    # Among the policies that share the minimum false-invalidation count, the most accurate one is the
    # practical recommendation, so say how many tie.
    ties = sum(1 for r in results if r["false_inv"] == best_fi)
    if ties > 1:
        print(f"  ({ties} policies tie at {best_fi} false invalidations; the most accurate is shown)")
    return {
        "grid": SWEEP_GRID,
        "n_cases": n,
        "top_by_strict": by_acc[:5],
        "min_false_invalidation": by_fi[0],
        "current": base_row,
    }


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", action="append", help="only cases in this category (repeatable)")
    ap.add_argument("--ids", nargs="*", help="only these case ids")
    ap.add_argument("--limit", type=int, help="first N cases (after --category)")
    ap.add_argument("--policy", help='JSON overrides for Policy, e.g. \'{"bears_min": 0.6}\'')
    ap.add_argument("--workers", type=int, default=8, help="concurrent requests (default 8)")
    ap.add_argument("--batch-size", type=int, default=20, help="memories per request in --batched mode (default 20)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--batched", action="store_true", help="pack memories sharing an event into one request (default)")
    mode.add_argument("--per-case", action="store_true", help="one memory per request")
    ap.add_argument("--dry-run", action="store_true", help="use the keyword FakeJudge; no API key needed")
    ap.add_argument("--from", dest="from_file", help="re-report and sweep from a saved results JSON; no API calls")
    ap.add_argument("--out", help="where to save raw votes (default evals/results/<timestamp>.json)")
    ap.add_argument("--no-sweep", action="store_true", help="skip the threshold sweep")
    ap.add_argument("--model", help="Jev model id override (default: SDK default)")
    args = ap.parse_args(argv)

    policy = build_policy(args.policy)

    if args.from_file:
        data = json.loads(Path(args.from_file).read_text())
        records = data["cases"]
        requests = data.get("requests", [])
        meta = data.get("meta", {})
        meta = {**meta, "mode": f"from:{args.from_file}"}
        if args.category:
            records = [r for r in records if r["category"] in set(args.category)]
        if args.ids:
            records = [r for r in records if r["id"] in set(args.ids)]
        if args.limit is not None:
            records = records[: args.limit]
        if not records:
            sys.exit("no cases selected")
        print_report(policy, records, requests, meta)
        if not args.no_sweep:
            sweep(records, policy)
        return 0

    cases = select_cases(args)
    if not cases:
        sys.exit("no cases selected")
    per_case = bool(args.per_case)
    mode_name = "per-case" if per_case else f"batched(<= {args.batch_size}/request)"

    if args.dry_run:
        judge = FakeJudge()
        judge_name = FakeJudge.model
    else:
        key = load_api_key()
        if not key:
            sys.exit(
                "TYPESAFE_API_KEY is not set. Export it, or put TYPESAFE_API_KEY=... in "
                f"{ROOT / '.env'}. Get a key at https://console.typesafe.ai/ . "
                "Use --dry-run to exercise the pipeline without a key."
            )
        from invalidate import JevJudge

        judge = JevJudge(api_key=key, model=args.model)
        judge_name = args.model or "jev (sdk default)"

    requests_plan = plan_requests(cases, per_case, args.batch_size)
    print(f"judging {len(cases)} cases in {len(requests_plan)} requests ({mode_name}, workers={args.workers}, judge={judge_name})")
    t0 = time.perf_counter()
    records, requests = judge_all(judge, requests_plan, args.workers)
    wall = time.perf_counter() - t0
    print(f"done in {wall:.1f}s")

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode_name,
        "judge": judge_name,
        "dry_run": args.dry_run,
        "policy": policy_thresholds(policy),
        "workers": args.workers,
        "wall_seconds": wall,
        "argv": sys.argv[1:],
        "model": next((r["model"] for r in requests if r.get("model")), None),
    }
    s = print_report(policy, records, requests, meta)
    sw = {} if args.no_sweep else sweep(records, policy)

    out = Path(args.out) if args.out else HERE / "results" / (datetime.now().strftime("%Y%m%d-%H%M%S") + ("-dryrun" if args.dry_run else "") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "summary": {
            "n": s["n"], "strict": s["strict"], "lenient": s["lenient"],
            "false_invalidations": s["false_invalidations"],
            "strict_acc": s["strict_acc"], "lenient_acc": s["lenient_acc"],
            "per_category": {k: dict(v) for k, v in s["per_category"].items()},
            "confusion": [{"expected": e, "predicted": p, "n": n} for (e, p), n in sorted(s["confusion"].items())],
        },
        "sweep": sw,
        "requests": requests,
        "cases": [{**r, "predicted": predict(policy, r)} for r in records],
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nsaved raw votes to {out}  (rerun offline: python evals/run_eval.py --from {out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
