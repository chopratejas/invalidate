#!/usr/bin/env python
"""Screening experiment: can Jev screen many events against many memories in ONE request?

Builds the full 157 x 157 matrix from evals/cases.py (every event against every memory) and
compares three screening forms:

  single   the current screen: state = {event, memories[]}, one question per memory
  compact  same shape, shorter question text
  pairs    state = {events[E], memories[M]}, one short question per (event, memory) pair

Reports, per form: recall of labelled bearing pairs (the diagonal, expected != unrelated) at the
screen threshold, how many off-diagonal pairs pass (screening efficiency), tokens per pair,
latency per request, and cost for the whole matrix.

    python evals/screen_matrix.py --events 10 --memories 100
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import CASES  # noqa: E402

from typesafe_sdk import Noul, NoulCriteria, TypeSafeClient  # noqa: E402

from invalidate.env import load_dotenv  # noqa: E402

USD_PER_M = 0.042
THRESHOLD = 0.3


def key() -> str:
    import os

    k = os.environ.get("TYPESAFE_API_KEY") or load_dotenv().get("TYPESAFE_API_KEY")
    if not k:
        sys.exit("TYPESAFE_API_KEY not set")
    return k


def mem_view(c: dict) -> dict:
    return {"fact": c["memory"], "kind": c["kind"], "source": "unknown"}


def evt_view(c: dict) -> dict:
    return {"text": c["event"], "source": c["event_source"]}


def q_single(i: int) -> Noul:
    return Noul(
        instructions=f"Does `event.text` give information about the same subject that `memories[{i}].fact` is about, whether it agrees with it or not?",
        criteria=NoulCriteria(true="Same thing, entity, setting, or preference", false="A different subject, even if a word or name is shared"),
    )


def q_compact(i: int) -> Noul:
    return Noul(
        instructions=f"Is `event.text` about the same subject as `memories[{i}].fact`?",
        criteria=NoulCriteria(true="Same thing, entity, setting, or preference, whether or not they agree", false="A different subject"),
    )


def q_pair(j: int, i: int) -> Noul:
    return Noul(
        instructions=f"Is `events[{j}].text` about the same subject as `memories[{i}].fact`?",
        criteria=NoulCriteria(true="Same thing, entity, setting, or preference, whether or not they agree", false="A different subject"),
    )


def call(client, state, qs):
    """system_one with exponential backoff on 429/529 (the SDK's own retry gives up too early at this rate)."""
    delay = 1.0
    for attempt in range(8):
        try:
            return client.system_one(state, qs)
        except Exception as e:  # noqa: BLE001
            name = type(e).__name__
            if "RateLimit" not in name and "Overloaded" not in name and "429" not in str(e) and "529" not in str(e):
                raise
            time.sleep(delay)
            delay = min(delay * 2, 20)
    return client.system_one(state, qs)


def run(client, form: str, E: int, M: int, workers: int):
    n = len(CASES)
    scores = [[None] * n for _ in range(n)]  # scores[j][i] = event j vs memory i
    jobs = []
    if form in ("single", "compact"):
        qf = q_single if form == "single" else q_compact
        for j in range(n):
            for m0 in range(0, n, M):
                jobs.append((j, [j], list(range(m0, min(n, m0 + M))), qf))
    else:
        for e0 in range(0, n, E):
            for m0 in range(0, n, M):
                jobs.append((None, list(range(e0, min(n, e0 + E))), list(range(m0, min(n, m0 + M))), None))

    def do(job):
        _, ev_idx, mem_idx, qf = job
        mems = [mem_view(CASES[i]) for i in mem_idx]
        if form == "pairs":
            state = {"events": [evt_view(CASES[j]) for j in ev_idx], "memories": mems}
            qs = {f"p_{jj}_{ii}": q_pair(jj, ii) for jj in range(len(ev_idx)) for ii in range(len(mem_idx))}
        else:
            state = {"event": evt_view(CASES[ev_idx[0]]), "memories": mems}
            qs = {f"s_{ii}": qf(ii) for ii in range(len(mem_idx))}
        t0 = time.perf_counter()
        resp = call(client, state, qs)
        dt = (time.perf_counter() - t0) * 1000
        out = []
        for jj, j in enumerate(ev_idx):
            for ii, i in enumerate(mem_idx):
                a = resp.answers[f"p_{jj}_{ii}" if form == "pairs" else f"s_{ii}"]
                out.append((j, i, float(a.noul)))
        return out, resp.usage.input_tokens, dt, len(qs)

    tokens = 0
    lat = []
    nq = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for out, tk, dt, q in pool.map(do, jobs):
            tokens += tk
            lat.append(dt)
            nq.append(q)
            for j, i, p in out:
                scores[j][i] = p
    return scores, tokens, lat, nq, len(jobs)


def report(form, scores, tokens, lat, nq, nreq, E, M):
    n = len(CASES)
    bearing = [(k, CASES[k]) for k in range(n) if CASES[k]["expected"] != "unrelated"]
    missed = [(c["id"], round(scores[k][k], 2), c["expected"]) for k, c in bearing if scores[k][k] < THRESHOLD]
    unrelated_diag = [k for k in range(n) if CASES[k]["expected"] == "unrelated"]
    unrel_pass = sum(1 for k in unrelated_diag if scores[k][k] >= THRESHOLD)
    off = [(j, i) for j in range(n) for i in range(n) if j != i]
    off_pass = sum(1 for j, i in off if scores[j][i] >= THRESHOLD)
    pairs = n * n
    print(f"\n== {form}" + (f" (E={E}, M={M})" if form == "pairs" else f" (M={M})"))
    print(f"requests {nreq}   questions/request {min(nq)}-{max(nq)}   latency/request p50 {statistics.median(lat):.0f} ms  max {max(lat):.0f} ms")
    print(f"tokens {tokens:,}  = {tokens / pairs:.0f} per pair   cost ${tokens * USD_PER_M / 1e6:.4f} for {pairs:,} pairs   (${tokens * USD_PER_M / 1e6 / pairs:.8f}/pair)")
    print(f"bearing diagonal recall @{THRESHOLD}: {len(bearing) - len(missed)}/{len(bearing)}   missed: {missed}")
    print(f"labelled-unrelated diagonal passing: {unrel_pass}/{len(unrelated_diag)}   off-diagonal passing: {off_pass}/{len(off)} ({100 * off_pass / len(off):.1f}%)")
    return {"form": form, "E": E, "M": M, "requests": nreq, "tokens": tokens, "tokens_per_pair": tokens / pairs,
            "latency_p50_ms": statistics.median(lat), "latency_max_ms": max(lat),
            "bearing_recall": (len(bearing) - len(missed)) / len(bearing), "missed": missed,
            "off_diag_pass_rate": off_pass / len(off), "scores": scores}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forms", default="single,compact,pairs")
    ap.add_argument("--events", type=int, default=10)
    ap.add_argument("--memories", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "results" / "screen_matrix.json"))
    a = ap.parse_args()
    client = TypeSafeClient(api_key=key(), timeout=60)
    results = []
    for form in a.forms.split(","):
        scores, tokens, lat, nq, nreq = run(client, form, a.events, a.memories, a.workers)
        results.append(report(form, scores, tokens, lat, nq, nreq, a.events, a.memories))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
