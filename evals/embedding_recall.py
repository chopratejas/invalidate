#!/usr/bin/env python
"""How wide must an embedding prefilter be to keep every bearing pair?

For each labelled bearing case (event -> the memory it bears on), rank all 157 memories by cosine
similarity to the event and report the rank of the true one. Two variants:

  whole      embed the whole event text
  sentences  split the event into sentences, rank memories by the best sentence (union of per-sentence nets)

Needs OPENAI_API_KEY (env or .env). Costs a fraction of a cent.

    python evals/embedding_recall.py
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import CASES  # noqa: E402

from invalidate.env import load_dotenv  # noqa: E402

MODEL = "text-embedding-3-small"


def embed(client, texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), 100):
        r = client.embeddings.create(model=MODEL, input=texts[i:i + 100])
        out.extend([d.embedding for d in r.data])
    return out


def cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+|;\s+", text) if p.strip()]
    return parts or [text]


def main() -> None:
    key = os.environ.get("OPENAI_API_KEY") or load_dotenv().get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY not set")
    from openai import OpenAI

    client = OpenAI(api_key=key)
    n = len(CASES)
    bearing = [k for k in range(n) if CASES[k]["expected"] != "unrelated"]
    mems = embed(client, [c["memory"] for c in CASES])
    whole = embed(client, [c["event"] for c in CASES])
    sent_lists = [sentences(c["event"]) for c in CASES]
    flat = [s for ss in sent_lists for s in ss]
    sent_vecs = embed(client, flat)
    per_event: list[list[list[float]]] = []
    pos = 0
    for ss in sent_lists:
        per_event.append(sent_vecs[pos:pos + len(ss)])
        pos += len(ss)

    def ranks(score) -> dict[int, int]:
        out = {}
        for k in bearing:
            order = sorted(range(n), key=lambda i: -score(k, i))
            out[k] = order.index(k) + 1
        return out

    r_whole = ranks(lambda k, i: cos(whole[k], mems[i]))
    r_sent = ranks(lambda k, i: max(cos(v, mems[i]) for v in per_event[k]))
    print(f"pool {n} memories, {len(bearing)} labelled bearing pairs, {MODEL}")
    print(f"{'net':>22s}  {'whole event':>12s}  {'best sentence':>14s}")
    for K in (1, 2, 4, 8, 16, 32, 64):
        hw = sum(1 for k in bearing if r_whole[k] <= K)
        hs = sum(1 for k in bearing if r_sent[k] <= K)
        print(f"top-{K:2d} ({100 * K / n:5.1f}% of pool)  {hw:3d}/{len(bearing)} {100 * hw / len(bearing):5.1f}%   {hs:3d}/{len(bearing)} {100 * hs / len(bearing):5.1f}%")
    print("\nworst by whole-event rank:")
    for k in sorted(bearing, key=lambda k: -r_whole[k])[:6]:
        c = CASES[k]
        print(f"  whole {r_whole[k]:3d}  sentence {r_sent[k]:3d}  {c['id']:6s} {c['expected']:12s} mem={c['memory'][:45]!r} ev={c['event'][:50]!r}")
    out = HERE / "results" / "embedding_recall.json"
    out.write_text(json.dumps({"model": MODEL, "whole": {CASES[k]["id"]: r_whole[k] for k in bearing},
                               "sentences": {CASES[k]["id"]: r_sent[k] for k in bearing}}, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
