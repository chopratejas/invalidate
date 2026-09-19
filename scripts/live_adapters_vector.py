"""Govern a real Chroma collection and a LangGraph InMemoryStore with the real Jev.

Seeds the README demo facts, syncs a flag-mode Governor, observes "we migrated to SQLite last
Tuesday", prints each host row's receipt and a governed query for "which database?".

Usage: TYPESAFE_API_KEY=... python scripts/live_adapters_vector.py   (or put the key in .env)
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import invalidate  # noqa: E402

invalidate.load_dotenv()
if not os.environ.get("TYPESAFE_API_KEY"):
    print("TYPESAFE_API_KEY not set (export it or put it in .env)")
    sys.exit(2)

import chromadb  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

from invalidate.adapters import Governor  # noqa: E402
from invalidate.adapters.chroma import ChromaAdapter, governed_query  # noqa: E402
from invalidate.adapters.langgraph import LangGraphStoreAdapter, filter_items  # noqa: E402

FACTS = [
    ("pg", "user prefers Postgres", "preference"),
    ("deploy", "deploys run at 2pm UTC", "schedule"),
    ("alice", "Alice owns the billing service", "fact"),
    ("email", "users.email is nullable", "config"),
    ("lunch", "lunch is at noon on Fridays", "fact"),
    ("replica", "prod reads go through the Postgres replica", "config"),
]
EVENT = "we migrated to SQLite last Tuesday"
QUERY = "which database?"


def _embedder():
    """Chroma's default (all-MiniLM, cached locally) when available; else a tiny bag-of-words stand-in."""
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        ef = DefaultEmbeddingFunction()
        ef(["warm up"])
        return ef, "all-MiniLM-L6-v2"
    except Exception:  # noqa: BLE001
        import hashlib

        class Bag:
            def __call__(self, input):  # noqa: A002 - chroma's signature
                out = []
                for text in input:
                    v = [0.0] * 64
                    for tok in text.lower().replace("?", "").split():
                        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 64] += 1.0
                    out.append(v)
                return out

            @staticmethod
            def name():
                return "bag"

        return Bag(), "bag-of-words fallback"


def chroma_demo() -> None:
    ef, ef_name = _embedder()
    client = chromadb.EphemeralClient()
    try:
        col = client.create_collection("invalidate-live", embedding_function=ef)
    except Exception:  # noqa: BLE001 - custom EFs need extra plumbing on some releases: embed by hand
        col = client.create_collection("invalidate-live", embedding_function=None)
    ids = [i for i, _, _ in FACTS]
    docs = [t for _, t, _ in FACTS]
    col.add(ids=ids, documents=docs, metadatas=[{"kind": k} for _, _, k in FACTS], embeddings=ef(docs))
    print(f"== Chroma ({chromadb.__version__}, embeddings: {ef_name}) ==")
    gov = Governor(ChromaAdapter(col, embed=lambda t: ef([t])[0]), ":memory:", mode="flag")
    print("sync:", gov.sync())
    rep = gov.observe(EVENT, source="slack")
    print("observe:", rep.summary())
    for p in rep.pushes:
        print(f"  push {p.action:6s} {p.host_id:8s} -> {p.status.value}" + (f"  ERROR {p.error}" if p.error else ""))
    got = col.get(include=["documents", "metadatas"])
    for rid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
        receipt = {k: v for k, v in (meta or {}).items() if k.startswith("invalidate_")}
        short = {k: (v if not isinstance(v, str) else v[:40]) for k, v in receipt.items()
                 if k in ("invalidate_status", "invalidate_disposition", "invalidate_still_true", "invalidate_event")}
        print(f"  {rid:8s} {doc!r:48s} {short or '(untouched)'}")
    res = governed_query(col, gov, query_embeddings=ef([QUERY]), n_results=6, include=["documents", "distances"])
    print(f"governed_query({QUERY!r}) ->")
    for rid, doc, d in zip(res["ids"][0], res["documents"][0], res["distances"][0]):
        print(f"  {d:.3f} {rid:8s} {doc!r}")
    print("  excluded:", sorted(gov.dead_ids()))
    gov.close()


def langgraph_demo() -> None:
    store = InMemoryStore()
    ns = ("user-1", "memories")
    for rid, text, kind in FACTS:
        store.put(ns, rid, {"content": text, "kind": kind})
    print("\n== LangGraph InMemoryStore ==")
    adapter = LangGraphStoreAdapter(store, ns)
    gov = Governor(adapter, ":memory:", mode="flag")
    print("sync:", gov.sync())
    rep = gov.observe(EVENT, source="slack")
    print("observe:", rep.summary())
    for p in rep.pushes:
        print(f"  push {p.action:6s} {p.host_id:8s} -> {p.status.value}" + (f"  ERROR {p.error}" if p.error else ""))
    for it in store.search(ns, limit=50):
        v = it.value
        short = {k: (v[k] if not isinstance(v[k], str) else v[k][:40]) for k in
                 ("invalidate_status", "invalidate_disposition", "invalidate_still_true", "invalidate_live") if k in v}
        print(f"  {it.key:8s} {v['content']!r:48s} {short or '(untouched)'}")
    live = filter_items(store.search(ns, limit=50), gov)
    print(f"filter_items(search) -> {[i.key for i in live]}")
    hostside = store.search(ns, filter=adapter.live_filter(), limit=50)
    print(f"search(filter=live_filter()) -> {[i.key for i in hostside]}")
    gov.close()


if __name__ == "__main__":
    chroma_demo()
    langgraph_demo()
