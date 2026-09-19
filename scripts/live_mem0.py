"""Live: invalidate as a layer in front of a real Mem0 (OSS 2.x) store.

Needs TYPESAFE_API_KEY and OPENAI_API_KEY (Mem0's default LLM/embedder) in the environment or ./.env.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from invalidate import load_dotenv  # noqa: E402

load_dotenv()
os.environ.setdefault("MEM0_TELEMETRY", "false")

from mem0 import Memory  # noqa: E402

from invalidate.adapters import Governor  # noqa: E402
from invalidate.adapters.mem0 import Mem0Adapter, governed_search, guard_add  # noqa: E402

UID = "live-test"
FILTERS = {"user_id": UID}
memory = Memory.from_config({"vector_store": {"provider": "qdrant", "config": {
    "path": "/tmp/invalidate-mem0-qdrant", "collection_name": "invalidate_live"}}})

# clean slate
for r in (memory.get_all(filters=FILTERS, top_k=1000) or {}).get("results", []):
    memory.delete(r["id"])

facts = ["user prefers Postgres for new services", "deploys run at 2pm UTC on weekdays", "Alice owns the billing service",
         "users.email is nullable", "lunch is at noon on Fridays", "prod reads go through the Postgres replica"]
for f in facts:
    memory.add(f, user_id=UID, infer=False, metadata={"src": "seed"})
print(f"seeded {len(facts)} verbatim memories into Mem0 (infer=False)")

gov = Governor(Mem0Adapter(memory, user_id=UID), ":memory:", mode="flag", successors=True)
print("sync:", gov.sync())

for text, src in [("Postgres replica was down for an hour this morning, all good now", "slack"),
                  ("we migrated to SQLite last Tuesday", "slack"),
                  ("Bob took over billing from Alice", "slack")]:
    t0 = time.perf_counter()
    r = gov.observe(text, source=src)
    print(f"\nobserve {text!r}  [{(time.perf_counter()-t0)*1000:.0f} ms]\n  {r.summary()}")
    for p in r.pushes:
        print(f"  push {p.action:6s} {p.host_id[:8]}… -> {p.status.value}{'  ERROR '+p.error if p.error else ''}")

print("\nMem0 now holds (verbatim text + invalidate metadata):")
for r in memory.get_all(filters=FILTERS, top_k=1000)["results"]:
    md = r.get("metadata") or {}
    st = md.get("invalidate_status", "active")
    ev = md.get("invalidate_event", "")
    print(f"  [{st:12s}] {r['memory'][:60]:60s} {('<- '+ev[:50]) if ev else ''}")

print("\ngoverned_search('which database should the new service use?'):")
for r in governed_search(memory, gov, "which database should the new service use?", filters=FILTERS, top_k=5):
    print(f"  {r.get('score', 0):.2f}  {r['memory']}")
print("plain Mem0 search for comparison:")
for r in memory.search("which database should the new service use?", filters=FILTERS, top_k=3)["results"]:
    print(f"  {r.get('score', 0):.2f}  {r['memory']}")

print("\nguard_add: a user message is judged before Mem0 stores it")
add = guard_add(memory, gov)
r = add([{"role": "user", "content": "Actually lunch moved to 1pm on Fridays"}], user_id=UID, infer=False)
print("  Mem0 add returned:", json.dumps(r)[:120])
print("  ledger:", [(gov.host_id(m)[:8], m.status.value, m.fact[:40]) for m in gov.mem.list() if m.status.value != "active"])
