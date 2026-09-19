"""Govern a CLAUDE.md with the real Jev: copy the README demo facts into a temp file, observe one event,
print the annotated file.

Usage: TYPESAFE_API_KEY=... python scripts/live_markdown.py   (or put the key in ./.env)
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from invalidate import load_dotenv  # noqa: E402

load_dotenv()
if not os.environ.get("TYPESAFE_API_KEY"):
    print("TYPESAFE_API_KEY not set (export it or put it in .env)")
    sys.exit(2)

from invalidate.adapters import Governor  # noqa: E402
from invalidate.adapters.markdown import MarkdownAdapter  # noqa: E402
from invalidate.cli import DEMO_FACTS  # noqa: E402

EVENT = "we migrated to SQLite last Tuesday"

tmp = tempfile.mkdtemp(prefix="invalidate-md-")
path = os.path.join(tmp, "CLAUDE.md")
with open(path, "w", encoding="utf-8") as fh:
    fh.write("# Project memory\n\nThings the agent should know.\n\n## Facts\n")
    for fact, kind, source in DEMO_FACTS:
        fh.write(f"- {fact}\n")

print(f"CLAUDE.md before ({path}):\n")
print(open(path, encoding="utf-8").read())

gov = Governor(MarkdownAdapter(path), os.path.join(tmp, "ledger.db"), mode="flag", successors=True)
print("sync:   ", gov.sync())
t0 = time.perf_counter()
rep = gov.observe(EVENT, source="slack")
dt = (time.perf_counter() - t0) * 1000
print(f"observe: {EVENT!r}  [{dt:.0f} ms wall, model={rep.report.model}]")
for v in rep.report.verdicts:
    m = gov.mem.get(v.memory_id)
    flip = "  <-- FLIP" if v.changed else ""
    print(f"  bears={v.votes.bears:.2f} still={v.votes.still_true:.2f} repl={v.votes.replaces:.2f}  "
          f"{v.disposition.value:12s} {v.from_status.value}->{v.to_status.value:12s} {m.fact!r}{flip}")
for p in rep.pushes:
    print(f"  push {p.action:6s} {p.host_id}  {p.status.value}" + (f"  ERROR {p.error}" if p.error else ""))
print("summary:", rep.summary())
print("re-sync:", gov.sync())
gov.close()

print(f"\nCLAUDE.md after:\n")
print(open(path, encoding="utf-8").read())
