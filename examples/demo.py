"""The headline demo as a script: remember six facts, watch evidence flip them, recall.

    export TYPESAFE_API_KEY=...   # https://console.typesafe.ai/
    python examples/demo.py
"""
import sys
import tempfile

from invalidate import Invalidate, MissingAPIKey
from invalidate.cli import load_dotenv

load_dotenv()  # the library itself never reads .env; the CLI and these examples do

FACTS = [
    ("user prefers Postgres", "preference", "chat"),
    ("deploys run at 2pm UTC", "fact", "wiki"),
    ("Alice owns the billing service", "fact", "wiki"),
    ("users.email is nullable", "schema", "migrations"),
    ("lunch is at noon on Fridays", "fact", "chat"),
    ("prod reads go through the Postgres replica", "fact", "runbook"),
]


def main() -> int:
    mem = Invalidate(tempfile.mktemp(suffix=".db", prefix="invalidate-demo-"))
    mem.judge  # fail fast on a missing TYPESAFE_API_KEY; remember() alone never needs one

    # 1. remember: verbatim, no model call, no embeddings.
    for fact, kind, source in FACTS:
        m = mem.remember(fact, kind=kind, source=source)
        print(f"remembered {m.id}  {m.fact}")

    # 2. observe: every live memory is judged against the event; the policy applies the write.
    for text in ("Postgres was down for an hour this morning", "we migrated to SQLite last Tuesday"):
        print(f"\nobserve: {text!r}")
        report = mem.observe(text, source="slack", remember_successor=True)
        for v in report.changed:
            print(f"  {mem.get(v.memory_id).fact!r}: {v.from_status.value} -> {v.to_status.value}"
                  f"  (still_true={v.votes.still_true:.2f}, replaces={v.votes.replaces:.2f})")
        if not report.changed:
            print("  no status changes")
        print(f"  {report.summary()}")

    # 3. recall: only live memories, ranked by Jev's relevance vote.
    query = "which database should the new service use?"
    print(f"\nrecall: {query!r}")
    rep = mem.recall(query)
    for r in rep.results:
        print(f"  {r.relevance:.2f}  {r.memory.fact}")
    if not rep.results:
        stale = [m for m in mem.list() if not m.status.live]
        print("  no live facts; withheld:", "; ".join(f"{m.fact} [{m.status.value}]" for m in stale))

    # 4. the audit trail: every vote is kept per (event, memory) pair.
    first = mem.list()[0]
    print(f"\nhistory of {first.fact!r}:")
    for v in mem.history(first.id):
        print(f"  {v.disposition.value:12} {v.from_status.value} -> {v.to_status.value}"
              f"  bears={v.votes.bears:.2f} still={v.votes.still_true:.2f} replaces={v.votes.replaces:.2f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MissingAPIKey as e:
        sys.exit(f"{e}")
