"""RAG cache governance: runbook chunks as leased memories, a PR description as evidence.

Each chunk is stored verbatim with kind="chunk", a hard ttl (re-index at least weekly), and its
source path. When a PR lands that changes the runbook, observe() judges every live chunk against
the PR text and supersedes the ones it rewrites. Nothing is re-embedded; nothing is rewritten.

    python examples/rag_cache.py
"""
import sys
import tempfile

from invalidate import Invalidate, MissingAPIKey, Status
from invalidate.cli import load_dotenv

load_dotenv()

RUNBOOK = {
    "runbook.md#deploy-window": "Production deploys run at 14:00 UTC, Monday to Thursday. No Friday deploys.",
    "runbook.md#rollback": "To roll back, run `deployctl rollback --last` from the ops box; it takes about 4 minutes.",
    "runbook.md#on-call": "Primary on-call is paged through PagerDuty; secondary is the #infra channel.",
    "runbook.md#db-failover": "Database failover is manual: promote the replica with `pg_ctl promote`.",
    "runbook.md#log-retention": "Application logs are retained for 30 days in Loki.",
}

PR_482 = (
    "PR #482 (merged): Move the production deploy window to 18:00 UTC and allow Friday deploys "
    "with a second approver. Replace `deployctl rollback --last` with `deployctl rollback --to <sha>`; "
    "rollbacks now complete in under a minute."
)

WEEK = 7 * 24 * 3600


def main() -> int:
    mem = Invalidate(tempfile.mktemp(suffix=".db", prefix="rag-cache-"), namespace="runbook")
    for src, text in RUNBOOK.items():
        mem.remember(text, kind="chunk", source=src, ttl=WEEK)
    print(f"indexed {len(RUNBOOK)} chunks (verbatim, hard ttl 7d)\n")

    report = mem.observe(PR_482, source="github")
    print(f"observed PR #482: {report.summary()}\n")
    for v in report.verdicts:
        m = mem.get(v.memory_id)
        flag = "  <- " + v.to_status.value if v.changed else ""
        print(f"  {m.source:28} {v.disposition.value:12} still_true={v.votes.still_true:.2f} "
              f"replaces={v.votes.replaces:.2f}{flag}")

    # Superseded chunks stay in the store (audit trail) but are never returned by recall().
    stale = mem.list(statuses=[Status.SUPERSEDED, Status.CONTRADICTED, Status.NEEDS_REVIEW])
    print(f"\n{len(stale)} chunk(s) no longer served:")
    for m in stale:
        print(f"  [{m.status.value}] {m.source}: {m.fact}")

    # Re-index the changed sections and link old -> new so the lineage is explicit.
    new = mem.remember("To roll back, run `deployctl rollback --to <sha>`; it completes in under a minute.",
                       kind="chunk", source="runbook.md#rollback@482", ttl=WEEK)
    for m in stale:
        if m.source == "runbook.md#rollback":
            mem.supersede(m.id, by=new.id)

    rep = mem.recall("how do I roll back a bad deploy?", limit=3)
    print("\nrecall 'how do I roll back a bad deploy?':")
    for r in rep.results:
        print(f"  {r.relevance:.2f}  {r.memory.source}: {r.memory.fact}")
    print(f"  ({rep.considered} live chunks considered, {rep.latency_ms:.0f} ms, ${rep.cost_usd:.5f})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MissingAPIKey as e:
        sys.exit(f"{e}")
