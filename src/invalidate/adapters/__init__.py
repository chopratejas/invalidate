"""Adapters: put invalidate in front of an existing memory system.

    from invalidate.adapters import Governor
    from invalidate.adapters.chroma import ChromaAdapter
    gov = Governor(ChromaAdapter(collection), "ledger.db", mode="flag")
    gov.sync(); gov.observe("we migrated to SQLite", source="slack")

Each adapter module imports its host SDK lazily, so importing this package needs nothing extra.
"""
from .base import DEAD, LIVE, Adapter, Governor, GovernorReport, HostMemory, InMemoryAdapter, Push, Reason, SyncReport

__all__ = ["Adapter", "Governor", "GovernorReport", "HostMemory", "InMemoryAdapter", "Push", "Reason", "SyncReport", "LIVE", "DEAD"]
