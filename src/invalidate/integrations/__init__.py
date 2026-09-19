"""Drop-in wrappers for LLM clients. Recall before the call, observe after.

    from invalidate.integrations.openai import wrap      # client.chat.completions.create
    from invalidate.integrations.anthropic import wrap   # client.messages.create

Neither module imports its SDK at import time; `make_client()` does, lazily.
"""
from __future__ import annotations

from ..engine import Invalidate
from ..types import ObserveReport

HEADER = "Known facts (verbatim, governed by invalidate):"


def memory_block(mem: Invalidate, query: str, *, limit: int = 8) -> str:
    """Recalled facts rendered as a system-prompt block. Empty string when nothing is relevant."""
    rep = mem.recall(query, limit=limit)
    if not rep.results:
        return ""
    return HEADER + "\n" + "\n".join(f"- {r.memory.fact}" for r in rep.results)


def observe_turn(mem: Invalidate, user_text: str, *, source: str = "user") -> ObserveReport:
    """Judge one user turn against every live memory so corrections invalidate stale facts."""
    return mem.observe(user_text, source=source)


def text_of(content: object) -> str:
    """Plain text of a message `content` (str, or a list of {'type': 'text'} parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def latest_user_text(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            return text_of(m.get("content")).strip()
    return ""


__all__ = ["HEADER", "memory_block", "observe_turn", "text_of", "latest_user_text"]
