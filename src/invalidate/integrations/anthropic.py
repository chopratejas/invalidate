"""Anthropic wrapper: `wrap(client, mem)` proxies `client.messages.create`.

    from invalidate import Invalidate
    from invalidate.integrations.anthropic import wrap
    client = wrap(Anthropic(), Invalidate("memories.db"))
    client.messages.create(model="claude-opus-5", max_tokens=16000, messages=[...])

Before each call the latest user message is used to recall live facts, which are appended to the
`system` kwarg (str or list of text blocks). After the call the same message is observed, so a
correction flips the stale fact. `anthropic` is imported lazily and only by `make_client()`.
"""
from __future__ import annotations

from typing import Any

from ..engine import Invalidate
from ..types import ObserveReport
from . import latest_user_text, memory_block, observe_turn

__all__ = ["wrap", "make_client", "memory_block", "observe_turn", "inject"]


def inject(system: Any, block: str) -> Any:
    """Merge `block` into an Anthropic `system` value: None, str, or a list of text blocks."""
    if not block:
        return system
    if system is None:
        return block
    if isinstance(system, str):
        return system.rstrip() + "\n\n" + block
    if isinstance(system, list):
        return [*system, {"type": "text", "text": block}]
    return system


class _Messages:
    def __init__(self, inner: Any, mem: Invalidate, opts: dict[str, Any]) -> None:
        self._inner, self.mem, self.opts = inner, mem, opts
        self.last_report: ObserveReport | None = None

    def create(self, *args: Any, **kwargs: Any) -> Any:
        text = latest_user_text(kwargs.get("messages") or [])
        if text and self.opts["observe"] and self.opts["observe_first"]:
            self.last_report = observe_turn(self.mem, text, source=self.opts["source"])
        if text and self.opts["inject"]:
            kwargs["system"] = inject(kwargs.get("system"), memory_block(self.mem, text, limit=self.opts["limit"]))
        resp = self._inner.create(*args, **kwargs)
        if text and self.opts["observe"] and not self.opts["observe_first"]:
            self.last_report = observe_turn(self.mem, text, source=self.opts["source"])
        return resp

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _Proxy:
    def __init__(self, inner: Any, messages: _Messages) -> None:
        self._inner, self.messages = inner, messages

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def wrap(
    client: Any,
    mem: Invalidate,
    *,
    inject: bool = True,
    observe: bool = True,
    observe_first: bool = False,
    limit: int = 8,
    source: str = "user",
) -> Any:
    """Proxy an Anthropic client so `messages.create` recalls before and observes after.

    inject=False        skip the memory block.
    observe=False       skip judging the user turn.
    observe_first=True  judge the turn before recalling, so a correction in this very turn is already
                        applied to what the model sees (one extra ~100 ms hop up front).
    """
    messages = getattr(client, "messages", None)
    if messages is None or not hasattr(messages, "create"):
        raise TypeError("wrap() expects an Anthropic client with .messages.create")
    opts = {"inject": inject, "observe": observe, "observe_first": observe_first, "limit": limit, "source": source}
    return _Proxy(client, _Messages(messages, mem, opts))


def make_client(**kwargs: Any) -> Any:
    """`anthropic.Anthropic(**kwargs)`, imported lazily with a clear error if the SDK is missing."""
    try:
        from anthropic import Anthropic
    except ImportError as e:  # pragma: no cover
        raise ImportError("the anthropic package is not installed: pip install 'invalidate[anthropic]'") from e
    return Anthropic(**kwargs)
