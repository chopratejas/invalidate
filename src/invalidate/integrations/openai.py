"""OpenAI wrapper: `wrap(client, mem)` proxies `client.chat.completions.create`.

    from invalidate import Invalidate
    from invalidate.integrations.openai import wrap
    client = wrap(OpenAI(), Invalidate("memories.db"))
    client.chat.completions.create(model=..., messages=[...])   # same call, governed memory

Before each call the latest user message is used to recall live facts, which are merged into
the system message. After the call the same message is observed, so "we moved off Postgres"
flips the stale fact. `openai` is imported lazily and only by `make_client()`.
"""
from __future__ import annotations

from typing import Any

from ..engine import Invalidate
from ..types import ObserveReport
from . import latest_user_text, memory_block, observe_turn

__all__ = ["wrap", "make_client", "memory_block", "observe_turn", "inject"]


def inject(messages: list[dict], block: str) -> list[dict]:
    """Return a copy of `messages` with `block` merged into the leading system/developer message."""
    if not block:
        return messages
    out = list(messages)
    if out and out[0].get("role") in ("system", "developer") and isinstance(out[0].get("content"), str):
        out[0] = {**out[0], "content": out[0]["content"].rstrip() + "\n\n" + block}
    else:
        out.insert(0, {"role": "system", "content": block})
    return out


class _Completions:
    def __init__(self, inner: Any, mem: Invalidate, opts: dict[str, Any]) -> None:
        self._inner, self.mem, self.opts = inner, mem, opts
        self.last_report: ObserveReport | None = None

    def create(self, *args: Any, **kwargs: Any) -> Any:
        text = latest_user_text(kwargs.get("messages") or [])
        if text and self.opts["observe"] and self.opts["observe_first"]:
            self.last_report = observe_turn(self.mem, text, source=self.opts["source"])
        if text and self.opts["inject"]:
            kwargs["messages"] = inject(kwargs["messages"], memory_block(self.mem, text, limit=self.opts["limit"]))
        resp = self._inner.create(*args, **kwargs)
        if text and self.opts["observe"] and not self.opts["observe_first"]:
            self.last_report = observe_turn(self.mem, text, source=self.opts["source"])
        return resp

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _Proxy:
    """Delegates everything to the wrapped object except one overridden attribute."""

    def __init__(self, inner: Any, name: str, value: Any) -> None:
        self._inner, self._name, self._value = inner, name, value

    def __getattr__(self, name: str) -> Any:
        return self._value if name == self._name else getattr(self._inner, name)


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
    """Proxy an OpenAI client so `chat.completions.create` recalls before and observes after.

    inject=False        skip the memory block.
    observe=False       skip judging the user turn.
    observe_first=True  judge the turn before recalling, so a correction in this very turn is already
                        applied to what the model sees (costs one extra ~100 ms hop up front).
    """
    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None)
    if completions is None or not hasattr(completions, "create"):
        raise TypeError("wrap() expects an OpenAI client with .chat.completions.create")
    opts = {"inject": inject, "observe": observe, "observe_first": observe_first, "limit": limit, "source": source}
    wrapped = _Completions(completions, mem, opts)
    proxy = _Proxy(client, "chat", _Proxy(chat, "completions", wrapped))
    proxy.completions = wrapped  # handy: client.completions.last_report
    return proxy


def make_client(**kwargs: Any) -> Any:
    """`openai.OpenAI(**kwargs)`, imported lazily with a clear error if the SDK is missing."""
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise ImportError("the openai package is not installed: pip install 'invalidate[openai]'") from e
    return OpenAI(**kwargs)
