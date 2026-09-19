"""LlamaIndex adapter: govern the facts a `FactExtractionMemoryBlock` renders into the prompt.

    from llama_index.core.memory import Memory, FactExtractionMemoryBlock
    from invalidate.adapters import Governor
    from invalidate.adapters.llamaindex import FactBlockAdapter, governed_facts, guard_put

    block = FactExtractionMemoryBlock(llm=llm)
    memory = Memory.from_defaults(session_id="alice", memory_blocks=[block])
    adapter = FactBlockAdapter(block)
    gov = Governor(adapter, "ledger.db", mode="flag", successors=True)
    gov.sync()
    gov.observe("we migrated to SQLite", source="slack")   # stale facts leave the prompt, live in adapter.hidden
    live = governed_facts(block, gov)                       # block.facts minus dead/reviewed, by the ledger
    put = guard_put(memory, gov)                            # judge user messages before memory.put()

Nothing from llama_index is imported here; the adapter relies on the block you pass in (any object with a
`facts: list[str]` attribute works). Signatures mirrored (llama-index-core 0.14.24):

  memory_blocks/fact.py:67   class FactExtractionMemoryBlock(BaseMemoryBlock[str])
  memory_blocks/fact.py:75   name: str = "ExtractedFacts"
  memory_blocks/fact.py:78   llm: LLM = Field(default_factory=get_default_llm)   (Settings.llm; MockLLM works)
  memory_blocks/fact.py:82   facts: List[str] = Field(default_factory=list)       <- the whole store; plain strings
  memory_blocks/fact.py:117  _aget() -> "\\n".join(f"<fact>{fact}</fact>" for fact in self.facts)
                             every string in `facts` reaches the prompt verbatim; there is no per-fact metadata,
                             no status field, no filter hook: the only way to keep a fact out is to not be in the list
  memory_blocks/fact.py:145  _aput(): `if fact not in self.facts: self.facts.append(fact)` (exact-match dedup)
  memory_blocks/fact.py:166  condense: `self.facts = condensed_facts` REPLACES the list when len > max_facts
  memory.py:111              BaseMemoryBlock.model_config = ConfigDict(arbitrary_types_allowed=True)
                             (no extra="allow": setting an unknown attribute on a block raises ValueError, verified)
  memory.py:217              Memory.memory_blocks: List[BaseMemoryBlock]
  memory.py:250              Memory.sql_store: AsyncDBChatStore = Field(exclude=True)  -> stores chat MESSAGES only;
                             block contents (facts) are never persisted by Memory, they live on the block object
  memory.py:460-463          Memory._get_memory_blocks_content(): `await block.aget(block_input, session_id=...)`
  memory.py:568              a str block result becomes `(block.name, [TextBlock(text=content)])`
  memory.py:51-85            DEFAULT_MEMORY_BLOCKS_TEMPLATE renders `<memory><{{ block_name }}>...</...></memory>`
  memory.py:788-797          flush: `block.aput(messages_to_flush, from_short_term_memory=True, session_id=...)`
  memory.py:811,859          Memory.aput(message: ChatMessage) / Memory.put(message)

Why flag() removes the fact from `block.facts` and keeps it in `adapter.hidden`
-----------------------------------------------------------------------------
A flagged fact must stay out of the prompt without being lost. The block offers no metadata per fact and
forbids extra attributes (memory.py:111), so the receipt cannot ride on the block. The adapter therefore
moves the string out of `block.facts` into its own `hidden` dict (host id -> text) with the receipt in
`receipts`; `pull()` still yields hidden facts (metadata `{"hidden": True}`) so the ledger row survives
`sync()`, and a flag with status ACTIVE (`gov.keep()`) moves the text back. The text is never rewritten.
Because Memory does not persist blocks either, the sidecar has exactly the block's lifetime: persist
`adapter.hidden` next to `block.facts` if you persist the block.

VectorMemoryBlock is not governed here. From memory_blocks/vector.py: `_aput` stores ONE `TextNode` per
flushed batch of messages, wrapped as `<message role='...'>...</message>` (vector.py:190-201), so the unit
is a conversation chunk, not a fact; `BasePydanticVectorStore` (vector_stores/types.py:334) has no
enumerate-all or metadata-update API (`get_nodes` :346 and `delete_nodes` :395 raise NotImplementedError
by default, `add` :363 is the only write) and the in-core `SimpleVectorStore` has `stores_text = False`
(vector_stores/simple.py:77), which `validate_vector_store` rejects (vector.py:71-74). For a concrete
store that can list, patch and delete nodes, use `adapters.vectorstore.CallableVectorStoreAdapter`.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from typing import Any

from .base import Governor, HostMemory, Reason
from ..types import Status


def fact_id(text: str) -> str:
    """Content-addressed host id for a fact string: sha1 of the exact text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _facts_of(block: Any) -> list[str]:
    facts = getattr(block, "facts", None)
    if not isinstance(facts, list):
        raise TypeError(f"{type(block).__name__} has no `facts: list[str]`; pass a FactExtractionMemoryBlock")
    return facts


class FactBlockAdapter:
    """Governs one FactExtractionMemoryBlock. Host ids are `fact_id(text)`.

    pull   -> one HostMemory per distinct string in `block.facts`, plus every hidden fact (`metadata={"hidden": True}`).
    flag   -> status ACTIVE: move the fact from `hidden` back into `block.facts` (appended).
              any other status (superseded, contradicted, needs_review): remove it from `block.facts`, keep it in
              `hidden[host_id]` with `receipts[host_id] = reason.as_metadata()`. Set `hide_review=False` to leave
              needs_review facts in the prompt (mirrors `Governor.filter(include_review=True)`).
    delete -> remove the fact from `block.facts` and from `hidden`.
    insert -> append the text to `block.facts` (no-op if already present) and return its id.
    """

    def __init__(
        self,
        block: Any,
        *,
        name: str | None = None,
        hide_review: bool = True,
        hidden: dict[str, str] | None = None,
        source: str = "llamaindex",
        metadata_prefix: str = "invalidate_",
    ) -> None:
        _facts_of(block)
        self.block = block
        self.name = name or f"llamaindex:{getattr(block, 'name', 'facts')}"
        self.hide_review = hide_review
        self.hidden: dict[str, str] = dict(hidden or {})
        self.receipts: dict[str, dict[str, Any]] = {}
        self.source = source
        self.prefix = metadata_prefix

    # -- host access ------------------------------------------------------------------
    @property
    def facts(self) -> list[str]:
        return _facts_of(self.block)

    def _find(self, host_id: str) -> str | None:
        for f in self.facts:
            if fact_id(f) == host_id:
                return f
        return None

    def _remove_all(self, host_id: str) -> str | None:
        """Drop every copy of the fact from `block.facts`; return its text if any was there."""
        found = None
        kept = []
        for f in self.facts:
            if fact_id(f) == host_id:
                found = f
            else:
                kept.append(f)
        if found is not None:
            self.facts[:] = kept  # in place: the block holds the same list object
        return found

    # -- Adapter protocol -------------------------------------------------------------
    def pull(self) -> Iterable[HostMemory]:
        out: list[HostMemory] = []
        seen: set[str] = set()
        for f in self.facts:
            if not isinstance(f, str) or not f.strip():
                continue
            hid = fact_id(f)
            if hid in seen:
                continue
            seen.add(hid)
            # The block's LLM re-extracted a fact we hid (fact.py:145 only dedups against the live list):
            # the host put it back, so it is live in the host again; the ledger keeps its status.
            self.hidden.pop(hid, None)
            out.append(HostMemory(id=hid, text=f, source=self.source))
        for hid, f in self.hidden.items():
            if hid not in seen:
                out.append(HostMemory(id=hid, text=f, source=self.source, metadata={"hidden": True}))
        return out

    def flag(self, host_id: str, reason: Reason) -> None:
        receipt = reason.as_metadata(self.prefix)
        hide = reason.status is not Status.ACTIVE and (self.hide_review or reason.status is not Status.NEEDS_REVIEW)
        if hide:
            text = self._remove_all(host_id)
            if text is None:
                text = self.hidden.get(host_id)
            if text is None:
                raise KeyError(f"{self.name}: no fact {host_id!r}")
            self.hidden[host_id] = text
            self.receipts[host_id] = receipt
            return
        text = self.hidden.pop(host_id, None)
        if text is not None:
            if self._find(host_id) is None:
                self.facts.append(text)
        elif self._find(host_id) is None:
            raise KeyError(f"{self.name}: no fact {host_id!r}")
        if reason.status is Status.ACTIVE:
            self.receipts.pop(host_id, None)
        else:
            self.receipts[host_id] = receipt  # needs_review left in the prompt (hide_review=False)

    def delete(self, host_id: str, reason: Reason) -> None:
        self._remove_all(host_id)
        self.hidden.pop(host_id, None)
        self.receipts.pop(host_id, None)

    def insert(self, text: str, source: str, metadata: dict[str, Any]) -> str:
        hid = fact_id(text)
        self.hidden.pop(hid, None)
        if self._find(hid) is None:
            self.facts.append(text)
        return hid


# -- helpers ----------------------------------------------------------------------------
def fact_blocks(memory: Any) -> list[Any]:
    """The blocks in `memory.memory_blocks` that hold a `facts: list[str]` (FactExtractionMemoryBlock)."""
    return [b for b in getattr(memory, "memory_blocks", []) or [] if isinstance(getattr(b, "facts", None), list)]


def governed_facts(block: Any, gov: Governor, *, include_review: bool = True, annotate: bool = False) -> list[Any]:
    """`block.facts` with facts that are dead (or under review) in the ledger dropped. Read-only: use this
    when you render facts yourself or want to double-check what the adapter left in the block.

    `annotate=True` keeps every fact and returns `[(fact, note), ...]`: facts are plain strings, so the
    label (`Governor.annotate`'s, None when live) rides alongside instead of inside."""
    if annotate:
        return gov.annotate(_facts_of(block), id_of=fact_id)
    return gov.filter(_facts_of(block), id_of=fact_id, include_review=include_review)


def user_texts(*args: Any, **kwargs: Any) -> list[str]:
    """The event texts inside a `Memory.put(message)` / `put_messages(messages)` call: user-role ChatMessage
    content (or plain strings / {role, content} dicts)."""
    x = args[0] if args else kwargs.get("message", kwargs.get("messages", ""))
    items = x if isinstance(x, list) else [x]
    out: list[str] = []
    for msg in items:
        if isinstance(msg, str):
            out.append(msg)
            continue
        if isinstance(msg, dict):
            role, content = msg.get("role", "user"), msg.get("content")
        else:
            role, content = getattr(msg, "role", "user"), getattr(msg, "content", None)
        role = getattr(role, "value", role)
        if role == "user" and isinstance(content, str) and content.strip():
            out.append(content)
    return out


def guard_put(memory: Any, gov: Governor, *, source: str = "user") -> Callable[..., Any]:
    """`memory.put(message)` wrapped so every user message is judged against the ledger (and flags pushed
    into the block) before LlamaIndex stores it; the block is re-synced afterwards."""
    return gov.guard(memory.put, text_of=user_texts, source=source)


__all__ = ["FactBlockAdapter", "fact_id", "fact_blocks", "governed_facts", "guard_put", "user_texts"]
