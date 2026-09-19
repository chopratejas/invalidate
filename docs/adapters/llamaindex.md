# LlamaIndex

Governs the facts a `FactExtractionMemoryBlock` renders into the prompt of a `llama_index.core.memory.Memory`.
LlamaIndex keeps its facts; invalidate keeps the ledger and moves stale facts out of the prompt without losing them.

## Install

```sh
pip install invalidate llama-index-core     # verified against llama-index-core 0.14.24
```

## Usage

```python
from llama_index.core.memory import Memory, FactExtractionMemoryBlock
from invalidate.adapters import Governor
from invalidate.adapters.llamaindex import FactBlockAdapter, governed_facts, guard_put

block = FactExtractionMemoryBlock(llm=llm)                     # the block your agent already uses
memory = Memory.from_defaults(session_id="alice", memory_blocks=[block])
adapter = FactBlockAdapter(block)
gov = Governor(adapter, "ledger.db", mode="flag", successors=True)
gov.sync()                                                     # block.facts -> ledger (content-addressed ids)
report = gov.observe("we migrated to SQLite", source="slack")  # judge; stale facts leave block.facts
print(report.summary(), adapter.hidden)                        # {sha1: "user prefers Postgres"}
put = guard_put(memory, gov)                                   # judge-before-write memory.put()
put(ChatMessage(role="user", content="we're on Postgres 16 now"))
```

## What each push does in the block

| action | effect on `FactExtractionMemoryBlock` | notes |
|---|---|---|
| pull | reads `block.facts` (fact.py:82) | one memory per distinct string, id `sha1(text)`; hidden facts are pulled too with `metadata={"hidden": True}` so `sync()` keeps their rows |
| flag | superseded / contradicted / needs_review: the string is removed from `block.facts` and kept in `adapter.hidden[id]`, receipt in `adapter.receipts[id]`; ACTIVE (`gov.keep()`): moved back | text never rewritten; `hide_review=False` leaves needs_review facts in the prompt |
| delete | removed from `block.facts` and from `adapter.hidden` | mode `"delete"` only, for contradicted / superseded; `gov.forget()` |
| insert | `block.facts.append(text)` unless already present; returns `sha1(text)` | successors are verbatim, exactly as the block would store an extracted fact |

`governed_facts(block, gov)` returns `block.facts` minus dead (and, by default, reviewed) facts according to the
ledger, without touching the block. `fact_blocks(memory)` lists the fact blocks in `memory.memory_blocks`.
`guard_put(memory, gov)` wraps `memory.put` with `gov.guard`, judging every user-role `ChatMessage` before
LlamaIndex stores it.

## Why flag removes the fact from the list

`FactExtractionMemoryBlock` renders every string in `facts` into the prompt verbatim
(`"\n".join(f"<fact>{fact}</fact>" ...)`, fact.py:117). There is no per-fact metadata, no status field and no filter
hook, and the pydantic model forbids extra attributes (`BaseMemoryBlock.model_config` is
`ConfigDict(arbitrary_types_allowed=True)` with no `extra="allow"`, memory.py:111; setting one raises `ValueError`).
So the only way to keep a flagged fact out of the prompt is for it not to be in the list, and the only place to keep it
is the adapter. `Memory` never persists block contents either (`sql_store` stores chat messages only and is
`exclude=True`, memory.py:250), so `adapter.hidden` has exactly the lifetime of `block.facts`.

## Signatures mirrored (llama-index-core 0.14.24)

| where | what |
|---|---|
| `memory/memory_blocks/fact.py:67` | `class FactExtractionMemoryBlock(BaseMemoryBlock[str])` |
| `fact.py:75` / `:78` / `:82` | `name: str = "ExtractedFacts"`, `llm: LLM` (default `Settings.llm`; `MockLLM` works), `facts: List[str]` |
| `fact.py:117` | `_aget()` joins `<fact>{fact}</fact>` lines |
| `fact.py:145-147` | `_aput()` appends new facts unless an exact match is already in `facts` |
| `fact.py:166` | condense replaces `facts` wholesale when `len(facts) > max_facts` |
| `memory/memory.py:111` | `BaseMemoryBlock.model_config = ConfigDict(arbitrary_types_allowed=True)` |
| `memory.py:217` | `Memory.memory_blocks: List[BaseMemoryBlock]` |
| `memory.py:250` | `Memory.sql_store: AsyncDBChatStore = Field(exclude=True)` (messages only) |
| `memory.py:460-463`, `:568`, `:51-85` | `block.aget(...)` per block, `str` results wrapped in a `TextBlock`, rendered by `DEFAULT_MEMORY_BLOCKS_TEMPLATE` into `<memory><ExtractedFacts>...` |
| `memory.py:788-797` | flush: `block.aput(messages_to_flush, from_short_term_memory=True, session_id=...)` |
| `memory.py:811`, `:859` | `Memory.aput(message: ChatMessage)`, `Memory.put(message)` |

## Limits

- **Persistence is yours.** LlamaIndex does not persist `block.facts`; if you serialize the block, also serialize
  `adapter.hidden` and pass it back as `FactBlockAdapter(block, hidden=...)`. `adapter.receipts` is in-process only;
  the durable receipt is the ledger (`gov.status_of(id)`, `gov.mem.get(gov.our_id(id))`).
- **Re-extraction.** `_aput` only dedups against the live list, so the block's LLM can re-extract a hidden fact into
  `block.facts`. The next `pull()`/`sync()` treats it as live in the host again (dropped from `hidden`); its ledger
  status is unchanged, so `governed_facts()` still excludes it and the next `observe()` that touches it hides it again.
- **Condense.** When `len(facts) > max_facts` the LLM rewrites the list (fact.py:166). Rewritten strings get new ids;
  `sync()` records them as new claims and marks the vanished ids deleted, like any host rewrite.
- **Ids are content-addressed.** Two blocks holding the same string share an id inside one adapter's namespace; the
  adapter dedups on pull. Use `name=` to give each block its own namespace.
- **`VectorMemoryBlock` is not governed.** Its `_aput` stores one `TextNode` per flushed batch of messages wrapped
  as `<message role='...'>...</message>` (vector.py:190-201), so the unit is a conversation chunk, not a fact; and
  `BasePydanticVectorStore` has no store-agnostic enumerate or metadata-update API (`get_nodes` and `delete_nodes`
  raise `NotImplementedError` by default, `vector_stores/types.py:346`, `:395`; `add` at `:363` is the only write).
  The in-core `SimpleVectorStore` has `stores_text = False` (`vector_stores/simple.py:77`) and is rejected by
  `validate_vector_store` (vector.py:71-74). For a concrete store that can list, patch and delete nodes, use
  `adapters.vectorstore.CallableVectorStoreAdapter` with your own callables.
- `StaticMemoryBlock` is constant instructions, not memories; it is not governed.
