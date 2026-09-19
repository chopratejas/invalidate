"""FactBlockAdapter against the REAL llama-index-core classes (0.14.24): FactExtractionMemoryBlock with a
MockLLM, and Memory.from_defaults() with its default in-memory SQLAlchemyChatStore. No network, no API key.
"""
from __future__ import annotations

import hashlib

import pytest
from conftest import CONTRADICT, FakeJudge, SUPERSEDE, UNCERTAIN

from invalidate.adapters import Governor, Reason
from invalidate.adapters.llamaindex import FactBlockAdapter, fact_blocks, fact_id, governed_facts, guard_put, user_texts
from invalidate.types import Status, now

llama = pytest.importorskip("llama_index.core.memory")
from llama_index.core.base.llms.types import ChatMessage  # noqa: E402
from llama_index.core.llms import MockLLM  # noqa: E402
from llama_index.core.memory import FactExtractionMemoryBlock, Memory, StaticMemoryBlock  # noqa: E402

FACTS = ["user prefers postgres", "user lives in berlin", "deploys run at 2pm UTC"]
PG = fact_id("user prefers postgres")


def _block(facts: list[str] | None = None) -> FactExtractionMemoryBlock:
    return FactExtractionMemoryBlock(llm=MockLLM(), facts=list(FACTS if facts is None else facts))


def _gov(adapter, fake: FakeJudge, **kw) -> Governor:
    return Governor(adapter, ":memory:", judge=fake, **kw)


def _reason(status: Status, event: str = "we migrated to sqlite") -> Reason:
    return Reason(status, status.value, event, "slack", "evt_x", 0.05, now())


def _prompt(memory: Memory) -> str:
    """The system message LlamaIndex would send: memory blocks rendered through the template (memory.py:660)."""
    msgs = memory.get()
    return "\n".join(m.content or "" for m in msgs if m.role.value == "system")


# -- pull / sync -------------------------------------------------------------------------
class TestPull:
    def test_round_trip_ids_are_content_addressed_and_text_verbatim(self, fake):
        block = _block()
        adapter = FactBlockAdapter(block)
        assert adapter.name == "llamaindex:ExtractedFacts"
        pulled = list(adapter.pull())
        assert [h.text for h in pulled] == FACTS
        assert pulled[0].id == hashlib.sha1(b"user prefers postgres").hexdigest()
        assert all(h.source == "llamaindex" and h.metadata == {} for h in pulled)
        gov = _gov(adapter, fake)
        assert gov.sync().added == 3
        assert gov.status_of(PG) is Status.ACTIVE
        assert gov.mem.get(gov.our_id(PG)).fact == "user prefers postgres"
        assert gov.sync().unchanged == 3  # idempotent

    def test_pull_dedups_and_skips_blank_strings(self):
        block = _block(["a", "a", "  ", "b"])
        assert [h.text for h in FactBlockAdapter(block).pull()] == ["a", "b"]

    def test_requires_a_facts_list(self):
        with pytest.raises(TypeError):
            FactBlockAdapter(StaticMemoryBlock(static_content="hello"))

    def test_fact_blocks_finds_the_fact_block_in_a_memory(self):
        block = _block()
        memory = Memory.from_defaults(session_id="s", memory_blocks=[StaticMemoryBlock(static_content="x"), block])
        assert fact_blocks(memory) == [block]


# -- flag ------------------------------------------------------------------------------------
class TestFlag:
    def test_flag_removes_the_fact_from_the_prompt_but_not_from_the_adapter(self, fake):
        block = _block()
        memory = Memory.from_defaults(session_id="s1", memory_blocks=[block], token_limit=2000)
        memory.put(ChatMessage(role="user", content="hi"))
        assert "<fact>user prefers postgres</fact>" in _prompt(memory)
        fake.script("postgres", SUPERSEDE)
        adapter = FactBlockAdapter(block)
        gov = _gov(adapter, fake)
        gov.sync()
        rep = gov.observe("we migrated to sqlite", source="slack")
        assert not rep.errors and [p.action for p in rep.pushes] == ["flag"]
        assert block.facts == ["user lives in berlin", "deploys run at 2pm UTC"]  # order of the others kept
        assert adapter.hidden == {PG: "user prefers postgres"}  # text kept, never rewritten
        assert adapter.receipts[PG]["invalidate_status"] == "superseded"
        assert adapter.receipts[PG]["invalidate_event"] == "we migrated to sqlite"
        prompt = _prompt(memory)
        assert "postgres" not in prompt and "<fact>user lives in berlin</fact>" in prompt
        # the hidden fact is still pulled, so the ledger row survives a resync
        pulled = {h.id: h for h in adapter.pull()}
        assert pulled[PG].text == "user prefers postgres" and pulled[PG].metadata == {"hidden": True}
        s = gov.sync()
        assert s.total == 3 and s.removed == 0 and s.unchanged == 3
        assert gov.status_of(PG) is Status.SUPERSEDED

    def test_keep_restores_the_fact_into_the_block(self, fake):
        block = _block()
        fake.script("postgres", CONTRADICT)
        adapter = FactBlockAdapter(block)
        gov = _gov(adapter, fake)
        gov.sync()
        gov.observe("we dropped postgres", source="slack")
        assert "user prefers postgres" not in block.facts
        gov.keep(PG)
        assert block.facts[-1] == "user prefers postgres" and adapter.hidden == {} and PG not in adapter.receipts
        assert gov.status_of(PG) is Status.ACTIVE

    def test_needs_review_is_hidden_by_default(self, fake):
        block = _block()
        fake.script("postgres", UNCERTAIN)
        adapter = FactBlockAdapter(block)
        gov = _gov(adapter, fake)
        gov.sync()
        gov.observe("maybe postgres is gone", source="slack")
        assert gov.status_of(PG) is Status.NEEDS_REVIEW
        assert "user prefers postgres" not in block.facts and PG in adapter.hidden

    def test_needs_review_stays_in_the_prompt_with_hide_review_false(self, fake):
        block = _block()
        fake.script("postgres", UNCERTAIN)
        adapter = FactBlockAdapter(block, hide_review=False)
        gov = _gov(adapter, fake)
        gov.sync()
        gov.observe("maybe postgres is gone", source="slack")
        assert "user prefers postgres" in block.facts and adapter.hidden == {}
        assert adapter.receipts[PG]["invalidate_status"] == "needs_review"
        assert governed_facts(block, gov) == ["user lives in berlin", "deploys run at 2pm UTC"]

    def test_flag_unknown_id_raises_and_is_reported_not_raised_by_the_governor(self, fake):
        adapter = FactBlockAdapter(_block())
        with pytest.raises(KeyError):
            adapter.flag("nope", _reason(Status.SUPERSEDED))
        with pytest.raises(KeyError):
            adapter.flag("nope", _reason(Status.ACTIVE))

    def test_re_extracted_hidden_fact_is_live_in_the_host_again(self, fake):
        block = _block()
        fake.script("postgres", SUPERSEDE)
        adapter = FactBlockAdapter(block)
        gov = _gov(adapter, fake)
        gov.sync()
        gov.observe("we migrated to sqlite", source="slack")
        block.facts.append("user prefers postgres")  # what fact.py:145-147 does when the LLM extracts it again
        pulled = {h.id: h for h in adapter.pull()}
        assert pulled[PG].metadata == {} and adapter.hidden == {}
        assert gov.sync().unchanged == 3 and gov.status_of(PG) is Status.SUPERSEDED  # the ledger remembers
        assert governed_facts(block, gov) == ["user lives in berlin", "deploys run at 2pm UTC"]

    def test_hidden_sidecar_can_be_restored_into_a_new_adapter(self, fake):
        block = _block(["user lives in berlin"])
        adapter = FactBlockAdapter(block, hidden={PG: "user prefers postgres"})
        assert {h.id for h in adapter.pull()} == {fact_id("user lives in berlin"), PG}
        gov = _gov(adapter, fake)
        assert gov.sync().added == 2


# -- delete / insert -----------------------------------------------------------------------
class TestDeleteInsert:
    def test_delete_mode_removes_the_fact(self, fake):
        block = _block()
        fake.script("postgres", CONTRADICT)
        adapter = FactBlockAdapter(block)
        gov = _gov(adapter, fake, mode="delete")
        gov.sync()
        rep = gov.observe("we dropped postgres", source="slack")
        assert rep.pushes[0].action == "delete"
        assert block.facts == ["user lives in berlin", "deploys run at 2pm UTC"]
        assert adapter.hidden == {} and PG not in adapter.receipts
        assert gov.status_of(PG) is Status.CONTRADICTED

    def test_forget_removes_a_hidden_fact_too(self, fake):
        block = _block()
        fake.script("postgres", SUPERSEDE)
        adapter = FactBlockAdapter(block)
        gov = _gov(adapter, fake)
        gov.sync()
        gov.observe("we migrated to sqlite", source="slack")
        gov.forget(PG)
        assert adapter.hidden == {} and "user prefers postgres" not in block.facts
        assert {h.id for h in adapter.pull()} == {fact_id(f) for f in FACTS[1:]}

    def test_successor_is_appended_verbatim(self, fake):
        block = _block()
        fake.script("prefers postgres", SUPERSEDE)
        adapter = FactBlockAdapter(block)
        gov = _gov(adapter, fake, successors=True)
        gov.sync()
        rep = gov.observe("We migrated to SQLite.", source="slack")
        assert rep.successor_host_id == fact_id("We migrated to SQLite.")
        assert block.facts == ["user lives in berlin", "deploys run at 2pm UTC", "We migrated to SQLite."]
        assert [p.action for p in rep.pushes] == ["flag", "insert"]
        assert gov.mem.get(gov.our_id(PG)).superseded_by == gov.our_id(rep.successor_host_id)
        assert gov.sync().unchanged == 4  # 3 originals (1 hidden) + successor, all stable

    def test_insert_is_idempotent_and_unhides(self):
        adapter = FactBlockAdapter(_block(["a"]))
        hid = adapter.insert("a", "slack", {})
        assert hid == fact_id("a") and adapter.block.facts == ["a"]
        adapter.hidden[fact_id("b")] = "b"
        adapter.insert("b", "slack", {})
        assert adapter.block.facts == ["a", "b"] and adapter.hidden == {}


# -- read side and write guard ---------------------------------------------------------
class TestHelpers:
    def test_governed_facts_filters_by_the_ledger_without_touching_the_block(self, fake):
        block = _block()
        fake.script("postgres", SUPERSEDE)
        gov = _gov(FactBlockAdapter(block), fake, mode="ledger")  # ledger mode: the block is never written
        gov.sync()
        gov.observe("we migrated to sqlite", source="slack")
        assert block.facts == FACTS
        assert governed_facts(block, gov) == ["user lives in berlin", "deploys run at 2pm UTC"]

    def test_user_texts_reads_chat_messages_dicts_and_strings(self):
        assert user_texts(ChatMessage(role="user", content="a")) == ["a"]
        assert user_texts(ChatMessage(role="assistant", content="b")) == []
        assert user_texts([{"role": "user", "content": "c"}, "d"]) == ["c", "d"]
        assert user_texts(message=ChatMessage(role="user", content=" ")) == []

    def test_guard_put_judges_user_messages_before_storing_them(self, fake):
        block = _block()
        memory = Memory.from_defaults(session_id="s2", memory_blocks=[block], token_limit=2000)
        fake.script("postgres", SUPERSEDE)
        adapter = FactBlockAdapter(block)
        gov = _gov(adapter, fake)
        gov.sync()
        put = guard_put(memory, gov)
        put(ChatMessage(role="user", content="we migrated to sqlite"))
        assert {e.text for e, _ in fake.observe_calls} == {"we migrated to sqlite"}
        assert gov.status_of(PG) is Status.SUPERSEDED and PG in adapter.hidden
        assert [m.content for m in memory.get_all()] == ["we migrated to sqlite"]  # stored after the judgment
        assert "postgres" not in _prompt(memory)
        put(ChatMessage(role="assistant", content="ok"))  # assistant turns are not events
        assert len({e.id for e, _ in fake.observe_calls}) == 1
