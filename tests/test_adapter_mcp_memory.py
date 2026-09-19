"""McpMemoryAdapter over a real temp JSONL file, plus Governor end-to-end with the fake judge."""
from __future__ import annotations

import datetime as dt
import json
import os

import pytest

from conftest import CONTRADICT, SUPERSEDE, UNCERTAIN, FakeJudge
from invalidate.adapters import Governor, Reason
from invalidate.adapters.markdown import content_hash
from invalidate.adapters.mcp_memory import McpMemoryAdapter, governed_read, split_marker
from invalidate.types import Status

ALICE = {"type": "entity", "name": "alice", "entityType": "person",
         "observations": ["user prefers Postgres", "deploys run at 2pm UTC"]}
BILLING = {"type": "entity", "name": "billing", "entityType": "service",
           "observations": ["Alice owns the billing service", "user prefers Postgres"]}
OWNS = {"type": "relation", "from": "alice", "to": "billing", "relationType": "owns"}
FILE = "\n".join(json.dumps(x, separators=(",", ":")) for x in (ALICE, BILLING, OWNS)) + "\n"


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return str(path)


def read(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def load_like_server(path):
    """Parse the file the way index.ts loadGraph does: blank/malformed lines skipped, typed lines validated."""
    entities, relations = [], []
    for line in read(path).split("\n"):
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("type") == "entity":
            assert isinstance(obj["name"], str) and isinstance(obj["entityType"], str)
            assert all(isinstance(o, str) for o in obj["observations"])
            entities.append(obj)
        elif obj.get("type") == "relation":
            assert all(isinstance(obj[k], str) for k in ("from", "to", "relationType"))
            relations.append(obj)
    return entities, relations


def reason(status=Status.SUPERSEDED, text="we migrated to SQLite", source="slack", p=0.12, at=0.0):
    return Reason(status, status.value, text, source, "evt_1", p, at)


def ids(adapter):
    return {hm.text: hm.id for hm in adapter.pull()}


@pytest.fixture
def path(tmp_path):
    return write(tmp_path / "memory.jsonl", FILE)


@pytest.fixture
def adapter(path):
    return McpMemoryAdapter(path)


# -- pull ---------------------------------------------------------------------------------
def test_pull_round_trip(adapter):
    got = list(adapter.pull())
    assert [hm.text for hm in got] == [
        "user prefers Postgres", "deploys run at 2pm UTC", "Alice owns the billing service",
        "user prefers Postgres", "alice owns billing",
    ]
    assert all(hm.source == "mcp_memory" and hm.kind == "fact" for hm in got)
    a = got[0]
    assert a.id == f"alice#{content_hash('user prefers Postgres')}"
    assert a.metadata == {"record": "observation", "entity": "alice", "entity_type": "person", "index": 0}
    rel = got[-1]
    assert rel.id == McpMemoryAdapter.relation_id("alice", "owns", "billing") == f"relation#{content_hash('alice owns billing')}"
    assert rel.metadata == {"record": "relation", "from": "alice", "to": "billing", "relation_type": "owns"}


def test_same_text_in_two_entities_gets_two_ids(adapter):
    got = ids(adapter)  # dict keeps the last, so check the full list
    assert len({hm.id for hm in adapter.pull() if hm.text == "user prefers Postgres"}) == 2
    assert got["user prefers Postgres"] == f"billing#{content_hash('user prefers Postgres')}"


def test_ids_stable_across_reordering_and_whitespace(tmp_path):
    p = write(tmp_path / "m.jsonl", FILE)
    before = {hm.id for hm in McpMemoryAdapter(p).pull()}
    shuffled = [OWNS, {**BILLING, "observations": list(reversed(BILLING["observations"]))},
                {**ALICE, "observations": ["  user   prefers Postgres ", "deploys run at 2pm UTC"]}]
    write(p, "\n".join(json.dumps(x) for x in shuffled) + "\n")
    assert {hm.id for hm in McpMemoryAdapter(p).pull()} == before


def test_duplicate_observations_collapse_and_relations_can_be_hidden(tmp_path):
    p = write(tmp_path / "m.jsonl", json.dumps({**ALICE, "observations": ["user prefers Postgres"] * 3}) + "\n" + json.dumps(OWNS) + "\n")
    assert [hm.text for hm in McpMemoryAdapter(p).pull()] == ["user prefers Postgres", "alice owns billing"]
    assert [hm.text for hm in McpMemoryAdapter(p, relations=False).pull()] == ["user prefers Postgres"]


def test_missing_file_pulls_nothing(tmp_path):
    assert list(McpMemoryAdapter(tmp_path / "nope.jsonl").pull()) == []


def test_path_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv("MEMORY_FILE_PATH", raising=False)
    with pytest.raises(ValueError):
        McpMemoryAdapter()
    monkeypatch.setenv("MEMORY_FILE_PATH", str(tmp_path / "m.jsonl"))
    assert McpMemoryAdapter().path == str(tmp_path / "m.jsonl")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert McpMemoryAdapter("~/m.jsonl").path == str(tmp_path / "m.jsonl")


# -- flag ---------------------------------------------------------------------------------
def test_flag_rewrites_observation_server_still_loads_it(adapter, path):
    hid = f"alice#{content_hash('user prefers Postgres')}"
    adapter.flag(hid, reason())
    entities, relations = load_like_server(path)
    today = dt.date.today().isoformat()  # reason.at == 0 falls back to today
    marked = f"user prefers Postgres [invalidate: superseded by “we migrated to SQLite” (slack, still true 12%), {today}]"
    assert entities[0]["observations"] == [marked, "deploys run at 2pm UTC"]
    assert entities[1] == BILLING and relations == [OWNS]  # billing's copy has a different id: untouched
    assert split_marker(marked) == ("user prefers Postgres", f"superseded by “we migrated to SQLite” (slack, still true 12%), {today}")


def test_flag_keeps_id_and_reports_status_on_next_pull(adapter):
    hid = f"alice#{content_hash('user prefers Postgres')}"
    adapter.flag(hid, reason(at=1_758_240_000.0))  # 2025-09-19 UTC
    hm = next(h for h in adapter.pull() if h.id == hid)
    assert hm.text == "user prefers Postgres"
    assert hm.metadata["invalidate_status"] == "superseded"
    assert hm.metadata["invalidate_note"] == "superseded by “we migrated to SQLite” (slack, still true 12%), 2025-09-19"
    assert "invalidate_status" not in next(h for h in adapter.pull() if h.text == "deploys run at 2pm UTC").metadata


@pytest.mark.parametrize("status,expect", [
    (Status.CONTRADICTED, "contradicted"), (Status.NEEDS_REVIEW, "needs_review"), (Status.FROZEN, "frozen"),
])
def test_flag_status_words_parse_back(adapter, status, expect):
    hid = f"alice#{content_hash('deploys run at 2pm UTC')}"
    adapter.flag(hid, reason(status))
    assert next(h for h in adapter.pull() if h.id == hid).metadata["invalidate_status"] == expect


def test_flag_twice_replaces_marker_and_active_strips_it(adapter, path):
    hid = f"alice#{content_hash('user prefers Postgres')}"
    adapter.flag(hid, reason())
    adapter.flag(hid, reason(Status.CONTRADICTED, text="Postgres is gone"))
    obs = load_like_server(path)[0][0]["observations"][0]
    assert obs.count("[invalidate:") == 1 and "contradicted by “Postgres is gone”" in obs
    adapter.flag(hid, Reason(Status.ACTIVE, "restored", "kept by a human", "human", "", 1.0, 0.0))
    assert read(path) == FILE


def test_flag_escapes_brackets_in_event_text(adapter):
    hid = f"alice#{content_hash('user prefers Postgres')}"
    adapter.flag(hid, reason(text="see [ticket-12] and [invalidate: nope]"))
    hm = next(h for h in adapter.pull() if h.id == hid)
    assert hm.text == "user prefers Postgres" and hm.metadata["invalidate_status"] == "superseded"


def test_flag_relation(adapter, path):
    hid = McpMemoryAdapter.relation_id("alice", "owns", "billing")
    adapter.flag(hid, reason(Status.CONTRADICTED, text="Bob owns billing now"))
    rel = load_like_server(path)[1][0]
    assert rel["from"] == "alice" and rel["to"] == "billing"
    assert rel["relationType"].startswith("owns [invalidate: contradicted by “Bob owns billing now”")
    hm = next(h for h in adapter.pull() if h.id == hid)
    assert hm.text == "alice owns billing" and hm.metadata["invalidate_status"] == "contradicted"
    assert hm.metadata["relation_type"] == "owns"


def test_flag_annotate_false_is_noop(path):
    ad = McpMemoryAdapter(path, annotate=False)
    ad.flag(f"alice#{content_hash('user prefers Postgres')}", reason())
    assert read(path) == FILE


def test_flag_unknown_id_raises(adapter, path):
    with pytest.raises(KeyError):
        adapter.flag("alice#0000000000", reason())
    with pytest.raises(KeyError):
        adapter.flag(f"nobody#{content_hash('user prefers Postgres')}", reason())
    assert read(path) == FILE


# -- delete -------------------------------------------------------------------------------
def test_delete_removes_only_that_observation(adapter, path):
    adapter.delete(f"alice#{content_hash('user prefers Postgres')}", reason())
    entities, relations = load_like_server(path)
    assert entities[0]["observations"] == ["deploys run at 2pm UTC"]
    assert entities[1] == BILLING and relations == [OWNS]
    assert len(list(adapter.pull())) == 4


def test_delete_relation_removes_the_line(adapter, path):
    adapter.delete(McpMemoryAdapter.relation_id("alice", "owns", "billing"), reason())
    assert load_like_server(path)[1] == []
    assert read(path).count("\n") == 2


def test_delete_last_observation_keeps_the_entity(tmp_path):
    p = write(tmp_path / "m.jsonl", json.dumps({**ALICE, "observations": ["user prefers Postgres"]}) + "\n")
    McpMemoryAdapter(p).delete(f"alice#{content_hash('user prefers Postgres')}", reason())
    assert load_like_server(p)[0] == [{**ALICE, "observations": []}]


# -- insert -------------------------------------------------------------------------------
def test_insert_into_named_entity_dedupes(adapter, path):
    hid = adapter.insert("we migrated to SQLite", "slack", {"entity": "alice"})
    assert hid == f"alice#{content_hash('we migrated to SQLite')}"
    obs = load_like_server(path)[0][0]["observations"]
    assert len(obs) == 3 and obs[2].startswith("we migrated to SQLite [invalidate: from slack, ")
    assert adapter.insert("we migrated to SQLite", "slack", {"entity": "alice"}) == hid
    assert len(load_like_server(path)[0][0]["observations"]) == 3
    hm = next(h for h in adapter.pull() if h.id == hid)
    assert hm.text == "we migrated to SQLite" and hm.metadata["invalidate_from"].startswith("from slack, ")
    assert "invalidate_status" not in hm.metadata


def test_insert_follows_superseded_entity_else_default(adapter, path):
    one = adapter.insert("we migrated to SQLite", "slack", {"invalidate_supersedes": [f"alice#{content_hash('user prefers Postgres')}"]})
    assert one.startswith("alice#")
    two = adapter.insert("two entities changed", "slack",
                         {"invalidate_supersedes": [f"alice#{content_hash('x')}", f"billing#{content_hash('y')}"]})
    assert two.startswith("invalidate#")
    entities, relations = load_like_server(path)
    assert [e["name"] for e in entities] == ["alice", "billing", "invalidate"]
    assert entities[2]["entityType"] == "invalidate" and relations == [OWNS]
    assert read(path).index('"name":"invalidate"') < read(path).index('"type":"relation"')
    three = McpMemoryAdapter(path, follow_superseded=False, default_entity="notes", default_entity_type="note").insert(
        "no following", "slack", {"invalidate_supersedes": [f"alice#{content_hash('x')}"]})
    assert three.startswith("notes#")
    notes = load_like_server(path)[0][3]
    assert notes == {"type": "entity", "name": "notes", "entityType": "note",
                     "observations": [f"no following [invalidate: from slack, {dt.date.today().isoformat()}]"]}


def test_insert_creates_missing_file(tmp_path):
    p = tmp_path / "deep" / "memory.jsonl"
    hid = McpMemoryAdapter(p).insert("brand new fact", "user", {})
    assert hid == f"invalidate#{content_hash('brand new fact')}"
    entities, _ = load_like_server(p)
    assert entities[0]["name"] == "invalidate" and read(p).endswith("\n")


def test_insert_terminates_unterminated_last_line(tmp_path):
    p = write(tmp_path / "m.jsonl", json.dumps(ALICE))  # no trailing newline
    McpMemoryAdapter(p).insert("brand new fact", "user", {})
    assert read(p).count("\n") == 2 and len(load_like_server(p)[0]) == 2


# -- preservation ---------------------------------------------------------------------------
def test_unknown_lines_extra_fields_and_bytes_preserved(tmp_path):
    raw = (
        json.dumps({"type": "meta", "version": 3}) + "\n"
        + "\n"
        + "{not json at all\n"
        + '{"type":"entity","name":"alice","entityType":"person","observations":["user prefers Postgres"],"createdAt":"2025-01-01"}\r\n'
        + json.dumps({"type": "entity", "name": "bob", "entityType": "person", "observations": ["Bob likes ünïcode"]}, ensure_ascii=False) + "\n"
        + '{"type":"relation","from":"alice","to":"bob","relationType":"knows"}'
    )
    p = write(tmp_path / "m.jsonl", raw)
    ad = McpMemoryAdapter(p)
    assert [hm.text for hm in ad.pull()] == ["user prefers Postgres", "Bob likes ünïcode", "alice knows bob"]
    ad.flag(f"alice#{content_hash('user prefers Postgres')}", reason())
    lines = read(p).split("\n")
    assert lines[0] == json.dumps({"type": "meta", "version": 3}) and lines[1] == "" and lines[2] == "{not json at all"
    alice = json.loads(lines[3].rstrip("\r"))
    assert lines[3].endswith("\r") and alice["createdAt"] == "2025-01-01"
    assert list(alice) == ["type", "name", "entityType", "observations", "createdAt"]
    assert alice["observations"][0].startswith("user prefers Postgres [invalidate: superseded by")
    assert lines[4] == raw.split("\n")[4] and "ünïcode" in lines[4]
    assert lines[5] == raw.split("\n")[5]
    ad.delete(f"alice#{content_hash('user prefers Postgres')}", reason())
    assert read(p).split("\n")[:3] == raw.split("\n")[:3]


def test_write_is_atomic_and_leaves_no_temp_files(adapter, path):
    adapter.flag(f"alice#{content_hash('user prefers Postgres')}", reason())
    assert sorted(os.listdir(os.path.dirname(path))) == ["memory.jsonl"]


# -- governed_read ------------------------------------------------------------------------
def test_governed_read_hides_dead_and_review(path, fake: FakeJudge):
    fake.script("user prefers Postgres", SUPERSEDE).script("deploys", UNCERTAIN).script("alice owns billing", CONTRADICT)
    with Governor(McpMemoryAdapter(path), ":memory:", judge=fake) as gov:
        gov.sync()
        gov.observe("we migrated to SQLite", source="slack")
        g = governed_read(None, gov)
        assert g["entities"][0]["observations"] == []
        assert g["entities"][1]["observations"] == ["Alice owns the billing service"]
        assert g["relations"] == []
        g = governed_read(path, gov, include_review=True)
        assert len(g["entities"][0]["observations"]) == 1
        assert g["entities"][0]["observations"][0].startswith("deploys run at 2pm UTC [invalidate: unclear after")
        assert "type" in g["entities"][0] and g["entities"][0]["name"] == "alice"


# -- Governor end-to-end --------------------------------------------------------------------
def test_governor_flag_mode_end_to_end(path, fake: FakeJudge):
    ad = McpMemoryAdapter(path)
    fake.script("mcp_memory:alice#" + content_hash("user prefers Postgres"), SUPERSEDE)  # keyed by ledger id
    with Governor(ad, ":memory:", judge=fake, successors=True) as gov:
        rep = gov.sync()
        assert rep.added == 5 and rep.total == 5
        old = f"alice#{content_hash('user prefers Postgres')}"
        twin = f"billing#{content_hash('user prefers Postgres')}"
        assert gov.status_of(old) is Status.ACTIVE
        out = gov.observe("we migrated to SQLite last Tuesday", source="slack")
        assert out.errors == []
        assert gov.status_of(old) is Status.SUPERSEDED
        assert gov.status_of(twin) is Status.ACTIVE  # judged by id: billing's copy is a different memory
        assert gov.status_of(McpMemoryAdapter.relation_id("alice", "owns", "billing")) is Status.ACTIVE
        entities, relations = load_like_server(path)
        assert entities[0]["observations"][0] == (
            "user prefers Postgres [invalidate: superseded by “we migrated to SQLite last Tuesday” (slack, still true 5%), "
            + entities[0]["observations"][0].rsplit(", ", 1)[1].rstrip("]") + "]")
        assert entities[0]["observations"][2].startswith("we migrated to SQLite last Tuesday [invalidate: from slack, ")
        assert entities[1] == BILLING and relations == [OWNS]
        assert out.successor_host_id == f"alice#{content_hash('we migrated to SQLite last Tuesday')}"
        assert gov.mem.get(gov.our_id(old)).superseded_by == gov.our_id(out.successor_host_id)
        # The annotated file syncs back as unchanged, plus the successor already known.
        rep = gov.sync()
        assert rep.added == 0 and rep.updated == 0 and rep.unchanged == 6
        assert gov.dead_ids() == {old}
        # keep clears the marker; forget removes the observation.
        gov.keep(old)
        assert load_like_server(path)[0][0]["observations"][0] == "user prefers Postgres"
        gov.forget(f"billing#{content_hash('Alice owns the billing service')}")
        assert load_like_server(path)[0][1]["observations"] == ["user prefers Postgres"]
        assert gov.sync().removed == 0


def test_governor_delete_mode_end_to_end(path, fake: FakeJudge):
    fake.script("user prefers Postgres", SUPERSEDE).script("deploys", UNCERTAIN)
    with Governor(McpMemoryAdapter(path), ":memory:", judge=fake, mode="delete") as gov:
        gov.sync()
        out = gov.observe("we migrated to SQLite", source="slack")
        assert {p.action for p in out.pushes} == {"delete", "flag"}
        entities, relations = load_like_server(path)
        assert entities[0]["observations"][0].startswith("deploys run at 2pm UTC [invalidate: unclear after")
        assert len(entities[0]["observations"]) == 1
        assert entities[1]["observations"] == ["Alice owns the billing service"]
        assert relations == [OWNS]
        assert gov.sync().removed == 0


def test_governor_ledger_mode_leaves_file_alone(path, fake: FakeJudge):
    fake.script("user prefers Postgres", SUPERSEDE)
    with Governor(McpMemoryAdapter(path), ":memory:", judge=fake, mode="ledger", successors=True) as gov:
        gov.sync()
        out = gov.observe("we migrated to SQLite", source="slack")
        assert out.pushes == [] and read(path) == FILE
        assert len(gov.dead_ids()) == 2


def test_governor_sees_server_edits_as_new_claims(path, fake: FakeJudge):
    fake.script("mcp_memory:alice#" + content_hash("user prefers Postgres"), SUPERSEDE)
    with Governor(McpMemoryAdapter(path), ":memory:", judge=fake) as gov:
        gov.sync()
        gov.observe("we migrated to SQLite", source="slack")
        old = f"alice#{content_hash('user prefers Postgres')}"
        # The MCP server (or a human) replaces the observation: new content-addressed id, old row stays dead.
        entities, relations = load_like_server(path)
        entities[0]["observations"][0] = "user prefers SQLite now"
        write(path, "\n".join(json.dumps(x, separators=(",", ":")) for x in entities + relations) + "\n")
        rep = gov.sync()
        assert rep.added == 1 and rep.removed == 0 and rep.unchanged == 4
        assert gov.status_of(old) is Status.SUPERSEDED
        assert gov.status_of(f"alice#{content_hash('user prefers SQLite now')}") is Status.ACTIVE
