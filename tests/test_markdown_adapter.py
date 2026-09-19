"""MarkdownAdapter over tmp_path files, plus Governor end-to-end with the fake judge."""
from __future__ import annotations

import os

import pytest

from conftest import CONTRADICT, SUPERSEDE, UNCERTAIN, FakeJudge
from invalidate.adapters import Governor, Reason
from invalidate.adapters.markdown import SECTION_HEADING, MarkdownAdapter, claude_code_paths, content_hash
from invalidate.types import Status

DOC = """---
title: project notes
tags: [a, b]
---
# Project notes

Some intro prose that is long enough to be a claim.

## Database
- user prefers Postgres
- deploys run at 2pm UTC
  * short
1. Alice owns the billing service
<!-- an html comment that is long enough -->
<!--
  a multi-line comment
  still inside
-->
```python
print("inside a fence, never a claim")
```
~~~
also fenced text that is long enough
~~~
| column | other |
|--------|-------|
| a row  | value |
- user prefers Postgres
***
"""


def write(path, text, encoding="utf-8"):
    with open(path, "w", encoding=encoding, newline="") as fh:
        fh.write(text)
    return str(path)


def read(path, encoding="utf-8"):
    with open(path, encoding=encoding, newline="") as fh:
        return fh.read()


def reason(status=Status.SUPERSEDED, text="we migrated to SQLite", source="slack", p=0.12):
    return Reason(status, status.value, text, source, "evt_1", p, 0.0)


@pytest.fixture
def doc(tmp_path):
    return write(tmp_path / "CLAUDE.md", DOC)


@pytest.fixture
def adapter(doc):
    return MarkdownAdapter(doc)


def ids(adapter):
    return {hm.text: hm.id for hm in adapter.pull()}


# -- pull ---------------------------------------------------------------------------------
def test_pull_skips_non_claims_and_strips_markers(adapter):
    got = adapter.pull()
    texts = [hm.text for hm in got]
    assert texts == [
        "Some intro prose that is long enough to be a claim.",
        "user prefers Postgres",
        "deploys run at 2pm UTC",
        "Alice owns the billing service",
    ]
    assert all(hm.source == "markdown" for hm in got)


def test_pull_ids_and_metadata(adapter, doc):
    hm = next(h for h in adapter.pull() if h.text == "user prefers Postgres")
    assert hm.id == f"CLAUDE.md#{content_hash('user prefers Postgres')}"
    assert hm.metadata == {"path": "CLAUDE.md", "line": 10}
    assert len(hm.id.split("#")[1]) == 10


def test_duplicate_lines_collapse_to_one_id(adapter):
    assert [h.text for h in adapter.pull()].count("user prefers Postgres") == 1


def test_ids_stable_across_line_moves(tmp_path):
    p = write(tmp_path / "a.md", "- user prefers Postgres\n- deploys run at 2pm UTC\n")
    before = set(ids(MarkdownAdapter(p)).values())
    write(p, "# heading\n\n* deploys run at 2pm UTC\n\n   1. user prefers   Postgres\n")
    after = set(ids(MarkdownAdapter(p)).values())
    assert before == after and len(after) == 2


def test_ids_include_relative_path_for_multiple_files(tmp_path):
    (tmp_path / ".claude" / "x").mkdir(parents=True)
    a = write(tmp_path / "CLAUDE.md", "- user prefers Postgres\n")
    b = write(tmp_path / ".claude" / "x" / "CLAUDE.md", "- user prefers Postgres\n")
    ad = MarkdownAdapter([a, b])
    got = sorted(h.id for h in ad.pull())
    assert got == [f".claude/x/CLAUDE.md#{content_hash('user prefers Postgres')}", f"CLAUDE.md#{content_hash('user prefers Postgres')}"]


def test_directory_and_glob_patterns(tmp_path):
    (tmp_path / "docs").mkdir()
    write(tmp_path / "docs" / "a.md", "- the first doc claim here\n")
    write(tmp_path / "docs" / "b.txt", "- the text file claim here\n")
    write(tmp_path / "docs" / "c.md", "- the third doc claim here\n")
    assert {h.text for h in MarkdownAdapter(str(tmp_path / "docs")).pull()} == {"the first doc claim here", "the third doc claim here"}
    assert {h.text for h in MarkdownAdapter(str(tmp_path / "docs"), glob="*.txt").pull()} == {"the text file claim here"}
    assert len(MarkdownAdapter(str(tmp_path / "docs" / "*")).pull()) == 3


def test_missing_file_is_ignored(tmp_path):
    assert MarkdownAdapter(str(tmp_path / "nope.md")).pull() == []


# -- flag ---------------------------------------------------------------------------------
def test_flag_annotates_exactly_the_target_lines(adapter, doc):
    before = read(doc)
    hid = ids(adapter)["user prefers Postgres"]
    adapter.flag(hid, reason())
    after = read(doc)
    note = ' <!-- invalidate: superseded by “we migrated to SQLite” (slack, still true 12%) -->'
    b, a = before.splitlines(keepends=True), after.splitlines(keepends=True)
    assert len(a) == len(b)
    changed = [i for i in range(len(b)) if a[i] != b[i]]
    assert changed == [9, 27]  # both copies of the duplicated line, nothing else
    for i in changed:
        assert a[i] == "- user prefers Postgres" + note + "\n"
    assert "".join(x for i, x in enumerate(a) if i not in changed) == "".join(x for i, x in enumerate(b) if i not in changed)


def test_flag_keeps_id_and_text(adapter):
    hid = ids(adapter)["user prefers Postgres"]
    adapter.flag(hid, reason())
    hm = next(h for h in adapter.pull() if h.id == hid)
    assert hm.text == "user prefers Postgres"


def test_reflag_replaces_comment(adapter, doc):
    hid = ids(adapter)["deploys run at 2pm UTC"]
    adapter.flag(hid, reason())
    adapter.flag(hid, reason(Status.CONTRADICTED, "deploys are manual now", "wiki", 0.03))
    text = read(doc)
    assert text.count("<!-- invalidate:") == 1
    assert "- deploys run at 2pm UTC <!-- invalidate: contradicted by “deploys are manual now” (wiki, still true 3%) -->\n" in text


def test_flag_needs_review_wording(adapter, doc):
    hid = ids(adapter)["deploys run at 2pm UTC"]
    adapter.flag(hid, reason(Status.NEEDS_REVIEW, "maybe deploys moved", "slack", 0.5))
    assert "<!-- invalidate: unclear after “maybe deploys moved” (slack, still true 50%) -->" in read(doc)


def test_keep_clears_comment_and_restores_bytes(adapter, doc):
    before = read(doc)
    hid = ids(adapter)["user prefers Postgres"]
    adapter.flag(hid, reason())
    assert read(doc) != before
    adapter.flag(hid, reason(Status.ACTIVE, "kept by a human", "human", 1.0))
    assert read(doc) == before


def test_flag_preserves_marker_and_indentation(tmp_path):
    p = write(tmp_path / "a.md", "  2) Alice owns the billing service\n\t- [ ] a checkbox item long enough\n")
    ad = MarkdownAdapter(p)
    got = ids(ad)
    ad.flag(got["Alice owns the billing service"], reason())
    ad.flag(got["a checkbox item long enough"], reason())
    lines = read(p).splitlines()
    assert lines[0].startswith("  2) Alice owns the billing service <!-- invalidate:")
    assert lines[1].startswith("\t- [ ] a checkbox item long enough <!-- invalidate:")


def test_flag_sanitizes_double_dash_in_event(adapter, doc):
    hid = ids(adapter)["deploys run at 2pm UTC"]
    adapter.flag(hid, reason(text="deploys --> moved -- really"))
    text = read(doc)
    line = next(l for l in text.splitlines() if "deploys run at 2pm UTC" in l)
    assert line.endswith("-->") and line.count("-->") == 1
    # The comment is still recognised and removable.
    adapter.flag(hid, reason(Status.ACTIVE))
    assert "invalidate:" not in read(doc)


def test_flag_unknown_id_raises(adapter):
    with pytest.raises(KeyError):
        adapter.flag("CLAUDE.md#0000000000", reason())
    with pytest.raises(KeyError):
        adapter.flag("other.md#0000000000", reason())


def test_annotate_false_makes_flag_a_noop(adapter, doc):
    before = read(doc)
    ad = MarkdownAdapter(doc, annotate=False)
    ad.flag(ids(ad)["user prefers Postgres"], reason())
    assert read(doc) == before


def test_flag_writes_atomically_no_temp_left(adapter, doc, tmp_path):
    adapter.flag(ids(adapter)["user prefers Postgres"], reason())
    assert sorted(os.listdir(tmp_path)) == ["CLAUDE.md"]


# -- delete -------------------------------------------------------------------------------
def test_delete_removes_line_and_comment(adapter, doc):
    before = read(doc).splitlines(keepends=True)
    hid = ids(adapter)["Alice owns the billing service"]
    adapter.flag(hid, reason())
    adapter.delete(hid, reason())
    after = read(doc).splitlines(keepends=True)
    assert after == [l for l in before if "Alice owns" not in l]
    assert hid not in {h.id for h in adapter.pull()}


def test_delete_removes_all_duplicates(adapter, doc):
    adapter.delete(ids(adapter)["user prefers Postgres"], reason())
    assert "user prefers Postgres" not in read(doc)


# -- insert -------------------------------------------------------------------------------
def test_insert_appends_section_once(tmp_path):
    a = write(tmp_path / "CLAUDE.md", "- user prefers Postgres")  # no trailing newline
    b = write(tmp_path / "AGENTS.md", "- deploys run at 2pm UTC\n")
    ad = MarkdownAdapter([a, b])
    hid = ad.insert("we migrated to SQLite last Tuesday", "slack", {})
    assert hid == f"CLAUDE.md#{content_hash('we migrated to SQLite last Tuesday')}"
    text = read(a)
    assert text.startswith("- user prefers Postgres\n\n" + SECTION_HEADING + "\n\n- we migrated to SQLite last Tuesday <!-- invalidate: from slack ")
    assert text.endswith(" -->\n")
    assert read(b) == "- deploys run at 2pm UTC\n"
    hid2 = ad.insert("Bob owns the billing service", "hr", {})
    text = read(a)
    assert text.count(SECTION_HEADING) == 1
    assert text.splitlines()[-1].startswith("- Bob owns the billing service <!-- invalidate: from hr ")
    pulled = {h.id: h for h in ad.pull()}
    assert pulled[hid].text == "we migrated to SQLite last Tuesday"
    assert pulled[hid2].text == "Bob owns the billing service"


def test_insert_is_idempotent_for_same_text(tmp_path):
    a = write(tmp_path / "CLAUDE.md", "- user prefers Postgres\n")
    ad = MarkdownAdapter(a)
    h1 = ad.insert("we migrated to SQLite", "slack", {})
    h2 = ad.insert("we migrated to SQLite", "slack", {})
    assert h1 == h2 and read(a).count("we migrated to SQLite") == 1


def test_insert_into_existing_section_before_next_heading(tmp_path):
    a = write(tmp_path / "CLAUDE.md", f"# top\n\n{SECTION_HEADING}\n\n- older successor line here\n\n# appendix\n- keep me here please\n")
    ad = MarkdownAdapter(a)
    ad.insert("newer successor line", "slack", {})
    lines = read(a).splitlines()
    assert lines[4] == "- older successor line here"
    assert lines[5].startswith("- newer successor line <!-- invalidate: from slack ")
    assert lines[-2:] == ["# appendix", "- keep me here please"]


def test_insert_creates_missing_file(tmp_path):
    p = str(tmp_path / "NEW.md")
    ad = MarkdownAdapter(p)
    hid = ad.insert("brand new fact here", "slack", {})
    assert hid.startswith("NEW.md#")
    assert read(p).startswith(SECTION_HEADING + "\n\n- brand new fact here")


# -- encodings ----------------------------------------------------------------------------
def test_crlf_file_round_trips(tmp_path):
    p = write(tmp_path / "win.md", "# t\r\n- user prefers Postgres\r\n- deploys run at 2pm UTC\r\n")
    ad = MarkdownAdapter(p)
    hid = ids(ad)["user prefers Postgres"]
    ad.flag(hid, reason())
    text = read(p)
    assert text.startswith("# t\r\n- user prefers Postgres <!-- invalidate: superseded by")
    assert text.endswith("-->\r\n- deploys run at 2pm UTC\r\n")
    assert "\n" not in text.replace("\r\n", "")
    ad.insert("we migrated to SQLite", "slack", {})
    assert read(p).endswith("-->\r\n") and "\r\n\r\n" + SECTION_HEADING + "\r\n" in read(p)
    ad.delete(hid, reason())
    assert read(p).startswith("# t\r\n- deploys run at 2pm UTC\r\n")


def test_unicode_content(tmp_path):
    line = "- l’équipe préfère Postgres — vraiment 🚀"
    p = write(tmp_path / "u.md", f"{line}\n- 日本語の行です、十分に長い\n")
    ad = MarkdownAdapter(p)
    got = ids(ad)
    assert "l’équipe préfère Postgres — vraiment 🚀" in got and "日本語の行です、十分に長い" in got
    ad.flag(got["l’équipe préfère Postgres — vraiment 🚀"], reason(text="on a migré vers SQLite"))
    text = read(p)
    assert text.startswith(line + " <!-- invalidate: superseded by “on a migré vers SQLite”")
    assert text.endswith("\n- 日本語の行です、十分に長い\n")


def test_short_lines_threshold(tmp_path):
    p = write(tmp_path / "s.md", "- 12345678901\n- 123456789012\n")
    assert [h.text for h in MarkdownAdapter(p).pull()] == ["123456789012"]


# -- claude_code_paths ----------------------------------------------------------------------
def test_claude_code_paths(tmp_path):
    proj = tmp_path / "proj"
    (proj / ".claude").mkdir(parents=True)
    write(proj / "CLAUDE.md", "x")
    write(proj / ".claude" / "CLAUDE.md", "x")
    home = tmp_path / "home"
    mem = home / ".claude" / "projects" / str(proj).replace(os.sep, "-") / "memory"
    mem.mkdir(parents=True)
    write(mem / "MEMORY.md", "index")
    write(mem / "b-note.md", "x")
    write(mem / "a-note.md", "x")
    got = claude_code_paths(str(proj), home=str(home))
    assert got == [str(proj / "CLAUDE.md"), str(proj / ".claude" / "CLAUDE.md"), str(mem / "a-note.md"), str(mem / "b-note.md")]
    write(proj / "AGENTS.md", "x")
    assert claude_code_paths(str(proj), home=str(home))[2] == str(proj / "AGENTS.md")
    assert claude_code_paths(str(tmp_path / "empty"), home=str(home)) == []


# -- Governor end-to-end --------------------------------------------------------------------
@pytest.fixture
def two_files(tmp_path):
    a = write(tmp_path / "CLAUDE.md", "# Project\n\n- user prefers Postgres\n- deploys run at 2pm UTC\n")
    b = write(tmp_path / "AGENTS.md", "- Alice owns the billing service\n- prod reads go through the Postgres replica\n")
    return a, b


def test_governor_flag_mode_end_to_end(two_files, fake: FakeJudge):
    a, b = two_files
    ad = MarkdownAdapter([a, b])
    fake.script("user prefers Postgres", SUPERSEDE).script("Postgres replica", CONTRADICT).script("deploys", UNCERTAIN)
    with Governor(ad, ":memory:", judge=fake, successors=True) as gov:
        rep = gov.sync()
        assert rep.added == 4 and rep.total == 4
        ids_ = ids(ad)
        assert gov.status_of(ids_["user prefers Postgres"]) is Status.ACTIVE
        out = gov.observe("we migrated to SQLite last Tuesday", source="slack")
        assert out.errors == []
        assert gov.status_of(ids_["user prefers Postgres"]) is Status.SUPERSEDED
        assert gov.status_of(ids_["prod reads go through the Postgres replica"]) is Status.CONTRADICTED
        assert gov.status_of(ids_["deploys run at 2pm UTC"]) is Status.NEEDS_REVIEW
        assert gov.status_of(ids_["Alice owns the billing service"]) is Status.ACTIVE
        ta, tb = read(a), read(b)
        assert "- user prefers Postgres <!-- invalidate: superseded by “we migrated to SQLite last Tuesday” (slack, still true 5%) -->\n" in ta
        assert "- deploys run at 2pm UTC <!-- invalidate: unclear after “we migrated to SQLite last Tuesday” (slack, still true 50%) -->\n" in ta
        assert "- prod reads go through the Postgres replica <!-- invalidate: contradicted by" in tb
        assert "- Alice owns the billing service\n" in tb
        assert out.successor_host_id == f"CLAUDE.md#{content_hash('we migrated to SQLite last Tuesday')}"
        assert SECTION_HEADING in ta and SECTION_HEADING not in tb
        assert gov.mem.get(gov.our_id(ids_["user prefers Postgres"])).superseded_by == gov.our_id(out.successor_host_id)
        # The annotated file syncs back as unchanged, plus the successor already known.
        rep = gov.sync()
        assert rep.added == 0 and rep.updated == 0 and rep.unchanged == 5
        assert gov.dead_ids() == {ids_["user prefers Postgres"], ids_["prod reads go through the Postgres replica"]}
        # keep clears the receipt; forget removes the line.
        gov.keep(ids_["deploys run at 2pm UTC"])
        assert "- deploys run at 2pm UTC\n" in read(a)
        gov.forget(ids_["Alice owns the billing service"])
        assert "Alice" not in read(b)
        assert gov.sync().removed == 0


def test_governor_delete_mode_end_to_end(two_files, fake: FakeJudge):
    a, b = two_files
    ad = MarkdownAdapter([a, b])
    fake.script("user prefers Postgres", SUPERSEDE).script("Alice", UNCERTAIN)
    with Governor(ad, ":memory:", judge=fake, mode="delete") as gov:
        gov.sync()
        out = gov.observe("we migrated to SQLite", source="slack")
        assert {p.action for p in out.pushes} == {"delete", "flag"}
        assert read(a) == "# Project\n\n- deploys run at 2pm UTC\n"
        assert "- Alice owns the billing service <!-- invalidate: unclear after" in read(b)
        assert gov.sync().removed == 0


def test_governor_ledger_mode_leaves_files_alone(two_files, fake: FakeJudge):
    a, b = two_files
    before = read(a), read(b)
    fake.script("user prefers Postgres", SUPERSEDE)
    with Governor(MarkdownAdapter([a, b]), ":memory:", judge=fake, mode="ledger", successors=True) as gov:
        gov.sync()
        out = gov.observe("we migrated to SQLite", source="slack")
        assert out.pushes == [] and (read(a), read(b)) == before
        assert len(gov.dead_ids()) == 1


def test_governor_sees_human_edits_as_new_claims(two_files, fake: FakeJudge):
    a, b = two_files
    fake.script("user prefers Postgres", SUPERSEDE)
    with Governor(MarkdownAdapter([a, b]), ":memory:", judge=fake) as gov:
        gov.sync()
        gov.observe("we migrated to SQLite", source="slack")
        old = ids(MarkdownAdapter([a, b]))["user prefers Postgres"]
        # A human rewrites the line: content-addressed id changes, so the old row is gone and a new one is added.
        write(a, read(a).replace("- user prefers Postgres <!-- invalidate: superseded by “we migrated to SQLite” (slack, still true 5%) -->", "- user prefers SQLite now"))
        rep = gov.sync()
        assert rep.added == 1 and rep.removed == 0 and rep.unchanged == 3  # dead row vanishing is not "removed"
        assert gov.status_of(old) is Status.SUPERSEDED
        assert gov.status_of(MarkdownAdapter([a, b]).id_for(a, "user prefers SQLite now")) is Status.ACTIVE
