"""Markdown / plain-text files as a memory host.

CLAUDE.md, AGENTS.md, .cursor/rules, Claude Code memory notes (~/.claude/projects/<proj>/memory/*.md),
docs: every claim line becomes one host memory. Ids are content-addressed
(`<relative path>#<sha1(normalized line)[:10]>`) so they survive line moves and re-ordering;
identical lines in one file collapse to one id.

Receipts go back into the file as a trailing HTML comment on the same line:

    - user prefers Postgres <!-- invalidate: superseded by “we migrated to SQLite” (slack, still true 5%) -->

`flag` rewrites exactly that line (marker and indentation preserved, every other byte untouched),
`delete` removes it, `insert` appends the successor under a `## invalidate: newer facts` section.
"""
from __future__ import annotations

import datetime as _dt
import glob as _glob
import hashlib
import os
import re
import tempfile
from collections.abc import Iterable
from typing import Any

from ..types import Status
from .base import HostMemory, Reason

MIN_CLAIM_LEN = 12
SECTION_HEADING = "## invalidate: newer facts"

_MARKER = re.compile(r"^(\s*)((?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?)?(.*)$", re.DOTALL)
_COMMENT = re.compile(r"\s*<!--\s*invalidate:.*?-->\s*$", re.DOTALL)
_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}(\s|$)")
_SETEXT = re.compile(r"^\s{0,3}(=+|-+)\s*$")
_TABLE = re.compile(r"^\s*\|")
_HR = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")


def _split_line(raw: str) -> tuple[str, str]:
    """('line without its terminator', 'terminator')."""
    for end in ("\r\n", "\n", "\r"):
        if raw.endswith(end):
            return raw[: -len(end)], end
    return raw, ""


def _strip_comment(body: str) -> str:
    return _COMMENT.sub("", body)


def _claim(body: str) -> tuple[str, str, str]:
    """Split a line body into (indent, list marker, claim text without any invalidate comment)."""
    m = _MARKER.match(body)
    indent, marker, rest = m.group(1), m.group(2) or "", m.group(3)
    return indent, marker, _strip_comment(rest).strip()


def normalize(text: str) -> str:
    return " ".join(text.split())


def content_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:10]


def _safe_comment(text: str) -> str:
    # "--" cannot appear inside an HTML comment; keep the receipt well-formed whatever the event says.
    return text.replace("--", "–").replace(">", "›")


def _iter_claims(lines: list[str]) -> Iterable[tuple[int, str, str, str]]:
    """Yield (index, indent, marker, claim) for every claim line. Skips headings, fences and their contents,
    blank lines, HTML comments, YAML frontmatter, table rows, rules and lines shorter than MIN_CLAIM_LEN."""
    in_fence: str | None = None
    in_comment = False
    i = 0
    n = len(lines)
    if n and lines[0].strip() == "---":
        j = 1
        while j < n and lines[j].strip() not in ("---", "..."):
            j += 1
        if j < n:
            i = j + 1
    while i < n:
        body = lines[i]
        s = body.strip()
        if in_comment:
            if "-->" in s:
                in_comment = False
            i += 1
            continue
        fm = _FENCE.match(body)
        if in_fence is not None:
            if fm and fm.group(1) == in_fence:
                in_fence = None
            i += 1
            continue
        if fm:
            in_fence = fm.group(1)
            i += 1
            continue
        if s.startswith("<!--"):
            if "-->" not in s:
                in_comment = True
            i += 1
            continue
        if not s or _HEADING.match(body) or _TABLE.match(body) or _HR.match(body) or _SETEXT.match(body):
            i += 1
            continue
        indent, marker, claim = _claim(body)
        if len(claim) >= MIN_CLAIM_LEN:
            yield i, indent, marker, claim
        i += 1


class MarkdownAdapter:
    """Govern the claims in one or more Markdown/text files.

        gov = Governor(MarkdownAdapter(["CLAUDE.md", "docs/"]), "ledger.db", mode="flag")
        gov.sync(); gov.observe("we migrated to SQLite last Tuesday", source="slack")

    `paths` may be files, directories (expanded with `glob`, default "*.md") or glob patterns.
    Ids are relative to `root` (default: the common directory of all paths). `annotate=False` turns
    `flag` into a no-op so only the ledger records verdicts; `delete` and `insert` still edit files.
    """

    name = "markdown"

    def __init__(self, paths: list[str] | str, *, glob: str | None = None, annotate: bool = True,
                 root: str | None = None) -> None:
        if isinstance(paths, str):
            paths = [paths]
        self.patterns = [os.fspath(p) for p in paths]
        self.glob = glob
        self.annotate = annotate
        self._root = os.path.abspath(root) if root else None

    # -- files ------------------------------------------------------------------
    def files(self) -> list[str]:
        out: list[str] = []
        for p in self.patterns:
            if os.path.isdir(p):
                found = sorted(_glob.glob(os.path.join(p, self.glob or "*.md"), recursive=True))
            elif _glob.has_magic(p):
                found = sorted(_glob.glob(p, recursive=True))
            else:
                found = [p]
            for f in found:
                f = os.path.abspath(f)
                if os.path.isfile(f) and f not in out:
                    out.append(f)
        return out

    @property
    def root(self) -> str:
        if self._root:
            return self._root
        files = self.files() or [os.path.abspath(p) for p in self.patterns]
        dirs = [os.path.dirname(f) for f in files]
        return os.path.commonpath(dirs) if dirs else os.getcwd()

    def rel(self, path: str) -> str:
        return os.path.relpath(os.path.abspath(path), self.root).replace(os.sep, "/")

    def id_for(self, path: str, text: str) -> str:
        return f"{self.rel(path)}#{content_hash(text)}"

    def _path_of(self, host_id: str) -> str:
        rel, _, _ = host_id.rpartition("#")
        for f in self.files():
            if self.rel(f) == rel:
                return f
        raise KeyError(host_id)

    @staticmethod
    def _read(path: str) -> list[str]:
        with open(path, encoding="utf-8", newline="") as fh:
            return fh.read().splitlines(keepends=True)

    @staticmethod
    def _write(path: str, lines: list[str]) -> None:
        d = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(prefix=".invalidate-", suffix=".tmp", dir=d)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
                fh.write("".join(lines))
            try:
                os.chmod(tmp, os.stat(path).st_mode & 0o7777)
            except OSError:
                pass
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # -- Adapter protocol ------------------------------------------------------
    def pull(self) -> Iterable[HostMemory]:
        out: list[HostMemory] = []
        for f in self.files():
            rel = self.rel(f)
            seen: set[str] = set()
            for idx, _indent, _marker, claim in _iter_claims([_split_line(l)[0] for l in self._read(f)]):
                hid = f"{rel}#{content_hash(claim)}"
                if hid in seen:
                    continue
                seen.add(hid)
                out.append(HostMemory(id=hid, text=claim, source="markdown", metadata={"path": rel, "line": idx + 1}))
        return out

    def _edit(self, host_id: str, fn) -> int:
        """Apply `fn(indent, marker, claim, terminator) -> new raw line | None (drop)` to every line with this id."""
        path = self._path_of(host_id)
        want = host_id.rpartition("#")[2]
        raw = self._read(path)
        bodies = [_split_line(l)[0] for l in raw]
        hits = {i: (ind, mk, cl) for i, ind, mk, cl in _iter_claims(bodies) if content_hash(cl) == want}
        if not hits:
            raise KeyError(host_id)
        out: list[str] = []
        for i, line in enumerate(raw):
            if i not in hits:
                out.append(line)
                continue
            indent, marker, claim = hits[i]
            new = fn(indent, marker, claim, _split_line(line)[1])
            if new is not None:
                out.append(new)
        if out != raw:
            self._write(path, out)
        return len(hits)

    def flag(self, host_id: str, reason: Reason) -> None:
        if not self.annotate:
            return
        if reason.status is Status.ACTIVE:
            self._edit(host_id, lambda ind, mk, cl, end: f"{ind}{mk}{cl}{end}")
            return
        note = f"<!-- {_safe_comment(reason.line())} -->"
        self._edit(host_id, lambda ind, mk, cl, end: f"{ind}{mk}{cl} {note}{end}")

    def delete(self, host_id: str, reason: Reason) -> None:
        self._edit(host_id, lambda ind, mk, cl, end: None)

    def insert(self, text: str, source: str, metadata: dict[str, Any]) -> str:
        files = self.files()
        path = files[0] if files else os.path.abspath(self.patterns[0])
        text = normalize(text)
        hid = self.id_for(path, text)
        raw = self._read(path) if os.path.exists(path) else []
        bodies = [_split_line(l)[0] for l in raw]
        end = next((e for _, e in (_split_line(l) for l in raw) if e), "\n")
        stamp = _dt.date.today().isoformat()
        line = f"- {text} <!-- invalidate: from {_safe_comment(source)} {stamp} -->{end}"

        head = next((i for i, b in enumerate(bodies) if b.strip() == SECTION_HEADING), None)
        if head is None:
            out = list(raw)
            if out and not _split_line(out[-1])[1]:
                out[-1] += end
            if out and out[-1].strip():
                out.append(end)
            out += [SECTION_HEADING + end, end, line]
        else:
            stop = len(bodies)
            for j in range(head + 1, len(bodies)):
                if _HEADING.match(bodies[j]):
                    stop = j
                    break
            for j in range(head + 1, stop):
                _, _, claim = _claim(bodies[j])
                if claim and content_hash(claim) == hid.rpartition("#")[2]:
                    return hid
            at = stop
            while at > head + 1 and not bodies[at - 1].strip():
                at -= 1
            out = list(raw)
            if at == len(out) and out and not _split_line(out[-1])[1]:
                out[-1] += end
            out.insert(at, line)
        self._write(path, out)
        return hid


def claude_code_paths(project_dir: str, *, home: str | None = None) -> list[str]:
    """The files Claude Code reads as memory for `project_dir`: CLAUDE.md, .claude/CLAUDE.md, AGENTS.md, and the
    project's memory notes under ~/.claude/projects/<dir with / replaced by ->/memory/*.md (except MEMORY.md).
    Only paths that exist are returned."""
    project_dir = os.path.abspath(project_dir)
    home = home or os.path.expanduser("~")
    out = [p for p in (os.path.join(project_dir, "CLAUDE.md"), os.path.join(project_dir, ".claude", "CLAUDE.md"),
                       os.path.join(project_dir, "AGENTS.md")) if os.path.isfile(p)]
    mem_dir = os.path.join(home, ".claude", "projects", project_dir.replace(os.sep, "-"), "memory")
    if os.path.isdir(mem_dir):
        out += [p for p in sorted(_glob.glob(os.path.join(mem_dir, "*.md"))) if os.path.basename(p) != "MEMORY.md"]
    return out


__all__ = ["MarkdownAdapter", "claude_code_paths", "content_hash", "normalize", "SECTION_HEADING"]
