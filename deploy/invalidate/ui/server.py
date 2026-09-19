"""Local web UI for invalidate: a stdlib HTTP server exposing a small JSON API over one `Invalidate` instance.

    from invalidate.ui import serve
    serve("invalidate.db")            # http://127.0.0.1:7411/ = playground (stateless), /store = dashboard

Binds localhost only. Every JSON response is `{"ok": true, ...}` or `{"ok": false, "error": "..."}`.
"""
from __future__ import annotations

import dataclasses
import enum
import json
import os
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from typesafe_sdk import TypeSafeError

from ..cli import DEMO_FACTS  # cli imports this module lazily (inside cmd_ui), so there is no cycle
from ..engine import Invalidate
from ..env import load_dotenv
from ..judge import MissingAPIKey
from ..types import Memory, Status, now
from . import check as playground

CONSOLE_URL = "https://console.typesafe.ai/"
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
ACTIONS = ("freeze", "unfreeze", "restore", "forget")
MAX_BODY = 64 * 1024  # bytes; the playground is meant to be hostable, so cap what one request can send


def jsonable(o: Any) -> Any:
    """Dataclasses -> dicts, enums -> .value, recursively."""
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {k: jsonable(v) for k, v in o.__dict__.items()}
    if isinstance(o, enum.Enum):
        return o.value
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(x) for x in o]
    return o


class Session:
    """One shared Invalidate plus the in-process tallies the store does not keep (cost, last latency, model)."""

    def __init__(self, db: str, namespace: str) -> None:
        self.db = db
        self.namespace = namespace
        self.mem = Invalidate(db, namespace=namespace)
        self.lock = threading.Lock()  # guards the tallies only; the store has its own RLock
        self.cost_usd_total = 0.0
        self.latency_ms_last: float | None = None
        self.model: str | None = None

    def tally(self, cost: float, latency_ms: float, model: str | None) -> None:
        with self.lock:
            self.cost_usd_total += cost
            self.latency_ms_last = latency_ms
            if model:
                self.model = model

    # -- serializers ------------------------------------------------------------
    def memory_dict(self, m: Memory, t: float | None = None) -> dict[str, Any]:
        d = m.to_dict()
        d["expired"] = m.is_expired(t)
        return d

    def memories(self) -> list[dict[str, Any]]:
        t = now()
        ms = [m for m in self.mem.list() if m.status is not Status.DELETED]
        ms.sort(key=lambda m: m.created_at, reverse=True)
        return [self.memory_dict(m, t) for m in ms]

    def state(self) -> dict[str, Any]:
        mems = self.memories()
        counts = {s.value: 0 for s in Status}
        for d in mems:
            counts[d["status"]] += 1
        with self.lock:
            stats = {
                "events": len(self.mem.events()),
                "verdicts_total": len(self.mem.store.list_verdicts()),
                "cost_usd_total": self.cost_usd_total,
                "latency_ms_last": self.latency_ms_last,
                "model": self.model,
                "db": self.db,
                "namespace": self.namespace,
                "has_key": bool(os.environ.get("TYPESAFE_API_KEY")),
            }
        return {"memories": mems, "counts": counts, "stats": stats,
                "demo_facts": [{"fact": f, "kind": k, "source": s} for f, k, s in DEMO_FACTS]}

    # -- API handlers -------------------------------------------------------------
    def remember(self, body: dict[str, Any]) -> dict[str, Any]:
        ttl = body.get("ttl_seconds")
        m = self.mem.remember(str(body.get("fact", "")), kind=str(body.get("kind") or "fact"),
                              source=str(body.get("source") or "human"),
                              ttl=float(ttl) if ttl not in (None, "") else None)
        return {"memory": self.memory_dict(m)}

    def observe(self, body: dict[str, Any]) -> dict[str, Any]:
        rep = self.mem.observe(str(body.get("text", "")), source=str(body.get("source") or "chat"),
                               dry_run=bool(body.get("dry_run", False)),
                               remember_successor=bool(body.get("remember_successor", True)))
        self.tally(rep.cost_usd, rep.latency_ms, rep.model)
        report = {
            "event": rep.event.to_dict(), "judged": rep.judged, "skipped": rep.skipped,
            "screened_out": rep.screened_out, "requests": rep.requests, "input_tokens": rep.input_tokens,
            "latency_ms": rep.latency_ms, "cost_usd": rep.cost_usd, "model": rep.model,
            "dry_run": bool(body.get("dry_run", False)),
            "successor": self.memory_dict(rep.successor) if rep.successor else None,
            "verdicts": [{"memory_id": v.memory_id, "disposition": v.disposition.value,
                          "from_status": v.from_status.value, "to_status": v.to_status.value,
                          "applied": v.applied, "votes": jsonable(v.votes)} for v in rep.verdicts],
        }
        return {"report": report, "memories": self.memories()}

    def recall(self, body: dict[str, Any]) -> dict[str, Any]:
        rep = self.mem.recall(str(body.get("query", "")), limit=int(body.get("limit") or 10),
                              include_review=bool(body.get("include_review", False)))
        self.tally(rep.cost_usd, rep.latency_ms, None)
        return {"results": [{"relevance": r.relevance, "memory": self.memory_dict(r.memory)} for r in rep.results],
                "considered": rep.considered, "requests": rep.requests, "latency_ms": rep.latency_ms,
                "cost_usd": rep.cost_usd}

    def action(self, memory_id: str, action: str) -> dict[str, Any]:
        m = getattr(self.mem, action)(memory_id)
        return {"memory": self.memory_dict(m)}

    def rewrite(self, memory_id: str, body: dict[str, Any]) -> dict[str, Any]:
        old = self.mem.get(memory_id)
        new = self.mem.remember(str(body.get("fact", "")), kind=str(body.get("kind") or old.kind),
                                source=str(body.get("source") or "human"),
                                metadata={"rewrites": old.id})
        old = self.mem.supersede(old.id, by=new.id)
        return {"old": self.memory_dict(old), "new": self.memory_dict(new)}

    def history(self, memory_id: str) -> dict[str, Any]:
        self.mem.get(memory_id)  # KeyError -> 404
        out = []
        for v in self.mem.history(memory_id):
            e = self.mem.store.get_event(v.event_id)
            d = jsonable(v)
            d["event_text"] = e.text if e else None
            d["event_source"] = e.source if e else None
            out.append(d)
        out.sort(key=lambda d: d["created_at"], reverse=True)
        return {"verdicts": out}

    def events(self, limit: int) -> dict[str, Any]:
        evs = self.mem.events(limit=limit)
        return {"events": [e.to_dict() for e in evs]}

    # -- playground (stateless; nothing touches the store) --------------------------
    def presets(self) -> dict[str, Any]:
        return {"presets": playground.PRESETS}

    def check(self, body: dict[str, Any]) -> dict[str, Any]:
        # `self.mem.judge` resolves the API key once and is shared; check() builds its own in-memory store.
        return playground.check(str(body.get("facts", "")), str(body.get("events", "")), self.mem.judge)


class Handler(BaseHTTPRequestHandler):
    session: Session  # set per server via make_server()
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet; errors still go through log_error
        pass

    # -- plumbing ----------------------------------------------------------------
    def _send(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _ok(self, payload: dict[str, Any]) -> None:
        self._send(HTTPStatus.OK, {"ok": True, **payload})

    def _err(self, status: int, message: str) -> None:
        self._send(status, {"ok": False, "error": message})

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        if n > MAX_BODY:
            raise ValueError(f"request body too large ({n} bytes; max {MAX_BODY})")
        data = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def _static(self, name: str) -> None:
        path = os.path.join(STATIC_DIR, name)
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            return self._err(HTTPStatus.NOT_FOUND, f"{name} missing")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _run(self, fn: Any, *args: Any) -> None:
        """Call an API handler and map exceptions to status codes."""
        try:
            self._ok(fn(*args))
        except MissingAPIKey:
            self._err(HTTPStatus.BAD_REQUEST,
                      f"TYPESAFE_API_KEY is not set. Get a key at {CONSOLE_URL} and put it in ./.env or export it.")
        except TypeSafeError as e:
            self._err(HTTPStatus.BAD_GATEWAY, f"TypeSafe API error: {type(e).__name__}: {e}")
        except KeyError as e:
            self._err(HTTPStatus.NOT_FOUND, f"no such memory: {e.args[0] if e.args else e}")
        except (ValueError, TypeError) as e:
            self._err(HTTPStatus.BAD_REQUEST, str(e))
        except Exception as e:  # noqa: BLE001 - surface anything else as a 500 rather than dropping the socket
            self.log_error("%s: %s", type(e).__name__, e)
            self._err(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}")

    # -- routing ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        url = urlsplit(self.path)
        parts = [p for p in url.path.split("/") if p]
        s = self.session
        if not parts:
            return self._static("playground.html")  # stateless playground is the landing page
        if parts == ["store"] or parts == ["index.html"]:
            return self._static("index.html")  # the dashboard over the on-disk store
        if parts == ["api", "state"]:
            return self._run(s.state)
        if parts == ["api", "presets"]:
            return self._run(s.presets)
        if parts == ["api", "events"]:
            q = parse_qs(url.query)
            return self._run(s.events, int(q.get("limit", ["50"])[0]))
        if len(parts) == 4 and parts[:2] == ["api", "memories"] and parts[3] == "history":
            return self._run(s.history, parts[2])
        self._err(HTTPStatus.NOT_FOUND, f"no route for GET {url.path}")

    def do_POST(self) -> None:  # noqa: N802
        url = urlsplit(self.path)
        parts = [p for p in url.path.split("/") if p]
        s = self.session
        try:
            body = self._body()
        except (ValueError, UnicodeDecodeError) as e:
            code = HTTPStatus.REQUEST_ENTITY_TOO_LARGE if "too large" in str(e) else HTTPStatus.BAD_REQUEST
            return self._err(code, f"bad JSON body: {e}")
        if parts == ["api", "remember"]:
            return self._run(s.remember, body)
        if parts == ["api", "observe"]:
            return self._run(s.observe, body)
        if parts == ["api", "recall"]:
            return self._run(s.recall, body)
        if parts == ["api", "check"]:
            return self._run(s.check, body)
        if len(parts) == 4 and parts[:2] == ["api", "memories"]:
            mid, action = parts[2], parts[3]
            if action in ACTIONS:
                return self._run(s.action, mid, action)
            if action == "rewrite":
                return self._run(s.rewrite, mid, body)
        self._err(HTTPStatus.NOT_FOUND, f"no route for POST {url.path}")


def make_server(db: str, host: str = "127.0.0.1", port: int = 7411, namespace: str = "default") -> ThreadingHTTPServer:
    """Build (but do not run) the server. Useful for tests: `srv.server_address` has the bound port."""
    load_dotenv()
    session = Session(db, namespace)
    handler = type("BoundHandler", (Handler,), {"session": session})
    ThreadingHTTPServer.allow_reuse_address = True
    ThreadingHTTPServer.daemon_threads = True
    srv = ThreadingHTTPServer((host, port), handler)
    srv.session = session  # type: ignore[attr-defined]
    return srv


def serve(db: str, host: str = "127.0.0.1", port: int = 7411, namespace: str = "default",
          open_browser: bool = True) -> None:
    """Serve the UI until Ctrl-C."""
    srv = make_server(db, host, port, namespace)
    url = f"http://{host}:{srv.server_address[1]}/"
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        srv.session.mem.close()  # type: ignore[attr-defined]
