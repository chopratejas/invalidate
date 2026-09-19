"""POST /api/chat {"facts": [{"fact","status","p_true","kind","source"}, ...], "text": "..."}
   -> {"ok": true, "reply": "...", "facts": [...], "changes": [...], "selected": [...], "summary": {...}}

Stateless: the browser holds the memory and sends it back each turn. Caps live in invalidate.ui.chat
(MAX_FACTS / MAX_TEXT); this file adds a body cap and a small per-instance, per-IP rate limit.
"""
import json
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from invalidate.judge import JevJudge, MissingAPIKey  # noqa: E402
from invalidate.ui.chat import turn  # noqa: E402

MAX_BODY = 64 * 1024          # bytes
RATE_LIMIT = 40               # turns ...
RATE_WINDOW = 10 * 60         # ... per this many seconds, per IP, per function instance

_lock = threading.Lock()
_judge = None
_hits: dict[str, deque] = {}


def _get_judge():
    """One JevJudge per instance, created on first use so a missing key surfaces as a clean 500."""
    global _judge
    if _judge is None:
        with _lock:
            if _judge is None:
                _judge = JevJudge()
    return _judge


def _rate_limited(ip: str) -> bool:
    now = time.monotonic()
    with _lock:
        q = _hits.setdefault(ip, deque())
        while q and now - q[0] > RATE_WINDOW:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return True
        q.append(now)
        if len(_hits) > 5000:
            for k in [k for k, v in _hits.items() if not v or now - v[-1] > RATE_WINDOW]:
                _hits.pop(k, None)
        return False


def _typesafe_error_types():
    try:
        from typesafe_sdk import TypeSafeError
        return (TypeSafeError,)
    except Exception:
        return ()


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, status: int, message: str) -> None:
        self._send(status, {"ok": False, "error": message})

    def _client_ip(self) -> str:
        xff = self.headers.get("x-forwarded-for") or ""
        return xff.split(",")[0].strip() or self.headers.get("x-real-ip") or "unknown"

    def do_GET(self):  # noqa: N802
        self._err(405, "POST a JSON body: {\"facts\": [...], \"text\": \"...\"}")

    def do_POST(self):  # noqa: N802
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            return self._err(400, "Bad Content-Length.")
        if length <= 0:
            return self._err(400, "Empty body. Send JSON: {\"facts\": [...], \"text\": \"...\"}")
        if length > MAX_BODY:
            return self._err(413, f"Body too large (max {MAX_BODY // 1024} KB).")

        if _rate_limited(self._client_ip()):
            return self._err(429, f"Easy there. The demo allows {RATE_LIMIT} turns per "
                                  f"{RATE_WINDOW // 60} minutes per IP. Try again in a few minutes.")

        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._err(400, "Body must be valid JSON.")
        if not isinstance(data, dict):
            return self._err(400, "Body must be a JSON object with \"facts\" and \"text\".")
        facts, text = data.get("facts", []), data.get("text", "")
        if facts is None:
            facts = []
        if not isinstance(facts, list) or not all(isinstance(f, dict) for f in facts):
            return self._err(400, "\"facts\" must be a list of objects.")
        if not isinstance(text, str):
            return self._err(400, "\"text\" must be a string.")

        try:
            judge = _get_judge()
        except MissingAPIKey:
            return self._err(500, "Server is missing TYPESAFE_API_KEY. Set it in the Vercel project's environment "
                                  "variables and redeploy.")

        try:
            result = turn(facts, text, judge, judge.client)
        except ValueError as e:
            return self._err(400, str(e))
        except _typesafe_error_types() as e:
            return self._err(502, f"The judge (TypeSafe) returned an error: {type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            return self._err(500, f"Unexpected error: {type(e).__name__}: {e}")
        return self._send(200, {"ok": True, **result})

    def log_message(self, *_):
        pass
