"""GET /api/presets -> {"ok": true, "presets": {...}}"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from invalidate.ui.check import PRESETS  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = json.dumps({"ok": True, "presets": PRESETS}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):  # keep function logs quiet
        pass
