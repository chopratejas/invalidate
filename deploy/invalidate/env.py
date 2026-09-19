"""Tiny .env loader so scripts and the CLI can keep TYPESAFE_API_KEY out of the shell."""
from __future__ import annotations

import os


def load_dotenv(path: str = ".env", *, override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE lines from `path` into os.environ. Returns what was loaded. Never raises."""
    loaded: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return loaded
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        if s.startswith("export "):
            s = s[7:]
        k, v = s.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if k and (override or k not in os.environ):
            os.environ[k] = v
            loaded[k] = v
    return loaded
