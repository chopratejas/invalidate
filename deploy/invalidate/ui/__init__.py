"""Local web UI for invalidate. stdlib only: `invalidate ui` serves static/index.html and a small JSON API."""
from .server import serve

__all__ = ["serve"]
