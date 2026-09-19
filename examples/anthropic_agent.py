"""Claude chat with governed memory in ~10 lines. Needs ANTHROPIC_API_KEY and TYPESAFE_API_KEY.

    pip install 'invalidate[anthropic]'
    python examples/anthropic_agent.py
"""
import sys

from invalidate import Invalidate, MissingAPIKey
from invalidate.cli import load_dotenv
from invalidate.integrations.anthropic import make_client, wrap

load_dotenv()
mem = Invalidate("agent.db")
if not mem.list():
    mem.remember("user prefers Postgres", kind="preference", source="onboarding")
    mem.remember("deploys run at 2pm UTC", source="wiki")

try:
    mem.judge  # validates TYPESAFE_API_KEY up front instead of on the first chat turn
    client = wrap(make_client(), mem)  # every .messages.create() now recalls before, observes after
    for turn in (
        "Which database should the new service use?",
        "Heads up: we migrated everything to SQLite last Tuesday.",
        "Which database should the new service use?",  # stale preference is now superseded, not injected
    ):
        resp = client.messages.create(
            model="claude-opus-5", max_tokens=16000,
            system="You are a terse infra assistant.",  # the memory block is appended to this
            messages=[{"role": "user", "content": turn}],
        )
        if resp.stop_reason == "refusal":
            print(f"> {turn}\n(refused: {resp.stop_details.category if resp.stop_details else 'unknown'})\n")
            continue
        print(f"> {turn}\n{''.join(b.text for b in resp.content if b.type == 'text')}\n")
        if client.messages.last_report and client.messages.last_report.changed:
            print(f"  [memory] {client.messages.last_report.summary()}\n")
except (MissingAPIKey, ImportError) as e:
    sys.exit(f"{e}")
