"""OpenAI chat with governed memory in ~10 lines. Needs OPENAI_API_KEY and TYPESAFE_API_KEY.

    pip install 'invalidate[openai]'
    python examples/openai_agent.py
"""
import sys

from invalidate import Invalidate, MissingAPIKey
from invalidate.cli import load_dotenv
from invalidate.integrations.openai import make_client, wrap

load_dotenv()
mem = Invalidate("agent.db")
if not mem.list():
    mem.remember("user prefers Postgres", kind="preference", source="onboarding")
    mem.remember("deploys run at 2pm UTC", source="wiki")

try:
    mem.judge  # validates TYPESAFE_API_KEY up front instead of on the first chat turn
    client = wrap(make_client(), mem)  # every .chat.completions.create() now recalls before, observes after
    for turn in (
        "Which database should the new service use?",
        "Heads up: we migrated everything to SQLite last Tuesday.",
        "Which database should the new service use?",  # stale preference is now superseded, not injected
    ):
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": turn}])
        print(f"> {turn}\n{resp.choices[0].message.content}\n")
        if client.completions.last_report and client.completions.last_report.changed:
            print(f"  [memory] {client.completions.last_report.summary()}\n")
except (MissingAPIKey, ImportError) as e:
    sys.exit(f"{e}")
