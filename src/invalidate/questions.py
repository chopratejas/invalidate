"""The questions Jev is asked. This module is the product.

Design notes (see DESIGN.md):
- Jev reads literally. Every question names the exact fields it compares with
  backticked paths into `state`, and each criterion says what belongs on each side.
- No dates, no counting, no arithmetic are asked of the model. Times are code's job.
- Each question is atomic. Code composes them (policy.py). Never ask Jev to rewrite a fact.
- All questions for one request are independent and run in parallel inside Jev.
"""
from __future__ import annotations

from typing import Any

from typesafe_sdk import Noul, NoulCriteria

from .types import Event, Memory

# Question id prefixes. Ids are for code only; Jev never sees them.
BEARS = "bears"
STILL_TRUE = "still_true"
REPLACES = "replaces"
HYPOTHETICAL = "hypothetical"
RELEVANT = "relevant"


def memory_view(m: Memory) -> dict[str, Any]:
    """The slice of a memory Jev sees. Verbatim fact, plus the two labels that help it judge."""
    return {"fact": m.fact, "kind": m.kind, "source": m.source}


def event_view(e: Event) -> dict[str, Any]:
    return {"text": e.text, "source": e.source}


def observe_state(event: Event, memories: list[Memory]) -> dict[str, Any]:
    return {"event": event_view(event), "memories": [memory_view(m) for m in memories]}


def observe_questions(n: int) -> dict[str, Noul]:
    """3 Nouls per memory + 1 for the event. `n` is len(state['memories'])."""
    qs: dict[str, Noul] = {
        HYPOTHETICAL: Noul(
            instructions={
                "question": "Is `event.text` a question, proposal, wish, plan, or hypothetical, rather than a statement of what is now the case?",
                "focus": "Judge the form of the event only. Do not judge whether it is true.",
            },
            criteria=NoulCriteria(
                true={
                    "what": "The event asks, suggests, wonders, or describes something that might happen or is being considered",
                    "examples": [
                        "Should we move to SQLite?",
                        "We might migrate off Postgres next quarter.",
                        "What if we dropped the Redis cache?",
                        "Proposal: switch the default region to eu-west.",
                    ],
                },
                false={
                    "what": "The event states something as a fact, decision, or completed change, even casually",
                    "examples": [
                        "We migrated to SQLite last Tuesday.",
                        "The user now prefers dark mode.",
                        "Decided: default region is eu-west from today.",
                        "Config change merged: cache TTL is 30s.",
                    ],
                },
            ),
        )
    }
    for i in range(n):
        f = f"`memories[{i}].fact`"
        qs[f"{BEARS}_{i}"] = Noul(
            instructions={
                "question": f"Does `event.text` give information about the same subject that {f} is about?",
                "compare": ["`event.text`", f],
                "focus": "Judge whether they concern the same thing, entity, setting, or preference. Ignore whether the event agrees or disagrees with the fact.",
            },
            criteria=NoulCriteria(
                true={
                    "what": "The event talks about the same thing the fact describes, whether it confirms, changes, or contradicts it",
                    "examples": [
                        "fact: 'user prefers Postgres' / event: 'we migrated to SQLite last Tuesday'",
                        "fact: 'user prefers Postgres' / event: 'still happily on Postgres here'",
                        "fact: 'deploys run at 2pm UTC' / event: 'the deploy window moved to 6pm'",
                    ],
                },
                false={
                    "what": "The event is about a different subject and says nothing about what the fact describes",
                    "examples": [
                        "fact: 'user prefers Postgres' / event: 'the marketing site got a new logo'",
                        "fact: 'user prefers Postgres' / event: 'lunch is at noon on Fridays'",
                    ],
                },
            ),
        )
        qs[f"{STILL_TRUE}_{i}"] = Noul(
            instructions={
                "question": f"Take `event.text` as accurate and more recent than {f}. Is {f} still true?",
                "compare": ["`event.text`", f],
                "focus": (
                    "A temporary condition, an outage, a one-off exception, a question, or a plan that has not happened "
                    "does not make the fact false. A stated change, reversal, correction, or replacement does."
                ),
            },
            criteria=NoulCriteria(
                true={
                    "what": "The event confirms the fact, restates it, describes only a temporary or partial situation, or leaves the fact unchanged",
                    "examples": [
                        "fact: 'user prefers Postgres' / event: 'still happily on Postgres here'",
                        "fact: 'user prefers Postgres' / event: 'Postgres was down for an hour this morning'",
                        "fact: 'user prefers Postgres' / event: 'the marketing site got a new logo'",
                    ],
                },
                false={
                    "what": "The event says the fact has changed, was wrong, was reversed, or no longer holds",
                    "examples": [
                        "fact: 'user prefers Postgres' / event: 'we migrated to SQLite last Tuesday'",
                        "fact: 'user prefers Postgres' / event: 'correction: I never liked Postgres, I meant MySQL'",
                        "fact: 'deploys run at 2pm UTC' / event: 'the deploy window moved to 6pm'",
                    ],
                },
            ),
        )
        qs[f"{REPLACES}_{i}"] = Noul(
            instructions={
                "question": f"Does `event.text` state a new current value, choice, or answer for the same thing that {f} asserts?",
                "compare": ["`event.text`", f],
                "focus": "Look for the replacement itself: a new name, setting, option, owner, status, or value, stated as now the case.",
            },
            criteria=NoulCriteria(
                true={
                    "what": "The event gives what is now true in place of the fact",
                    "examples": [
                        "fact: 'user prefers Postgres' / event: 'we migrated to SQLite last Tuesday'",
                        "fact: 'deploys run at 2pm UTC' / event: 'the deploy window moved to 6pm'",
                        "fact: 'Alice owns the billing service' / event: 'Bob took over billing from Alice'",
                    ],
                },
                false={
                    "what": "The event only says the fact is wrong or outdated without giving a replacement, confirms the fact, or is about something else",
                    "examples": [
                        "fact: 'user prefers Postgres' / event: 'we are no longer using Postgres'",
                        "fact: 'user prefers Postgres' / event: 'still happily on Postgres here'",
                        "fact: 'user prefers Postgres' / event: 'the marketing site got a new logo'",
                    ],
                },
            ),
        )
    return qs


def recall_state(query: str, memories: list[Memory]) -> dict[str, Any]:
    return {"query": query, "memories": [memory_view(m) for m in memories]}


def recall_questions(n: int) -> dict[str, Noul]:
    qs: dict[str, Noul] = {}
    for i in range(n):
        f = f"`memories[{i}].fact`"
        qs[f"{RELEVANT}_{i}"] = Noul(
            instructions={
                "question": f"Would knowing {f} help answer or act on `query`?",
                "compare": ["`query`", f],
                "focus": "Judge usefulness for this query, not whether the fact is interesting in general.",
            },
            criteria=NoulCriteria(
                true={
                    "what": "The fact directly answers the query, constrains the answer, or is a preference the answer should respect",
                    "examples": [
                        "query: 'which database should the new service use?' / fact: 'user prefers Postgres'",
                        "query: 'when can I deploy?' / fact: 'deploys run at 2pm UTC'",
                    ],
                },
                false={
                    "what": "The fact is about something the query does not touch",
                    "examples": [
                        "query: 'which database should the new service use?' / fact: 'lunch is at noon on Fridays'",
                    ],
                },
            ),
        )
    return qs
