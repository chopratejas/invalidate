#!/usr/bin/env python
"""LongMemEval with and without invalidate, same host, same answer model, official grader.

Host ("turns"): every user turn is stored verbatim as one memory with its session date; retrieval is
top-k by OpenAI embedding similarity. This is the benchmark's own turn-level RAG baseline.

Treatment ("+invalidate"): the same host, with invalidate in lazy mode. Every user turn is also an event.
At question time the host's top-2k candidates are validated against the events they have not seen
(Governor.filter), retired ones are hidden, and the first k live memories go to the answer model.

Answer model: Claude (default claude-sonnet-5). Grader: gpt-4o-2024-08-06 with the official
get_anscheck_prompt templates from the LongMemEval repo; label = "yes" in response.

    python evals/longmemeval/run.py --split oracle --types knowledge-update --arms base,inv
    python evals/longmemeval/run.py --split oracle --arms base,inv            # all 500
    python evals/longmemeval/run.py --split s --types knowledge-update --arms base,inv --k 10

Results are cached per (split, arm, answer model, k, question) under evals/longmemeval/results/ so a
run can be resumed and re-scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from invalidate import Invalidate, Policy  # noqa: E402
from invalidate.adapters import Governor, InMemoryAdapter  # noqa: E402
from invalidate.env import load_dotenv  # noqa: E402
from invalidate.judge import JevJudge  # noqa: E402


class CachingJudge(JevJudge):
    """JevJudge whose raw responses are cached on disk by a hash of (state, questions). Jev is deterministic for
    the same input, so a rerun of the same questions (a k sweep, a re-score) costs nothing after the first pass."""

    def __init__(self, cache_dir: Path, **kw: Any) -> None:
        super().__init__(**kw)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _call(self, state: Any, questions: Any) -> Any:
        payload = json.dumps({"state": state, "questions": {k: _q_dump(v) for k, v in questions.items()}, "model": self.model},
                             sort_keys=True, default=str)
        key = hashlib.sha1(payload.encode()).hexdigest()
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            self.hits += 1
            d = json.loads(f.read_text())
            return _Resp(d["answers"], d["input_tokens"], d["model"])
        resp = super()._call(state, questions)
        self.misses += 1
        answers = {k: float(getattr(a, "noul")) for k, a in resp.answers.items()}
        usage = getattr(resp, "usage", None)
        toks = int(getattr(usage, "input_tokens", 0) or 0)
        f.write_text(json.dumps({"answers": answers, "input_tokens": toks, "model": getattr(resp, "model", None)}))
        return _Resp(answers, toks, getattr(resp, "model", None))


def _q_dump(q: Any) -> Any:
    for attr in ("model_dump", "dict"):
        fn = getattr(q, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001
                pass
    return repr(q)


class _Ans:
    def __init__(self, p: float) -> None:
        self.noul = p


class _Resp:
    def __init__(self, answers: dict[str, float], input_tokens: int, model: Any) -> None:
        self.answers = {k: _Ans(v) for k, v in answers.items()}
        self.usage = type("U", (), {"input_tokens": input_tokens})()
        self.model = model

ENV = load_dotenv(str(ROOT / ".env"))


def env(name: str) -> str:
    v = os.environ.get(name) or ENV.get(name)
    if not v:
        sys.exit(f"{name} not set")
    return v


# ----------------------------------------------------------------------------- grading (official)
def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == "temporal-reasoning":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == "knowledge-update":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == "single-session-preference":
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        else:
            raise NotImplementedError(task)
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    return template.format(question, answer, response)


# ----------------------------------------------------------------------------- models
class Models:
    def __init__(self, answer_model: str, grader_model: str, embed_model: str) -> None:
        from anthropic import Anthropic
        from openai import OpenAI

        self.anthropic = Anthropic(api_key=env("ANTHROPIC_API_KEY"))
        self.openai = OpenAI(api_key=env("OPENAI_API_KEY"))
        self.answer_model = answer_model
        self.grader_model = grader_model
        self.embed_model = embed_model
        self._emb_cache: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def embed(self, texts: list[str]) -> list[list[float]]:
        todo = [t for t in dict.fromkeys(texts) if t not in self._emb_cache]
        for i in range(0, len(todo), 96):
            chunk = todo[i:i + 96]
            for attempt in range(6):
                try:
                    r = self.openai.embeddings.create(model=self.embed_model, input=[t[:6000] for t in chunk])
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(1.5 * (attempt + 1))
            else:
                raise RuntimeError("embedding failed")
            with self._lock:
                for t, d in zip(chunk, r.data):
                    self._emb_cache[t] = d.embedding
        return [self._emb_cache[t] for t in texts]

    def answer(self, system: str, user: str) -> str:
        for attempt in range(6):
            try:
                r = self.anthropic.messages.create(model=self.answer_model, max_tokens=400,
                                                   system=system, messages=[{"role": "user", "content": user}])
                return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
            except Exception as e:  # noqa: BLE001
                if attempt == 5:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("unreachable")

    def grade(self, prompt: str) -> bool:
        for attempt in range(6):
            try:
                r = self.openai.chat.completions.create(model=self.grader_model, temperature=0, max_tokens=8,
                                                        messages=[{"role": "user", "content": prompt}])
                return "yes" in (r.choices[0].message.content or "").lower()
            except Exception:  # noqa: BLE001
                if attempt == 5:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("unreachable")


# ----------------------------------------------------------------------------- host
def parse_date(s: str) -> datetime:
    # "2023/04/10 (Mon) 17:50"
    return datetime.strptime(re.sub(r"\s*\(\w+\)\s*", " ", s).strip(), "%Y/%m/%d %H:%M")


def cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-12)


class TurnsHost:
    """Verbatim user turns with dates; top-k by embedding similarity."""

    def __init__(self, models: Models) -> None:
        self.models = models
        self.rows: list[dict] = []  # {id, text, date, session, has_answer}
        self.vecs: list[list[float]] = []

    def add(self, text: str, date: str, session: str, has_answer: bool) -> str:
        hid = f"t{len(self.rows)}"
        self.rows.append({"id": hid, "text": text, "date": date, "session": session, "has_answer": has_answer})
        return hid

    def index(self) -> None:
        self.vecs = self.models.embed([r["text"] for r in self.rows])

    def search(self, query: str, k: int) -> list[dict]:
        q = self.models.embed([query])[0]
        scored = sorted(((cos(q, v), i) for i, v in enumerate(self.vecs)), reverse=True)
        return [self.rows[i] for _, i in scored[:k]]


EXTRACT_SYSTEM = (
    "You extract the user's personal facts from one message they sent to an assistant. Return a JSON array of "
    "short, self-contained statements about the user, each starting with 'The user', keeping names, numbers, "
    "dates and places exactly as written. Include only things the user states as true about themselves or their "
    "life (possessions, plans they have made, preferences, events that happened, current values). Do not include "
    "questions, requests for advice, hypotheticals, or facts about the assistant. Return [] if there are none. "
    "Output only the JSON array."
)


class FactsHost(TurnsHost):
    """Mem0/Zep-style host: an LLM extracts atomic facts from each user turn; facts are the memories."""

    def __init__(self, models: Models, extract_model: str, cache_dir: Path) -> None:
        super().__init__(models)
        self.extract_model = extract_model
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.turn_facts: list[tuple[str, str, str, bool, list[str]]] = []  # (turn, date, session, has_answer, facts)

    def extract(self, text: str) -> list[str]:
        key = hashlib.sha1((self.extract_model + "\x00" + text).encode()).hexdigest()
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text())
        raw = ""
        for attempt in range(6):
            try:
                r = self.models.anthropic.messages.create(model=self.extract_model, max_tokens=800, system=EXTRACT_SYSTEM,
                                                          messages=[{"role": "user", "content": text}])
                raw = "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
                break
            except Exception:  # noqa: BLE001
                if attempt == 5:
                    raise
                time.sleep(2 * (attempt + 1))
        m = re.search(r"\[.*\]", raw, re.S)
        facts: list[str] = []
        if m:
            try:
                facts = [str(x).strip() for x in json.loads(m.group(0)) if str(x).strip()]
            except json.JSONDecodeError:
                facts = []
        f.write_text(json.dumps(facts))
        return facts

    def add_turn(self, text: str, date: str, session: str, has_answer: bool) -> list[str]:
        ids = []
        for fact in self.extract(text):
            ids.append(self.add(fact, date, session, has_answer))
        return ids


class HostAdapter(InMemoryAdapter):
    """invalidate adapter over TurnsHost rows (ledger mode: the host is not modified)."""

    name = "lme_turns"

    def __init__(self, host: TurnsHost) -> None:
        super().__init__()
        self.host = host

    def pull(self):
        from invalidate.adapters import HostMemory

        return [HostMemory(id=r["id"], text=f"[{r['date']}] {r['text']}", metadata={"session": r["session"]}) for r in self.host.rows]


# ----------------------------------------------------------------------------- one question
SYSTEM = (
    "You are a helpful assistant with a memory of past conversations with the user. You are given memories, "
    "each with the date it was said, and a question with the date it is asked. Answer the question using the "
    "memories. Be concise. If the memories do not contain the information needed, say that you do not know."
)


def fmt_memories(rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda r: parse_date(r["date"]))
    out = []
    for r in rows:
        line = f"[{r['date']}] user said: {r['text']}"
        if r.get("note"):
            line += f"\n    -> {r['note']}"
        out.append(line)
    return "\n\n".join(out)


def annotate(gov: Governor, rows: list[dict]) -> list[dict]:
    """Serving mode 'annotate': keep every candidate, but label the ones the ledger has retired with what
    retired them. The model sees the change chain explicitly instead of inferring it from dates."""
    out = []
    for r in rows:
        m = gov.mem.store.get_memory(gov.our_id(r["id"]))
        r = dict(r)
        if m is not None and m.status.value in ("superseded", "contradicted"):
            last = [v for v in gov.mem.history(m.id) if v.applied and v.to_status is m.status]
            if last:
                e = gov.mem.store.get_event(last[-1].event_id)
                ev = re.sub(r"^\[[^\]]*\]\s*", "", e.text) if e else ""
                when = re.match(r"^\[([^\]]*)\]", e.text).group(1) if e and e.text.startswith("[") else ""
                word = "replaced" if m.status.value == "superseded" else "no longer true"
                r["note"] = f"OUTDATED, {word} as of {when}: {ev}" if ev else "OUTDATED"
            else:
                r["note"] = "OUTDATED"
        out.append(r)
    return out


def run_question(q: dict, arm: str, k: int, models: Models, policy_kw: dict, host_kind: str = "facts",
                 extract_model: str = "claude-haiku-4-5-20251001", serve_review: object = False) -> dict:
    t0 = time.perf_counter()
    order = sorted(range(len(q["haystack_sessions"])), key=lambda i: parse_date(q["haystack_dates"][i]))
    host = FactsHost(models, extract_model, HERE / "results" / "extract_cache") if host_kind == "facts" else TurnsHost(models)
    gov = None
    if arm == "inv":
        judge = CachingJudge(HERE / "results" / "jev_cache", api_key=env("TYPESAFE_API_KEY"))
        gov = Governor(HostAdapter(host), ":memory:", mode="ledger", lazy=True, policy=Policy(**policy_kw), judge=judge)
    n_turns = 0
    for i in order:
        sid = q["haystack_session_ids"][i]
        date = q["haystack_dates"][i]
        for turn in q["haystack_sessions"][i]:
            if turn["role"] != "user" or not turn["content"].strip():
                continue
            text = turn["content"].strip()
            units = host.extract(text) if host_kind == "facts" else [text]
            if gov is not None:
                # Each unit is evidence about the world first (event), then a memory born current.
                for u in units:
                    gov.mem.observe(f"[{date}] {u}", source="user", defer=True)
            for u in units:
                host.add(u, date, sid, bool(turn.get("has_answer")))
            if gov is not None:
                gov.sync()
            n_turns += 1
    host.index()
    jev_tokens = 0
    jev_requests = 0
    hidden: list[dict] = []
    if gov is None:
        rows = host.search(q["question"], k)
    else:
        cands = host.search(q["question"], 2 * k)
        rep = gov.validate([r["id"] for r in cands])
        jev_tokens, jev_requests = rep.input_tokens, rep.requests
        if serve_review == "annotate":
            live = annotate(gov, cands)
        else:
            live = gov.filter(cands, id_of=lambda r: r["id"], validate=False, include_review=bool(serve_review))
        live_ids = {r["id"] for r in live}
        hidden = [{"id": r["id"], "text": r["text"][:160], "date": r["date"], "has_answer": r["has_answer"],
                   "status": (gov.status_of(r["id"]) or "?").value if gov.status_of(r["id"]) else "?"}
                  for r in cands if r["id"] not in live_ids or r.get("note")]
        rows = live[:k]
        if serve_review == "annotate":
            rows = live[:k]  # same k as the baseline; annotations add text, not memories
    user = f"Memories:\n\n{fmt_memories(rows)}\n\nQuestion date: {q['question_date']}\nQuestion: {q['question']}"
    response = models.answer(SYSTEM, user)
    abst = q["question_id"].endswith("_abs")
    correct = models.grade(get_anscheck_prompt(q["question_type"], q["question"], q["answer"], response, abstention=abst))
    return {
        "question_id": q["question_id"], "type": q["question_type"], "arm": arm, "correct": correct,
        "response": response, "answer": q["answer"], "n_turns": n_turns, "served": [r["id"] for r in rows],
        "served_rows": [{"id": r["id"], "date": r["date"], "has_answer": r["has_answer"], "note": r.get("note"),
                         "text": r["text"][:120]} for r in rows],
        "evidence_dates": sorted({r["date"] for r in host.rows if r["has_answer"]}, key=parse_date),
        "served_has_answer": sum(1 for r in rows if r["has_answer"]),
        "hidden": hidden, "hidden_has_answer": sum(1 for h in hidden if h["has_answer"]),
        "jev_tokens": jev_tokens, "jev_requests": jev_requests, "seconds": time.perf_counter() - t0,
    }


# ----------------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="oracle", choices=["oracle", "s"])
    ap.add_argument("--types", default="", help="comma list of question types; default all")
    ap.add_argument("--arms", default="base,inv")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--answer-model", default="claude-sonnet-5")
    ap.add_argument("--grader-model", default="gpt-4o-2024-08-06")
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--host", default="facts", choices=["facts", "turns"])
    ap.add_argument("--extract-model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--serve", default="hide", choices=["hide", "review", "annotate"],
                    help="inv arm: hide = drop dead and reviewed; review = drop dead only; "
                         "annotate = keep everything, label dead facts with what replaced them (default hide)")
    a = ap.parse_args()

    data = json.load(open(HERE / "data" / f"longmemeval_{a.split}"))
    types = set(a.types.split(",")) if a.types else None
    qs = [q for q in data if types is None or q["question_type"] in types]
    if a.limit:
        qs = qs[: a.limit]
    models = Models(a.answer_model, a.grader_model, a.embed_model)
    policy_kw: dict = {}
    out_dir = HERE / "results" / f"{a.split}_{a.host}_{a.answer_model}_k{a.k}{('_' + a.tag) if a.tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(q, arm) for arm in a.arms.split(",") for q in qs]
    done = {}
    for q, arm in jobs:
        f = out_dir / f"{arm}_{q['question_id']}.json"
        if f.exists():
            done[(q["question_id"], arm)] = json.load(open(f))
    todo = [(q, arm) for q, arm in jobs if (q["question_id"], arm) not in done]
    print(f"{len(qs)} questions x {a.arms} = {len(jobs)} runs; {len(done)} cached, {len(todo)} to do -> {out_dir}")

    def work(q, arm):
        try:
            r = run_question(q, arm, a.k, models, policy_kw, host_kind=a.host, extract_model=a.extract_model,
                             serve_review={"hide": False, "review": True, "annotate": "annotate"}[a.serve])
        except Exception as e:  # noqa: BLE001
            r = {"question_id": q["question_id"], "type": q["question_type"], "arm": arm, "error": repr(e)}
        (out_dir / f"{arm}_{q['question_id']}.json").write_text(json.dumps(r, indent=1))
        return r

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(work, q, arm) for q, arm in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            done[(r["question_id"], r["arm"])] = r
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)

    # ---- report
    arms = a.arms.split(",")
    by = defaultdict(lambda: defaultdict(list))
    errors = Counter()
    for (qid, arm), r in done.items():
        if "error" in r:
            errors[arm] += 1
            continue
        by[r["type"]][arm].append(r)
        by["ALL"][arm].append(r)
    print(f"\nsplit={a.split} host={a.host} k={a.k} answer={a.answer_model} grader={a.grader_model}" + (f"  errors={dict(errors)}" if errors else ""))
    print(f"| type | n | " + " | ".join(arms) + " |")
    print("|---|---|" + "---|" * len(arms))
    for t in [t for t in ["knowledge-update", "temporal-reasoning", "multi-session", "single-session-user",
                          "single-session-assistant", "single-session-preference", "ALL"] if t in by]:
        cells = []
        for arm in arms:
            rs = by[t][arm]
            cells.append(f"{100 * sum(r['correct'] for r in rs) / len(rs):.1f}% ({sum(r['correct'] for r in rs)}/{len(rs)})" if rs else "-")
        print(f"| {t} | {len(by[t][arms[0]])} | " + " | ".join(cells) + " |")
    if "inv" in arms:
        rs = by["ALL"]["inv"]
        hid = sum(len(r["hidden"]) for r in rs)
        hid_ans = sum(r["hidden_has_answer"] for r in rs)
        tok = sum(r["jev_tokens"] for r in rs)
        print(f"\ninvalidate: {hid} candidate memories hidden across {len(rs)} questions ({hid_ans} of them evidence turns; "
              f"for knowledge-update the old value is evidence too), Jev {tok:,} tokens = ${tok * 0.042 / 1e6:.3f}, "
              f"{sum(r['jev_requests'] for r in rs)} requests")
        for t in by:
            if t == "ALL":
                continue
            rs_t = by[t]["inv"]
            print(f"  {t:26s} hidden {sum(len(r['hidden']) for r in rs_t):4d}  hidden evidence {sum(r['hidden_has_answer'] for r in rs_t):3d}")
        # gain slice: knowledge-update questions where the served set held evidence from an earlier
        # evidence date but none from the latest one (the old value came back, the update did not)
        for arm in arms:
            rs = [r for r in by.get("knowledge-update", {}).get(arm, []) if r.get("served_rows") and r.get("evidence_dates")]
            if not rs:
                continue
            old_only = [r for r in rs if any(x["has_answer"] and x["date"] != r["evidence_dates"][-1] for x in r["served_rows"])
                        and not any(x["has_answer"] and x["date"] == r["evidence_dates"][-1] for x in r["served_rows"])]
            neither = [r for r in rs if not any(x["has_answer"] for x in r["served_rows"])]
            both = [r for r in rs if r not in old_only and r not in neither]
            def acc(xs):
                return f"{sum(x['correct'] for x in xs)}/{len(xs)}" if xs else "-"
            print(f"\n{arm}: served old value only {len(old_only)} (correct {acc(old_only)}), "
                  f"update present {len(both)} (correct {acc(both)}), no evidence served {len(neither)} (correct {acc(neither)})")
        # flips
        if "base" in arms:
            b = {r["question_id"]: r["correct"] for r in by["ALL"]["base"]}
            i = {r["question_id"]: r["correct"] for r in by["ALL"]["inv"]}
            gained = [q for q in i if i[q] and not b.get(q, True)]
            lost = [q for q in i if not i[q] and b.get(q, False)]
            print(f"\nflips: +{len(gained)} gained, -{len(lost)} lost")
            for q in lost[:10]:
                print(f"  lost {q}")


if __name__ == "__main__":
    main()
