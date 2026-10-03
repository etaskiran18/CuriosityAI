"""The judge: a separate, blind check of every claim the organism makes about a quote.

A language model that grades its own evidence is lenient with itself. In a
one-hour run, half of the "confirmations" a 7B model gave itself did not
support the claim at all (Plato's "Whereas the physician and the carpenter have
different natures?" was taken to confirm "cultural differences in the balance
of novelty and understanding").

The judge sees only a claim, a quote and who wrote the quote. It never sees
the organism's reasoning or its own verdict, so it cannot simply agree. It can
run on a different, larger model than the organism (organism.judge.model).
Its verdicts decide what counts as evidence: only a quote the judge accepts can
confirm an expectation or ground a belief. It also rates how close new
questions are to the main topic, so the organism does not drift away from it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from ..schema import _as_list, _as_str
from . import prompts as P
from .textutil import one_line

VERDICTS = ("supports", "contradicts", "neither")


@dataclass
class Pair:
    claim: str
    quote: str
    source: str = ""  # who wrote the quote, e.g. "John Dewey, How We Think"


@dataclass
class Verdict:
    verdict: str  # supports | contradicts | neither | unjudged
    reason: str = ""


class Judge:
    def __init__(
        self,
        llm,
        *,
        temperature: float = 0.0,
        max_tokens: int = 500,
        before_thinking: Callable[[], None] | None = None,
        trace: Callable[[str, str, Any], None] | None = None,
    ):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.before_thinking = before_thinking or (lambda: None)
        self.trace = trace or (lambda step, prompt, reply: None)

    def judge(
        self,
        pairs: list[Pair],
        *,
        questions: list[str] | None = None,
        topic: str = "",
        errors: list[str] | None = None,
    ) -> tuple[list[Verdict], list[float | None]]:
        """Verdicts for the pairs, and relevance (0..1) of each new question to the topic.

        A verdict the judge did not give is "unjudged"; a rating it did not give is None.
        """
        questions = questions or []
        if not pairs and not questions:
            return [], []
        parts: list[str] = []
        if pairs:
            parts.append("Pairs to judge:\n\n" + "\n\n".join(
                f"{i}. CLAIM: {one_line(p.claim, 400)}\n   QUOTE"
                + (f" (from {one_line(p.source, 120)})" if p.source else "")
                + f': "{one_line(p.quote, 500)}"'
                for i, p in enumerate(pairs, start=1)
            ))
        else:
            parts.append("There are no pairs to judge: leave verdicts empty.")
        if questions:
            parts.append(
                f'Main topic: {topic}\n\nAlso rate each question below: how directly does it serve the main topic? '
                "3 = central to it, 2 = clearly related, 1 = loosely related, 0 = off the topic.\n\n"
                + "\n".join(f"{i}. {one_line(q, 300)}" for i, q in enumerate(questions, start=1))
            )
        else:
            parts.append("There are no questions to rate: leave ratings empty.")
        user = "\n\n".join(parts)
        self.before_thinking()
        try:
            data = self.llm.json_chat(P.JUDGE, user, P.JUDGE_SCHEMA, temperature=self.temperature, max_tokens=self.max_tokens)
        except Exception as exc:  # an absent judge must not stop a heartbeat; nothing gets its approval
            if errors is not None:
                errors.append(f"judge: {type(exc).__name__}: {one_line(str(exc), 160)}")
            self.trace("judge", user, f"ERROR {exc}")
            return [Verdict("unjudged") for _ in pairs], [None for _ in questions]
        self.trace("judge", user, data)
        data = data if isinstance(data, dict) else {}
        return _parse_verdicts(data.get("verdicts"), len(pairs)), _parse_ratings(data.get("ratings"), len(questions))


    def text_relevance(self, title: str, excerpt: str, *, question: str, topic: str, errors: list[str] | None = None) -> float | None:
        """How useful a text found by a search is for the question (0..1), or None if the judge could not say."""
        user = (
            f"Main topic: {topic}\n\nQuestion: {question}\n\nA text found in a search:\nTitle: {one_line(title, 200)}\n"
            f"Beginning: {one_line(excerpt, 900)}\n\nHow useful is this text for answering the question, within the main "
            "topic? 3 = directly useful, 2 = useful, 1 = only loosely related, 0 = unrelated."
        )
        self.before_thinking()
        try:
            data = self.llm.json_chat(P.JUDGE_TEXT, user, P.TEXT_RATING_SCHEMA, temperature=self.temperature, max_tokens=120)
        except Exception as exc:
            if errors is not None:
                errors.append(f"judge: {type(exc).__name__}: {one_line(str(exc), 160)}")
            self.trace("judge-text", user, f"ERROR {exc}")
            return None
        self.trace("judge-text", user, data)
        return rating_of((data or {}).get("rating")) if isinstance(data, dict) else None


def _index(value: Any, default: int) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    match = re.search(r"\d+", _as_str(value))
    return int(match.group()) if match else default


def verdict_of(value: Any) -> str:
    s = _as_str(value).strip().lower()
    if not s or "|" in s or " or " in s:
        return "unjudged"
    if s.startswith(("support", "confirm", "yes", "agree")):
        return "supports"
    if s.startswith(("contradict", "oppos", "refut", "disagree")):
        return "contradicts"
    if s.startswith(("neither", "none", "no", "irrelevant", "unrelated", "not")):
        return "neither"
    return "unjudged"


def _parse_verdicts(value: Any, n: int) -> list[Verdict]:
    out = [Verdict("unjudged") for _ in range(n)]
    for position, item in enumerate(_as_list(value), start=1):
        if isinstance(item, dict):
            idx = _index(item.get("pair", item.get("id")), position)
            verdict, reason = verdict_of(item.get("verdict")), one_line(_as_str(item.get("reason")), 200)
        else:
            idx, verdict, reason = position, verdict_of(item), ""
        if 1 <= idx <= n and out[idx - 1].verdict == "unjudged":
            out[idx - 1] = Verdict(verdict, reason)
    return out


def rating_of(raw: Any) -> float | None:
    """A 0-3 rating as relevance from 0 to 1 ("2" -> 0.67)."""
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        number = float(raw)
    else:
        match = re.search(r"\d+(\.\d+)?", _as_str(raw))
        if not match:
            return None
        number = float(match.group())
    if number != int(number) and number <= 1.0:
        return max(0.0, number)  # already a fraction
    return max(0.0, min(1.0, number / 3.0))


def _parse_ratings(value: Any, n: int) -> list[float | None]:
    out: list[float | None] = [None] * n
    for position, item in enumerate(_as_list(value), start=1):
        if isinstance(item, dict):
            idx = _index(item.get("question", item.get("id")), position)
            rating = rating_of(item.get("rating", item.get("score")))
        else:
            idx, rating = position, rating_of(item)
        if rating is not None and 1 <= idx <= n and out[idx - 1] is None:
            out[idx - 1] = rating
    return out
