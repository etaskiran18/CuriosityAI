"""A scripted stand-in for the local LLM, so the organism's life cycle can be tested offline.

It behaves like a small, slightly sloppy model: it quotes real passages when
asked, but also fabricates one quote, returns numbers as strings, and echoes
schema hints, which are exactly the things the organism must survive. It also
plays the blind judge, the exam grader and the researcher's topic setup.
"""
from __future__ import annotations

import re
from typing import Any, Callable

_PASSAGE_RE = re.compile(r"\[(S\d+)\] [^\n]*\n(.*?)(?=\n\n\[S\d+\] |\n\n(?:For each expectation|List at most)|\Z)", re.DOTALL)
_PAIR_RE = re.compile(r"^(\d+)\. CLAIM: (.*)\n\s+QUOTE[^:]*: \"(.*)\"$", re.MULTILINE)
_STEPS = ("ANTICIPATE", "COMPARE", "WONDER", "SKEPTIC", "SETTLE", "NOTICE", "REFLECT", "LIBRARIAN", "JUDGE", "GRADE", "EXAM", "TOPIC")


def step_of(system: str) -> str:
    match = re.search(r"\[(" + "|".join(_STEPS) + r")\]", system)
    return match.group(1) if match else "UNKNOWN"


def passages_in(prompt: str) -> dict[str, str]:
    return {label: text for label, text in _PASSAGE_RE.findall(prompt)}


def pairs_in(prompt: str) -> list[tuple[int, str, str]]:
    return [(int(n), claim, quote) for n, claim, quote in _PAIR_RE.findall(prompt)]


def questions_to_rate(prompt: str) -> list[str]:
    if "Also rate each question below" not in prompt:
        return []
    tail = prompt.split("Also rate each question below", 1)[1]
    return re.findall(r"^\d+\. (.+)$", tail, re.MULTILINE)


def words(text: str, start: int, n: int) -> str:
    return " ".join(text.split()[start:start + n])


def best_quote(passages: dict[str, str], claim: str, n: int = 10) -> tuple[str, str]:
    """(label, quote): the n-word stretch that shares most words with the claim, as a careful reader would copy."""
    from curiosity_ai.organism.textutil import token_set

    wanted = token_set(claim)
    best = (-1, "", "")
    for label, text in passages.items():
        tokens = text.split()
        for start in range(max(1, len(tokens) - n + 1)):
            quote = " ".join(tokens[start:start + n])
            score = len(wanted & token_set(quote))
            if score > best[0]:
                best = (score, label, quote)
    return best[1], best[2]


DEFINITE = [
    {"author": "Plato", "claim": "Philosophy begins in wonder, the feeling of a philosopher", "probability": "0.8"},
    {"author": "Dewey", "claim": "Curiosity becomes intellectual through problems found in observation", "probability": 0.6},
    {"author": "the texts", "claim": "Reason can answer every question it raises", "probability": "30%"},
]

HEDGED = [
    {"author": "the texts", "claim": "The texts may discuss the role of wonder in philosophy", "probability": 0.7},
    {"author": "the texts", "claim": "Some texts might connect curiosity with problems", "probability": 0.6},
]


class ScriptedLLM:
    def __init__(
        self,
        *,
        fail_steps: tuple[str, ...] = (),
        settle: dict[str, Any] | None = None,
        contradict_for_real: bool = False,
        echo_unexpected: bool = False,
        hedge: str = "never",  # "never", "first" (only before the retry) or "always"
        judge: Callable[[str, str], str] | str | None = None,
        relevance: Callable[[str], Any] | Any = 3,
        grade: str = "correct",
        stance: str = "REVISE",
        text_rating: Any = 3,
    ):
        self.fail_steps = set(fail_steps)
        self.settle_override = settle
        self.contradict_for_real = contradict_for_real
        self.echo_unexpected = echo_unexpected
        self.hedge = hedge
        self.judge_rule = judge
        self.relevance = relevance
        self.grade = grade
        self.stance = stance
        self.text_rating = text_rating
        self.calls: list[str] = []
        self.prompts: dict[str, str] = {}  # the last prompt of each step
        self.history: list[tuple[str, str]] = []  # every (step, prompt), in order
        self.on_call = lambda step: None  # e.g. advance a fake clock: thinking takes time

    def _enter(self, system: str, user: str) -> str:
        step = step_of(system)
        self.calls.append(step)
        self.prompts[step] = user
        self.history.append((step, user))
        self.on_call(step)
        if step in self.fail_steps:
            raise ConnectionError(f"{step.lower()} model offline")
        return step

    def chat(self, system: str, user: str, *, temperature: float | None = None, json_mode: bool = False, max_tokens: int | None = None) -> str:
        step = self._enter(system, user)
        if step == "WONDER" and "Begin your reply with exactly one word" in user:
            return f"{self.stance}: the passage speaks of a feeling, so wonder starts inquiry but is not yet curiosity. I now hold that wonder precedes curiosity."
        if step == "WONDER":
            return "WONDER: What puzzles me is that wonder starts inquiry yet seems to fade once answers arrive. Perhaps wonder feeds on gaps. Does understanding kill wonder?"
        if step == "SKEPTIC":
            return "You assume wonder and curiosity are one thing. The passage speaks of a feeling, not a method. What do you mean by wonder?"
        return "..."

    def json_chat(self, system: str, user: str, schema_hint: str, *, temperature: float | None = None, max_tokens: int | None = None) -> dict[str, Any]:
        step = self._enter(system, user)
        if step == "ANTICIPATE":
            retry = "You first wrote these as hedged guesses" in user
            hedged = self.hedge == "always" or (self.hedge == "first" and not retry)
            return {
                "answer": "Wonder is the feeling that starts philosophical inquiry.",
                "confidence": "0.4",
                "expectations": HEDGED if hedged else DEFINITE,
            }
        if step == "COMPARE":
            passages = passages_in(user)
            labels = list(passages)
            first, last = labels[0], labels[-1]
            e1 = re.search(r"^E1: (.*)$", user, re.MULTILINE)
            e1_source, e1_quote = best_quote(passages, e1.group(1)) if e1 else (first, words(passages[first], 0, 10))
            third = {
                "expectation": "E3",
                "status": "contradicted",
                "source": last,
                "quote": words(passages[last], 2, 9) if self.contradict_for_real else "Reason can answer every single question that it ever raises",
            }
            if self.echo_unexpected:
                quote = words(passages[first], 0, 10)
                return {
                    "checks": [{"expectation": "E1", "status": "confirmed", "source": first, "quote": quote}],
                    "unexpected": [{"finding": "the same thing again", "source": first, "quote": quote}],
                }
            return {
                "checks": [
                    {"expectation": "E1", "status": "confirmed", "source": e1_source, "quote": e1_quote},
                    {"expectation": 2, "status": "confirmed or contradicted or not_addressed"},
                    third,
                ],
                "unexpected": [
                    {"finding": "The passage links wonder to perplexity", "source": labels[1] if len(labels) > 1 else first,
                     "quote": words(passages[labels[1] if len(labels) > 1 else first], 3, 8)},
                    {"finding": "made up", "source": first, "quote": "This sentence does not appear anywhere in the sources at all"},
                ],
            }
        if step == "JUDGE" and "A text found in a search" in user:
            return {"rating": self.text_rating, "reason": "scripted"}
        if step == "JUDGE":
            verdicts = []
            for n, claim, quote in pairs_in(user):
                if callable(self.judge_rule):
                    verdict = self.judge_rule(claim, quote)
                elif isinstance(self.judge_rule, str):
                    verdict = self.judge_rule
                else:
                    verdict = "contradicts" if "every question" in claim.lower() else "supports"
                verdicts.append({"pair": n, "verdict": verdict, "reason": "scripted"})
            ratings = []
            for n, question in enumerate(questions_to_rate(user), start=1):
                rating = self.relevance(question) if callable(self.relevance) else self.relevance
                ratings.append({"question": n, "rating": rating})
            return {"verdicts": verdicts, "ratings": ratings}
        if step == "SETTLE":
            if self.settle_override is not None:
                return self.settle_override
            return {
                "answer": "Wonder, the feeling of a philosopher, is the felt perplexity that starts inquiry; curiosity is the pursuit it sets in motion.",
                "would_be_wrong_if": "A text showed curiosity starting without any felt perplexity.",
                "confidence": "0.55",
                "learned": [
                    {"belief": "Plato treats wonder as the feeling that marks a philosopher.", "source": "S1"},
                    "Wonder and curiosity differ as a feeling differs from a pursuit.",
                ],
                "contradicts": [],
                "new_questions": [
                    {"question": "Does understanding something end our wonder about it, or deepen it?", "trigger": "surprise", "importance": "0.7"},
                    {"question": "Plato says philosophy begins in wonder: what exactly is wonder, and is it the same thing as curiosity?", "trigger": "gap"},
                ],
                "unanswerable": False,
                "insight": "Wonder is a beginning, not an answer.",
            }
        if step == "LIBRARIAN":
            return {
                "topics": ["Curiosity"],
                "books": [{"author": "Thomas Hobbes", "title": "Leviathan"}],
                "papers": ["The Information Gap Theory of Curiosity: A Review and Reinterpretation of the Evidence"],
            }
        if step == "NOTICE":
            return {"questions": [{"question": "Why does boredom feel like the opposite of curiosity rather than its absence?", "importance": 0.8}]}
        if step == "REFLECT":
            return {
                "understanding_of_curiosity": "Curiosity is the pursuit that perplexity sets in motion; it lives on gaps it can partly see.",
                "reflection": "I keep returning to wonder; I should test my ideas against Dewey more directly.",
                "focus_question": "How does Dewey's distinction between social and intellectual curiosity apply to a machine that asks questions?",
            }
        if step == "EXAM":
            notes = re.findall(r"^- \[([BQ]\d+)\]", user, re.MULTILINE)
            return {"answer": "My notes say wonder is the feeling of a philosopher.", "notes_used": notes[:1]}
        if step == "GRADE":
            return {"grade": self.grade, "reason": "scripted"}
        if step == "TOPIC":
            return {
                "meaning": "How lithium-ion cells lose capacity and power as they are cycled and stored.",
                "title": "Battery aging",
                "keywords": ["lithium-ion", "capacity fade", "solid electrolyte interphase", "cycling"],
                "questions": [
                    "Why does the solid electrolyte interphase keep growing during cycling?",
                    "How much does fast charging speed up capacity fade in lithium-ion cells?",
                ],
            }
        return {}
