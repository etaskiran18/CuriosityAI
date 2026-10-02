"""A scripted stand-in for the local LLM, so the organism's life cycle can be tested offline.

It behaves like a small, slightly sloppy model: it quotes real passages when
asked, but also fabricates one quote, returns numbers as strings, and echoes
schema hints, which are exactly the things the organism must survive.
"""
from __future__ import annotations

import re
from typing import Any

_PASSAGE_RE = re.compile(r"\[(S\d+)\] [^\n]*\n(.*?)(?=\n\n\[S\d+\] |\n\n(?:For each expectation|List at most)|\Z)", re.DOTALL)


def step_of(system: str) -> str:
    match = re.search(r"\[(ANTICIPATE|COMPARE|WONDER|SKEPTIC|SETTLE|NOTICE|REFLECT|LIBRARIAN)\]", system)
    return match.group(1) if match else "UNKNOWN"


def passages_in(prompt: str) -> dict[str, str]:
    return {label: text for label, text in _PASSAGE_RE.findall(prompt)}


def words(text: str, start: int, n: int) -> str:
    return " ".join(text.split()[start:start + n])


class ScriptedLLM:
    def __init__(self, *, fail_steps: tuple[str, ...] = (), settle: dict[str, Any] | None = None, contradict_for_real: bool = False, echo_unexpected: bool = False):
        self.fail_steps = set(fail_steps)
        self.settle_override = settle
        self.contradict_for_real = contradict_for_real
        self.echo_unexpected = echo_unexpected
        self.calls: list[str] = []
        self.prompts: dict[str, str] = {}
        self.on_call = lambda step: None  # e.g. advance a fake clock: thinking takes time

    def _enter(self, system: str, user: str) -> str:
        step = step_of(system)
        self.calls.append(step)
        self.prompts[step] = user
        self.on_call(step)
        if step in self.fail_steps:
            raise ConnectionError(f"{step.lower()} model offline")
        return step

    def chat(self, system: str, user: str, *, temperature: float | None = None, json_mode: bool = False, max_tokens: int | None = None) -> str:
        step = self._enter(system, user)
        if step == "WONDER":
            return "WONDER: What puzzles me is that wonder starts inquiry yet seems to fade once answers arrive. Perhaps wonder feeds on gaps. Does understanding kill wonder?"
        if step == "SKEPTIC":
            return "You assume wonder and curiosity are one thing. The passage speaks of a feeling, not a method. What do you mean by wonder?"
        return "..."

    def json_chat(self, system: str, user: str, schema_hint: str, *, temperature: float | None = None, max_tokens: int | None = None) -> dict[str, Any]:
        step = self._enter(system, user)
        if step == "ANTICIPATE":
            return {
                "answer": "Wonder is the feeling that starts philosophical inquiry.",
                "confidence": "0.4",
                "expectations": [
                    "Plato will say that philosophy begins in wonder",
                    "Dewey will connect curiosity with problems found in observation",
                    "The texts will claim that reason can answer every question it raises",
                ],
            }
        if step == "COMPARE":
            passages = passages_in(user)
            labels = list(passages)
            first, last = labels[0], labels[-1]
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
                    {"expectation": "E1", "status": "confirmed", "source": first, "quote": words(passages[first], 0, 10)},
                    {"expectation": 2, "status": "confirmed or contradicted or not_addressed"},
                    third,
                ],
                "unexpected": [
                    {"finding": "The passage links wonder to perplexity", "source": labels[1] if len(labels) > 1 else first,
                     "quote": words(passages[labels[1] if len(labels) > 1 else first], 3, 8)},
                    {"finding": "made up", "source": first, "quote": "This sentence does not appear anywhere in the sources at all"},
                ],
            }
        if step == "SETTLE":
            if self.settle_override is not None:
                return self.settle_override
            return {
                "answer": "Wonder is the felt perplexity that starts inquiry; curiosity is the pursuit it sets in motion.",
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
                "papers": ["information gap theory of curiosity"],
            }
        if step == "NOTICE":
            return {"questions": [{"question": "Why does boredom feel like the opposite of curiosity rather than its absence?", "importance": 0.8}]}
        if step == "REFLECT":
            return {
                "understanding_of_curiosity": "Curiosity is the pursuit that perplexity sets in motion; it lives on gaps it can partly see.",
                "reflection": "I keep returning to wonder; I should test my ideas against Dewey more directly.",
                "focus_question": "How does Dewey's distinction between social and intellectual curiosity apply to a machine that asks questions?",
            }
        return {}
