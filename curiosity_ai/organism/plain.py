"""The model without the loop: a baseline for the blind ratings (docs/CURIOSITY_RECORDS.md).

The same model reads passages from the same library and is asked the same two things the organism asks (what
puzzles you; what exactly do you not know, and what would answer it), with no predictions, no judge, no
decisions and no memory. Its curiosities can look as curious as the organism's: what the loop adds shows in
behaviour (ending, waiting, fading), and in what raters see when the source is hidden.
"""
from __future__ import annotations

from typing import Any

from ..schema import _as_list
from . import prompts as P
from .organism import CuriosityOrganism, _clean_question, _question_fields
from .textutil import clip


def plain_records(org: CuriosityOrganism, *, n: int = 20, seed: int = 1) -> list[dict[str, Any]]:
    sample = getattr(org.senses.library, "sample", None)
    passages = sample(n, seed=seed) if sample else []
    items: list[dict[str, Any]] = []
    for i, obs in enumerate(passages, start=1):
        errors: list[str] = []
        user = (
            f'A passage, from {obs.heading}:\n---\n{clip(obs.text, 1500)}\n---\n\n'
            f"What genuinely puzzles you about it? Ask one specific question about {org.state.topic.title}."
        )
        data = org._json(org._sys("NOTICE"), user, P.NOTICE_SCHEMA, temperature=0.5, errors=errors, step="plain-notice")
        texts = [_clean_question(t, its_own=True) for t, _ in (_question_fields(item) for item in _as_list(data.get("questions")))]
        text = next((t for t in texts if t), "")
        if not text:
            continue
        record = org._conceive(text, errors=errors) or {}
        items.append({
            "id": f"P{i}", "question": text, "unknown": record.get("unknown", ""), "known": "; ".join(record.get("known", [])),
            "answer_form": record.get("answer_form", ""), "interest": record.get("interest", ""),
            "reason": record.get("wish_reason", ""), "source": obs.heading,
        })
    return items
