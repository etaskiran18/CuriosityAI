"""The persistent mind of the curiosity organism.

Everything the organism knows, wonders about and has lived through is stored
in one human-readable JSON file (``mind.json``), so a person can open it and
watch the organism's beliefs, open questions and temperament change over time.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..schema import utc_now_iso

QuestionStatus = Literal["open", "settled", "dormant", "unanswerable"]
Trigger = Literal["seed", "human", "observation", "surprise", "contradiction", "gap", "objection", "reflection"]
TRIGGERS: tuple[str, ...] = ("seed", "human", "observation", "surprise", "contradiction", "gap", "objection", "reflection")


class Evidence(BaseModel):
    """A verified quote: the words were found in the passage that was read."""

    citation: str
    quote: str
    source_title: str = ""


class Visit(BaseModel):
    """One investigation of a question; the raw material of learning progress."""

    heartbeat: int
    confidence_before: float
    confidence_after: float
    prediction_error: float  # surprise: share of my expectations the texts went against
    informativeness: float  # share of my expectations the texts addressed at all
    support: int = 0  # expectations confirmed by a verified quote


class Question(BaseModel):
    id: str
    text: str
    trigger: Trigger = "seed"
    parent_id: str | None = None
    born_at: int = 0
    status: QuestionStatus = "open"
    status_reason: str = ""
    status_changed_at: int = 0
    confidence: float = 0.25
    answer: str = ""
    importance: float = 0.5
    inherited_surprise: float = 0.0
    visits: list[Visit] = Field(default_factory=list)
    rest_marker: int = 0
    last_visited: int | None = None
    last_library_visit: int | None = None
    related_beliefs: list[str] = Field(default_factory=list)

    @property
    def active_visits(self) -> list[Visit]:
        """Visits since the question last woke from dormancy."""
        return self.visits[self.rest_marker:]


class Belief(BaseModel):
    id: str
    statement: str
    confidence: float
    evidence: list[Evidence] = Field(default_factory=list)
    origin_question: str | None = None
    born_at: int = 0
    updated_at: int = 0
    status: Literal["held", "doubted", "retracted"] = "held"
    contradicted_by: list[str] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)

    @property
    def interpretive(self) -> bool:
        return not self.evidence


class Temperament(BaseModel):
    """Weights of the curiosity drive: the organism's character.

    Homeostasis nudges these during reflection, so the organism's way of
    being curious changes with its own experience of inquiry.
    """

    gap: float = 0.30
    learning_progress: float = 0.30
    surprise: float = 0.15
    novelty: float = 0.10
    importance: float = 0.15
    boredom_patience: int = 3
    exploration_temperature: float = 0.15

    def weights(self) -> dict[str, float]:
        return {
            "gap": self.gap,
            "learning_progress": self.learning_progress,
            "surprise": self.surprise,
            "novelty": self.novelty,
            "importance": self.importance,
        }


class SelfModel(BaseModel):
    understanding_of_curiosity: str = (
        "Curiosity is the felt pull of a gap between what I know and what I could know. "
        "I do not yet understand it; I intend to find out by practicing it."
    )
    last_reflection: str = ""
    diagnosis: str = "healthy_wonder"
    reflections: list[str] = Field(default_factory=list)


class Episode(BaseModel):
    """A compact record of one heartbeat."""

    heartbeat: int
    created_at: str = Field(default_factory=utc_now_iso)
    question_id: str
    question: str
    trigger: str = ""
    drive: dict[str, float] = Field(default_factory=dict)
    prior_answer: str = ""
    prior_confidence: float = 0.0
    expectations: list[str] = Field(default_factory=list)
    checks: list[dict[str, Any]] = Field(default_factory=list)
    unexpected: list[dict[str, Any]] = Field(default_factory=list)
    rejected_quotes: int = 0
    sources: list[str] = Field(default_factory=list)
    prediction_error: float = 0.0
    informativeness: float = 0.0
    dialogue: list[dict[str, str]] = Field(default_factory=list)
    answer: str = ""
    confidence: float = 0.0
    insight: str = ""
    new_belief_ids: list[str] = Field(default_factory=list)
    grounded_new_beliefs: int = 0  # new beliefs backed by a verified quote
    reinforced_belief_ids: list[str] = Field(default_factory=list)
    doubted_belief_ids: list[str] = Field(default_factory=list)
    new_question_ids: list[str] = Field(default_factory=list)
    reawakened_question_ids: list[str] = Field(default_factory=list)
    status_after: str = "open"
    policy: str = "curiosity"
    acquisitions: list[str] = Field(default_factory=list)  # what it fetched from the library this heartbeat
    library_misses: list[str] = Field(default_factory=list)  # what it looked for but did not find
    library_owned: list[str] = Field(default_factory=list)  # what it wished for but already had
    errors: list[str] = Field(default_factory=list)


class MindState(BaseModel):
    version: int = 1
    name: str = "Curiosity"
    born_at: str = Field(default_factory=utc_now_iso)
    heartbeat: int = 0
    next_question_num: int = 1
    next_belief_num: int = 1
    questions: dict[str, Question] = Field(default_factory=dict)
    beliefs: dict[str, Belief] = Field(default_factory=dict)
    temperament: Temperament = Field(default_factory=Temperament)
    self_model: SelfModel = Field(default_factory=SelfModel)
    recent_episodes: list[Episode] = Field(default_factory=list)
    unread_inbox: list[str] = Field(default_factory=list)
    vitals_history: list[dict[str, Any]] = Field(default_factory=list)

    def add_question(self, text: str, **fields: Any) -> Question:
        qid = f"Q{self.next_question_num}"
        self.next_question_num += 1
        question = Question(id=qid, text=text, born_at=self.heartbeat, status_changed_at=self.heartbeat, **fields)
        self.questions[qid] = question
        return question

    def add_belief(self, statement: str, confidence: float, **fields: Any) -> Belief:
        bid = f"B{self.next_belief_num}"
        self.next_belief_num += 1
        belief = Belief(
            id=bid,
            statement=statement,
            confidence=confidence,
            born_at=self.heartbeat,
            updated_at=self.heartbeat,
            **fields,
        )
        self.beliefs[bid] = belief
        return belief

    def open_questions(self) -> list[Question]:
        return [q for q in self.questions.values() if q.status == "open"]

    def held_beliefs(self) -> list[Belief]:
        return [b for b in self.beliefs.values() if b.status != "retracted"]

    def remember_episode(self, episode: Episode, keep: int = 30) -> None:
        self.recent_episodes.append(episode)
        if len(self.recent_episodes) > keep:
            self.recent_episodes = self.recent_episodes[-keep:]


def load_mind(path: str | Path) -> MindState | None:
    path = Path(path)
    if not path.exists():
        return None
    return MindState.model_validate(json.loads(path.read_text(encoding="utf-8")))


def save_mind(state: MindState, path: str | Path) -> None:
    """Write atomically, so an interrupted heartbeat never corrupts the mind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".mind-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; the mind is meant to be read
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
