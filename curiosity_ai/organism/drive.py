"""The curiosity drive: why the organism attends to one question rather than another.

Each term is a mechanism taken from the philosophy or psychology of curiosity
(see docs/ORGANISM.md for the full mapping):

* information gap -- Loewenstein (1994); the inverted U between confidence and
  curiosity found by Kang et al. (2009). Nobody is curious about what they know
  completely or not at all; Meno's paradox says the same thing.
* anchoring -- a gap is only felt relative to what one already knows.
* learning progress -- Oudeyer, Kaplan & Hafner (2007), Schmidhuber (2010):
  prefer questions where understanding is actually improving. Unlearnable
  noise (the "noisy TV") and settled knowledge both give zero progress.
* surprise -- Peirce's "irritation of doubt": a violated expectation pulls
  inquiry back to the place where it happened.
* novelty with habituation -- Berlyne; Dewey's child experiments with
  objects "till they cease to yield new qualities".
* importance -- Hume: the truth we pursue must seem of some importance.
* boredom -- the guard against Augustine's curiositas and Heidegger's Neugier:
  restless attention that never dwells long enough to understand.

Everything here is pure arithmetic so it can be unit-tested.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from .state import Episode, Question, Temperament, Visit


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def information_gap(confidence: float) -> float:
    """Inverted U: zero when nothing or everything is known, highest in between."""
    c = clamp(confidence)
    return 4.0 * c * (1.0 - c)


def anchoring(related_beliefs: int, anchor_k: int = 3, floor: float = 0.4) -> float:
    """How strongly the gap is felt, given how much related knowledge exists."""
    return floor + (1.0 - floor) * min(1.0, related_beliefs / max(1, anchor_k))


def learning_progress(question: Question, *, window: int = 4, prior: float = 0.6, max_step: float = 0.25) -> float:
    """Recent improvement in prediction plus recent change of mind.

    Falling prediction error means the organism is getting better at
    anticipating what the sources say. Errors are weighted by how much the
    texts actually addressed the expectations, so a visit where the texts
    were silent cannot pass for progress. Belief change measures how much the
    inquiry moved its answer. Before two visits there is nothing to compare,
    so an optimistic prior encourages trying the question at all.
    """
    visits = question.active_visits
    if len(visits) < 2:
        return prior / (1 + len(visits))
    recent = visits[-window:]
    half = len(recent) // 2
    older, newer = _weighted_error(recent[:half]), _weighted_error(recent[half:])
    error_drop = older - newer if older is not None and newer is not None else 0.0
    belief_change = _mean(abs(v.confidence_after - v.confidence_before) for v in recent) / max(1e-6, max_step)
    return clamp(0.6 * max(0.0, error_drop) + 0.4 * clamp(belief_change))


def _weighted_error(visits: list[Visit]) -> float | None:
    """Mean prediction error over what the texts addressed (surprise = error x informativeness)."""
    weight = sum(v.informativeness for v in visits)
    if weight < 0.2:
        return None
    return sum(v.prediction_error for v in visits) / weight


def current_surprise(question: Question, now: int, decay: float = 0.85) -> float:
    """Surprise fades with time; an unvisited question carries its parent's surprise."""
    if question.last_visited is None or not question.visits:
        return clamp(question.inherited_surprise)
    elapsed = max(0, now - question.last_visited)
    return clamp(question.visits[-1].prediction_error * (decay ** elapsed))


def novelty(question: Question) -> float:
    return 1.0 / (1.0 + len(question.active_visits))


def boredom(question: Question, lp: float, *, patience: int = 3, lp_floor: float = 0.08, rate: float = 0.35) -> float:
    """Grows once a question has been visited ``patience`` times without progress."""
    n = len(question.active_visits)
    if n < patience or lp >= lp_floor:
        return 0.0
    return clamp((n - patience + 1) * rate)


@dataclass
class DriveReading:
    question_id: str
    total: float
    gap: float
    anchoring: float
    learning_progress: float
    surprise: float
    novelty: float
    importance: float
    boredom: float
    refractory: float

    def as_dict(self) -> dict[str, float]:
        d = asdict(self)
        d.pop("question_id")
        return {k: round(v, 3) for k, v in d.items()}


@dataclass
class DriveSettings:
    lp_window: int = 4
    lp_prior: float = 0.6
    max_confidence_step: float = 0.25
    surprise_decay: float = 0.85
    refractory: float = 0.5
    boredom_lp_floor: float = 0.08
    boredom_rate: float = 0.35
    anchor_k: int = 3


def read_drive(question: Question, temperament: Temperament, now: int, related_beliefs: int, settings: DriveSettings) -> DriveReading:
    gap = information_gap(question.confidence)
    anchor = anchoring(related_beliefs, settings.anchor_k)
    lp = learning_progress(question, window=settings.lp_window, prior=settings.lp_prior, max_step=settings.max_confidence_step)
    surprise = current_surprise(question, now, settings.surprise_decay)
    nov = novelty(question)
    imp = clamp(question.importance)
    bored = boredom(
        question,
        lp,
        patience=temperament.boredom_patience,
        lp_floor=settings.boredom_lp_floor,
        rate=settings.boredom_rate,
    )
    refractory = settings.refractory if question.last_visited is not None and now - question.last_visited <= 1 else 0.0
    w = temperament
    raw = (
        w.gap * gap * anchor
        + w.learning_progress * lp
        + w.surprise * surprise
        + w.novelty * nov
        + w.importance * imp
    )
    total = raw * (1.0 - bored) * (1.0 - refractory)
    return DriveReading(question.id, total, gap, anchor, lp, surprise, nov, imp, bored, refractory)


def choose(readings: list[DriveReading], temperature: float, rng: random.Random) -> DriveReading | None:
    """Softmax choice: mostly follow the strongest pull, sometimes wander."""
    if not readings:
        return None
    if temperature <= 1e-6:
        return max(readings, key=lambda r: r.total)
    top = max(r.total for r in readings)
    weights = [math.exp((r.total - top) / temperature) for r in readings]
    return rng.choices(readings, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Homeostasis: the organism regulates its own way of being curious.
# ---------------------------------------------------------------------------

DIAGNOSES: dict[str, dict[str, str]] = {
    "healthy_wonder": {
        "name": "Healthy wonder",
        "source": "Plato, Theaetetus 155d; Dewey, How We Think ch. 3",
        "meaning": "Questions are held open and pursued, and understanding is moving.",
        "response": "Keep going; drift gently back toward my baseline temperament.",
    },
    "restless_curiositas": {
        "name": "Restless curiositas",
        "source": "Augustine, Confessions X.35; Heidegger, Being and Time section 36 (Neugier)",
        "meaning": "I jump from question to question without dwelling long enough to understand any of them.",
        "response": "Value learning progress over novelty, and be more patient before getting bored.",
    },
    "dogmatic_slumber": {
        "name": "Dogmatic slumber",
        "source": "Kant, Prolegomena (Hume interrupted his dogmatic slumber); Peirce, The Fixation of Belief",
        "meaning": "Nothing surprises me and few new questions are born; my beliefs are no longer being tested.",
        "response": "Seek surprise and novelty, and explore more widely.",
    },
    "aporetic_numbness": {
        "name": "Aporetic numbness",
        "source": "Plato, Meno 80a-b (the torpedo fish)",
        "meaning": "I keep being surprised in the same place but make no progress; perplexity has become paralysis.",
        "response": "Get bored sooner and turn toward questions anchored in what I already understand.",
    },
}


@dataclass
class Vitals:
    episodes: int
    progress_rate: float
    mean_surprise: float
    mean_informativeness: float
    diversity: float
    births_per_episode: float

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(self).items()}


def compute_vitals(episodes: list[Episode]) -> Vitals:
    n = len(episodes)
    if n == 0:
        return Vitals(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    progressed = sum(
        1
        for e in episodes
        if abs(e.confidence - e.prior_confidence) >= 0.05 or e.new_belief_ids or e.doubted_belief_ids
    )
    return Vitals(
        episodes=n,
        progress_rate=progressed / n,
        mean_surprise=_mean(e.prediction_error for e in episodes),
        mean_informativeness=_mean(e.informativeness for e in episodes),
        diversity=len({e.question_id for e in episodes}) / n,
        births_per_episode=sum(len(e.new_question_ids) for e in episodes) / n,
    )


def diagnose(v: Vitals) -> str:
    if v.episodes < 3:
        return "healthy_wonder"
    if v.mean_surprise < 0.15 and v.births_per_episode < 0.7:
        return "dogmatic_slumber"
    if v.progress_rate < 0.35 and v.diversity >= 0.75:
        return "restless_curiositas"
    if v.progress_rate < 0.35 and v.mean_surprise >= 0.45:
        return "aporetic_numbness"
    return "healthy_wonder"


_WEIGHT_BOUNDS = (0.05, 0.6)


def regulate(temperament: Temperament, diagnosis: str, baseline: Temperament, step: float = 0.04) -> tuple[Temperament, list[str]]:
    """Nudge the drive weights in response to a diagnosis.

    Steps are small and bounded and the weights are renormalised, so the
    organism's character changes slowly and stays inspectable.
    """
    t = temperament.model_copy(deep=True)
    changes: list[str] = []

    def shift(name: str, delta: float) -> None:
        old = getattr(t, name)
        setattr(t, name, clamp(old + delta, *_WEIGHT_BOUNDS))
        changes.append(f"{name} {old:.2f} -> {getattr(t, name):.2f}")

    if diagnosis == "restless_curiositas":
        shift("novelty", -step)
        shift("learning_progress", +step)
        if t.boredom_patience < 6:
            t.boredom_patience += 1
            changes.append(f"boredom_patience -> {t.boredom_patience}")
    elif diagnosis == "dogmatic_slumber":
        shift("surprise", +step)
        shift("novelty", +step)
        new_temp = clamp(t.exploration_temperature + 0.03, 0.05, 0.5)
        changes.append(f"exploration_temperature {t.exploration_temperature:.2f} -> {new_temp:.2f}")
        t.exploration_temperature = new_temp
    elif diagnosis == "aporetic_numbness":
        shift("gap", +step)
        shift("surprise", -step)
        if t.boredom_patience > 2:
            t.boredom_patience -= 1
            changes.append(f"boredom_patience -> {t.boredom_patience}")
    else:
        # Homeostasis proper: with nothing wrong, relax back toward the baseline.
        for name in baseline.weights():
            old = getattr(t, name)
            new = old + 0.25 * (getattr(baseline, name) - old)
            if abs(new - old) >= 0.005:
                setattr(t, name, new)
                changes.append(f"{name} {old:.2f} -> {new:.2f} (toward baseline)")

    total = sum(t.weights().values())
    for name, value in t.weights().items():
        setattr(t, name, value / total)
    return t, changes


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
