from __future__ import annotations

import random

import pytest

from curiosity_ai.organism.drive import (
    DriveSettings,
    Vitals,
    boredom,
    choose,
    compute_vitals,
    current_surprise,
    diagnose,
    information_gap,
    learning_progress,
    novelty,
    read_drive,
    regulate,
)
from curiosity_ai.organism.state import Episode, Question, Temperament, Visit


def visited(qid: str, errors: list[float], confidences: list[float] | None = None, last: int | None = None) -> Question:
    confidences = confidences or [0.5] * (len(errors) + 1)
    q = Question(id=qid, text=f"question {qid}", confidence=confidences[-1])
    for i, err in enumerate(errors):
        q.visits.append(Visit(heartbeat=i + 1, confidence_before=confidences[i], confidence_after=confidences[i + 1], prediction_error=err, informativeness=1.0))
    q.last_visited = last if last is not None else (len(errors) or None)
    return q


def test_information_gap_is_an_inverted_u():
    assert information_gap(0.0) == 0.0
    assert information_gap(1.0) == 0.0
    assert information_gap(0.5) == pytest.approx(1.0)
    assert information_gap(0.25) == pytest.approx(information_gap(0.75))
    assert information_gap(0.25) < information_gap(0.5)


def test_unvisited_question_gets_optimistic_progress_and_full_novelty():
    q = Question(id="Q1", text="fresh")
    assert learning_progress(q, prior=0.6) == pytest.approx(0.6)
    assert novelty(q) == 1.0


def test_learning_progress_rewards_falling_prediction_error():
    learnable = visited("L", [0.9, 0.7, 0.4, 0.2], [0.3, 0.35, 0.45, 0.55, 0.65])
    known = visited("K", [0.05, 0.05, 0.05, 0.05])
    noisy = visited("N", [0.9, 0.9, 0.9, 0.9])
    assert learning_progress(learnable) > 0.3
    assert learning_progress(known) < 0.05
    assert learning_progress(noisy) < 0.05


def test_a_silent_visit_is_not_progress():
    """Surprise followed by silence must not look like learning."""
    q = Question(id="Q", text="q")
    q.visits = [
        Visit(heartbeat=1, confidence_before=0.3, confidence_after=0.3, prediction_error=0.8, informativeness=1.0),
        Visit(heartbeat=2, confidence_before=0.3, confidence_after=0.3, prediction_error=0.0, informativeness=0.0),
    ]
    assert learning_progress(q) == 0.0


def test_noisy_tv_loses_to_a_learnable_question():
    """A question that stays surprising without ever getting easier must not capture attention."""
    settings = DriveSettings()
    temperament = Temperament()
    noisy = visited("N", [0.9, 0.9, 0.9, 0.9, 0.9], last=5)
    learnable = visited("L", [0.9, 0.7, 0.5, 0.3, 0.2], [0.3, 0.35, 0.4, 0.5, 0.55, 0.6], last=5)
    now = 8
    n = read_drive(noisy, temperament, now, related_beliefs=2, settings=settings)
    l = read_drive(learnable, temperament, now, related_beliefs=2, settings=settings)
    assert n.boredom > 0.5
    assert l.boredom == 0.0
    assert l.total > 2 * n.total


def test_boredom_needs_patience_to_run_out():
    q = visited("Q", [0.6, 0.6])
    assert boredom(q, lp=0.0, patience=3) == 0.0
    q = visited("Q", [0.6, 0.6, 0.6, 0.6])
    assert boredom(q, lp=0.0, patience=3) > 0.0
    assert boredom(q, lp=0.5, patience=3) == 0.0


def test_surprise_fades_and_is_inherited_before_the_first_visit():
    child = Question(id="Q2", text="child", inherited_surprise=0.7)
    assert current_surprise(child, now=10) == pytest.approx(0.7)
    q = visited("Q", [0.8], last=1)
    assert current_surprise(q, now=1) == pytest.approx(0.8)
    assert current_surprise(q, now=5) < 0.8 * 0.6


def test_refractory_period_discourages_rumination():
    settings = DriveSettings(refractory=0.5)
    q = visited("Q", [0.4], last=4)
    just_now = read_drive(q, Temperament(), now=5, related_beliefs=0, settings=settings)
    later = read_drive(q, Temperament(), now=7, related_beliefs=0, settings=settings)
    assert just_now.refractory == 0.5
    assert later.refractory == 0.0


def test_choose_is_greedy_at_zero_temperature_and_reproducible_with_a_seed():
    settings = DriveSettings()
    readings = [read_drive(Question(id=f"Q{i}", text="x", importance=i / 10), Temperament(), 1, 0, settings) for i in range(5)]
    assert choose(readings, 0.0, random.Random(1)).question_id == "Q4"
    a = [choose(readings, 0.3, random.Random(f"seed:{i}")).question_id for i in range(20)]
    b = [choose(readings, 0.3, random.Random(f"seed:{i}")).question_id for i in range(20)]
    assert a == b
    assert choose([], 0.2, random.Random()) is None


@pytest.mark.parametrize(
    "vitals, expected",
    [
        (Vitals(6, progress_rate=0.2, mean_surprise=0.3, mean_informativeness=0.5, diversity=1.0, births_per_episode=1.5), "restless_curiositas"),
        (Vitals(6, progress_rate=0.6, mean_surprise=0.05, mean_informativeness=0.8, diversity=0.5, births_per_episode=0.2), "dogmatic_slumber"),
        (Vitals(6, progress_rate=0.1, mean_surprise=0.7, mean_informativeness=0.6, diversity=0.3, births_per_episode=1.0), "aporetic_numbness"),
        (Vitals(6, progress_rate=0.7, mean_surprise=0.35, mean_informativeness=0.7, diversity=0.6, births_per_episode=1.2), "healthy_wonder"),
        (Vitals(2, progress_rate=0.0, mean_surprise=0.0, mean_informativeness=0.0, diversity=1.0, births_per_episode=0.0), "healthy_wonder"),
    ],
)
def test_diagnosis(vitals, expected):
    assert diagnose(vitals) == expected


def test_restless_curiosity_is_regulated_toward_dwelling():
    base = Temperament()
    t, changes = regulate(base, "restless_curiositas", baseline=base)
    assert t.novelty < base.novelty
    assert t.learning_progress > base.learning_progress
    assert t.boredom_patience == base.boredom_patience + 1
    assert sum(t.weights().values()) == pytest.approx(1.0)
    assert changes


def test_dogmatic_slumber_is_regulated_toward_surprise():
    base = Temperament()
    t, _ = regulate(base, "dogmatic_slumber", baseline=base)
    assert t.surprise > base.surprise
    assert t.exploration_temperature > base.exploration_temperature
    assert sum(t.weights().values()) == pytest.approx(1.0)


def test_healthy_wonder_relaxes_back_to_baseline():
    base = Temperament()
    drifted = Temperament(gap=0.2, learning_progress=0.45, surprise=0.15, novelty=0.05, importance=0.15)
    t, _ = regulate(drifted, "healthy_wonder", baseline=base)
    assert abs(t.learning_progress - base.learning_progress) < abs(drifted.learning_progress - base.learning_progress)


def test_regulation_keeps_weights_bounded_over_a_long_life():
    t = Temperament()
    for _ in range(200):
        t, _ = regulate(t, "restless_curiositas", baseline=Temperament())
    assert all(0.0 < w <= 0.6 + 1e-9 for w in t.weights().values())
    assert t.boredom_patience <= 6


def test_unsupported_lessons_do_not_count_as_progress():
    def ep(qid: str, **kw) -> Episode:
        return Episode(heartbeat=1, question_id=qid, question=qid, prior_confidence=0.4, confidence=0.42, **kw)

    talk = [ep(f"Q{i}", new_belief_ids=["B1"], new_question_ids=[f"Q{i + 10}"], prediction_error=0.3) for i in range(4)]
    assert compute_vitals(talk).progress_rate == 0.0
    real = [ep("Q1", new_belief_ids=["B1"], grounded_new_beliefs=1), ep("Q2", doubted_belief_ids=["B2"]),
            Episode(heartbeat=3, question_id="Q3", question="Q3", prior_confidence=0.3, confidence=0.5), ep("Q4")]
    assert compute_vitals(real).progress_rate == 0.75
    assert diagnose(compute_vitals(talk)) == "restless_curiositas"
