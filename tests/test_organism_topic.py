"""Staying on topic, and an honest diagnosis of restless curiosity."""
from __future__ import annotations

import pytest

from curiosity_ai.organism import CuriosityOrganism
from curiosity_ai.organism.drive import DriveSettings, Vitals, diagnose, read_drive, regulate, topic_factor
from curiosity_ai.organism.state import Question, Temperament
from curiosity_ai.organism.textutil import token_set, topic_relevance

from .organism_fakes import ScriptedLLM

NEW = [
    {"question": "Does understanding something end our wonder about it, or deepen it?", "trigger": "surprise", "importance": 0.9},
    {"question": "How can we develop adaptive learning strategies for diverse cultural contexts?", "trigger": "gap", "importance": 0.95},
    {"question": "Is perplexity a feeling or a judgment about what we lack?", "trigger": "objection", "importance": 0.8},
]


def proposing(new_questions, **kw) -> ScriptedLLM:
    return ScriptedLLM(settle={"answer": "a", "would_be_wrong_if": "b c d e", "confidence": 0.4, "learned": [], "new_questions": new_questions}, **kw)


def test_a_question_off_the_topic_pulls_less():
    q = Question(id="Q9", text="How can we develop mechanisms for self-correction in models?")
    t, s = Temperament(), DriveSettings()
    central = read_drive(q, t, 1, 0, s, relevance=1.0)
    off = read_drive(q, t, 1, 0, s, relevance=0.0)
    assert off.total == pytest.approx(central.total * (1 - t.topic_anchor))
    assert topic_factor(0.5, 0.6) == pytest.approx(0.7)


def test_the_topic_words_estimate_relevance_when_nobody_rated_it():
    words = token_set("curiosity wonder doubt inquiry novelty understanding")
    assert topic_relevance("Is doubt the engine of inquiry or its enemy?", words) == 1.0
    assert topic_relevance("How can we develop mechanisms for self-correction in models?", words) == 0.0


def test_a_confused_judge_cannot_throw_away_a_question_made_of_topic_words(config):
    """A small judge sometimes rates everything 0; the topic's own words then keep the question."""
    org = CuriosityOrganism(config, llm=proposing(NEW[:1], relevance=0))
    ep = org.heartbeat()
    assert len(ep.new_question_ids) == 1 and ep.set_aside_questions == []


def test_off_topic_questions_are_set_aside(config):
    rate = lambda question: 0 if "cultural" in question else 3
    org = CuriosityOrganism(config, llm=proposing(NEW[1:2], relevance=rate))
    ep = org.heartbeat()
    assert ep.new_question_ids == []
    assert ep.set_aside_questions == [NEW[1]["question"]]
    assert "Set aside as off my topic" in (org.home / "diary.md").read_text(encoding="utf-8")


def test_only_one_question_is_born_per_heartbeat_the_most_central(config):
    rate = lambda question: 3 if "perplexity" in question else (0 if "cultural" in question else 2)
    side = {"question": "Which experiments did Michotte run on perceived causality?", "trigger": "gap", "importance": 0.9}
    org = CuriosityOrganism(config, llm=proposing([side, NEW[1], NEW[2]], relevance=rate))
    ep = org.heartbeat()
    assert len(ep.new_question_ids) == 1
    child = org.state.questions[ep.new_question_ids[0]]
    assert "perplexity" in child.text and child.relevance == pytest.approx(1.0)
    assert ep.set_aside_questions == [NEW[1]["question"]]


def test_a_child_question_does_not_matter_more_than_its_parent(config):
    org = CuriosityOrganism(config, llm=proposing(NEW[:1]))
    parent = org.state.questions["Q1"]
    ep = org.heartbeat()
    child = org.state.questions[ep.new_question_ids[0]]
    assert child.importance == pytest.approx(parent.importance) and child.importance < 0.9
    assert parent.importance == pytest.approx(0.6)  # fertility is not importance


def test_unrated_questions_from_before_fall_back_to_the_topic_words(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    old = org.state.add_question("What specific examples of adaptive learning strategies have been effective?", trigger="gap")
    assert old.relevance is None
    assert org.relevance(old) < 0.3
    assert org.relevance(org.state.questions["Q1"]) == 1.0


def test_an_old_life_learns_its_topic(config, tmp_path):
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    org.state.topic.keywords = []
    org.save()
    again = CuriosityOrganism(config, llm=ScriptedLLM())
    assert again.state.topic.title == "the philosophy of curiosity" and "wonder" in again.state.topic.keywords


@pytest.mark.parametrize("vitals, expected", [
    # The one-hour run: questions multiplied (1.3 per heartbeat), none settled, every heartbeat a new question.
    (Vitals(10, 0.8, 0.05, 0.5, 1.0, 1.3, 0.0, 0.9), "restless_curiositas"),
    # Drifting away from the topic.
    (Vitals(10, 0.6, 0.3, 0.5, 0.6, 0.3, 0.1, 0.3), "restless_curiositas"),
    # Dwelling, settling, on topic.
    (Vitals(10, 0.6, 0.3, 0.5, 0.6, 0.4, 0.1, 0.9), "healthy_wonder"),
    # Nothing surprises it and few questions are born.
    (Vitals(10, 0.6, 0.05, 0.5, 0.5, 0.2, 0.1, 0.9), "dogmatic_slumber"),
])
def test_the_diagnosis_sees_multiplying_and_drifting(vitals, expected):
    assert diagnose(vitals) == expected


def test_restlessness_anchors_it_closer_to_its_topic():
    t, _ = regulate(Temperament(), "restless_curiositas", baseline=Temperament())
    assert t.topic_anchor > Temperament().topic_anchor
    for _ in range(50):
        t, _ = regulate(t, "restless_curiositas", baseline=Temperament())
    assert t.topic_anchor <= 0.9
    relaxed, _ = regulate(t, "healthy_wonder", baseline=Temperament())
    assert relaxed.topic_anchor < t.topic_anchor


def test_reflection_keeps_its_theory_when_the_new_one_is_too_vague(config):
    config.organism.reflect_every = 1

    class VagueReflection(ScriptedLLM):
        def json_chat(self, system, user, schema_hint, **kw):
            if "[REFLECT]" in system:
                self.calls.append("REFLECT")
                return {"understanding": "Curiosity is a dynamic, multifaceted interplay of various factors.", "reflection": "r", "focus_question": ""}
            return super().json_chat(system, user, schema_hint, **kw)

    org = CuriosityOrganism(config, llm=VagueReflection())
    before = org.state.self_model.understanding_of_curiosity
    org.heartbeat()
    assert org.state.self_model.understanding_of_curiosity == before
    assert "too vague to be wrong, so I kept the old one" in (org.home / "diary.md").read_text(encoding="utf-8")
