"""The decision to pursue (records architecture): being curious is one thing, acting on it another (Inan).

A curiosity is born; then a separate step decides whether to pursue it now or let it wait, and keeps the reason.
"""
from __future__ import annotations

import pytest

from curiosity_ai.organism import CuriosityOrganism
from curiosity_ai.organism.organism import _parse_record

from .organism_fakes import RECORD, ScriptedLLM

CHILD = "understanding"  # the child question the scripted SETTLE proposes: "Does understanding something end our wonder ...?"


@pytest.fixture
def records(config):
    config.organism.curiosity.architecture = "records"
    return config


def conceive_child(**fields):
    """The seed question gets the ordinary record; the child question gets these fields."""
    def conceive(text: str) -> dict:
        return dict(RECORD, **fields) if CHILD in text.lower() else dict(RECORD)
    return conceive


def child_of(org, ep):
    return org.state.questions[ep.new_question_ids[0]]


def test_a_curiosity_it_wants_to_pursue_later_waits(records):
    llm = ScriptedLLM(recognize="no", conceive=conceive_child(pursue_now="no", why="I should first understand wonder itself."))
    org = CuriosityOrganism(records, llm=llm)
    ep = org.heartbeat()
    child = child_of(org, ep)
    assert child.status == "deferred" and child.decision == "defer"
    assert child.decision_reason == "I want to come back to it later: I should first understand wonder itself."
    assert ep.decisions == [{
        "question_id": child.id, "decision": "defer", "kind": "wish", "reason": child.decision_reason, "when": "birth", "was": "open",
    }]
    diary = (org.home / "diary.md").read_text(encoding="utf-8")
    assert f"I am curious about **{child.id}**, but I do not pursue it now, because I want to come back to it later" in diary
    assert any("deferred" in line for line in child.history)


def test_a_curiosity_no_text_speaks_of_waits_when_it_cannot_fetch_texts(records):
    llm = ScriptedLLM(recognize="no", conceive=conceive_child(unknown="the spectral index of chorus emissions near the plasmapause"))
    org = CuriosityOrganism(records, llm=llm)
    ep = org.heartbeat()
    child = child_of(org, ep)
    assert child.status == "deferred"
    assert ep.decisions[0]["kind"] == "no-texts"
    assert child.decision_reason == "no text I have speaks of it, and I cannot fetch new ones now"


def test_a_curiosity_its_texts_speak_of_is_pursued(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM(recognize="no"))
    ep = org.heartbeat()
    child = child_of(org, ep)
    assert child.status == "open" and child.decision == "pursue"
    assert ep.decisions[0]["kind"] == "texts" and "Plato" in ep.decisions[0]["reason"]
    assert f"I pursue **{child.id}** now: my texts speak of it" in (org.home / "diary.md").read_text(encoding="utf-8")


def test_a_persons_question_is_always_pursued(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM())
    q = org.ask("What would it take for a machine to be curious about something?")
    org._apply_record(q, dict(_parse_record(RECORD), wish="later"))
    q.status = "deferred"
    entries = org._decide(q, when="touch")
    assert q.status == "open" and entries[-1]["kind"] == "person"


def _record(org, text: str, unknown: str, *, importance: float, intensity: float):
    q = org.state.add_question(text, trigger="gap", importance=importance)
    org._apply_record(q, dict(_parse_record(RECORD), unknown=unknown))
    q.intensity = intensity
    return q


def test_at_most_max_pursued_curiosities_are_pursued(records):
    records.organism.curiosity.max_pursued = 2
    org = CuriosityOrganism(records, llm=ScriptedLLM())
    org.state.questions["Q1"].status = "deferred"
    strong = _record(org, "Which feeling marks a philosopher in Plato?", "the feeling that marks a philosopher", importance=0.9, intensity=0.9)
    middle = _record(org, "How does wonder relate to perplexity?", "the feeling of a philosopher in perplexity", importance=0.6, intensity=0.6)

    weak = _record(org, "Is wonder a feeling of the philosopher alone?", "the feeling of a philosopher alone", importance=0.1, intensity=0.1)
    entries = org._decide(weak, when="birth")
    assert weak.status == "deferred" and entries[-1]["kind"] == "room"
    assert strong.status == "open" and middle.status == "open"

    stronger = _record(org, "Who called Iris the child of wonder?", "the genealogist who called Iris the child of wonder", importance=1.0, intensity=1.0)
    entries = org._decide(stronger, when="birth")
    assert stronger.status == "open"
    assert middle.status == "deferred" and middle.status_reason == f"{stronger.id}, a stronger curiosity, took its place"
    assert [e["when"] for e in entries] == ["room", "birth"]


def test_a_waiting_curiosity_is_pursued_when_a_text_speaks_of_it(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM(recognize="no"))
    waiting = _record(org, "Which feeling marks a philosopher?", "the feeling that marks a philosopher and with which philosophy begins",
                      importance=0.5, intensity=0.5)
    waiting.status = "deferred"
    ep = org.heartbeat()
    assert ep.question_id == "Q1"
    assert waiting.status == "open"
    touch = [d for d in ep.decisions if d["when"] == "touch"]
    assert touch and touch[0]["question_id"] == waiting.id and touch[0]["decision"] == "pursue"
    assert f"**{waiting.id}** was waiting, and what I read spoke of it" in (org.home / "diary.md").read_text(encoding="utf-8")


def test_without_the_decision_every_curiosity_is_pursued_at_once(records):
    records.organism.curiosity.decision = False
    llm = ScriptedLLM(recognize="no", conceive=conceive_child(pursue_now="no", why="later"))
    org = CuriosityOrganism(records, llm=llm)
    ep = org.heartbeat()
    assert child_of(org, ep).status == "open" and ep.decisions == []


def test_a_session_counts_decisions(records):
    llm = ScriptedLLM(recognize="no", conceive=conceive_child(pursue_now="no", why="I should first understand wonder itself."))
    org = CuriosityOrganism(records, llm=llm)
    report = org.run_session(heartbeats=1)
    m = report.metrics
    assert m["deferred_at_birth"] == 1 and m["pursued_at_birth"] == 0
    assert m["deferral_reasons"] == {"wish": 1}
    assert "**Curious, but not acting:**" in report.markdown and "it wanted to come back later 1" in report.markdown


def test_an_unknown_in_the_topics_own_words_is_judged_by_all_its_words(records):
    """"the feeling with which philosophy begins": only "feeling" is not one of the topic's own words."""
    org = CuriosityOrganism(records, llm=ScriptedLLM())
    q = _record(org, "What feeling begins philosophy?", "the feeling with which philosophy begins", importance=0.5, intensity=0.5)
    assert org._touch_share(q, "wonder is the feeling of a philosopher, and philosophy begins in wonder") == 0.0
    assert org._texts_on(q)  # Plato's passage speaks of it
