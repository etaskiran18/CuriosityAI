"""The records architecture (v10): curiosity as a record of what exactly is unknown, ended by a verified answer.

After Inan (2012), curiosity has an object, something one can name but not yet identify, and it is satisfied
when one comes to know that object; after Ady et al. (2022), curiosity ends once it is satisfied.
"""
from __future__ import annotations

import pytest

from curiosity_ai.config import CuriosityConfig
from curiosity_ai.organism import CuriosityOrganism
from curiosity_ai.organism.judge import yes_no
from curiosity_ai.organism.organism import _parse_record
from curiosity_ai.organism.state import load_mind

from .conftest import SEED
from .organism_fakes import RECORD, ScriptedLLM


@pytest.fixture
def records(config):
    config.organism.curiosity.architecture = "records"
    return config


def test_the_drive_is_the_default_and_makes_no_records(config):
    llm = ScriptedLLM()
    org = CuriosityOrganism(config, llm=llm)
    org.heartbeat()
    assert "CONCEIVE" not in llm.calls and "SATISFY" not in llm.calls
    assert not org.state.questions["Q1"].has_record
    assert org.state.questions["Q1"].status == "open"


def test_parts_can_be_switched_off_only_in_the_records_architecture():
    cc = CuriosityConfig(architecture="records", decision=False)
    assert cc.records and cc.ablated() == ["decision"]
    assert not cc.on("decision") and cc.on("satisfaction")
    drive = CuriosityConfig()
    assert not drive.records and drive.ablated() == [] and not drive.on("satisfaction")


def test_the_chosen_question_is_made_precise_first(records):
    llm = ScriptedLLM()
    org = CuriosityOrganism(records, llm=llm)
    ep = org.heartbeat()
    q = org.state.questions["Q1"]
    assert llm.calls[0] == "CONCEIVE" and llm.calls[1] == "ANTICIPATE"
    assert ep.conceived
    assert q.unknown == RECORD["unknown"] and q.answer_form == RECORD["answer_form"]
    assert q.known == RECORD["known"] and q.interest == RECORD["interest"]
    assert q.reachable is True and q.wish == "now"
    first = next(prompt for step, prompt in llm.history if step == "CONCEIVE")
    assert f"Your question: {SEED}" in first
    diary = (org.home / "diary.md").read_text(encoding="utf-8")
    assert f"*What I want to know:* {RECORD['unknown']}" in diary and "I made it precise just now" in diary


def test_a_verified_quote_the_judge_accepts_ends_the_curiosity(records):
    llm = ScriptedLLM()
    org = CuriosityOrganism(records, llm=llm)
    ep = org.heartbeat()
    q = org.state.questions["Q1"]

    assert q.status == "satisfied" and ep.status_after == "satisfied"
    assert len(ep.satisfied) == 1 and not ep.satisfied[0]["incidental"]
    quote = q.satisfied_by[0].quote
    assert "feeling of a philosopher" in quote and q.satisfied_by[0].source_title.startswith("Plato")
    assert q.identified_as.startswith("Philosophy begins in wonder")
    assert any(e.quote == quote for e in q.support)
    satisfaction = [j for j in ep.judgments if j["kind"] == "satisfaction"]
    assert satisfaction and satisfaction[0]["judge"] == "supports" and satisfaction[0]["quote"] == quote
    assert "Curiosity satisfied: Q1" in (org.home / "diary.md").read_text(encoding="utf-8")

    # Cessation: an answered curiosity is not pursued again.
    later = org.heartbeat()
    assert later is None or later.question_id != "Q1"


def test_the_judge_can_refuse_a_satisfaction(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM(satisfies="no"))
    ep = org.heartbeat()
    assert org.state.questions["Q1"].status == "open"
    assert not ep.satisfied and ep.satisfaction_refused
    assert ep.satisfaction_refused[0]["why"].startswith("the judge did not accept it")
    assert "it stays open" in (org.home / "diary.md").read_text(encoding="utf-8")


def test_an_echoed_schema_hint_is_not_a_yes(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM(satisfies="yes or no: does the quote answer it?"))
    ep = org.heartbeat()
    assert org.state.questions["Q1"].status == "open"
    assert ep.satisfaction_refused[0]["why"] == "the judge could not check it"


def test_an_invented_quote_ends_nothing(records):
    llm = ScriptedLLM(recognize="invented")
    org = CuriosityOrganism(records, llm=llm)
    ep = org.heartbeat()
    assert org.state.questions["Q1"].status == "open"
    assert ep.satisfaction_refused[0]["why"] == "the words it quoted are not in the passage"
    assert not any(j["kind"] == "satisfaction" for j in ep.judgments)  # the judge never saw it


def test_not_recognizing_the_unknown_leaves_it_open(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM(recognize="no"))
    ep = org.heartbeat()
    assert org.state.questions["Q1"].status == "open"
    assert not ep.satisfied and not ep.satisfaction_refused


def test_a_question_is_born_only_when_its_unknown_can_be_named(records):
    """The child question about understanding gets only the schema hint back: no unknown, so it is not born."""
    def conceive(text: str) -> dict:
        if "understanding" in text.lower():
            return {"unknown": "the one thing you cannot yet identify, written as: the ... that ...", "answer_form": "what would answer it"}
        return dict(RECORD)

    org = CuriosityOrganism(records, llm=ScriptedLLM(conceive=conceive, recognize="no"))
    ep = org.heartbeat()
    assert ep.new_question_ids == []
    assert ep.not_conceived and "understanding" in ep.not_conceived[0].lower()
    assert len(org.state.questions) == 1
    assert "could not say what exactly I do not know" in (org.home / "diary.md").read_text(encoding="utf-8")


def test_a_new_question_is_born_with_its_record(records):
    llm = ScriptedLLM(recognize="no")
    org = CuriosityOrganism(records, llm=llm)
    ep = org.heartbeat()
    child = org.state.questions[ep.new_question_ids[0]]
    assert child.has_record and child.unknown == RECORD["unknown"]
    assert "What drew your attention" in llm.history[-1][1] or any(
        step == "CONCEIVE" and "Does understanding" in prompt for step, prompt in llm.history
    )
    assert "*What I do not know:*" in (org.home / "diary.md").read_text(encoding="utf-8")


def test_a_waiting_curiosity_can_end_while_reading_for_another(records):
    """A curiosity it is not pursuing ends when the answer comes along: here in the passages read for Q1."""
    org = CuriosityOrganism(records, llm=ScriptedLLM())
    waiting = org.state.add_question("Which feeling marks a philosopher?", trigger="gap", importance=0.5)
    org._apply_record(waiting, dict(_parse_record(RECORD), unknown="the feeling that marks a philosopher and with which philosophy begins"))
    waiting.status = "deferred"

    ep = org.heartbeat()
    assert ep.question_id == "Q1"
    assert waiting.status == "satisfied" and waiting.satisfied_incidentally
    assert [s["question_id"] for s in ep.satisfied] == ["Q1", waiting.id]
    assert ep.satisfied[1]["incidental"]


def test_satisfaction_checks_can_be_limited(records):
    records.organism.curiosity.satisfaction_checks = 0
    org = CuriosityOrganism(records, llm=ScriptedLLM())
    waiting = org.state.add_question("Which feeling marks a philosopher?", trigger="gap", importance=0.5)
    org._apply_record(waiting, dict(_parse_record(RECORD), unknown="the feeling that marks a philosopher and with which philosophy begins"))
    waiting.status = "deferred"
    org.heartbeat()
    assert waiting.status == "deferred"


def test_an_answered_curiosity_is_not_asked_again(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM())
    org.heartbeat()
    assert org.state.questions["Q1"].status == "satisfied"
    q, outcome = org._adopt_question(SEED, trigger="gap", parent_id=None, importance=0.5, inherited_surprise=0.0)
    assert q.id == "Q1" and outcome == "already-satisfied"
    assert org.state.questions["Q1"].status == "satisfied"


def test_a_person_can_ask_an_answered_question_again(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM())
    org.heartbeat()
    q = org.ask(SEED)
    assert q.id == "Q1" and q.status == "open"


def test_without_satisfaction_nothing_is_checked(records):
    records.organism.curiosity.satisfaction = False
    llm = ScriptedLLM()
    org = CuriosityOrganism(records, llm=llm)
    ep = org.heartbeat()
    assert "SATISFY" not in llm.calls
    assert ep.conceived and org.state.questions["Q1"].status == "open"


def test_the_persons_own_draft_cannot_end_a_curiosity(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM())
    q = org.state.questions["Q1"]
    org._apply_record(q, _parse_record(RECORD))
    from curiosity_ai.organism.senses import Observation

    draft = Observation("S1", "[INBOX:x]", "My article (draft)", "Human observer", "Philosophy begins in wonder, the feeling of a philosopher.", "inbox", own=True)
    from curiosity_ai.organism.state import Episode

    ep = Episode(heartbeat=1, question_id="Q1", question=q.text)
    org._satisfaction(q, ep, [draft], [])
    assert q.status == "open" and not ep.satisfied


def test_records_and_their_end_are_kept_in_the_mind(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM())
    org.heartbeat()
    again = load_mind(org.mind_path)
    q = again.questions["Q1"]
    assert q.unknown == RECORD["unknown"] and q.status == "satisfied" and q.satisfied_by


def test_a_session_reports_what_happened_to_its_curiosities(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM())
    report = org.run_session(heartbeats=2, label="records")
    m = report.metrics
    assert m["architecture"] == "records" and m["ablated"] == []
    assert m["records_made"] >= 1 and m["curiosities_satisfied"] >= 1 and m["questions_satisfied"] >= 1
    assert m["looks_to_satisfaction"] == pytest.approx(1.0)
    assert "curiosity **records**" in report.markdown and "**Curiosities:**" in report.markdown
    assert "## Curiosities that ended" in report.markdown


def test_parse_record_refuses_schema_hints_and_keeps_answers():
    assert _parse_record({"unknown": "the one thing you cannot yet identify", "answer_form": "a mechanism"}) is None
    assert _parse_record({"unknown": "the cause of the stain", "answer_form": "what would answer it: a text"}) is None
    record = _parse_record({
        "unknown": "the cause of the dark stain on the wall?", "known": "it is wet\nit grew overnight", "answer_form": "a text naming the cause",
        "reachable": "No", "pursue_now": "yes or no", "why": "x",
    })
    assert record["unknown"] == "the cause of the dark stain on the wall"
    assert record["known"] == ["it is wet", "it grew overnight"]  # one string of lines becomes a list
    assert record["reachable"] is False and record["wish"] == "" and record["wish_reason"] == ""


@pytest.mark.parametrize("raw, expected", [
    ("yes", True), ("Yes.", True), ("no", False), ("No, it does not", False), (True, True),
    ("yes or no: does the quote answer it?", None), ("yes/no", None), ("maybe", None), ("", None),
])
def test_yes_no(raw, expected):
    assert yes_no(raw) is expected


def test_the_research_map_shows_curiosities_and_their_end(records):
    from curiosity_ai.organism.research_map import render_research_map

    org = CuriosityOrganism(records, llm=ScriptedLLM())
    org.heartbeat()
    text = render_research_map(org)
    assert "## What it is curious about" in text and "## Curiosities that ended" in text
    assert f"**Q1**: {RECORD['unknown']}" in text and "feeling of a philosopher" in text
