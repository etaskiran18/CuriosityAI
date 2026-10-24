"""Voluntary exposure and coherent long-term learning (records architecture).

Ady et al.'s agent learns to return to the place that makes it curious (the bookstore), not to the page that
answered it. Here: with nothing to pursue, or after a rest, the organism goes back to the text that has awakened
most curiosities so far and reads on. Its interests grow from the curiosities it pursued and had answered, and
new curiosities near them start stronger.
"""
from __future__ import annotations

import pytest

from curiosity_ai.organism import CuriosityOrganism
from curiosity_ai.organism.organism import _parse_record
from curiosity_ai.organism.state import SourceStats

from .organism_fakes import RECORD, ScriptedLLM

PLATO = "Plato, Theaetetus"


@pytest.fixture
def records(config):
    config.organism.curiosity.architecture = "records"
    return config


def _quiet(org) -> None:
    """Nothing left to pursue."""
    for q in org.state.questions.values():
        q.status = "faded"


def test_with_nothing_to_pursue_it_goes_back_to_the_text_that_made_it_curious(records):
    llm = ScriptedLLM(recognize="no")
    org = CuriosityOrganism(records, llm=llm)
    org.state.sources[PLATO] = SourceStats(read=2, awakened=1)
    org.state.sources["John Dewey, How We Think"] = SourceStats(read=5, awakened=1)
    _quiet(org)

    ep = org.heartbeat()
    assert "RETURN" in llm.calls and llm.calls.index("RETURN") < llm.calls.index("ANTICIPATE")
    assert f'You go back to "{PLATO}"' in llm.prompts["RETURN"]  # the higher yield: 1 of 2 against 1 of 5
    born = org.state.returns[-1]["questions"]
    assert born and org.state.questions[born[0]].trigger == "exposure" and org.state.questions[born[0]].has_record
    assert org.state.questions[born[0]].origin_sources == [PLATO]
    stats = org.state.sources[PLATO]
    assert stats.returned == 1 and stats.awakened == 2 and stats.next_passage >= 1
    assert ep is not None and ep.question_id == born[0]  # what it found there, it then pursues
    diary = (org.home / "diary.md").read_text(encoding="utf-8")
    assert f"Spare time: I went back to *{PLATO}*" in diary and "It made me wonder again" in diary


def test_after_a_rest_it_goes_back_first(records):
    llm = ScriptedLLM(recognize="no")
    org = CuriosityOrganism(records, llm=llm)
    org.state.sources[PLATO] = SourceStats(read=1, awakened=1)
    org._just_rested = True
    org.heartbeat()
    assert llm.calls.index("RETURN") < llm.calls.index("ANTICIPATE")
    assert not org._just_rested


def test_no_text_has_made_it_curious_yet_so_there_is_nothing_to_go_back_to(records):
    llm = ScriptedLLM(recognize="no")
    org = CuriosityOrganism(records, llm=llm)
    org._just_rested = True
    org.heartbeat()
    assert "RETURN" not in llm.calls


def test_it_reads_on_where_it_stopped(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM(recognize="no"))
    org.state.sources[PLATO] = SourceStats(read=1, awakened=1, next_passage=0)
    first = org.senses.reread(PLATO, start=0, k=1)
    assert first and first[0].label == "S1" and first[0].heading == PLATO
    assert org._spare_time([])
    assert org.state.sources[PLATO].next_passage >= 1


def test_without_exposure_it_neither_goes_back_nor_learns_interests(records):
    records.organism.curiosity.exposure = False
    llm = ScriptedLLM(recognize="no")
    org = CuriosityOrganism(records, llm=llm)
    org.state.sources[PLATO] = SourceStats(read=1, awakened=1)
    org._just_rested = True
    _quiet(org)
    org.heartbeat()
    assert "RETURN" not in llm.calls and org.state.interests == {}


def test_interests_grow_from_pursuit_and_more_from_answers(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM())
    org.heartbeat()  # Q1 is made precise and a quote answers it (2); its child, with the same record, is pursued (1)
    words = org._distinctive(RECORD["unknown"])
    assert words and all(org.state.interests.get(w) == pytest.approx(3.0) for w in words)

    near = org.state.add_question("Which feeling marks a philosopher, for Plato?", trigger="gap", importance=0.2)
    org._apply_record(near, dict(_parse_record(RECORD), unknown="the feeling that marks a philosopher in Plato"))
    far = org.state.add_question("How should soups be seasoned?", trigger="gap", importance=0.2)
    org._apply_record(far, dict(_parse_record(RECORD), unknown="the seasoning that suits a vegetable soup"))
    assert near.interest_fit > 0.3 and far.interest_fit == 0.0
    assert near.intensity > far.intensity  # curiosity is stronger near what it already cares about


def test_the_research_map_names_the_texts_that_made_it_curious_and_its_interests(records):
    from curiosity_ai.organism.research_map import render_research_map

    org = CuriosityOrganism(records, llm=ScriptedLLM(recognize="no"))
    org.heartbeat()
    org.state.sources[PLATO].awakened = max(1, org.state.sources[PLATO].awakened)
    text = render_research_map(org)
    assert "## Texts that made it curious" in text and f"*{PLATO}*" in text
    assert "## What it has come to be interested in" in text


def test_a_session_reports_exposure(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM(recognize="no"))
    org.state.sources[PLATO] = SourceStats(read=1, awakened=1)
    org._just_rested = True
    report = org.run_session(heartbeats=2)
    m = report.metrics
    assert m["returns_to_texts"] == 1 and m["born_on_return"] >= 1
    assert 0.0 < m["read_from_curious_texts"] <= 1.0
    assert "**Voluntary exposure and interests:**" in report.markdown


def test_passages_from_a_text_come_in_order_and_wrap_around(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM())
    library = org.senses.library
    all_of_kant = library.passages_from("Immanuel Kant, The Critique of Pure Reason", k=10)
    assert len(all_of_kant) == 1
    again = library.passages_from("Immanuel Kant, The Critique of Pure Reason", start=1, k=1)
    assert again[0].text == all_of_kant[0].text
    assert library.passages_from("Nobody, Nothing") == []
