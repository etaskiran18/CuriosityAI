"""Two layers and transience (records architecture).

The pre-conceptual layer only draws attention: a confident prediction a text contradicted, an unexpected passage,
a text never read before. A question is born when an unknown is named. A curiosity lasts while attention stays
with it: untouched it fades; what nothing could answer is let go; a curiosity looked at too long without an
answer waits.
"""
from __future__ import annotations

import pytest

from curiosity_ai.organism import CuriosityOrganism
from curiosity_ai.organism.organism import _parse_record
from curiosity_ai.organism.state import Episode, Visit

from .organism_fakes import RECORD, ScriptedLLM


@pytest.fixture
def records(config):
    config.organism.curiosity.architecture = "records"
    return config


def _record(org, text: str, unknown: str, *, intensity: float = 0.5, trigger: str = "gap"):
    q = org.state.add_question(text, trigger=trigger, importance=0.5)
    org._apply_record(q, dict(_parse_record(RECORD), unknown=unknown))
    q.intensity = intensity
    return q


def test_attention_comes_from_surprise_unexpected_passages_and_new_texts(records):
    llm = ScriptedLLM(recognize="no", expect_contradiction=True, contradict_for_real=True)
    org = CuriosityOrganism(records, llm=llm)
    ep = org.heartbeat()
    kinds = [a["kind"] for a in ep.attention]
    assert "surprise" in kinds and "unexpected" in kinds and "novelty" in kinds
    surprise = next(a for a in ep.attention if a["kind"] == "surprise")
    assert surprise["strength"] == pytest.approx(0.7) and surprise["quote"]
    assert "*What drew my attention:* a prediction a text contradicted" in (org.home / "diary.md").read_text(encoding="utf-8")

    # The texts it has now read are no longer new.
    read = set(org.state.sources)
    assert read and all(org.state.sources[h].read >= 1 for h in read)
    later = org.heartbeat()
    assert not any(a["kind"] == "novelty" and a["source"] in read for a in later.attention)


def test_without_the_attention_layer_an_unnamed_question_is_born_all_the_same(records):
    records.organism.curiosity.attention = False

    def conceive(text: str) -> dict:
        return {"unknown": "", "answer_form": ""} if "understanding" in text.lower() else dict(RECORD)

    org = CuriosityOrganism(records, llm=ScriptedLLM(recognize="no", conceive=conceive))
    ep = org.heartbeat()
    child = org.state.questions[ep.new_question_ids[0]]
    assert not child.has_record and ep.not_conceived == []


def test_a_curiosity_untouched_fades_but_one_it_began_with_does_not(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM())
    q = org.state.questions["Q1"]
    org._apply_record(q, _parse_record(RECORD))
    weak = _record(org, "What is the opposite of wonder?", "the feeling opposite to wonder", intensity=0.16)
    seed = _record(org, "Where does wonder come from?", "the source of wonder", intensity=0.16, trigger="seed")
    weak.status = "deferred"
    ep = Episode(heartbeat=org.state.heartbeat, question_id="Q1", question=q.text)
    org._transience(q, [], ep)
    assert weak.status == "faded" and weak.intensity == 0.0 and ep.faded == [weak.id]
    assert "my attention stayed elsewhere" in weak.status_reason
    assert seed.status == "open" and seed.intensity == pytest.approx(0.144)
    assert q.intensity == pytest.approx(1.0)  # what it looked at grows stronger


def test_what_touches_a_curiosity_makes_it_stronger(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM(recognize="no"))
    near = _record(org, "Which feeling marks a philosopher?", "the feeling that marks a philosopher and with which philosophy begins", intensity=0.3)
    near.status = "deferred"
    far = _record(org, "How are soups made?", "the order in which vegetables go into a soup", intensity=0.3)
    far.status = "deferred"
    org.heartbeat()
    assert near.intensity > 0.3 and near.touched_at == org.state.heartbeat
    assert far.intensity == pytest.approx(0.27)


def test_its_doubt_alone_lets_nothing_go(records):
    """A curiosity it doubts anything could answer waits at birth; it goes only if a look finds nothing."""
    def conceive(text: str) -> dict:
        return dict(RECORD, unknowable="yes") if "understanding" in text.lower() else dict(RECORD)

    org = CuriosityOrganism(records, llm=ScriptedLLM(recognize="no", conceive=conceive))
    ep = org.heartbeat()
    child = org.state.questions[ep.new_question_ids[0]]
    assert child.status == "deferred" and ep.decisions[0]["kind"] == "doubtful"


def test_a_look_that_finds_nothing_confirms_the_doubt(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM(recognize="no"))
    q = org.state.questions["Q1"]
    org._apply_record(q, dict(_parse_record(RECORD), reachable=False))
    silent = Episode(heartbeat=1, question_id="Q1", question=q.text, informativeness=0.0)
    org._transience(q, [], silent)
    assert q.status == "unanswerable" and silent.decisions[0]["kind"] == "unreachable"
    assert "the texts said nothing of it" in q.status_reason


def test_a_look_that_finds_something_lifts_the_doubt(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM(recognize="no"))
    q = org.state.questions["Q1"]
    org._apply_record(q, dict(_parse_record(RECORD), reachable=False))
    spoke = Episode(heartbeat=1, question_id="Q1", question=q.text, informativeness=0.6)
    org._transience(q, [], spoke)
    assert q.status == "open" and q.reachable is None


def test_a_curiosity_looked_at_too_long_waits(records):
    records.organism.curiosity.pursuit_limit = 2
    org = CuriosityOrganism(records, llm=ScriptedLLM(recognize="no"))
    q = org.state.questions["Q1"]
    q.visits.append(Visit(heartbeat=0, confidence_before=0.25, confidence_after=0.3, prediction_error=0.2, informativeness=0.5))
    ep = org.heartbeat()
    assert ep.question_id == "Q1" and q.status == "deferred"
    assert q.status_reason == "I have looked 2 times without finding what would answer it"


def test_the_word_list_gives_way_to_the_models_judgment(records):
    unknown = "What undiscovered forms of curiosity remain to be identified in machines?"
    settle = {
        "answer": "Wonder is the feeling of a philosopher.", "would_be_wrong_if": "A text showed otherwise about wonder.",
        "confidence": 0.5, "learned": [], "new_questions": [{"question": unknown, "trigger": "surprise", "importance": 0.6}],
    }
    org = CuriosityOrganism(records, llm=ScriptedLLM(recognize="no", settle=settle, expect_contradiction=True, contradict_for_real=True))
    ep = org.heartbeat()
    assert ep.unknowable_questions == [] and ep.new_question_ids  # made precise and judged by the model, not by its words

    records.organism.curiosity.transience = False
    records.organism.home = records.organism.home + "-without"
    org = CuriosityOrganism(records, llm=ScriptedLLM(recognize="no", settle=settle, expect_contradiction=True, contradict_for_real=True))
    ep = org.heartbeat()
    assert ep.unknowable_questions and not ep.new_question_ids


def test_a_curiosity_remembers_the_texts_it_was_born_from(records):
    llm = ScriptedLLM(recognize="no", expect_contradiction=True, contradict_for_real=True)
    org = CuriosityOrganism(records, llm=llm)
    ep = org.heartbeat()
    child = org.state.questions[ep.new_question_ids[0]]
    surprising = [a["source"] for a in ep.attention if a["kind"] in ("surprise", "unexpected")]
    assert child.origin_sources and set(child.origin_sources) <= set(surprising)
    assert all(org.state.sources[h].awakened == 1 for h in child.origin_sources)


def test_a_session_reports_attention_and_transience(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM(recognize="no"))
    report = org.run_session(heartbeats=2)
    m = report.metrics
    assert m["attention_heartbeats"] >= 1 and m["attention_events"].get("novelty", 0) >= 1
    assert 0.0 <= m["attention_to_curiosity"] <= 1.0
    assert "**Attention and transience:**" in report.markdown


def test_a_faded_curiosity_comes_back_when_it_is_brought_up_again(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM())
    gone = _record(org, "What is the opposite of wonder?", "the feeling opposite to wonder", intensity=0.0)
    org._set_status(gone, "faded", "my attention stayed elsewhere")
    q, outcome = org._adopt_question("What is the opposite of wonder?", trigger="gap", parent_id="Q1", importance=0.5, inherited_surprise=0.0)
    assert q is gone and outcome == "reawakened"
    assert gone.status == "open" and gone.intensity == pytest.approx(0.5)


def test_a_curiosity_that_comes_back_still_respects_the_room(records):
    records.organism.curiosity.max_pursued = 2
    org = CuriosityOrganism(records, llm=ScriptedLLM())
    for q in list(org.state.questions.values()):
        q.status = "deferred"
    a = _record(org, "Which feeling marks a philosopher?", "the feeling that marks a philosopher", intensity=0.9)
    b = _record(org, "Why did Plato call Iris the child of wonder?", "the reason Iris was called the child of wonder", intensity=0.2)
    waiting = _record(org, "What is the opposite of wonder?", "the feeling opposite to wonder", intensity=0.6)
    waiting.status = "deferred"
    org._reopen(waiting, "a new question touched it")
    assert waiting.status == "open" and b.status == "deferred" and a.status == "open"
