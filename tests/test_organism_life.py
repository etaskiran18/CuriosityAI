from __future__ import annotations

import json

import pytest

from curiosity_ai.organism import CuriosityOrganism, OrganismError
from curiosity_ai.organism.organism import Check, Comparison

from .conftest import SEED
from .organism_fakes import ScriptedLLM


def test_birth_creates_seed_questions_and_a_diary(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    assert org.newborn
    assert [q.text for q in org.state.questions.values()] == [SEED]
    assert (org.home / "mind.json").exists()
    assert "Diary of Curiosity" in (org.home / "diary.md").read_text(encoding="utf-8")


def test_a_heartbeat_learns_only_from_verified_quotes(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    ep = org.heartbeat()

    assert ep.question_id == "Q1"
    statuses = {c["expectation"]: c["status"] for c in ep.checks}
    assert statuses == {1: "confirmed", 2: "not_addressed", 3: "unverified"}
    assert ep.rejected_quotes == 2  # the invented contradiction and the invented "unexpected" finding
    assert len(ep.unexpected) == 1
    # Brier: E1 was predicted with p=0.8 and confirmed, (1 - 0.8)^2 = 0.04; the finding counts half.
    assert ep.prediction_error == pytest.approx((0.04 + 0.5) / 3.5)
    assert ep.informativeness == pytest.approx(1.5 / 3.5)
    assert ep.brier == pytest.approx(0.04)

    grounded = org.state.beliefs[ep.new_belief_ids[0]]
    interpretive = org.state.beliefs[ep.new_belief_ids[1]]
    assert grounded.evidence and grounded.confidence == pytest.approx(0.6)
    assert grounded.evidence[0].citation.startswith("[LOCAL:")
    assert interpretive.interpretive and interpretive.confidence == pytest.approx(0.35)


def test_a_question_folded_inside_the_current_one_is_not_new(config):
    llm = ScriptedLLM(settle={"answer": "a", "confidence": 0.4, "new_questions": ["Is wonder the same thing as curiosity?"]})
    org = CuriosityOrganism(config, llm=llm)
    ep = org.heartbeat()
    assert ep.new_question_ids == []
    assert len(org.state.questions) == 1


def test_new_questions_are_born_and_restatements_are_dropped(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    ep = org.heartbeat()
    assert len(ep.new_question_ids) == 1
    child = org.state.questions[ep.new_question_ids[0]]
    assert child.parent_id == "Q1"
    assert child.trigger == "surprise"
    assert child.inherited_surprise == pytest.approx(ep.prediction_error)
    assert len(org.state.questions) == 2


FALSIFIABLE = "A passage showing wonder without any perplexity would refute this."


def test_confidence_moves_in_bounded_steps(config):
    config.organism.evidence_ceiling_base = 1.0
    llm = ScriptedLLM(stance="DEFEND", settle={"answer": "Certain now.", "would_be_wrong_if": FALSIFIABLE, "confidence": 1.0, "learned": [], "new_questions": []})
    org = CuriosityOrganism(config, llm=llm)
    ep = org.heartbeat()
    assert ep.prior_confidence == pytest.approx(0.4)
    assert ep.confidence == pytest.approx(0.4 + config.organism.max_confidence_step)


def _sure(answer: str) -> dict:
    return {"answer": answer, "would_be_wrong_if": FALSIFIABLE, "confidence": 1.0, "learned": [], "new_questions": []}


def test_confidence_cannot_outrun_the_evidence(config):
    """An answer is held at most as firmly as the quotes the judge accepted for it allow (0.5 + 0.1 each)."""
    org = CuriosityOrganism(config, llm=ScriptedLLM(stance="DEFEND", settle=_sure("Wonder is the feeling of a philosopher.")))
    ep = org.heartbeat()
    q = org.state.questions["Q1"]
    assert q.visits[-1].support == 1 and len(q.support) == 1 and "feeling of a philosopher" in q.support[0].quote
    assert ep.confidence == pytest.approx(config.organism.evidence_ceiling_base + config.organism.evidence_ceiling_per_support)


def test_an_answer_no_quote_supports_stays_at_the_base(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM(stance="DEFEND", settle=_sure("Certain now.")))
    ep = org.heartbeat()
    assert org.state.questions["Q1"].support == [] and ep.confidence == pytest.approx(config.organism.evidence_ceiling_base)


def test_a_new_answer_does_not_inherit_the_old_answers_confidence(config):
    """A 7B model held 0.70 for an answer a paper confirmed, swapped it for an unsupported guess and kept 0.70."""
    from curiosity_ai.organism.research_map import render_research_map
    from curiosity_ai.organism.state import Evidence, Visit

    guess = "Lightning polarization sets which way the energy goes in the inner sky."
    org = CuriosityOrganism(config, llm=ScriptedLLM(stance="DEFEND", settle=_sure(guess)))
    q = org.state.questions["Q1"]
    q.confidence, q.answer = 0.7, "Philosophy begins in wonder, the feeling of a philosopher."
    q.visits.append(Visit(heartbeat=0, confidence_before=0.5, confidence_after=0.7, prediction_error=0.0, informativeness=0.5, support=2))
    q.support = [Evidence(citation="[BOOK:Theaetetus]", quote="wonder is the feeling of a philosopher", source_title="Plato, Theaetetus")]
    ep = org.heartbeat()
    assert ep.answer == guess and ep.unsupported_answer and q.support == []
    assert ep.confidence == pytest.approx(config.organism.evidence_ceiling_base)
    assert "cannot keep the confidence the old answer had" in (org.home / "diary.md").read_text(encoding="utf-8")
    assert "no quote the judge accepted for this answer" in render_research_map(org)


def test_certainty_cannot_grow_without_evidence(config):
    llm = ScriptedLLM(fail_steps=("COMPARE",), settle={"answer": "Certain now.", "confidence": 0.95, "learned": [], "new_questions": []})
    org = CuriosityOrganism(config, llm=llm)
    ep = org.heartbeat()
    assert ep.confidence <= ep.prior_confidence + 0.05 + 1e-9
    assert any(e.startswith("compare:") for e in ep.errors)


def test_contradiction_doubts_a_belief_and_reopens_what_relied_on_it(config):
    config.organism.seed_questions = ["Can reason answer every question that it raises, or are some questions beyond it?"]
    llm = ScriptedLLM(settle={"answer": "No.", "confidence": 0.5, "contradicts": ["B1 is contradicted by Kant", "B99"], "new_questions": []})
    org = CuriosityOrganism(config, llm=llm)
    belief = org.state.add_belief("Reason can answer every question that it raises.", confidence=0.6)
    unrelated = org.state.add_belief("Soup needs salt and a bay leaf.", confidence=0.6)
    settled = org.state.add_question("Is reason sufficient for every inquiry it begins?", status="settled", confidence=0.85, related_beliefs=[belief.id])

    ep = org.heartbeat()

    assert ep.doubted_belief_ids == [belief.id]
    assert org.state.beliefs[belief.id].status == "doubted"
    assert org.state.beliefs[belief.id].confidence == pytest.approx(0.45)
    assert org.state.beliefs[unrelated.id].confidence == pytest.approx(0.6)
    assert org.state.questions[settled.id].status == "open"
    assert org.state.questions[settled.id].confidence == pytest.approx(0.7)


def test_the_mind_survives_a_restart(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    org.heartbeat()
    again = CuriosityOrganism(config, llm=ScriptedLLM())
    assert not again.newborn
    assert again.state.heartbeat == 1
    assert again.state.questions.keys() == org.state.questions.keys()
    assert again.state.questions["Q1"].visits[0].prediction_error == pytest.approx(0.54 / 3.5)
    assert "Heartbeat 1" in (org.home / "diary.md").read_text(encoding="utf-8")
    episodes = (org.home / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(episodes[0])["question_id"] == "Q1"


def test_a_human_question_is_adopted_or_merged(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    same = org.ask("Plato says philosophy begins in wonder: what is wonder, and is it the same as curiosity?")
    assert same.id == "Q1" and same.importance == pytest.approx(0.9)
    new = org.ask("Can a machine be curious, or only act as if it were?")
    assert new.trigger == "human" and new.id == "Q2"


def test_shared_observations_are_noticed_and_become_part_of_the_library(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    org.feed("Boredom is not the absence of curiosity but its frustrated form: a restless wish for something worth attending to.", title="On boredom")
    org.heartbeat()
    noticed = [q for q in org.state.questions.values() if q.trigger == "observation"]
    assert len(noticed) == 1 and noticed[0].importance >= 0.7
    assert not org.state.unread_inbox
    assert org.senses.library.search("boredom frustrated restless", k=1)[0].kind == "inbox"
    assert "A human shared something" in (org.home / "diary.md").read_text(encoding="utf-8")


def test_a_silent_model_does_not_kill_the_organism(config):
    llm = ScriptedLLM(fail_steps=("ANTICIPATE", "COMPARE", "WONDER", "SKEPTIC", "SETTLE"))
    org = CuriosityOrganism(config, llm=llm)
    ep = org.heartbeat()
    assert ep.status_after == "no-thought"
    assert org.state.questions["Q1"].visits == []
    with pytest.raises(OrganismError):
        org.live(3)


def test_a_failed_dialogue_still_leaves_a_lesson(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM(fail_steps=("WONDER",)))
    ep = org.heartbeat()
    assert ep.dialogue == []
    assert ep.status_after == "open"
    assert ep.new_belief_ids
    assert any(e.startswith("wonder:") for e in ep.errors)


def test_dialogue_alternates_wonder_and_skeptic(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    ep = org.heartbeat()
    assert [t["voice"] for t in ep.dialogue] == ["Wonder", "Skeptic", "Wonder"]
    assert not ep.dialogue[0]["text"].lower().startswith("wonder:")


def test_reflection_updates_the_self_model_and_temperament(config):
    config.organism.reflect_every = 2
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    org.live(2)
    st = org.state
    assert "perplexity" in st.self_model.understanding_of_curiosity
    assert st.vitals_history and st.vitals_history[-1]["heartbeat"] == 2
    assert any(q.trigger == "reflection" for q in st.questions.values())
    assert "Reflection after heartbeat 2" in (org.home / "diary.md").read_text(encoding="utf-8")


def test_grounded_episodes_become_training_data(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    org.heartbeat()
    sft = (org.home / "experience" / "sft.jsonl").read_text(encoding="utf-8").splitlines()
    example = json.loads(sft[0])
    assert "Evidence:" in example["messages"][1]["content"]
    assert not (org.home / "experience" / "preference_pairs.jsonl").exists()


def test_a_verified_contradiction_yields_a_preference_pair(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM(contradict_for_real=True))
    ep = org.heartbeat()
    assert any(c["status"] == "contradicted" for c in ep.checks)
    pair = json.loads((org.home / "experience" / "preference_pairs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert pair["rejected"] == ep.prior_answer and pair["chosen"] == ep.answer


def test_archive_moves_a_life_aside(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    org.heartbeat()
    archived = CuriosityOrganism.archive(config)
    assert archived is not None and (archived / "mind.json").exists()
    assert CuriosityOrganism(config, llm=ScriptedLLM()).newborn


def test_surprise_scoring():
    comp = Comparison([Check(1, "confirmed", "S1", "q"), Check(2, "contradicted", "S2", "q"), Check(3)], [])
    assert comp.scores(3) == pytest.approx((1 / 3, 2 / 3))
    assert Comparison([Check(1, "contradicted", "S1", "q"), Check(2, "contradicted", "S1", "r")], []).scores(2) == (1.0, 1.0)
    assert Comparison([Check(1), Check(2)], []).scores(2) == (0.0, 0.0)
    assert Comparison([], [{"finding": "f", "source": "S1", "quote": "q"}]).scores(0) == (0.5, 0.5)


def test_silence_and_side_remarks_are_not_much_of_a_surprise():
    """Texts that ignore my expectations and only say unrelated things do not shock me."""
    unrelated = [{"finding": "typhoid", "source": "S1", "quote": "q1"}, {"finding": "deduction", "source": "S1", "quote": "q2"}]
    surprise, informativeness = Comparison([Check(i) for i in range(1, 5)], unrelated).scores(4)
    assert surprise == pytest.approx(0.2)
    assert informativeness == pytest.approx(0.2)


def test_the_same_words_cannot_be_both_expected_and_unexpected(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM(echo_unexpected=True))
    ep = org.heartbeat()
    assert ep.unexpected == []
    assert ep.rejected_quotes == 0
    assert ep.prediction_error == pytest.approx(0.04 / 3)  # only E1's small Brier error: nothing unexpected


def test_a_voice_cannot_speak_for_the_other():
    from curiosity_ai.organism.organism import _strip_voice_prefix

    text = "SKEPTIC: You assume doubt is good.\n\nWONDER: How can we clarify this?"
    assert _strip_voice_prefix(text, "Skeptic") == "You assume doubt is good."


def test_with_nothing_left_open_it_reflects_and_finds_a_new_question(config):
    llm = ScriptedLLM()
    org = CuriosityOrganism(config, llm=llm)
    org.state.questions["Q1"].status = "settled"
    ep = org.heartbeat()
    assert "REFLECT" in llm.calls
    assert ep is not None
    assert org.state.questions[ep.question_id].trigger == "reflection"


def test_living_forever_waits_for_a_silent_model_instead_of_dying(config):
    from curiosity_ai.config import BodyConfig
    from curiosity_ai.organism.body import Body

    waits: list[float] = []

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) == 3:
            raise KeyboardInterrupt  # the human stops it

    body = Body(BodyConfig(enabled=False), sleep=sleep, clock=lambda: 0.0)
    llm = ScriptedLLM(fail_steps=("ANTICIPATE", "COMPARE", "WONDER", "SKEPTIC", "SETTLE"))
    org = CuriosityOrganism(config, llm=llm, body=body)
    with pytest.raises(KeyboardInterrupt):
        org.live(forever=True)
    assert waits == [60, 120, 240]  # 1, 2, 4 minutes, and on up to 15


def test_a_judge_that_fails_does_not_erase_the_quotes_an_answer_had(config):
    from curiosity_ai.organism.state import Evidence, Visit

    answer = "Philosophy begins in wonder, the feeling of a philosopher."
    org = CuriosityOrganism(config, llm=ScriptedLLM(stance="DEFEND", settle=_sure(answer), fail_steps=("JUDGE",)))
    q = org.state.questions["Q1"]
    q.confidence, q.answer = 0.6, answer
    q.visits.append(Visit(heartbeat=0, confidence_before=0.5, confidence_after=0.6, prediction_error=0.0, informativeness=0.5, support=1))
    kept = Evidence(citation="[BOOK:Theaetetus]", quote="wonder is the feeling of a philosopher", source_title="Plato, Theaetetus")
    q.support = [kept]
    ep = org.heartbeat()
    assert kept in q.support and not ep.unsupported_answer and ep.confidence >= 0.6 - 1e-9

