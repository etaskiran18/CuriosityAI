"""Staying on topic, and an honest diagnosis of restless curiosity."""
from __future__ import annotations

import pytest

from curiosity_ai.organism import CuriosityOrganism
from curiosity_ai.organism.drive import DriveSettings, Vitals, diagnose, read_drive, regulate, topic_factor
from curiosity_ai.organism.state import Question, Temperament, Visit
from curiosity_ai.organism.textutil import asks_about_the_unknown, token_set, topic_relevance

from .organism_fakes import ScriptedLLM

NEW = [
    {"question": "Does understanding something end our wonder about it, or deepen it?", "trigger": "surprise", "importance": 0.9},
    {"question": "How can we develop adaptive learning strategies for diverse cultural contexts?", "trigger": "gap", "importance": 0.95},
    {"question": "Is perplexity a feeling or a judgment about what we lack?", "trigger": "objection", "importance": 0.8},
]


def first_look(org) -> None:
    """Q1 has been looked at once already, so gaps and objections may now give birth."""
    org.state.questions["Q1"].visits.append(Visit(heartbeat=0, confidence_before=0.25, confidence_after=0.25, prediction_error=0.0, informativeness=0.0))


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
    first_look(org)
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
    first_look(org)
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



# -- depth before breadth --------------------------------------------------------------


GAP = {"question": "Is perplexity a feeling or a judgment about what we lack?", "trigger": "gap", "importance": 0.8}


def test_a_gap_question_waits_for_a_second_look(config):
    org = CuriosityOrganism(config, llm=proposing([GAP]))
    ep = org.heartbeat()
    assert ep.new_question_ids == [] and ep.held_back_questions == [GAP["question"]]
    assert "Kept for a second look" in (org.home / "diary.md").read_text(encoding="utf-8")
    again = CuriosityOrganism(config, llm=proposing([GAP]))
    again.state.questions["Q1"].last_visited = None  # no refractory pause in this test
    ep2 = again.heartbeat()
    assert ep2.question_id == "Q1" and len(ep2.new_question_ids) == 1  # second look: the gap is real


def test_a_surprise_without_evidence_is_only_a_gap(config):
    surprise = {**GAP, "trigger": "surprise"}
    ep = CuriosityOrganism(config, llm=proposing([surprise], judge="neither")).heartbeat()
    assert ep.new_question_ids == [] and ep.held_back_questions  # the judge accepted no evidence


def test_doubting_an_old_guess_without_evidence_is_no_contradiction(config):
    """In a real run a 3B model doubted an earlier belief at almost every heartbeat, with no quote the
    judge accepted, and each doubt let a "contradiction" question be born at once. In the one test mistral
    doubted 55 times in 45 heartbeats, guesses included, while the texts contradicted one prediction."""
    contradiction = {**GAP, "trigger": "contradiction"}

    def live(judge: str):
        settle = {"answer": "a", "would_be_wrong_if": "b c d e", "confidence": 0.4, "learned": [], "contradicts": ["B1"],
                  "new_questions": [contradiction]}
        org = CuriosityOrganism(config, llm=ScriptedLLM(settle=settle, judge=judge))
        belief = org.state.add_belief("Perplexity is only a feeling of lack.", confidence=0.5)
        org.state.questions["Q1"].related_beliefs.append(belief.id)
        return org.heartbeat()

    ep = live("neither")
    assert ep.doubted_belief_ids == [] and ep.paper_doubt_ids == ["B1"]  # words alone doubt nothing
    assert ep.new_question_ids == [] and ep.held_back_questions == [GAP["question"]]
    config.organism.home += "-with-evidence"
    ep = live("supports")
    assert ep.doubted_belief_ids == [] and ep.paper_doubt_ids == ["B1"]  # a confirmation is no contradiction either


def test_a_real_surprise_gives_birth_at_once(config):
    surprise = {**GAP, "trigger": "surprise"}
    ep = CuriosityOrganism(config, llm=proposing([surprise])).heartbeat()
    assert len(ep.new_question_ids) == 1  # confirmed evidence and a real prediction error


def test_new_texts_bring_it_back_to_the_question():
    q = Question(id="Q1", text="q", last_visited=5)
    q.visits.append(Visit(heartbeat=5, confidence_before=0.3, confidence_after=0.3, prediction_error=0.0, informativeness=0.0))
    t, s = Temperament(), DriveSettings()
    pause = read_drive(q, t, 6, 0, s)
    q.news = True
    back = read_drive(q, t, 6, 0, s)
    assert pause.refractory == 0.5 and back.refractory == 0.0
    assert back.news == pytest.approx(0.25) and back.total > pause.total * 2


# -- keeping guesses from taking over ---------------------------------------------------------


@pytest.mark.parametrize("question, unknowable", [
    ("How do undiscovered plasma instabilities interact with radiation-belt particles?", True),
    ("What other yet-to-be-identified structures might influence the propagation of whistlers?", True),
    ("What are the specific roles of lesser-known magnetospheric structures in whistler propagation?", True),
    ("How can we distinguish the influence of unidentified magnetospheric structures?", True),
    ("What is the evidence for overlooked structures or frequency ranges?", True),
    ("How does the plasmapause guide lightning-generated whistlers?", False),
    ("Why does a known density notch duct whistlers at 4 to 8 kHz?", False),
])
def test_a_question_about_things_nobody_has_identified(question, unknowable):
    assert asks_about_the_unknown(question) is unknowable


def test_questions_no_text_can_answer_are_not_asked(config):
    unknown = {"question": "How do undiscovered kinds of wonder shape philosophy?", "trigger": "surprise", "importance": 0.9}
    org = CuriosityOrganism(config, llm=proposing([unknown]))
    ep = org.heartbeat()
    assert ep.new_question_ids == [] and ep.unknowable_questions == [unknown["question"]]
    assert "which no text can answer" in (org.home / "diary.md").read_text(encoding="utf-8")


def test_reflection_builds_on_what_the_texts_support(config):
    class Reflecting(ScriptedLLM):
        def json_chat(self, system, user, schema_hint, **kw):
            if "[REFLECT]" in system:
                self.calls.append("REFLECT")
                self.prompts["REFLECT"] = user
                return {"understanding": "What wonder is remains unclear and is not yet fully understood.", "reflection": "r",
                        "focus_question": "What undiscovered feelings come before wonder?"}
            return super().json_chat(system, user, schema_hint, **kw)

    llm = Reflecting()
    org = CuriosityOrganism(config, llm=llm)
    before = org.state.self_model.understanding_of_curiosity
    org.heartbeat()
    org.reflect()
    prompt = llm.prompts["REFLECT"]
    assert "What the texts support so far" in prompt and "Your guesses (beliefs no quote supports yet)" in prompt
    assert org.state.self_model.understanding_of_curiosity == before  # "not yet understood" is no theory
    assert not any(q.trigger == "reflection" for q in org.state.questions.values())  # no paper can answer the focus


def test_a_bare_stance_word_is_no_stance(config):
    class Terse(ScriptedLLM):
        def chat(self, system, user, **kw):
            if "[WONDER]" in system and "Begin your reply with exactly one word" in user:
                self._enter(system, user)
                return "REVISE"
            return super().chat(system, user, **kw)

    ep = CuriosityOrganism(config, llm=Terse()).heartbeat()
    wonder = [t for t in ep.dialogue if t["voice"] == "Wonder"]
    assert len(wonder) >= 2 and "stance" not in wonder[-1] and ep.stance == ""




@pytest.mark.parametrize("asked, kept", [
    ("What is the direct interaction between the plasmapause and the ionospheric FAI, and how does this interaction impact "
     "the overall propagation and interaction of lightning-generated whistlers in the inner magnetosphere?",
     "What is the direct interaction between the plasmapause and the ionospheric FAI?"),
    ("What are the specific mechanisms that shape the propagation pathways of nonducted lightning-generated whistlers in the "
     "inner magnetosphere, as suggested by the discovery of SR whistlers?",
     "What are the specific mechanisms that shape the propagation pathways of nonducted lightning-generated whistlers in the inner magnetosphere?"),
    ("What factors beyond the specific lightning events and ionospheric conditions contribute to the observed discrepancies in "
     "the literature regarding the role of specular reflection and the quasielectrostatic field in lightning-generated whistler "
     "propagation to the inner magnetosphere?", ""),  # one question, but too long to be answered by one paper
    ("Does understanding something end our wonder about it, or deepen it?", "Does understanding something end our wonder about it, or deepen it?"),
])
def test_its_own_questions_are_single_and_short(asked, kept):
    """In the one test new questions grew to 45 words, two questions in one."""
    from curiosity_ai.organism.organism import _clean_question

    assert _clean_question(asked) == kept


def test_a_persons_question_keeps_its_words(config):
    long_question = ("How does the plasmapause organize the pathways of lightning-generated whistlers, and what does this mean "
                     "for the electron density we infer from them at different L-shells?")
    q = CuriosityOrganism(config, llm=ScriptedLLM()).ask(long_question)
    assert q.text == long_question
