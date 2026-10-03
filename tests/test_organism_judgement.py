"""Predictions that can be wrong, a blind judge, and a debate with consequences."""
from __future__ import annotations

import pytest

from curiosity_ai.organism import CuriosityOrganism
from curiosity_ai.organism.judge import Judge, Pair, rating_of, verdict_of
from curiosity_ai.organism.organism import Check, Comparison, _quotes_a_passage, _stance
from curiosity_ai.organism.senses import Observation
from curiosity_ai.organism.textutil import is_hedged, vagueness

from .organism_fakes import ScriptedLLM

FALSIFIABLE = "A passage showing wonder without any perplexity would refute this."


def settle(**overrides):
    base = {
        "answer": "Wonder is the felt perplexity that starts inquiry.",
        "would_be_wrong_if": FALSIFIABLE,
        "confidence": 0.9,
        "learned": [],
        "new_questions": [],
    }
    return {**base, **overrides}


# -- predictions -----------------------------------------------------------------


def test_brier_scoring_rewards_confident_truths_and_punishes_confident_errors():
    comp = Comparison([Check(1, "confirmed", "S1", "q"), Check(2, "contradicted", "S1", "r"), Check(3)], [])
    surprise, informativeness = comp.scores([0.9, 0.9, 0.5])
    assert surprise == pytest.approx((0.01 + 0.81) / 3)
    assert informativeness == pytest.approx(2 / 3)
    assert comp.brier([0.9, 0.9, 0.5]) == pytest.approx(0.41)
    assert Comparison([Check(1)], []).brier([0.7]) is None  # nothing addressed: nothing to score


def test_hedges_are_recognised():
    assert is_hedged("The texts may discuss the role of wonder")
    assert is_hedged("Some texts might connect curiosity with problems")
    assert not is_hedged("Dewey holds that curiosity dies if it is not used at the right moment")


def test_hedged_predictions_are_asked_again_once(config):
    llm = ScriptedLLM(hedge="first")
    org = CuriosityOrganism(config, llm=llm)
    ep = org.heartbeat()
    assert llm.calls.count("ANTICIPATE") == 2
    assert [p["hedged"] for p in ep.predictions] == [False, False, False]
    assert ep.predictions[0]["author"] == "Plato" and ep.predictions[0]["probability"] == pytest.approx(0.8)
    assert ep.predictions[2]["probability"] == pytest.approx(0.3)  # "30%"


def test_a_model_that_keeps_hedging_is_recorded_as_hedging(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM(hedge="always"))
    ep = org.heartbeat()
    assert ep.predictions and all(p["hedged"] for p in ep.predictions)
    assert "hedged: it could never be wrong" in (org.home / "diary.md").read_text(encoding="utf-8")


def test_without_retry_one_call_is_made(config):
    config.organism.prediction_retry = False
    llm = ScriptedLLM(hedge="always")
    CuriosityOrganism(config, llm=llm).heartbeat()
    assert llm.calls.count("ANTICIPATE") == 1


# -- the judge --------------------------------------------------------------------


def test_the_judge_sees_the_claim_and_quote_but_not_the_verdict(config):
    llm = ScriptedLLM()
    CuriosityOrganism(config, llm=llm).heartbeat()
    prompt = next(p for step, p in llm.history if step == "JUDGE")
    assert "CLAIM:" in prompt and "QUOTE" in prompt
    assert "confirmed" not in prompt.lower() and "wonder" not in prompt.split("CLAIM:")[0].lower()


def test_a_quote_beside_the_point_confirms_nothing(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM(judge="neither"))
    ep = org.heartbeat()
    first = ep.checks[0]
    assert first["claimed"] == "confirmed" and first["judge"] == "neither" and first["status"] == "not_addressed"
    assert ep.support == 0 and ep.unexpected == []  # the finding was not accepted either
    assert all(org.state.beliefs[b].interpretive for b in ep.new_belief_ids)  # nothing grounded
    assert ep.confidence <= ep.prior_confidence + 0.05 + 1e-9
    diary = (org.home / "diary.md").read_text(encoding="utf-8")
    assert "the judge found the quote beside the point" in diary


def test_the_judge_can_turn_a_confirmation_into_a_contradiction(config):
    rule = lambda claim, quote: "contradicts" if "philosophy begins in wonder" in claim.lower() else "supports"
    ep = CuriosityOrganism(config, llm=ScriptedLLM(judge=rule)).heartbeat()
    assert ep.checks[0]["claimed"] == "confirmed" and ep.checks[0]["status"] == "contradicted"
    assert ep.contradicted == 1


def test_beliefs_are_grounded_only_with_the_judges_approval(config):
    rule = lambda claim, quote: "neither" if "marks a philosopher" in claim else "supports"
    org = CuriosityOrganism(config, llm=ScriptedLLM(judge=rule))
    ep = org.heartbeat()
    plato = next(org.state.beliefs[b] for b in ep.new_belief_ids if "philosopher" in org.state.beliefs[b].statement)
    assert plato.interpretive and plato.confidence == pytest.approx(0.35)
    assert any(j["kind"] == "belief" and j["judge"] == "neither" for j in ep.judgments)


def test_an_approved_belief_keeps_the_quote_the_judge_saw(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    ep = org.heartbeat()
    grounded = [org.state.beliefs[b] for b in ep.new_belief_ids if org.state.beliefs[b].evidence]
    assert len(grounded) == 1 and len(grounded[0].evidence) == 1
    judged = next(j for j in ep.judgments if j["kind"] == "belief")
    assert judged["quote"] == grounded[0].evidence[0].quote


def test_without_the_judges_approval_nothing_counts(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM(fail_steps=("JUDGE",)))
    ep = org.heartbeat()
    assert ep.checks[0]["claimed"] == "confirmed" and ep.checks[0]["judge"] == "unjudged"
    assert ep.checks[0]["status"] == "not_addressed" and ep.support == 0 and ep.unexpected == []
    assert any(e.startswith("judge:") for e in ep.errors)
    assert all(org.state.beliefs[b].interpretive for b in ep.new_belief_ids)
    assert "the judge did not check it" in (org.home / "diary.md").read_text(encoding="utf-8")


def test_without_a_judge_the_organism_grades_itself_as_before(config):
    config.organism.judge.enabled = False
    llm = ScriptedLLM(judge="neither")
    org = CuriosityOrganism(config, llm=llm)
    ep = org.heartbeat()
    assert "JUDGE" not in llm.calls and ep.judgments == []
    assert ep.checks[0]["status"] == "confirmed" and ep.support == 1


def test_a_separate_judge_model_can_be_given(config):
    judge_llm = ScriptedLLM(judge="neither")
    org = CuriosityOrganism(config, llm=ScriptedLLM(), judge_llm=judge_llm)
    ep = org.heartbeat()
    assert "JUDGE" in judge_llm.calls and ep.support == 0


def test_verdict_and_rating_parsing():
    assert verdict_of("Supports.") == "supports" and verdict_of("contradicts") == "contradicts"
    assert verdict_of("neither") == "neither" and verdict_of("supports or contradicts or neither") == "unjudged"
    assert rating_of("2") == pytest.approx(2 / 3) and rating_of(3) == 1.0 and rating_of(0) == 0.0
    assert rating_of("0.5") == pytest.approx(0.5) and rating_of("none") is None


def test_the_judge_tolerates_a_short_answer():
    class Lazy:
        def json_chat(self, *a, **k):
            return {"verdicts": [{"pair": 2, "verdict": "supports"}]}

    verdicts, ratings = Judge(Lazy()).judge([Pair("a b c d", "e f g h"), Pair("i j k l", "m n o p")], questions=["Why?"])
    assert [v.verdict for v in verdicts] == ["unjudged", "supports"] and ratings == [None]


# -- the debate -------------------------------------------------------------------------


@pytest.mark.parametrize("text, stance", [
    ("CONCEDE: you are right about the word.", "concede"),
    ("**Defend** - the passage says so.", "defend"),
    ("REVISED. I change one part.", "revise"),
    ("I think the objection fails.", ""),
])
def test_stances_are_read(text, stance):
    assert _stance(text)[0] == stance


def test_conceding_an_objection_never_raises_confidence(config):
    config.organism.evidence_ceiling_base = 1.0
    org = CuriosityOrganism(config, llm=ScriptedLLM(stance="CONCEDE", settle=settle()))
    ep = org.heartbeat()
    assert ep.stance == "concede" and ep.confidence <= ep.prior_confidence


def test_revising_raises_confidence_only_a_little(config):
    config.organism.evidence_ceiling_base = 1.0
    ep = CuriosityOrganism(config, llm=ScriptedLLM(stance="REVISE", settle=settle())).heartbeat()
    assert ep.stance == "revise" and ep.confidence == pytest.approx(ep.prior_confidence + 0.10)


def test_an_answer_too_vague_to_be_wrong_earns_no_confidence(config):
    config.organism.evidence_ceiling_base = 1.0
    vague = "Curiosity is a complex, multifaceted interplay of various factors that differ across cultural contexts."
    assert vagueness(vague) >= 2
    org = CuriosityOrganism(config, llm=ScriptedLLM(stance="DEFEND", settle=settle(answer=vague)))
    ep = org.heartbeat()
    assert ep.vague and ep.confidence <= ep.prior_confidence
    assert "too vague to be wrong" in (org.home / "diary.md").read_text(encoding="utf-8")


def test_an_answer_that_cannot_say_what_would_refute_it_gains_little(config):
    config.organism.evidence_ceiling_base = 1.0
    ep = CuriosityOrganism(config, llm=ScriptedLLM(stance="DEFEND", settle=settle(would_be_wrong_if=""))).heartbeat()
    assert ep.falsifier == "" and ep.confidence == pytest.approx(ep.prior_confidence + 0.05)


def test_the_skeptic_quoting_a_real_passage_is_noticed():
    obs = Observation("S1", "c", "Theaetetus", "Plato", "for wonder is the feeling of a philosopher, and philosophy begins in wonder.", "corpus")
    assert _quotes_a_passage('Plato says "wonder is the feeling of a philosopher, and philosophy begins in wonder" here.', [obs])
    assert not _quotes_a_passage('He says "curiosity is the engine of all progress in the world" here.', [obs])


def test_a_copied_schema_hint_is_not_a_probability(config):
    from curiosity_ai.organism.organism import CuriosityOrganism as Org

    org = Org(config, llm=ScriptedLLM())
    parsed = org._parse_anticipation({"expectations": [
        {"author": "Dewey", "claim": "Curiosity dies unless it is used at the right moment", "probability": "0.05 to 0.95"},
        {"author": "who will say it, e.g. Dewey", "claim": "what they will say, stated plainly", "probability": "0.5"},
    ]})
    assert len(parsed.predictions) == 1  # the echoed example is dropped
    assert parsed.predictions[0].probability == pytest.approx(config.organism.default_probability)


def test_a_copied_instruction_after_the_stance_is_dropped():
    stance, rest = _stance("DEFEND (the objection fails: show why from the evidence)\nThe text says so.")
    assert stance == "defend" and rest == "The text says so."


def test_a_finding_that_only_copies_its_quote_is_judged_for_relevance(config):
    class CopyingModel(ScriptedLLM):
        def json_chat(self, system, user, schema_hint, **kw):
            data = super().json_chat(system, user, schema_hint, **kw)
            if "[COMPARE]" in system:
                for u in data.get("unexpected", []):
                    u["finding"] = u["quote"]
            return data

    llm = CopyingModel()
    CuriosityOrganism(config, llm=llm).heartbeat()
    first_judgement = next(prompt for step, prompt in llm.history if step == "JUDGE")
    assert "CLAIM: This passage bears directly on the question: Plato says philosophy begins in wonder" in first_judgement


def test_a_fragment_too_short_to_say_anything_is_not_a_finding(config):
    class Fragments(ScriptedLLM):
        def json_chat(self, system, user, schema_hint, **kw):
            data = super().json_chat(system, user, schema_hint, **kw)
            if "[COMPARE]" in system:
                from .organism_fakes import passages_in, words

                first = next(iter(passages_in(user).values()))
                data["unexpected"] = [{"finding": "outer barbarians.", "source": "S1", "quote": words(first, 0, 5)}]
            return data

    ep = CuriosityOrganism(config, llm=Fragments()).heartbeat()
    assert ep.unexpected == [] and ep.rejected_quotes == 1  # only the invented E3 quote counts as invented


# -- from the first research run (mistral 7B, a space-physics topic) ----------------------


def test_a_reply_cut_off_by_the_length_limit_keeps_what_was_complete():
    from curiosity_ai.utils import extract_json_object

    cut = (
        '{ "answer": "Lightning-generated whistlers indirectly illuminate the inner magnetosphere.", "confidence": 0.3, '
        '"expectations": [{"author": "Carpenter", "claim": "Whistlers trace the plasmapause", "probability": 0.7}, '
        '{"author": "Tu", "claim": "Lightning heats the D region by quasi-electrostatic fie'
    )
    data = extract_json_object(cut)
    assert data["answer"].startswith("Lightning") and data["expectations"][0]["claim"] == "Whistlers trace the plasmapause"
    with pytest.raises(ValueError):
        extract_json_object("no json here at all")


def test_will_discuss_is_a_hedge():
    assert is_hedged("Carpenter will discuss the structure of electromagnetic pulses")
    assert is_hedged("Gallagher will provide an overview of the waves")
    assert not is_hedged("Lightning will heat the lower ionosphere by ten kelvin")


def test_a_long_author_list_becomes_one_name(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    parsed = org._parse_anticipation({"expectations": [{
        "author": "J.-N. Tu, J. T. Emmert, R. A. Marshall, Chih-Te Hsu and Roderick A. Heelis",
        "claim": "Joule heating from lightning currents is negligible above 100 km", "probability": 0.6,
    }]})
    assert parsed.predictions[0].author == "J.-N. Tu"


def test_citations_from_memory_are_flagged_in_the_debate(config):
    class Citing(ScriptedLLM):
        def chat(self, system, user, **kw):
            text = super().chat(system, user, **kw)
            if "[WONDER]" in system and "Begin your reply" in user:
                return text + ' [1] T. Nakamura et al., "Whistler Observations During Storms," Journal of Geophysical Research, vol. 82, 1977.'
            return text

    org = CuriosityOrganism(config, llm=Citing())
    ep = org.heartbeat()
    assert ep.dialogue[-1].get("citations") == "unverified"
    assert "cites papers it was not shown" in (org.home / "diary.md").read_text(encoding="utf-8")
    assert "never cite a paper you were not shown" in org.llm.prompts["SKEPTIC"].lower()


def test_a_quoted_passage_that_cites_others_is_not_a_citation_from_memory():
    from curiosity_ai.organism.organism import _cites_from_memory

    assert not _cites_from_memory('[S4] says: "the whistler intensity was measured by Inan et al., 1990 on board the satellite"')
    assert _cites_from_memory('As shown by Inan et al. (1990), whistlers precipitate electrons.')
    assert _cites_from_memory('Nakamura et al., "Whistler observations of the inner magnetosphere", show it.')
