"""Predictions that can be wrong, a blind judge, and a debate with consequences."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from curiosity_ai.organism import CuriosityOrganism
from curiosity_ai.organism.judge import Judge, Pair, rating_of, verdict_of
from curiosity_ai.organism.organism import Check, Comparison, _invented_quotes, _name_sources, _quotes_a_passage, _stance, _without_belief_labels
from curiosity_ai.organism.senses import Observation
from curiosity_ai.organism.state import Evidence
from curiosity_ai.organism.textutil import is_hedged, is_influence_only, is_non_answer, is_strawman_falsifier, lexically_related, token_set, vagueness

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
    assert is_hedged("Carpenter will explain the difference between ordinary and extraordinary modes of whistler propagation")
    assert is_hedged("Insights are provided on the interaction between the plasmapause and the ionosphere")
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


def test_a_hedged_answer_gains_little(config):
    """"Structures may play a role" survives any finding; a 7B model settled questions at 0.85 with such answers."""
    config.organism.evidence_ceiling_base = 1.0
    hedged = "The plasmapause may guide whistlers and could potentially affect the radiation belts."
    org = CuriosityOrganism(config, llm=ScriptedLLM(stance="DEFEND", settle=settle(answer=hedged)))
    ep = org.heartbeat()
    assert ep.hedged_answer and not ep.vague and ep.confidence == pytest.approx(ep.prior_confidence + 0.05)
    assert "hedged (may, could)" in (org.home / "diary.md").read_text(encoding="utf-8")
    config.organism.home += "-plain"
    plain = CuriosityOrganism(config, llm=ScriptedLLM(stance="DEFEND", settle=settle())).heartbeat()
    assert not plain.hedged_answer and plain.confidence > plain.prior_confidence + 0.05


def test_the_voices_see_what_was_expected_not_its_labels(config):
    """Shown only "did not address: E1, E2, E3", a 7B model took E1, E2 and E3 for structures in the magnetosphere."""
    llm = ScriptedLLM()
    CuriosityOrganism(config, llm=llm).heartbeat()
    for step in ("WONDER", "SKEPTIC", "SETTLE"):
        prompt = llm.prompts[step]
        assert 'What you expected, "Plato: Philosophy begins in wonder' in prompt or 'What you expected, "Philosophy begins in wonder' in prompt
        assert "did not clearly address what you expected:" in prompt and "Curiosity becomes intellectual" in prompt
        assert not re.search(r"\bE\d\b", prompt), step


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
    (Path(config.corpus.path) / "tu.md").write_text(
        "---\ntitle: Lightning currents and the lower ionosphere\nauthor: J.-N. Tu\n---\n\n" + "Lightning heats the lower ionosphere. " * 20,
        encoding="utf-8",
    )
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
    # the passage it read cites Singh itself: a paraphrase of that citation is not one from memory
    skeptic = "In [S2], Singh et al. (1992, 1993) state that the dispersion of whistlers yields the electron density."
    assert _cites_from_memory(skeptic)
    assert not _cites_from_memory(skeptic, ["The dispersion property (Singh et al., 1992, 1993, 1997) is widely used."])
    assert _cites_from_memory(skeptic, ["The dispersion property (Sazhin et al., 1992) is widely used."])


# -- after the third research run ---------------------------------------------------------


@pytest.mark.parametrize("falsifier", [
    "A study that demonstrates that neither resonance nor plasma instabilities play any role in the propagation.",
    "Evidence showing that the interplay has no influence on space weather effects.",
    "If the anisotropy instability were the only factor shaping the propagation.",
    "A study that clearly demonstrates that lower band chorus whistlers are the sole cause of electron loss.",
    "Plasma instabilities are always the dominant mechanism, regardless of conditions.",
    "Evidence that the plasmapause's dynamics are solely determined by factors independent of lightning.",
])
def test_a_falsifier_that_only_denies_any_role_is_a_strawman(falsifier):
    assert is_strawman_falsifier(falsifier)


@pytest.mark.parametrize("falsifier", [
    "Evidence showing that resonance is a primary mechanism under different plasmapause conditions.",
    "Evidence showing that the interplay dominates under low-density and isotropic plasma conditions.",
    "Whistler arrival times at L=3 lag the density model by more than 0.5 s.",
    "If densities inverted from whistlers differ from in situ densities by more than 20 percent.",
])
def test_a_falsifier_that_names_a_finding_is_not(falsifier):
    assert not is_strawman_falsifier(falsifier)


def test_an_answer_wrong_only_if_nothing_were_at_play_gains_little(config):
    config.organism.evidence_ceiling_base = 1.0
    strawman = "Evidence showing that wonder plays no role at all in starting philosophical inquiry."
    org = CuriosityOrganism(config, llm=ScriptedLLM(stance="DEFEND", settle=settle(would_be_wrong_if=strawman)))
    ep = org.heartbeat()
    assert ep.strawman_falsifier and ep.confidence == pytest.approx(ep.prior_confidence + 0.05)
    assert "nothing at all, or one thing alone" in (org.home / "diary.md").read_text(encoding="utf-8")


def test_an_answer_that_is_a_question_is_no_answer(config):
    question = "Can we infer, from the available evidence, when wonder turns into inquiry?"
    ep = CuriosityOrganism(config, llm=ScriptedLLM(settle=settle(answer=question))).heartbeat()
    assert ep.answer == "Wonder is the feeling that starts philosophical inquiry."  # its answer from before looking


def test_a_hedged_belief_stays_an_interpretation(config):
    learned = [{"belief": "Wonder may be the feeling that starts philosophy", "source": "S1"}]
    org = CuriosityOrganism(config, llm=ScriptedLLM(settle=settle(learned=learned)))
    ep = org.heartbeat()
    belief = org.state.beliefs[ep.new_belief_ids[0]]
    assert belief.interpretive and belief.confidence == pytest.approx(0.35)


def test_a_belief_resting_on_a_text_yields_only_to_a_contradiction(config):
    """Peirce: no paper doubt. A 7B model doubted the person's own article 36 times while nothing contradicted it."""
    config.organism.seed_questions = ["Can reason answer every question that it raises, or are some questions beyond it?"]

    class Believing(ScriptedLLM):
        """Gives the prediction that the texts will contradict ("Reason can answer every question") p = 0.7."""

        def json_chat(self, system, user, schema_hint, **kw):
            reply = super().json_chat(system, user, schema_hint, **kw)
            if "[ANTICIPATE]" in system:
                reply["expectations"] = [dict(e, probability=0.7) if "every question" in e["claim"] else e for e in reply["expectations"]]
            return reply

    def live(contradict_for_real: bool, believed: bool = True):
        kind = Believing if believed else ScriptedLLM
        llm = kind(contradict_for_real=contradict_for_real, settle=settle(contradicts=["B1"]))
        org = CuriosityOrganism(config, llm=llm)
        quote = Evidence(citation="[BOOK:Kant]", quote="human reason is burdened by questions it cannot dismiss", source_title="Kant")
        belief = org.state.add_belief("Reason can answer every question that it raises.", confidence=0.6, evidence=[quote])
        return org, belief, org.heartbeat()

    org, belief, ep = live(contradict_for_real=False)
    assert ep.contradicted == 0 and ep.doubted_belief_ids == [] and ep.paper_doubt_ids == [belief.id]
    assert org.state.beliefs[belief.id].status == "held" and org.state.beliefs[belief.id].confidence == pytest.approx(0.6)
    assert "gives way only to a text" in (org.home / "diary.md").read_text(encoding="utf-8")
    config.organism.home += "-contradicted"
    org, belief, ep = live(contradict_for_real=True)
    assert ep.contradicted == 1 and ep.doubted_belief_ids == [belief.id]
    # Contradicting what it never believed (p = 0.3) is no surprising fact: no doubt from it.
    config.organism.home += "-unbelieved"
    org, belief, ep = live(contradict_for_real=True, believed=False)
    assert ep.contradicted == 1 and ep.doubted_belief_ids == [] and ep.paper_doubt_ids == [belief.id]



def test_a_belief_a_text_has_just_repeated_is_not_doubted(config):
    """mistral doubted, and retracted, "The interaction between electrons and the primary whistler wave packet plays a
    role ..." in the heartbeats in which the judge accepted "The electrons interacted with the primary whistler wave
    packet" for a prediction."""
    config.organism.seed_questions = ["Can reason answer every question that it raises, or are some questions beyond it?"]
    llm = ScriptedLLM(contradict_for_real=True, expect_contradiction=True, settle=settle(contradicts=["B1"]))
    org = CuriosityOrganism(config, llm=llm)
    org.heartbeat()  # (to learn which quote the judge accepts for E1)
    agreeing = next(c["quote"] for c in org.state.recent_episodes[-1].checks if c["status"] == "confirmed")
    config.organism.home += "-again"
    org = CuriosityOrganism(config, llm=ScriptedLLM(contradict_for_real=True, expect_contradiction=True, settle=settle(contradicts=["B1"])))
    # it speaks of what a text contradicted, but repeats what a quote accepted just now says
    belief = org.state.add_belief(f"Reason can answer every question it raises: {agreeing}.", confidence=0.35)
    ep = org.heartbeat()
    assert ep.contradicted == 1 and ep.doubted_belief_ids == [] and ep.agreed_doubt_ids == [belief.id]
    assert "says much the same: no reason for doubt" in (org.home / "diary.md").read_text(encoding="utf-8")


def test_a_contradiction_needs_a_second_look(config):
    """mistral called "We report the discovery of specularly reflected (SR) whistler in which the lightning energy ...
    reaches the magnetosphere" a contradiction of "SR whistlers provide a more efficient low-latitude channel ...";
    from that a question about a discrepancy in the literature was born."""
    config.organism.seed_questions = ["Can reason answer every question that it raises, or are some questions beyond it?"]
    llm = ScriptedLLM(contradict_for_real=True, expect_contradiction=True, contradiction_holds=False)
    ep = CuriosityOrganism(config, llm=llm).heartbeat()
    assert ep.contradicted == 0 and llm.calls.count("JUDGE") >= 2
    judged = [j for j in ep.judgments if j["kind"] == "expectation" and "every question" in j["claim"]]
    assert judged and judged[0]["judge"] == "neither" and "second look" in judged[0]["reason"]
    config.organism.home += "-confirmed"
    ep = CuriosityOrganism(config, llm=ScriptedLLM(contradict_for_real=True, expect_contradiction=True)).heartbeat()
    assert ep.contradicted == 1  # the judge said yes again: a real surprise


def test_a_quote_in_none_of_the_texts_is_marked_as_invented():
    passage = "We report the discovery of specularly reflected (SR) whistler in which the lightning energy injected into the ionosphere at low latitudes reaches the magnetosphere."
    skeptic = ('[S2] directly contradicts it, stating that "previous claims that lightning energy injected into the ionosphere at '
               'low latitudes can reach the magnetosphere may need revision".')
    assert _invented_quotes(skeptic, [passage]) == [
        "previous claims that lightning energy injected into the ionosphere at low latitudes can reach the magnetosphere may need revision"]
    honest = 'As [S2] says, "the lightning energy injected into the ionosphere at low latitudes reaches the magnetosphere".'
    assert _invented_quotes(honest, [passage]) == []
    titled = 'In "Specularly reflected whistler: A low-latitude channel" the authors report it.'
    assert _invented_quotes(titled, [passage, "Specularly reflected whistler: A low-latitude channel to couple lightning energy"]) == []


def test_a_guess_is_not_doubted_on_words_alone(config):
    """In the one test mistral doubted 55 times in 45 heartbeats, nearly always a guess, while the texts contradicted
    one prediction; "The ionosphere acts as a waveguide for lightning-generated whistlers" was retracted."""
    config.organism.seed_questions = ["Can reason answer every question that it raises, or are some questions beyond it?"]

    def live(**kw):
        org = CuriosityOrganism(config, llm=ScriptedLLM(settle=settle(contradicts=["B1"]), **kw))
        belief = org.state.add_belief("Reason can answer every question that it raises.", confidence=0.35)  # a guess
        return org, belief, org.heartbeat()

    org, belief, ep = live()
    assert ep.doubted_belief_ids == [] and ep.paper_doubt_ids == [belief.id]
    assert org.state.beliefs[belief.id].status == "held"
    assert "no text I read contradicted it" in (org.home / "diary.md").read_text(encoding="utf-8")
    config.organism.home += "-contradicted"
    org, belief, ep = live(contradict_for_real=True, expect_contradiction=True)
    assert ep.doubted_belief_ids == [belief.id] and org.state.beliefs[belief.id].confidence == pytest.approx(0.2)


# -- after the fourth research run --------------------------------------------------------

TOPIC = token_set("lightning whistlers VLF inner magnetosphere plasmapause radiation belt electromagnetic energy ionosphere plasma")
CENTRAL = "The large-scale plasma environment is expected to play a central role in selecting these propagation pathways."


@pytest.mark.parametrize("claim, quote", [
    ("The role of lightning polarization is context-dependent in the Earth's inner magnetosphere.", CENTRAL),
    ("Plasma instabilities resonate with the frequency of lightning-generated whistlers", CENTRAL),
    ("Plasma instabilities amplify the energy of lightning-generated whistlers", "SR whistlers to the lightning energy reaching the magnetosphere."),
    ("Resonant scattering of electrons by whistler waves is a primary mechanism for electron scattering loss.",
     "Lightning-generated whistlers carry electromagnetic energy from lightning into Earth's inner magnetosphere"),
])
def test_a_quote_that_shares_no_real_words_with_a_claim_never_reaches_the_judge(claim, quote):
    assert not lexically_related(claim, quote, TOPIC)


@pytest.mark.parametrize("claim, quote", [
    ("The plasma environment plays a crucial role in shaping the pathways of lightning-generated waves.", CENTRAL),
    ("The plasmapause acts as a refracting point for whistler-mode energy",
     "The plasmapause can act as a one-sided VLF waveguide, allowing whistler-mode energy to propagate along its density gradient"),
    ("Specularly reflected whistlers provide an additional pathway for lightning energy to reach the magnetosphere.",
     "Specularly reflected (SR) whistlers provide an additional pathway by which lightning energy injected at low latitudes"),
])
def test_a_quote_that_restates_a_claim_does(claim, quote):
    assert lexically_related(claim, quote, TOPIC)


@pytest.mark.parametrize("answer, open_", [
    ("Further research is crucial to uncover the specific plasma conditions that shape the pathways.", True),
    ("The specific conditions are yet to be clearly defined, but higher density may play a role.", True),
    ("The precise role of polarization is not definitively known.", True),
    ("The role of lightning polarization needs further exploration to improve predictions.", True),
    ("The plasmapause organizes the pathways of lightning-generated waves.", False),
    ("Whistlers inside the plasmasphere are ducted; outside it they propagate unducted.", False),
])
def test_an_answer_that_only_says_the_question_is_open(answer, open_):
    assert is_non_answer(answer) is open_


def test_an_answer_that_only_says_the_question_is_open_earns_no_confidence(config):
    config.organism.evidence_ceiling_base = 1.0
    still_open = "What wonder is, exactly, is not yet fully understood, and further research is needed."
    ep = CuriosityOrganism(config, llm=ScriptedLLM(stance="DEFEND", settle=settle(answer=still_open))).heartbeat()
    assert ep.vague and ep.confidence <= ep.prior_confidence


def test_a_belief_its_quote_does_not_mention_stays_an_interpretation(config):
    learned = [{"belief": "Soup needs salt and a bay leaf to taste right.", "source": "S1"}]
    org = CuriosityOrganism(config, llm=ScriptedLLM(settle=settle(learned=learned)))  # the scripted judge accepts anything
    ep = org.heartbeat()
    assert org.state.beliefs[ep.new_belief_ids[0]].interpretive
    assert not any(j["kind"] == "belief" for j in ep.judgments)  # the pair never reached the judge



_SOURCES = {
    "S1": Observation("S1", "c1", "Numerical modelling of subionospheric VLF propagation with perturbations", "A. Author", "t", "papers"),
    "S2": Observation("S2", "c2", "Whistler (radio)", "Wikipedia", "t", "web"),
    "S3": Observation("S3", "c3", "My article (draft)", "Human observer", "t", "inbox", own=True),
}


@pytest.mark.parametrize("text, belief, answer", [
    ("The model in [S1] considers lightning.",
     "The model in \u201cNumerical modelling of subionospheric VLF propagation\u2026\u201d considers lightning.", None),
    ("Dispersion depends on electron density (S2).",
     "Dispersion depends on electron density.", "Dispersion depends on electron density (\u201cWhistler (radio)\u201d)."),
    ("Both [S1] and [S2] show that density falls.",
     "Both \u201cNumerical modelling of subionospheric VLF propagation\u2026\u201d and \u201cWhistler (radio)\u201d show that density falls.", None),
    ("Density falls [S1] and [S2].", "Density falls.", None),
    ("[S3] says density falls.", "Your draft says density falls.", None),
    ("The model in [S9] considers lightning.", "The model in one of the texts considers lightning.", None),
])
def test_a_source_label_becomes_the_texts_name(text, belief, answer):
    """'[S1]' means nothing outside its heartbeat; cutting it out left "The model in considers"."""
    assert _name_sources(text, _SOURCES, keep_citations=False) == belief
    if answer is not None:
        assert _name_sources(text, _SOURCES, keep_citations=True) == answer


def test_the_theory_says_guess_without_labels():
    """The map's reader does not know what [B4] is."""
    theory = "These waves are affected by diffraction (guess: [B4]), and their properties reveal density (guess: [B5], [B6])."
    assert _without_belief_labels(theory) == "These waves are affected by diffraction (a guess), and their properties reveal density (a guess)."
    assert _without_belief_labels("Whistlers reveal the density [B3].") == "Whistlers reveal the density."
    assert _without_belief_labels("A text says so (S1).") == "A text says so (S1)."


def test_beliefs_and_answers_name_their_texts(config):
    learned = [{"belief": "The account in [S1] holds that wonder starts philosophy.", "source": "S1"}]
    answer = "Wonder is the felt perplexity that starts inquiry, as argued in [S1]."
    org = CuriosityOrganism(config, llm=ScriptedLLM(settle=settle(learned=learned, answer=answer)))
    ep = org.heartbeat()
    title = re.sub(r"^\[S1\] (?:[^,]*, )?", "", ep.sources[0]).split()[0]
    statement = org.state.beliefs[ep.new_belief_ids[0]].statement
    assert "[S1]" not in statement and f"in \u201c{title}" in statement
    assert "[S1]" not in ep.answer and f"as argued in \u201c{title}" in ep.answer


# -- predictions a text could contradict ----------------------------------------------------


@pytest.mark.parametrize("claim", [
    "The plasmapause significantly influences the propagation of lightning-generated waves in the inner magnetosphere.",
    "The frequency of lightning-generated waves plays a crucial role in determining their pathways in the inner magnetosphere.",
    "Ionospheric plasma density affects the specific pathways of lightning-generated waves in the inner magnetosphere.",
    "The magnetic field strength plays a role in the propagation of lightning-generated waves in the inner magnetosphere.",
])
def test_a_claim_that_only_names_an_influence_cannot_be_contradicted(claim):
    assert is_influence_only(claim)


@pytest.mark.parametrize("claim", [
    "The plasmapause plays a crucial role in the refraction and reflection of lightning-generated waves.",
    "Higher plasma density slows the group velocity of whistlers.",
    "The UM propagation mode is more dominant under certain plasmapause conditions.",
    "Whistlers are ducted along field lines inside the plasmasphere but not outside it.",
    "Wonder is the feeling of a philosopher.",
])
def test_a_claim_with_a_direction_size_or_mechanism_can(claim):
    assert not is_influence_only(claim)


def test_predictions_that_only_name_an_influence_are_asked_again(config):
    llm = ScriptedLLM(weak="first")
    ep = CuriosityOrganism(config, llm=llm).heartbeat()
    assert llm.calls.count("ANTICIPATE") == 2
    assert not any(p["weak"] or p["hedged"] for p in ep.predictions)
    retry_prompt = [prompt for step, prompt in llm.history if step == "ANTICIPATE"][1]
    assert "only say that one thing influences another" in retry_prompt


def test_confirming_a_prediction_that_could_not_fail_is_no_support(config):
    """Popper: only a prediction that could have failed is tested when it comes true."""
    config.organism.evidence_ceiling_base = 1.0
    org = CuriosityOrganism(config, llm=ScriptedLLM(weak="always", stance="DEFEND", settle=settle()))
    ep = org.heartbeat()
    assert ep.predictions[0]["weak"] and ep.checks[0]["status"] == "confirmed"
    assert org.state.questions["Q1"].visits[-1].support == 0
    assert ep.confidence <= ep.prior_confidence + 0.05 + 1e-9
    assert "no text could contradict it" in (org.home / "diary.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("claim, hedged", [
    ("The paper discusses the role of whistler waves generated by lightning in revealing the electron density", True),
    ("Carpenter examines the impact of the plasmapause on whistler propagation", True),
    ("The study provides insights into the coupling of lightning energy", True),
    ("Shastun discusses the relationship between electron density and the cutoff frequency", True),
    ("Carpenter describes ducted whistlers as confined to density enhancements", False),
    ("The study finds that whistler arrival times reveal the equatorial density", False),
])
def test_naming_a_topic_in_the_present_tense_is_no_claim(claim, hedged):
    assert is_hedged(claim) is hedged


def test_a_probability_written_into_the_claim_is_taken_out(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    ant = org._parse_anticipation({"expectations": [
        {"author": "Golden", "claim": "Most whistlers reach the equator inside the plasmasphere, with a probability of 0.75"},
        {"author": "Golden", "claim": "The density drops by a factor of 5 at the plasmapause (probability: 60%)", "probability": 0.4},
    ]})
    first, second = ant.predictions
    assert first.claim == "Most whistlers reach the equator inside the plasmasphere" and first.probability == pytest.approx(0.75)
    assert second.claim == "The density drops by a factor of 5 at the plasmapause" and second.probability == pytest.approx(0.4)


def test_a_retry_that_only_denies_its_own_predictions_is_not_kept(config):
    """mistral, asked for claims a text could contradict, negated them ("X is not crucial", p = 0.05)."""
    believed = [
        {"author": "Plato", "claim": "Wonder plays a significant role in philosophy, the feeling of a philosopher", "probability": 0.8},
        {"author": "Dewey", "claim": "Curiosity becomes intellectual through problems found in observation", "probability": 0.65},
    ]
    denials = [
        {"author": "Plato", "claim": "Wonder is not the feeling of a philosopher", "probability": 0.05},
        {"author": "Dewey", "claim": "Curiosity never becomes intellectual through problems", "probability": 0.1},
    ]

    class Denying(ScriptedLLM):
        def json_chat(self, system, user, schema_hint, **kw):
            reply = super().json_chat(system, user, schema_hint, **kw)
            if "[ANTICIPATE]" in system:
                reply["expectations"] = denials if "Write all your predictions again" in user else believed
            return reply

    llm = Denying()
    ep = CuriosityOrganism(config, llm=llm).heartbeat()
    assert llm.calls.count("ANTICIPATE") == 2  # the influence-only claim was asked again...
    assert [p["claim"] for p in ep.predictions] == [b["claim"] for b in believed]  # ...but the denials were not kept
    retry_prompt = [prompt for step, prompt in llm.history if step == "ANTICIPATE"][1]
    assert "Do not turn them into denials" in retry_prompt


def test_an_author_the_library_does_not_have_is_dropped_from_a_prediction(config):
    """mistral wrote "Smith's research shows that ..." and even "Dewey found that ..." in a space-physics library."""
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    parsed = org._parse_anticipation({"expectations": [
        {"author": "Smith", "claim": "Smith's research shows that changes in electron density alter the speed of whistlers", "probability": 0.8},
        {"author": "Plato", "claim": "Plato holds that philosophy begins in wonder", "probability": 0.7},
        {"author": "Johnson", "claim": "According to Johnson et al. (2019), whistlers start near the equator", "probability": 0.6},
        {"author": "Theaetetus", "claim": "Wonder is the feeling of a philosopher", "probability": 0.7},
    ]})
    smith, plato, johnson, title = parsed.predictions
    assert smith.author == "" and smith.claim == "Changes in electron density alter the speed of whistlers"
    assert plato.author == "Plato" and plato.claim.startswith("Plato holds")  # Plato is in the library
    assert johnson.author == "" and johnson.claim == "Whistlers start near the equator"
    assert title.author == ""  # a title's first word is nobody's name (mistral made "Thunderstorms" an author)

