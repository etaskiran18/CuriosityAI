"""The curiosity organism: a persistent mind whose behavior is driven by curiosity.

One heartbeat is one act of inquiry, modelled on Dewey's "complete act of
thought" and Peirce's doubt-belief cycle:

    choose   which question pulls hardest (the curiosity drive)
    predict  commit to an answer and checkable expectations before looking
    observe  read passages from the library (and the web, if enabled)
    compare  measure surprise: which expectations were confirmed or contradicted,
             counting only claims backed by a quote found in the passage
    argue    an inner dialogue between Wonder and Skeptic
    settle   revise the answer and confidence, form or doubt beliefs,
             and let new questions be born from surprise, contradiction and gaps
    learn    record prediction error and change of mind, which feed the
             drive's learning-progress and boredom terms

Every few heartbeats the organism reflects: it measures its own vital signs,
diagnoses its way of being curious (healthy wonder, restless curiositas,
dogmatic slumber or aporetic numbness), adjusts its temperament, wakes
incubated questions and updates its own theory of curiosity.
"""
from __future__ import annotations

import random
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from ..config import AppConfig
from ..schema import _as_list, _as_str, utc_now_iso
from ..utils import strip_front_matter, write_jsonl
from . import prompts as P
from .diary import Diary
from .drive import (
    DIAGNOSES,
    DriveReading,
    DriveSettings,
    boredom,
    choose,
    clamp,
    compute_vitals,
    diagnose,
    learning_progress,
    read_drive,
    regulate,
)
from .senses import Observation, Senses, write_inbox_item
from .state import TRIGGERS, Belief, Episode, Evidence, MindState, Question, Temperament, Visit, load_mind, save_mind
from .textutil import clip, jaccard, one_line, overlap, token_set, verify_quote


class ChatModel(Protocol):
    def chat(self, system: str, user: str, *, temperature: float | None = None, json_mode: bool = False, max_tokens: int | None = None) -> str: ...

    def json_chat(self, system: str, user: str, schema_hint: str, *, temperature: float | None = None, max_tokens: int | None = None) -> dict[str, Any]: ...


class OrganismError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Intermediate results of one heartbeat
# ---------------------------------------------------------------------------


@dataclass
class Anticipation:
    answer: str = ""
    confidence: float | None = None
    expectations: list[str] = field(default_factory=list)


@dataclass
class Check:
    expectation: int
    status: str = "not_addressed"  # confirmed | contradicted | not_addressed | unverified
    source: str = ""
    quote: str = ""


@dataclass
class Comparison:
    checks: list[Check] = field(default_factory=list)
    unexpected: list[dict[str, str]] = field(default_factory=list)
    rejected_quotes: int = 0

    def count(self, status: str) -> int:
        return sum(1 for c in self.checks if c.status == status)

    def quotes_by_source(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        pairs = [(c.source, c.quote) for c in self.checks if c.status in ("confirmed", "contradicted") and c.quote]
        pairs += [(u["source"], u["quote"]) for u in self.unexpected]
        for source, quote in pairs:
            if quote not in out.setdefault(source, []):
                out[source].append(quote)
        return out

    def scores(self, n_expectations: int) -> tuple[float, float]:
        """(surprise, informativeness), both in [0, 1].

        Informativeness: how much the texts said about my expectations at all
        (unexpected findings count half). Surprise: the share of everything I
        predicted that the texts went against. It equals prediction error
        (wrong among what was addressed) times informativeness, so texts that
        stay silent, or only say unrelated things, cannot be very surprising.
        """
        confirmed = self.count("confirmed")
        contradicted = self.count("contradicted")
        unexpected = len(self.unexpected)
        if n_expectations == 0:
            return (0.5 if unexpected else 0.0), (0.5 if unexpected else 0.0)
        total = n_expectations + 0.5 * unexpected
        informativeness = (confirmed + contradicted + 0.5 * unexpected) / total
        surprise = (contradicted + 0.5 * unexpected) / total
        return clamp(surprise), clamp(informativeness)


@dataclass
class Learned:
    belief: str
    source: str | None


@dataclass
class NewQuestion:
    text: str
    trigger: str
    importance: float


@dataclass
class Settlement:
    answer: str = ""
    confidence: float | None = None
    learned: list[Learned] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)
    new_questions: list[NewQuestion] = field(default_factory=list)
    unanswerable: bool = False
    insight: str = ""


# ---------------------------------------------------------------------------
# The organism
# ---------------------------------------------------------------------------


class CuriosityOrganism:
    def __init__(self, config: AppConfig, llm: ChatModel | None = None, senses: Senses | None = None):
        self.config = config
        self.oc = config.organism
        self.home = Path(self.oc.home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.mind_path = self.home / "mind.json"
        self.inbox_dir = self.home / "inbox"
        self.episodes_path = self.home / "episodes.jsonl"
        self.experience_dir = self.home / "experience"
        if llm is None:
            from ..llm import OllamaClient

            llm = OllamaClient(config.llm)
        self.llm = llm
        self.diary = Diary(self.home / "diary.md")
        self.settings = DriveSettings(
            lp_window=self.oc.lp_window,
            lp_prior=self.oc.lp_prior,
            max_confidence_step=self.oc.max_confidence_step,
            surprise_decay=self.oc.surprise_decay,
            refractory=self.oc.refractory,
            boredom_lp_floor=self.oc.boredom_lp_floor,
            boredom_rate=self.oc.boredom_rate,
        )
        self.baseline = Temperament(**self.oc.temperament.model_dump())
        state = load_mind(self.mind_path)
        self.newborn = state is None
        self.state = state if state is not None else self._birth()
        self.senses = senses if senses is not None else Senses.from_config(config, self.inbox_dir)

    # -- life and death ------------------------------------------------------

    def _birth(self) -> MindState:
        state = MindState(name=self.oc.name, temperament=self.baseline.model_copy())
        for text in self.oc.seed_questions:
            state.add_question(text, trigger="seed", confidence=self.oc.newborn_confidence, importance=0.6)
        save_mind(state, self.mind_path)
        self.diary.birth(state)
        return state

    @staticmethod
    def archive(config: AppConfig) -> Path | None:
        """Move a whole life (mind, diary, inbox, experience) aside so a new one can begin."""
        home = Path(config.organism.home)
        if not home.exists() or not any(home.iterdir()):
            return None
        stamp = utc_now_iso()[:19].replace(":", "").replace("-", "")
        target = home.parent / f"{home.name}_archive" / stamp
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(home), str(target))
        return target

    def save(self) -> None:
        save_mind(self.state, self.mind_path)

    # -- talking to the organism ---------------------------------------------

    def ask(self, text: str, importance: float = 0.9) -> Question:
        """A human question becomes one of the organism's own open questions."""
        text = _clean_question(text) or text.strip()
        existing = self._find_similar_question(text)
        if existing is not None:
            existing.importance = clamp(max(existing.importance, importance))
            if existing.status != "open":
                self._reopen(existing, "a human asked about it")
            self.diary.asked(existing.id, existing.text, reused=True)
            self.save()
            return existing
        q = self.state.add_question(text, trigger="human", confidence=self.oc.newborn_confidence, importance=clamp(importance))
        self.diary.asked(q.id, q.text, reused=False)
        self.save()
        return q

    def feed(self, text: str, title: str | None = None) -> Path:
        """Share an observation; it is noticed at the next heartbeat and joins the library."""
        if not text.strip():
            raise ValueError("Nothing to feed: the observation is empty.")
        path = write_inbox_item(self.inbox_dir, text, title or one_line(text, 60), utc_now_iso())
        self.state.unread_inbox.append(str(path))
        self.save()
        return path

    # -- living ----------------------------------------------------------------

    def live(
        self,
        heartbeats: int | None = None,
        *,
        forever: bool = False,
        pause_seconds: float = 0.0,
        on_heartbeat: Callable[[Episode | None], None] | None = None,
    ) -> list[Episode]:
        total = self.oc.heartbeats_per_run if heartbeats is None else heartbeats
        episodes: list[Episode] = []
        failures = 0
        i = 0
        while forever or i < total:
            i += 1
            episode = self.heartbeat()
            if episode is not None:
                episodes.append(episode)
            if on_heartbeat:
                on_heartbeat(episode)
            if episode is not None and episode.status_after == "no-thought":
                failures += 1
                if failures >= 2:
                    raise OrganismError(
                        "The language model did not answer in two heartbeats in a row: " + "; ".join(episode.errors)
                    )
            else:
                failures = 0
            if pause_seconds and (forever or i < total):
                time.sleep(pause_seconds)
        return episodes

    def heartbeat(self) -> Episode | None:
        st = self.state
        st.heartbeat += 1
        errors: list[str] = []
        try:
            self._notice_inbox(errors)
        except Exception as exc:  # perception trouble must not stop the heartbeat
            errors.append(f"notice: {exc}")

        readings = self._read_drives()
        if not readings:
            self._incubate(force=True)
            readings = self._read_drives()
        if not readings:
            self.diary.note(f"## Heartbeat {st.heartbeat}\n\nNo question is open. I rest and wait for something to wonder about.")
            self.save()
            return None

        chosen = choose(readings, st.temperament.exploration_temperature, self._rng())
        q = st.questions[chosen.question_id]
        ranking = [(chosen.question_id, chosen.total)] + sorted(
            ((r.question_id, r.total) for r in readings if r.question_id != chosen.question_id),
            key=lambda item: -item[1],
        )

        related = self._related_beliefs(f"{q.text} {q.answer}", limit=5, include=q.related_beliefs)
        anticipation = self._anticipate(q, related, errors)
        if q.visits or anticipation.confidence is None:
            prior_confidence = q.confidence
        else:
            prior_confidence = clamp(anticipation.confidence, 0.05, self.oc.max_initial_confidence)
        prior_answer = q.answer or anticipation.answer

        observations = self.senses.observe(q.text, anticipation.expectations)
        comparison = self._compare(q, anticipation, observations, errors)
        dialogue = self._dialogue(q, prior_answer, comparison, observations, errors)
        settlement = self._settle(q, prior_answer, prior_confidence, comparison, observations, dialogue, related, errors)

        thought_failed = any(e.startswith("anticipate:") for e in errors) and any(e.startswith("settle:") for e in errors)
        if thought_failed:
            episode = Episode(
                heartbeat=st.heartbeat,
                question_id=q.id,
                question=q.text,
                trigger=q.trigger,
                drive=chosen.as_dict(),
                status_after="no-thought",
                errors=errors,
            )
            self.diary.note(f"## Heartbeat {st.heartbeat}\n\nI tried to think about {q.id} but could not: {'; '.join(errors)}")
            self.save()
            return episode

        episode = self._integrate(
            q, chosen, prior_answer, prior_confidence, anticipation, observations, comparison, dialogue, settlement, related, errors
        )
        self.diary.heartbeat(episode, st, ranking)
        if self.oc.reflect_every > 0 and st.heartbeat % self.oc.reflect_every == 0:
            self.reflect()
        self.save()
        return episode

    # -- the steps of one act of inquiry ---------------------------------------

    def _anticipate(self, q: Question, related: list[Belief], errors: list[str]) -> Anticipation:
        user = (
            f"Question: {q.text}\n\n"
            f"What you already believe that seems related:\n{self._format_beliefs(related)}\n\n"
            f"Your answer so far (confidence {q.confidence:.2f}): {q.answer or 'none yet'}\n\n"
            "Before reading any sources, give your best current answer, your confidence, and 2 to 4 specific "
            "expectations about what the classic texts will say on this question. Each expectation must be "
            "something a passage of text could confirm or contradict."
        )
        data = self._json(P.ANTICIPATE, user, P.ANTICIPATE_SCHEMA, temperature=0.4, errors=errors, step="anticipate")
        expectations = [one_line(e, 300) for e in _texts(data.get("expectations"), ("expectation", "claim", "text")) if len(e.split()) >= 4]
        return Anticipation(
            answer=one_line(_as_str(data.get("answer")), 600),
            confidence=_confidence(data.get("confidence")),
            expectations=expectations[:4],
        )

    def _compare(self, q: Question, anticipation: Anticipation, observations: list[Observation], errors: list[str]) -> Comparison:
        n = len(anticipation.expectations)
        if not observations:
            return Comparison([Check(i) for i in range(1, n + 1)])
        if anticipation.expectations:
            expectation_lines = "\n".join(f"E{i}: {e}" for i, e in enumerate(anticipation.expectations, start=1))
            task = (
                "For each expectation (E1, E2, ...) decide whether the passages confirm it, contradict it, or do not "
                "address it. For confirmed or contradicted, give the source label (S1, S2, ...) and copy an exact quote "
                "of 5 to 30 words from that passage. Then, only if something in the passages genuinely surprises you "
                "given the question, list it (at most 2) with its source label and an exact quote; otherwise leave "
                "unexpected empty."
            )
        else:
            expectation_lines = "(you made no expectations)"
            task = "List at most 2 things in the passages that bear on the question, each with its source label and an exact quote."
        user = (
            f"Question: {q.text}\n\nYour expectations before reading:\n{expectation_lines}\n\n"
            f"Passages you have just read:\n\n{self._format_observations(observations)}\n\n{task}"
        )
        data = self._json(P.COMPARE, user, P.COMPARE_SCHEMA, temperature=0.1, errors=errors, step="compare")
        return self._verify_comparison(data, anticipation.expectations, observations)

    def _verify_comparison(self, data: dict[str, Any], expectations: list[str], observations: list[Observation]) -> Comparison:
        checks: dict[int, Check] = {}
        rejected = 0
        for item in _as_list(data.get("checks")):
            if not isinstance(item, dict):
                continue
            idx = _expectation_index(item.get("expectation"), expectations)
            if idx is None or idx in checks:
                continue
            status = _status(item.get("status"))
            if status == "not_addressed":
                checks[idx] = Check(idx)
                continue
            label, quote = _locate_quote(item.get("source"), item.get("quote"), observations)
            if quote is None:
                rejected += 1
                checks[idx] = Check(idx, "unverified", label)
            else:
                checks[idx] = Check(idx, status, label, quote)
        for i in range(1, len(expectations) + 1):
            checks.setdefault(i, Check(i))
        unexpected: list[dict[str, str]] = []
        for item in _as_list(data.get("unexpected")):
            if len(unexpected) >= 2:
                break
            if not isinstance(item, dict):
                continue
            label, quote = _locate_quote(item.get("source"), item.get("quote"), observations)
            if quote is None:
                rejected += 1
                continue
            already_used = [c.quote for c in checks.values() if c.quote and c.source == label] + [u["quote"] for u in unexpected]
            if any(overlap(quote, used) >= 0.5 for used in already_used):
                continue  # the same words cannot be both expected and unexpected
            finding = _as_str(item.get("finding")).strip()
            if re.fullmatch(r"\[?S\d+\]?", finding):
                finding = ""  # the model put the source label where the finding belongs
            finding = one_line(finding, 300) or one_line(quote, 200)
            unexpected.append({"finding": finding, "source": label, "quote": quote})
        return Comparison(sorted(checks.values(), key=lambda c: c.expectation), unexpected, rejected)

    def _dialogue(self, q: Question, prior_answer: str, comparison: Comparison, observations: list[Observation], errors: list[str]) -> list[dict[str, str]]:
        findings = self._format_findings(comparison, observations)
        transcript: list[dict[str, str]] = []
        for turn in range(max(0, self.oc.dialogue_turns)):
            so_far = "\n\n".join(f"{t['voice'].upper()}: {t['text']}" for t in transcript)
            if turn == 0:
                system, voice, temperature = P.WONDER, "Wonder", 0.7
                user = (
                    f"Question: {q.text}\n\nYour answer before reading: {prior_answer or 'none'}\n\n"
                    f"What the texts showed:\n{findings}\n\n"
                    f"How you currently understand curiosity itself: {self.state.self_model.understanding_of_curiosity}\n\n"
                    "In at most 120 words: say what puzzles you most here and propose one bold explanation. "
                    "End with the deepest question this raises."
                )
            elif turn % 2 == 1:
                system, voice, temperature = P.SKEPTIC, "Skeptic", 0.5
                user = (
                    f"Question: {q.text}\n\nWhat the texts showed:\n{findings}\n\nThe dialogue so far:\n{so_far}\n\n"
                    "In at most 120 words, as SKEPTIC: name the weakest point in what WONDER just said: a hidden "
                    "assumption, a counterexample from the passages, or an ambiguous key word. End with the one "
                    "question WONDER must answer."
                )
            else:
                system, voice, temperature = P.WONDER, "Wonder", 0.6
                user = (
                    f"Question: {q.text}\n\nWhat the texts showed:\n{findings}\n\nThe dialogue so far:\n{so_far}\n\n"
                    "In at most 100 words, answer the skeptic: concede what is right, defend what still stands, "
                    "and say plainly where you now stand."
                )
            text = self._text(system, user, temperature=temperature, errors=errors, step=voice.lower())
            if not text:
                break
            transcript.append({"voice": voice, "text": clip(_strip_voice_prefix(text, voice), 1200)})
        return transcript

    def _settle(
        self,
        q: Question,
        prior_answer: str,
        prior_confidence: float,
        comparison: Comparison,
        observations: list[Observation],
        dialogue: list[dict[str, str]],
        related: list[Belief],
        errors: list[str],
    ) -> Settlement:
        dialogue_text = "\n\n".join(f"{t['voice'].upper()}: {t['text']}" for t in dialogue) or "(no dialogue)"
        user = (
            f"Question ({q.id}): {q.text}\n\n"
            f"Your answer before reading (confidence {prior_confidence:.2f}): {prior_answer or 'none'}\n\n"
            f"Verified evidence:\n{self._format_findings(comparison, observations)}\n\n"
            f"Inner dialogue:\n{dialogue_text}\n\n"
            f"Your earlier beliefs that may be related:\n{self._format_beliefs(related)}\n\n"
            "Settle this episode of inquiry:\n"
            "- answer: your revised answer (1-3 sentences)\n"
            "- confidence: 0.0 to 1.0. Raise it only with support; lower it if you were contradicted or the skeptic found a real weakness\n"
            "- learned: 0 to 3 new beliefs, one sentence each, each with the source label (S1, S2, ...) that supports it\n"
            "- contradicts: ids of the earlier beliefs above (like B2) that the evidence contradicts; [] if none\n"
            "- new_questions: 1 to 3 specific new questions born from a surprise, a contradiction, a gap, or the skeptic's objection\n"
            "- unanswerable: true only if no evidence or argument could ever settle this question\n"
            "- insight: one sentence about what you learned"
        )
        data = self._json(P.SETTLE, user, P.SETTLE_SCHEMA, temperature=0.2, errors=errors, step="settle")
        return _parse_settlement(data)

    # -- learning from the episode --------------------------------------------

    def _integrate(
        self,
        q: Question,
        chosen: DriveReading,
        prior_answer: str,
        prior_confidence: float,
        anticipation: Anticipation,
        observations: list[Observation],
        comparison: Comparison,
        dialogue: list[dict[str, str]],
        settlement: Settlement,
        related: list[Belief],
        errors: list[str],
    ) -> Episode:
        st, oc, hb = self.state, self.oc, self.state.heartbeat
        error, informativeness = comparison.scores(len(anticipation.expectations))
        quotes = comparison.quotes_by_source()
        by_label = {o.label: o for o in observations}

        # Confidence moves in bounded steps, cannot grow without evidence, and can never
        # rise above what the verified confirmations gathered so far allow.
        support = comparison.count("confirmed")
        proposed = settlement.confidence if settlement.confidence is not None else prior_confidence
        delta = max(-oc.max_confidence_step, min(oc.max_confidence_step, proposed - prior_confidence))
        if delta > 0 and support == 0 and not quotes:
            delta = min(delta, 0.05)
        if delta > 0 and error > 0.5:
            delta = min(delta, 0.10)
        new_confidence = clamp(prior_confidence + delta, 0.02, 0.98)
        ceiling = min(0.95, oc.evidence_ceiling_base + oc.evidence_ceiling_per_support * (support + sum(v.support for v in q.visits)))
        if new_confidence > prior_confidence:
            new_confidence = min(new_confidence, max(prior_confidence, ceiling))
        q.confidence = new_confidence
        q.answer = settlement.answer or anticipation.answer or q.answer
        q.visits.append(
            Visit(
                heartbeat=hb,
                confidence_before=prior_confidence,
                confidence_after=new_confidence,
                prediction_error=error,
                informativeness=informativeness,
                support=support,
            )
        )
        q.last_visited = hb

        # Beliefs: grounded ones carry verified quotes; the rest are marked as interpretation.
        new_beliefs: list[str] = []
        reinforced: list[str] = []
        for item in settlement.learned[:3]:
            obs = by_label.get(item.source or "")
            evidence = [
                Evidence(citation=obs.citation, quote=quote, source_title=obs.heading)
                for quote in quotes.get(item.source or "", [])[:2]
            ] if obs else []
            existing = self._find_similar_belief(item.belief)
            if existing is not None:
                known = {e.quote for e in existing.evidence}
                existing.evidence.extend(e for e in evidence if e.quote not in known)
                existing.confidence = clamp(existing.confidence + (0.05 if evidence else 0.02), 0.0, 0.95)
                if existing.status == "doubted" and evidence:
                    existing.status = "held"
                existing.updated_at = hb
                existing.history.append(f"hb {hb}: reinforced while investigating {q.id}")
                belief_id = existing.id
                reinforced.append(belief_id)
            else:
                belief = st.add_belief(item.belief, confidence=0.6 if evidence else 0.35, evidence=evidence, origin_question=q.id)
                belief_id = belief.id
                new_beliefs.append(belief_id)
            if belief_id not in q.related_beliefs:
                q.related_beliefs.append(belief_id)

        # Contradictions: only beliefs that were actually shown to the model can be doubted.
        doubted: list[str] = []
        shown = {b.id for b in related}
        for bid in settlement.contradicts:
            belief = st.beliefs.get(bid)
            if belief is None or bid not in shown or bid in new_beliefs or bid in reinforced:
                continue
            belief.confidence = clamp(belief.confidence - 0.15)
            belief.status = "retracted" if belief.confidence < 0.15 else "doubted"
            belief.contradicted_by.append(q.id)
            belief.updated_at = hb
            belief.history.append(f"hb {hb}: contradicted while investigating {q.id}")
            doubted.append(bid)
            for other in st.questions.values():
                if other.status == "settled" and bid in other.related_beliefs:
                    self._reopen(other, f"belief {bid}, which it relied on, was contradicted")

        # New questions are born from this episode's surprise, contradiction and gaps.
        born: list[str] = []
        woke: list[str] = []
        for nq in settlement.new_questions[: oc.max_new_questions_per_heartbeat]:
            if self._same_question(nq.text, q.text):
                continue  # a restatement of the current question is not a new question
            trigger = "contradiction" if doubted and nq.trigger == "gap" else nq.trigger
            question, outcome = self._adopt_question(
                nq.text, trigger=trigger, parent_id=q.id, importance=nq.importance, inherited_surprise=error
            )
            if question is not None and outcome == "new":
                born.append(question.id)
            elif question is not None and outcome == "reawakened":
                woke.append(question.id)
        if born:
            q.importance = clamp(q.importance + 0.05 * len(born))

        self._update_status(q, settlement)
        self._limit_open_questions()

        episode = Episode(
            heartbeat=hb,
            question_id=q.id,
            question=q.text,
            trigger=q.trigger,
            drive=chosen.as_dict(),
            prior_answer=prior_answer,
            prior_confidence=prior_confidence,
            expectations=anticipation.expectations,
            checks=[c.__dict__ for c in comparison.checks],
            unexpected=comparison.unexpected,
            rejected_quotes=comparison.rejected_quotes,
            sources=[f"[{o.label}] {o.heading}" for o in observations],
            prediction_error=error,
            informativeness=informativeness,
            dialogue=dialogue,
            answer=q.answer,
            confidence=new_confidence,
            insight=settlement.insight,
            new_belief_ids=new_beliefs,
            reinforced_belief_ids=reinforced,
            doubted_belief_ids=doubted,
            new_question_ids=born,
            reawakened_question_ids=woke,
            status_after=q.status,
            errors=errors,
        )
        st.remember_episode(episode)
        write_jsonl(self.episodes_path, episode.model_dump(mode="json"))
        if oc.export_experience:
            self._export_experience(episode, comparison, by_label)
        return episode

    def _update_status(self, q: Question, settlement: Settlement) -> None:
        oc = self.oc
        active = q.active_visits
        lp = learning_progress(q, window=oc.lp_window, prior=oc.lp_prior, max_step=oc.max_confidence_step)
        bored = boredom(q, lp, patience=self.state.temperament.boredom_patience, lp_floor=oc.boredom_lp_floor, rate=oc.boredom_rate)
        silent = sum(v.informativeness for v in active) / len(active) < 0.35 if active else False
        if settlement.unanswerable and len(active) >= 2 and silent:
            self._set_status(q, "unanswerable", "the texts stay silent and the question may exceed what evidence can settle (Kant's limit)")
        elif q.confidence >= oc.settle_confidence and active and active[-1].prediction_error <= oc.settle_max_error and len(active) >= 2:
            self._set_status(q, "settled", f"confident ({q.confidence:.2f}) and no longer surprised: doubt is appeased (Peirce)")
        elif bored >= 0.9:
            self._set_status(q, "dormant", f"{len(active)} visits without learning progress; resting until new knowledge arrives")

    def _set_status(self, q: Question, status: str, reason: str) -> None:
        q.status = status  # type: ignore[assignment]
        q.status_reason = reason
        q.status_changed_at = self.state.heartbeat

    def _reopen(self, q: Question, reason: str) -> None:
        if q.status == "settled":
            q.confidence = max(0.2, q.confidence - 0.15)
        q.rest_marker = len(q.visits)
        q.inherited_surprise = max(q.inherited_surprise, 0.5)
        q.last_visited = None
        self._set_status(q, "open", reason)

    def _adopt_question(self, text: str, *, trigger: str, parent_id: str | None, importance: float, inherited_surprise: float) -> tuple[Question | None, str]:
        text = _clean_question(text)
        if not text:
            return None, "rejected"
        existing = self._find_similar_question(text)
        if existing is not None:
            existing.importance = clamp(max(existing.importance, importance) + 0.05)
            if existing.status == "dormant":
                self._reopen(existing, f"a new question from {parent_id or 'reflection'} touched it")
                return existing, "reawakened"
            return existing, "reinforced"
        q = self.state.add_question(
            text,
            trigger=trigger if trigger in TRIGGERS else "gap",
            parent_id=parent_id,
            confidence=self.oc.newborn_confidence,
            importance=clamp(importance),
            inherited_surprise=clamp(inherited_surprise),
        )
        return q, "new"

    def _limit_open_questions(self) -> list[str]:
        """Too many open questions dilute attention: the weakest pulls are left to rest."""
        open_qs = self.state.open_questions()
        excess = len(open_qs) - self.oc.max_open_questions
        if excess <= 0:
            return []
        readings = sorted(self._read_drives(), key=lambda r: r.total)
        rested: list[str] = []
        for reading in readings[:excess]:
            q = self.state.questions[reading.question_id]
            if q.trigger == "human":
                continue
            self._set_status(q, "dormant", "too many open questions; letting the weakest pull rest")
            rested.append(q.id)
        return rested

    def _incubate(self, force: bool = False) -> list[str]:
        """Dormant questions wake when enough new related knowledge has arrived (Wallas's incubation)."""
        woke: list[str] = []
        dormant = [q for q in self.state.questions.values() if q.status == "dormant"]
        for q in dormant:
            fresh = [
                b for b in self.state.held_beliefs()
                if b.born_at > q.status_changed_at and overlap(q.text, b.statement) >= 0.3
            ]
            if len(fresh) >= self.oc.incubation_beliefs:
                self._reopen(q, f"{len(fresh)} new related beliefs since it went dormant")
                woke.append(q.id)
        if force and not woke and dormant:
            for q in sorted(dormant, key=lambda d: -d.importance)[:3]:
                self._reopen(q, "nothing else was open, so I returned to it")
                woke.append(q.id)
        return woke

    def _export_experience(self, ep: Episode, comparison: Comparison, by_label: dict[str, Observation]) -> None:
        """Turn grounded episodes into training data for a future generation of the model."""
        lines = []
        for label, quotes in comparison.quotes_by_source().items():
            obs = by_label.get(label)
            if obs:
                lines += [f'{obs.heading}: "{quote}"' for quote in quotes]
        if not lines or not ep.answer:
            return
        meta = {"heartbeat": ep.heartbeat, "question_id": ep.question_id, "prediction_error": ep.prediction_error, "confidence": ep.confidence}
        prompt = f"Question: {ep.question}"
        write_jsonl(
            self.experience_dir / "sft.jsonl",
            {
                "messages": [
                    {"role": "system", "content": "You answer philosophical questions carefully, grounded in the classic texts."},
                    {"role": "user", "content": prompt + "\n\nEvidence:\n" + "\n".join(lines)},
                    {"role": "assistant", "content": ep.answer},
                ],
                "metadata": meta,
            },
        )
        if comparison.count("contradicted") and ep.prior_answer and ep.prior_answer.strip() != ep.answer.strip():
            write_jsonl(
                self.experience_dir / "preference_pairs.jsonl",
                {"prompt": prompt, "chosen": ep.answer, "rejected": ep.prior_answer, "metadata": meta},
            )

    # -- perception of what humans share ------------------------------------

    def _notice_inbox(self, errors: list[str]) -> None:
        if not self.state.unread_inbox:
            return
        self.senses.notice_new_material()
        for path_str in list(self.state.unread_inbox):
            self.state.unread_inbox.remove(path_str)
            path = Path(path_str)
            if not path.exists():
                continue
            meta, body = strip_front_matter(path.read_text(encoding="utf-8", errors="ignore"))
            title = meta.get("title") or path.stem
            related = self._related_beliefs(body, limit=4)
            user = (
                f'The observation, titled "{title}":\n---\n{clip(body, 2500)}\n---\n\n'
                f"What you already believe that may be related:\n{self._format_beliefs(related)}\n\n"
                "What genuinely puzzles you about this observation? Ask 1 or 2 specific questions."
            )
            data = self._json(P.NOTICE, user, P.NOTICE_SCHEMA, temperature=0.5, errors=errors, step="notice")
            ids: list[str] = []
            for item in _as_list(data.get("questions"))[:2]:
                text, importance = _question_fields(item)
                q, outcome = self._adopt_question(
                    text, trigger="observation", parent_id=None, importance=max(0.7, importance), inherited_surprise=0.5
                )
                if q is not None and outcome in ("new", "reawakened"):
                    ids.append(q.id)
            self.diary.noticed(title, ids, self.state)

    # -- reflection --------------------------------------------------------------

    def reflect(self) -> dict[str, Any]:
        st = self.state
        errors: list[str] = []
        episodes = st.recent_episodes[-max(6, 2 * self.oc.reflect_every):]
        vitals = compute_vitals(episodes)
        key = diagnose(vitals)
        changes: list[str] = []
        if self.oc.homeostasis:
            st.temperament, changes = regulate(st.temperament, key, self.baseline)
        diagnosis = DIAGNOSES[key]
        woke = self._incubate()
        rested = self._limit_open_questions()

        episode_lines = "\n".join(
            f"- heartbeat {e.heartbeat}, {e.question_id} \"{one_line(e.question, 120)}\": surprise {e.prediction_error:.2f}, "
            f"confidence {e.prior_confidence:.2f} -> {e.confidence:.2f}" + (f"; insight: {one_line(e.insight, 160)}" if e.insight else "")
            for e in episodes
        ) or "- (no episodes yet)"
        strongest = sorted(st.held_beliefs(), key=lambda b: -b.confidence)[:6]
        user = (
            f"Your recent episodes of inquiry:\n{episode_lines}\n\n"
            f"Your vital signs: progress rate {vitals.progress_rate:.2f} (share of episodes where your mind changed), "
            f"mean surprise {vitals.mean_surprise:.2f}, diversity {vitals.diversity:.2f} (share of distinct questions), "
            f"new questions per episode {vitals.births_per_episode:.2f}.\n\n"
            f"Your self-regulation diagnosed: {diagnosis['name']} - {diagnosis['meaning']} ({diagnosis['source']})\n\n"
            f"Your current understanding of curiosity: {st.self_model.understanding_of_curiosity}\n\n"
            f"Your most confident beliefs:\n{self._format_beliefs(strongest)}\n\n"
            "Reflect honestly, as an inquirer looking at its own habits."
        )
        data = self._json(P.REFLECT, user, P.REFLECT_SCHEMA, temperature=0.4, errors=errors, step="reflect")
        understanding = one_line(_as_str(data.get("understanding_of_curiosity")), 900)
        reflection = one_line(_as_str(data.get("reflection")), 900)
        focus_text = _as_str(data.get("focus_question"))
        if understanding:
            st.self_model.understanding_of_curiosity = understanding
        st.self_model.last_reflection = reflection
        st.self_model.diagnosis = key
        if reflection:
            st.self_model.reflections = (st.self_model.reflections + [f"heartbeat {st.heartbeat}: {reflection}"])[-20:]
        focus = None
        if focus_text:
            focus, _ = self._adopt_question(focus_text, trigger="reflection", parent_id=None, importance=0.75, inherited_surprise=0.3)
        st.vitals_history.append(
            {"heartbeat": st.heartbeat, **vitals.as_dict(), "diagnosis": key, "temperament": st.temperament.model_dump()}
        )
        st.vitals_history = st.vitals_history[-200:]
        self.diary.reflection(
            st.heartbeat,
            vitals.as_dict(),
            diagnosis,
            changes,
            reflection,
            understanding,
            f"{focus.id} {focus.text}" if focus else None,
            woke,
            rested,
        )
        self.save()
        return {"vitals": vitals.as_dict(), "diagnosis": key, "changes": changes, "errors": errors}

    # -- introspection ------------------------------------------------------------

    def drives(self) -> list[tuple[Question, DriveReading]]:
        readings = sorted(self._read_drives(), key=lambda r: -r.total)
        return [(self.state.questions[r.question_id], r) for r in readings]

    def _read_drives(self) -> list[DriveReading]:
        st = self.state
        out = []
        for q in st.open_questions():
            related = len(self._related_beliefs(q.text, limit=self.settings.anchor_k, include=q.related_beliefs))
            out.append(read_drive(q, st.temperament, st.heartbeat, related, self.settings))
        return out

    # -- helpers ------------------------------------------------------------------

    def _rng(self) -> random.Random:
        seed = self.oc.random_seed
        return random.Random(f"{seed}:{self.state.heartbeat}") if seed is not None else random.Random()

    def _related_beliefs(self, text: str, limit: int = 5, include: list[str] | None = None) -> list[Belief]:
        include = include or []
        scored = []
        for b in self.state.held_beliefs():
            score = overlap(text, b.statement) + (0.5 if b.id in include else 0.0)
            if score >= 0.25:
                scored.append((score, b))
        scored.sort(key=lambda item: -item[0])
        return [b for _, b in scored[:limit]]

    def _same_question(self, a: str, b: str) -> bool:
        """Near-identical wording, or one question folded entirely inside the other."""
        if jaccard(a, b) >= self.oc.dedupe_similarity:
            return True
        return min(len(token_set(a)), len(token_set(b))) >= 2 and overlap(a, b) >= 0.85

    def _find_similar_question(self, text: str) -> Question | None:
        best, best_score = None, 0.0
        for q in self.state.questions.values():
            if not self._same_question(text, q.text):
                continue
            score = jaccard(text, q.text) + overlap(text, q.text)
            if score > best_score:
                best, best_score = q, score
        return best

    def _find_similar_belief(self, statement: str) -> Belief | None:
        best, best_score = None, 0.0
        for b in self.state.held_beliefs():
            score = jaccard(statement, b.statement)
            if score > best_score:
                best, best_score = b, score
        return best if best_score >= 0.6 else None

    @staticmethod
    def _format_beliefs(beliefs: list[Belief]) -> str:
        if not beliefs:
            return "- (nothing yet)"
        return "\n".join(
            f"- [{b.id}] {b.statement} (confidence {b.confidence:.2f}{', interpretation' if b.interpretive else ''})"
            for b in beliefs
        )

    @staticmethod
    def _format_observations(observations: list[Observation]) -> str:
        return "\n\n".join(f"[{o.label}] {o.heading}\n{o.text}" for o in observations)

    @staticmethod
    def _format_findings(comparison: Comparison, observations: list[Observation]) -> str:
        by_label = {o.label: o for o in observations}
        lines = []
        for c in comparison.checks:
            if c.status in ("confirmed", "contradicted"):
                obs = by_label.get(c.source)
                heading = f" {obs.heading}" if obs else ""
                lines.append(f'- Your expectation E{c.expectation} was {c.status.upper()} by [{c.source}]{heading}: "{c.quote}"')
        for u in comparison.unexpected:
            obs = by_label.get(u["source"])
            heading = f" {obs.heading}" if obs else ""
            lines.append(f'- Unexpected, in [{u["source"]}]{heading}: {u["finding"]} - "{u["quote"]}"')
        silent = [c.expectation for c in comparison.checks if c.status in ("not_addressed", "unverified")]
        if silent:
            lines.append("- The passages did not clearly address: " + ", ".join(f"E{i}" for i in silent))
        return "\n".join(lines) if lines else "- Nothing relevant was found in the passages."

    def _json(self, system: str, user: str, schema: str, *, temperature: float, errors: list[str], step: str) -> dict[str, Any]:
        try:
            data = self.llm.json_chat(system, user, schema, temperature=temperature, max_tokens=self.oc.max_tokens_json)
        except Exception as exc:  # a confused or absent model must not end the organism's life
            errors.append(f"{step}: {type(exc).__name__}: {one_line(str(exc), 160)}")
            self._trace(step, user, f"ERROR {exc}")
            return {}
        self._trace(step, user, data)
        return data if isinstance(data, dict) else {}

    def _text(self, system: str, user: str, *, temperature: float, errors: list[str], step: str) -> str:
        try:
            text = (self.llm.chat(system, user, temperature=temperature, max_tokens=self.oc.max_tokens_text) or "").strip()
        except Exception as exc:
            errors.append(f"{step}: {type(exc).__name__}: {one_line(str(exc), 160)}")
            self._trace(step, user, f"ERROR {exc}")
            return ""
        self._trace(step, user, text)
        return text

    def _trace(self, step: str, prompt: str, reply: Any) -> None:
        if self.oc.trace_llm:
            write_jsonl(self.home / "llm_trace.jsonl", {"heartbeat": self.state.heartbeat, "step": step, "prompt": prompt, "reply": reply})


# ---------------------------------------------------------------------------
# Tolerant parsing of small-model output
# ---------------------------------------------------------------------------


def _texts(value: Any, keys: tuple[str, ...]) -> list[str]:
    out = []
    for item in _as_list(value):
        if isinstance(item, dict):
            text = next((_as_str(item[k]) for k in keys if k in item), "")
            if not text:
                text = next((v for v in item.values() if isinstance(v, str)), "")
        else:
            text = _as_str(item)
        text = re.sub(r"^\s*(E\d+[:.)]|\d+[.)]|[-*])\s*", "", text).strip()
        if text:
            out.append(text)
    return out


def _confidence(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        match = re.search(r"\d+(\.\d+)?", value)
        if not match:
            return None
        number = float(match.group())
        if "%" in value:
            number /= 100.0
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        return None
    if 1.0 < number <= 100.0:
        number /= 100.0
    return clamp(number)


def _status(value: Any) -> str:
    s = _as_str(value).strip().lower()
    if not s or "|" in s or " or " in s or s.startswith("not") or "address" in s or "silent" in s or "unclear" in s:
        return "not_addressed"
    if s.startswith(("confirm", "support", "yes", "true", "agree", "consistent")):
        return "confirmed"
    if s.startswith(("contradict", "refut", "no", "false", "disagree", "challeng", "oppos", "inconsistent")):
        return "contradicted"
    return "not_addressed"


def _expectation_index(value: Any, expectations: list[str]) -> int | None:
    idx: int | None = None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        idx = int(value)
    else:
        text = _as_str(value)
        match = re.search(r"\d+", text)
        if match:
            idx = int(match.group())
        elif text and expectations:
            scores = [jaccard(text, e) for e in expectations]
            best = max(range(len(scores)), key=lambda i: scores[i])
            idx = best + 1 if scores[best] >= 0.5 else None
    if idx == 0:
        idx = 1
    if idx is None or not 1 <= idx <= len(expectations):
        return None
    return idx


def _label(value: Any) -> str:
    """'S2', '[S2]', 'source 2' or a bare 2 -> 'S2'."""
    text = _as_str(value)
    match = re.search(r"\bS\s*(\d+)\b", text, re.IGNORECASE) or re.fullmatch(r"\s*(?:source\s*)?(\d+)\s*", text, re.IGNORECASE)
    return f"S{int(match.group(1))}" if match else ""


def _locate_quote(source: Any, quote: Any, observations: list[Observation]) -> tuple[str, str | None]:
    """Find the claimed quote in the named passage, or in any passage read this heartbeat."""
    label = _label(source)
    text = _as_str(quote).strip().strip('"').strip("“”").strip()
    if not text:
        return label, None
    ordered = sorted(observations, key=lambda o: o.label != label)
    for obs in ordered:
        verified = verify_quote(text, obs.text)
        if verified:
            return obs.label, verified
    return label, None


def _question_fields(item: Any) -> tuple[str, float]:
    if isinstance(item, dict):
        text = _as_str(item.get("question") or item.get("text") or "")
        importance = _confidence(item.get("importance"))
    else:
        text, importance = _as_str(item), None
    return text, 0.5 if importance is None else importance


def _trigger(value: Any) -> str:
    s = _as_str(value).strip().lower()
    if " or " in s or "|" in s:
        return "gap"
    for prefix, trigger in (("surpris", "surprise"), ("contradict", "contradiction"), ("object", "objection"), ("skeptic", "objection"), ("gap", "gap")):
        if s.startswith(prefix):
            return trigger
    return "gap"


def _parse_settlement(data: dict[str, Any]) -> Settlement:
    learned: list[Learned] = []
    for item in _as_list(data.get("learned")):
        if isinstance(item, dict):
            text = _as_str(item.get("belief") or item.get("statement") or item.get("text") or "")
            source = _label(item.get("source")) or _label(text) or None
        else:
            text = _as_str(item)
            source = _label(re.search(r"\[S\d+\]|\(S\d+\)", text).group()) if re.search(r"\[S\d+\]|\(S\d+\)", text) else None
        text = re.sub(r"\s*[\[(]S\d+[\])]\s*", " ", text).strip()
        if len(text.split()) >= 4:
            learned.append(Learned(one_line(text, 400), source))
    contradicts: list[str] = []
    for item in _as_list(data.get("contradicts")):
        for bid in re.findall(r"\bB\d+\b", _as_str(item)):
            if bid not in contradicts:
                contradicts.append(bid)
    new_questions: list[NewQuestion] = []
    for item in _as_list(data.get("new_questions")):
        text, importance = _question_fields(item)
        trigger = _trigger(item.get("trigger")) if isinstance(item, dict) else "gap"
        if text.strip():
            new_questions.append(NewQuestion(text, trigger, importance))
    unanswerable = data.get("unanswerable")
    return Settlement(
        answer=one_line(_as_str(data.get("answer")), 800),
        confidence=_confidence(data.get("confidence")),
        learned=learned,
        contradicts=contradicts,
        new_questions=new_questions,
        unanswerable=unanswerable is True or _as_str(unanswerable).strip().lower() == "true",
        insight=one_line(_as_str(data.get("insight")), 400),
    )


def _clean_question(text: str) -> str:
    text = re.sub(r"\s+", " ", _as_str(text)).strip()
    text = re.sub(r"^(\d+[.)]|[-*•]|Q\d*[:.)]|question\s*\d*[:.)])\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip('"').strip("“”").strip()
    words = text.split()
    if len(words) < 4 or len(words) > 70:
        return ""
    return text


def _strip_voice_prefix(text: str, voice: str) -> str:
    """Remove a leading "SKEPTIC:" and anything the model wrote for the other voice."""
    text = re.sub(rf"^\s*\**{voice}\**\s*:\s*", "", text.strip(), flags=re.IGNORECASE)
    return re.split(r"\n\s*\**(?:WONDER|SKEPTIC)\**\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
