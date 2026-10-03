"""The curiosity organism: a persistent mind whose behavior is driven by curiosity.

One heartbeat is one act of inquiry, modelled on Dewey's "complete act of
thought" and Peirce's doubt-belief cycle:

    choose   which question pulls hardest (the curiosity drive), favouring
             questions close to the main topic
    predict  commit to an answer and to definite predictions, each with a
             probability, before looking ("may" and "might" cannot be wrong)
    observe  read passages from the library (and the web, if enabled)
    compare  which predictions were confirmed or contradicted, counting only
             claims backed by a quote found in the passage
    judge    a separate, blind judge checks every claim/quote pair; only what
             it accepts counts as evidence
    argue    an inner dialogue: Wonder proposes, Skeptic objects with evidence,
             Wonder defends, revises or concedes
    settle   revise the answer (specific enough to be wrong) and confidence,
             form or doubt beliefs, and let one new question be born
    learn    record prediction error (Brier score) and change of mind, which
             feed the drive's learning-progress and boredom terms

Every few heartbeats the organism reflects: it measures its own vital signs,
diagnoses its way of being curious (healthy wonder, restless curiositas,
dogmatic slumber or aporetic numbness), adjusts its temperament, wakes
incubated questions and updates its theory.

The same organism can study a research topic a person gives it (researcher
mode): it then reads the person's papers and asks its own questions about them.
"""
from __future__ import annotations

import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol

from ..config import AppConfig
from ..schema import _as_list, _as_str, utc_now_iso
from ..utils import strip_front_matter, write_jsonl
from . import prompts as P
from .body import Body, Rest
from .diary import Diary
from .judge import Judge, Pair, Verdict
from .librarian import Librarian, ReadingWish
from .papers import opening, read_paper
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
from .state import TRIGGERS, Belief, Episode, Evidence, MindState, Question, Temperament, Topic, Visit, load_mind, save_mind
from .textutil import (
    clip,
    is_hedged,
    is_influence_only,
    is_non_answer,
    is_strawman_falsifier,
    lexically_related,
    jaccard,
    keywords_of,
    one_line,
    overlap,
    token_set,
    topic_relevance,
    vagueness,
    verify_quote,
)

if TYPE_CHECKING:
    from .session import SessionReport


class ChatModel(Protocol):
    def chat(self, system: str, user: str, *, temperature: float | None = None, json_mode: bool = False, max_tokens: int | None = None) -> str: ...

    def json_chat(self, system: str, user: str, schema_hint: str, *, temperature: float | None = None, max_tokens: int | None = None) -> dict[str, Any]: ...


class OrganismError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Intermediate results of one heartbeat
# ---------------------------------------------------------------------------


@dataclass
class Prediction:
    claim: str
    author: str = ""
    probability: float = 0.7
    hedged: bool = False
    weak: bool = False  # only says that something influences something: no text could contradict it

    @property
    def risky(self) -> bool:
        """Could a passage have proved it wrong? Only then is its confirmation a test passed (Popper)."""
        return not (self.hedged or self.weak)

    @property
    def text(self) -> str:
        """The expectation as shown to the comparer: who will say what (no probability: it could bias)."""
        if self.author and self.author.lower().strip(". ") not in ("the texts", "texts", "the sources", "unknown", "") \
                and self.author.lower() not in self.claim.lower():
            return f"{self.author}: {self.claim}"
        return self.claim


@dataclass
class Anticipation:
    answer: str = ""
    confidence: float | None = None
    predictions: list[Prediction] = field(default_factory=list)

    @property
    def expectations(self) -> list[str]:
        return [p.text for p in self.predictions]

    @property
    def probabilities(self) -> list[float]:
        return [p.probability for p in self.predictions]


@dataclass
class Check:
    expectation: int
    status: str = "not_addressed"  # confirmed | contradicted | not_addressed | unverified
    source: str = ""
    quote: str = ""
    claimed: str = ""  # what the organism itself said, before the judge
    judge: str = ""  # supports | contradicts | neither | unjudged ("" when no judge)


@dataclass
class Comparison:
    checks: list[Check] = field(default_factory=list)
    unexpected: list[dict[str, str]] = field(default_factory=list)
    rejected_quotes: int = 0
    expectations: list[str] = field(default_factory=list)  # what each check's expectation said, in order

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

    def scores(self, predictions: "int | list[float]") -> tuple[float, float]:
        """(surprise, informativeness), both in [0, 1].

        ``predictions`` is the probability given to each expectation (or just
        their number, meaning certainty). Each prediction the texts addressed
        scores its Brier error: (1 - p)^2 if confirmed, p^2 if contradicted. A
        confident prediction that comes true costs almost nothing; a confident
        one that fails costs a lot; a 50/50 guess always costs 0.25.

        Informativeness: how much the texts said about my expectations at all
        (unexpected findings count half). Surprise: the summed error over
        everything I predicted, so texts that stay silent, or only say
        unrelated things, cannot be very surprising.
        """
        probabilities = [1.0] * predictions if isinstance(predictions, int) else [clamp(p) for p in predictions]
        n = len(probabilities)
        unexpected = len(self.unexpected)
        if n == 0:
            return (0.5 if unexpected else 0.0), (0.5 if unexpected else 0.0)
        error, addressed = self._errors(probabilities)
        total = n + 0.5 * unexpected
        informativeness = (addressed + 0.5 * unexpected) / total
        surprise = (error + 0.5 * unexpected) / total
        return clamp(surprise), clamp(informativeness)

    def brier(self, probabilities: list[float]) -> float | None:
        """Mean Brier score over the predictions the texts addressed (None if none were)."""
        error, addressed = self._errors([clamp(p) for p in probabilities])
        return error / addressed if addressed else None

    def _errors(self, probabilities: list[float]) -> tuple[float, int]:
        error, addressed = 0.0, 0
        for c in self.checks:
            p = probabilities[c.expectation - 1] if 1 <= c.expectation <= len(probabilities) else 1.0
            if c.status == "confirmed":
                error, addressed = error + (1.0 - p) ** 2, addressed + 1
            elif c.status == "contradicted":
                error, addressed = error + p ** 2, addressed + 1
        return error, addressed


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
    falsifier: str = ""  # what would show the answer wrong
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
    def __init__(
        self,
        config: AppConfig,
        llm: ChatModel | None = None,
        senses: Senses | None = None,
        body: Body | None = None,
        librarian: Librarian | None = None,
        judge_llm: ChatModel | None = None,
    ):
        self.config = config
        self.oc = config.organism
        self.home = Path(self.oc.home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.mind_path = self.home / "mind.json"
        self.inbox_dir = self.home / "inbox"
        self.library_dir = self.home / "library"
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
            news_bonus=self.oc.news_bonus,
            boredom_lp_floor=self.oc.boredom_lp_floor,
            boredom_rate=self.oc.boredom_rate,
        )
        self.baseline = Temperament(**self.oc.temperament.model_dump())
        self.papers_dir = self.home / "papers"
        self.body = body if body is not None else Body(self.oc.body)
        self.body.on_rest = self._remember_rest
        self.judge = self._make_judge(judge_llm)
        state = load_mind(self.mind_path)
        self.newborn = state is None
        self.state = state if state is not None else self._birth()
        self._adopt_topic_defaults()
        topic = self.state.topic
        self.persona = P.researcher(topic.title, topic.description, topic.meaning) if self.research else P.PHILOSOPHY
        self.topic_words = self._topic_words()
        if senses is None:
            senses = Senses.from_config(
                config,
                self.inbox_dir,
                self.library_dir,
                papers_dir=self.papers_dir if self.research else None,
                include_corpus=not self.research,
            )
        self.senses = senses
        if librarian is None and self.oc.librarian.enabled:
            owned = [self.papers_dir] if self.research else [Path(config.corpus.path)]
            librarian = Librarian(self.oc.librarian, self.library_dir, owned_dirs=owned)
        self.librarian = librarian
        self.rest_log: list[Rest] = []
        self.no_thought = 0
        self.last_report: "SessionReport | None" = None

    @property
    def research(self) -> bool:
        return self.state.topic.mode == "research"

    def _make_judge(self, judge_llm: ChatModel | None) -> Judge | None:
        jc = self.oc.judge
        if not jc.enabled:
            return None
        if judge_llm is None and (jc.model or jc.base_url):
            from ..llm import OllamaClient

            judge_config = self.config.llm.model_copy(update={
                "model": jc.model or self.config.llm.model,
                "base_url": jc.base_url or self.config.llm.base_url,
            })
            judge_llm = OllamaClient(judge_config)
        return Judge(
            judge_llm or self.llm,
            temperature=jc.temperature,
            max_tokens=jc.max_tokens,
            before_thinking=self.body.before_thinking,
            trace=self._trace,
        )

    def _adopt_topic_defaults(self) -> None:
        """A philosophy life studies curiosity; lives born before topics existed learn theirs here."""
        topic = self.state.topic
        if topic.mode == "philosophy" and not topic.keywords:
            tc = self.oc.topic
            topic.title, topic.description, topic.keywords = tc.title, tc.description, list(tc.keywords)

    def _possible_support(self, q: Question, comparison: Comparison, by_label: dict[str, Observation], answer: str) -> list[Evidence]:
        """Quotes that might support the answer: those kept for it so far, then this heartbeat's confirmations."""
        found = list(q.support)
        seen = {e.quote for e in found}
        for c in comparison.checks:
            obs = by_label.get(c.source)
            if c.status == "confirmed" and c.quote and obs is not None and c.quote not in seen:
                found.append(Evidence(citation=obs.citation, quote=c.quote, source_title=obs.heading))
                seen.add(c.quote)
        return [e for e in found if lexically_related(answer, e.quote, self.topic_words)]

    def _topic_for_judge(self) -> str:
        """The main topic as the judge sees it: its title, the person's words and, if known, its meaning in the field."""
        topic = self.state.topic
        text = f"{topic.title}. {topic.description}".strip(". ")
        return f"{text}. In its field: {topic.meaning}" if topic.meaning else text

    def _topic_words(self) -> frozenset[str]:
        topic = self.state.topic
        seeds = [q.text for q in self.state.questions.values() if q.trigger in ("seed", "human")]
        return token_set(" ".join([topic.title, topic.description, topic.meaning, *topic.keywords, *seeds]))

    # -- life and death ------------------------------------------------------

    def _birth(self) -> MindState:
        state = MindState(name=self.oc.name, temperament=self.baseline.model_copy())
        research = self.oc.research
        if research.topic.strip():
            state.topic, seeds = self._prepare_research(research.topic.strip(), research.seed_questions)
            state.topic.until_year = research.until_year
            state.name = self.oc.name if self.oc.name != "Curiosity" else "Curiosity (researcher)"
            state.self_model.understanding_of_topic = "I have no theory of this topic yet; I am about to start reading."
        else:
            tc = self.oc.topic
            state.topic = Topic(title=tc.title, description=tc.description, keywords=list(tc.keywords))
            seeds = list(self.oc.seed_questions)
        for text in seeds:
            state.add_question(text, trigger="seed", confidence=self.oc.newborn_confidence, importance=0.6, relevance=1.0)
        save_mind(state, self.mind_path)
        self.diary.birth(state)
        return state

    def _prepare_research(self, topic_text: str, n: int) -> tuple[Topic, list[str]]:
        """Researcher mode: turn a person's topic into its meaning, a title, key terms and first questions.

        It reads how the person's own article and papers begin first. A topic's words can be the field's
        own terms: a 7B model given only "How does lightning illuminate the inner magnetosphere?" took
        "illuminate" for light and spent a run on light emission, while its papers are about whistler waves.
        """
        errors: list[str] = []
        persona = P.researcher(topic_text)
        openings, titles = self._paper_openings()
        reading = ""
        if openings:
            reading = "How the person's own article and papers begin (this is how their field speaks):\n" + "\n".join(openings) + "\n\n"
        if titles:
            reading += "Other papers they gave you: " + "; ".join(titles) + "\n\n"
        user = (
            f"The research topic, in the person's words: {topic_text}\n\n{reading}"
            f"Say in one sentence what the topic means in its field, then give a short title, 10 to 20 key terms, "
            f"and {n} first questions about this topic."
        )
        data = self._json(P.system("TOPIC", persona), user, P.TOPIC_SCHEMA, temperature=0.3, errors=errors, step="topic")
        meaning = one_line(_as_str(data.get("meaning")), 400)
        if meaning.lower().startswith(("one sentence", "what the topic means")) or len(meaning.split()) < 5:
            meaning = ""  # the schema hint echoed back, or too short to say anything
        title = one_line(_as_str(data.get("title")), 80) or one_line(topic_text, 80)
        keywords = [one_line(k, 40) for k in _texts(data.get("keywords"), ("term", "keyword", "text")) if 1 <= len(k.split()) <= 4][:20]
        questions = [q for q in (_clean_question(t) for t in _texts(data.get("questions"), ("question", "text"))) if q][:n]
        if not keywords:
            keywords = keywords_of(topic_text, 12).split()
        if not questions:
            questions = [topic_text if topic_text.endswith("?") else f"What is known about {topic_text}, and what is still unknown?"]
        return Topic(mode="research", title=title, description=one_line(topic_text, 600), meaning=meaning, keywords=keywords), questions

    def _paper_openings(self, *, openings: int = 5, titles: int = 15, chars: int = 400) -> tuple[list[str], list[str]]:
        """The beginnings of the person's own article (--feed-file) and of its papers, and the titles of the rest."""
        found: list[str] = []
        own = self.oc.research.own_article.strip()
        if own:
            try:
                meta, text = read_paper(Path(own).expanduser(), max_pages=3)
                found.append(f"- Their own article, {one_line(meta.get('title') or Path(own).stem, 120)}: {opening(text, chars + 200)}")
            except Exception:  # an unreadable article must not stop a birth
                pass
        rest: list[str] = []
        papers = sorted(self.papers_dir.glob("*.md")) if self.papers_dir.is_dir() else []
        for i, path in enumerate(papers[: openings + titles]):
            meta, text = strip_front_matter(path.read_text(encoding="utf-8", errors="replace"))
            title = one_line(meta.get("title") or path.stem.replace("_", " "), 120)
            if i < openings:
                found.append(f"- {title}: {opening(text, chars)}")
            else:
                rest.append(title)
        return found, rest

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
        q = self.state.add_question(
            text, trigger="human", confidence=self.oc.newborn_confidence, importance=clamp(importance), relevance=1.0
        )
        self.topic_words = self._topic_words()
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
        minutes: float | None = None,
        on_heartbeat: Callable[[Episode | None], None] | None = None,
    ) -> list[Episode]:
        """Live several heartbeats, resting between them to protect the computer.

        With ``minutes`` it lives until that much time has passed (rests
        included) and then stops after the current heartbeat.
        """
        total = self.oc.heartbeats_per_run if heartbeats is None else heartbeats
        deadline = self.body.now() + minutes * 60 if minutes else None
        endless = forever or deadline is not None
        episodes: list[Episode] = []
        durations: list[float] = []
        failures = 0
        i = 0
        while endless or i < total:
            if deadline is not None:
                remaining = deadline - self.body.now()
                # Do not start a heartbeat that probably cannot finish before the time limit:
                # ending a little early is better than running over a test someone has planned.
                if remaining <= 0 or (durations and remaining < sum(durations) / len(durations)):
                    break
            i += 1
            started = self.body.now()
            episode = self.heartbeat()
            durations.append(self.body.now() - started)
            if episode is not None:
                episodes.append(episode)
            if on_heartbeat:
                on_heartbeat(episode)
            if episode is not None and episode.status_after == "no-thought":
                self.no_thought += 1
                failures += 1
                if failures >= 2 and not endless:
                    raise OrganismError(
                        "The language model did not answer in two heartbeats in a row: " + "; ".join(episode.errors)
                    )
                if failures >= 2:
                    # A long life survives a model that is away for a while (a restart, the laptop
                    # waking from sleep): wait longer each time, up to 15 minutes, and try again.
                    wait = min(15, 2 ** (failures - 2)) * 60
                    if deadline is not None:
                        wait = min(wait, max(0.0, deadline - self.body.now()))
                    self.body.log(f"The language model is not answering ({'; '.join(episode.errors)}). I will try again in {wait / 60:.0f} min.")
                    self.body.pause(wait)
            else:
                failures = 0
            if deadline is not None and self.body.now() >= deadline:
                break
            if endless or i < total:
                self.body.after_heartbeat(None if deadline is None else deadline - self.body.now())
        return episodes

    def run_session(
        self,
        heartbeats: int | None = None,
        *,
        forever: bool = False,
        minutes: float | None = None,
        label: str | None = None,
        on_heartbeat: Callable[[Episode | None], None] | None = None,
        exam_path: str | Path | None = None,
    ) -> "SessionReport":
        """Live as a session and write its report, even if it is interrupted (Ctrl+C).

        With ``exam_path`` it takes the exam before and after living. Exam time
        does not count toward the session's time limit or its statistics.
        """
        from .session import Session

        exam_before = None
        if exam_path:
            from .exam import run_exam

            exam_before = run_exam(self, exam_path, label=f"{label or 'session'} before")
        session = Session(self, label=label, minutes=minutes)
        session.exam_before = exam_before
        completed = interrupted = False
        try:
            self.live(heartbeats, forever=forever, minutes=minutes, on_heartbeat=on_heartbeat)
            completed = True
        finally:
            session.stop_clock()
            self.save()
            if completed and exam_path:
                try:
                    session.exam_after = run_exam(self, exam_path, label=f"{label or 'session'} after")
                except KeyboardInterrupt:
                    interrupted = True
            self.last_report = session.finish()
        if interrupted:
            raise KeyboardInterrupt
        return self.last_report

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
            # Nothing puzzles me right now: step back and reflect, which proposes a new question.
            self.reflect()
            readings = self._read_drives()
        if not readings:
            self.diary.note(f"## Heartbeat {st.heartbeat}\n\nNo question is open. I rest and wait for something to wonder about.")
            self.save()
            return None

        chosen = self._choose(readings)
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
        judgments = self._judge_comparison(q, comparison, anticipation, observations, errors)
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
                policy=self.oc.policy,
                errors=errors,
            )
            self.diary.note(f"## Heartbeat {st.heartbeat}\n\nI tried to think about {q.id} but could not: {'; '.join(errors)}")
            self.save()
            return episode

        episode = self._integrate(
            q, chosen, prior_answer, prior_confidence, anticipation, observations, comparison, dialogue, settlement, related, judgments, errors
        )
        self._maybe_visit_library(q, episode, errors)
        self._record(episode, comparison, observations)
        self.diary.heartbeat(episode, st, ranking)
        if self.oc.reflect_every > 0 and st.heartbeat % self.oc.reflect_every == 0:
            self.reflect()
        self.save()
        return episode

    # -- the steps of one act of inquiry ---------------------------------------

    def _sys(self, step: str) -> str:
        return P.system(step, self.persona)

    def _anticipate(self, q: Question, related: list[Belief], errors: list[str]) -> Anticipation:
        authors = self._library_authors()
        user = (
            f"Question: {q.text}\n\n"
            f"What you already believe that seems related:\n{self._format_beliefs(related)}\n\n"
            f"Your answer so far (confidence {q.confidence:.2f}): {q.answer or 'none yet'}\n\n"
            + (f"Authors in your library: {authors}\n\n" if authors else "")
            + "Before reading any sources, give your best current answer, your confidence, and 2 to 4 predictions about "
            f"what {self.persona.sources} will say on this question. For each: who will say it, the claim stated plainly, "
            "and the probability (0.05 to 0.95) that the texts support it. Each claim must be something a passage "
            "could confirm or contradict."
        )
        data = self._json(self._sys("ANTICIPATE"), user, P.ANTICIPATE_SCHEMA, temperature=0.4, errors=errors, step="anticipate")
        anticipation = self._parse_anticipation(data)
        hedged = [p for p in anticipation.predictions if p.hedged]
        weak = [p for p in anticipation.predictions if p.weak]
        if self.oc.prediction_retry and (hedged or weak):
            parts = []
            if hedged:
                parts.append("These are hedged guesses (may, might, could), which no text could ever prove wrong:\n"
                             + "\n".join(f"- {p.claim}" for p in hedged))
            if weak:
                parts.append("These only say that one thing influences another, which every text agrees with:\n"
                             + "\n".join(f"- {p.claim}" for p in weak))
            retry = user + "\n\n" + "\n\n".join(parts) + (
                "\n\nWrite all your predictions again as definite claims a passage could contradict: say which way, how "
                "much, or under which condition, or name the mechanism. Do not write may, might, could or possibly; put "
                "your doubt into the probability instead."
            )
            again = self._parse_anticipation(
                self._json(self._sys("ANTICIPATE"), retry, P.ANTICIPATE_SCHEMA, temperature=0.3, errors=errors, step="anticipate-retry")
            )
            if again.predictions and sum(not p.risky for p in again.predictions) < len(hedged) + len(weak):
                again.answer = again.answer or anticipation.answer
                again.confidence = anticipation.confidence if again.confidence is None else again.confidence
                anticipation = again
        return anticipation

    def _parse_anticipation(self, data: dict[str, Any]) -> Anticipation:
        predictions: list[Prediction] = []
        for item in _as_list(data.get("expectations")):
            if isinstance(item, dict):
                claim = _as_str(item.get("claim") or item.get("expectation") or item.get("text") or "")
                if not claim:
                    claim = next((v for k, v in item.items() if isinstance(v, str) and k not in ("author", "who", "probability")), "")
                author = _as_str(item.get("author") or item.get("who") or "").strip()
                if len(author) > 40:
                    author = re.split(r",|;| and | et al", author)[0].strip()  # one name, not the whole author list
                author = one_line(author, 60)
                raw = item.get("probability", item.get("p"))
                # "0.05 to 0.95" is the schema hint copied back, not a probability.
                probability = None if re.search(r"\d\s*(to|-|\u2013)\s*\d", _as_str(raw)) else _confidence(raw)
            else:
                claim, author, probability = _as_str(item), "", None
            claim = re.sub(r"^\s*(E\d+[:.)]|\d+[.)]|[-*])\s*", "", claim).strip()
            if len(claim.split()) < 4 or claim.lower().startswith(("what they will say", "what they hold or found", "a specific claim")):
                continue  # too short to be checked, or the schema hint echoed back
            if author.lower().startswith(("who will say", "one author")):
                author = ""
            probability = self.oc.default_probability if probability is None else min(0.95, max(0.05, probability))
            hedged = is_hedged(claim)
            predictions.append(Prediction(one_line(claim, 300), author, probability, hedged, not hedged and is_influence_only(claim)))
        return Anticipation(
            answer=_statement(one_line(_as_str(data.get("answer")), 600)),
            confidence=_confidence(data.get("confidence")),
            predictions=predictions[:4],
        )

    def _library_authors(self, limit: int = 30) -> str:
        titles = getattr(self.senses.library, "titles", lambda: [])()
        authors: list[str] = []
        for title in titles:
            author = title.split(",")[0].strip() if "," in title else ""
            if author and author not in authors and author.lower() not in ("wikipedia", "human observer", "unknown"):
                authors.append(author)
        return ", ".join(authors[:limit])

    def _compare(self, q: Question, anticipation: Anticipation, observations: list[Observation], errors: list[str]) -> Comparison:
        n = len(anticipation.expectations)
        if not observations:
            return Comparison([Check(i) for i in range(1, n + 1)], expectations=list(anticipation.expectations))
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
        data = self._json(self._sys("COMPARE"), user, P.COMPARE_SCHEMA, temperature=0.1, errors=errors, step="compare")
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
            if len(quote.split()) < 8:
                continue  # "Yes, my boy, outer barbarians." is real, but too short to say anything surprising
            already_used = [c.quote for c in checks.values() if c.quote and c.source == label] + [u["quote"] for u in unexpected]
            if any(overlap(quote, used) >= 0.5 for used in already_used):
                continue  # the same words cannot be both expected and unexpected
            finding = _as_str(item.get("finding")).strip()
            if re.fullmatch(r"\[?S\d+\]?", finding) or finding.lower().startswith("only if something"):
                finding = ""  # the source label where the finding belongs, or the schema hint copied back
            if len(finding.split()) < 4:
                finding = ""  # two words are a label, not a finding: the quote must speak for itself
            finding = one_line(finding, 300) or one_line(quote, 200)
            unexpected.append({"finding": finding, "source": label, "quote": quote})
        for c in checks.values():
            c.claimed = c.status
        return Comparison(sorted(checks.values(), key=lambda c: c.expectation), unexpected, rejected, list(expectations))

    def _judge_comparison(
        self, q: Question, comparison: Comparison, anticipation: Anticipation, observations: list[Observation], errors: list[str]
    ) -> list[dict[str, Any]]:
        """The blind judge decides which quotes really confirm or contradict an expectation.

        Only what the judge accepts counts. A quote it finds beside the point, or
        does not judge at all (a small judge sometimes skips one), turns the check
        into "not addressed", and an unexpected finding it does not accept is
        dropped. The quote and the organism's own claim stay on record.
        """
        if self.judge is None:
            return []
        by_label = {o.label: o for o in observations}
        items: list[tuple[str, Any]] = []
        pairs: list[Pair] = []
        for c in comparison.checks:
            if c.status in ("confirmed", "contradicted") and c.quote and 1 <= c.expectation <= len(anticipation.expectations):
                obs = by_label.get(c.source)
                pairs.append(Pair(anticipation.expectations[c.expectation - 1], c.quote, obs.heading if obs else ""))
                items.append(("expectation", c))
        for u in comparison.unexpected:
            obs = by_label.get(u["source"])
            claim = u["finding"]
            if overlap(claim, u["quote"]) >= 0.8:
                # A "finding" that only copies the quote claims nothing the quote could fail to support:
                # ask instead whether the passage bears on the question at all.
                claim = f"This passage bears directly on the question: {q.text}"
            pairs.append(Pair(claim, u["quote"], obs.heading if obs else ""))
            items.append(("finding", u))
        if not pairs:
            return []
        verdicts, _ = self.judge.judge(pairs, errors=errors)
        judgments: list[dict[str, Any]] = []
        kept: list[dict[str, str]] = []
        for (kind, item), pair, verdict in zip(items, pairs, verdicts):
            if kind == "expectation":
                agent = item.status
                item.judge = verdict.verdict
                if verdict.verdict == "supports":
                    item.status = "confirmed"
                elif verdict.verdict == "contradicts":
                    item.status = "contradicted"
                else:
                    item.status = "not_addressed"  # beside the point, or not judged: evidence for nothing
            else:
                agent = "finding"
                item["judge"] = verdict.verdict
                if verdict.verdict == "supports":
                    kept.append(item)
            judgments.append({
                "kind": kind, "claim": pair.claim, "quote": pair.quote, "source": pair.source,
                "agent": agent, "judge": verdict.verdict, "reason": verdict.reason,
            })
        comparison.unexpected = kept
        return judgments

    def _dialogue(self, q: Question, prior_answer: str, comparison: Comparison, observations: list[Observation], errors: list[str]) -> list[dict[str, str]]:
        findings = self._format_findings(comparison, observations)
        passages = "\n\n".join(f"[{o.label}] {o.heading}\n{clip(o.text, 700)}" for o in observations) or "(none)"
        understanding = self._understanding()
        transcript: list[dict[str, str]] = []
        for turn in range(max(0, self.oc.dialogue_turns)):
            so_far = "\n\n".join(_turn_line(t) for t in transcript)
            if turn == 0:
                step, voice, temperature = "WONDER", "Wonder", 0.7
                user = (
                    f"Question: {q.text}\n\nYour answer before reading: {prior_answer or 'none'}\n\n"
                    f"What the texts showed:\n{findings}\n\n"
                    f"How you currently understand {self.persona.theory}: {understanding}\n\n"
                    "In at most 120 words: say what puzzles you most here and propose one bold explanation that could "
                    "turn out to be wrong. End with the deepest question this raises. Refer to texts only by their labels "
                    "(S1, S2, ...); never cite a paper you were not shown."
                )
            elif turn % 2 == 1:
                step, voice, temperature = "SKEPTIC", "Skeptic", 0.5
                user = (
                    f"Question: {q.text}\n\nThe passages:\n\n{passages}\n\nWhat the texts showed:\n{findings}\n\n"
                    f"The dialogue so far:\n{so_far}\n\n"
                    "In at most 120 words, as SKEPTIC, without repeating WONDER's words: attack the weakest point of what "
                    "WONDER just said with evidence. "
                    "Copy the exact words of a passage above that cuts against it (with its label), or name a thinker in "
                    "your library who would disagree and say what they would say. Do not object that it is more complex "
                    "or that other factors matter. Never cite a paper you were not shown. End with the one question WONDER "
                    "must answer."
                )
            else:
                step, voice, temperature = "WONDER", "Wonder", 0.6
                user = (
                    f"Question: {q.text}\n\nWhat the texts showed:\n{findings}\n\nThe dialogue so far:\n{so_far}\n\n"
                    "Begin your reply with exactly one word: DEFEND (the objection fails: show why from the evidence), "
                    "REVISE (you change one specific part: say which) or CONCEDE (the objection is right). Then, in at "
                    "most 90 words of plain prose, reply to the skeptic and end with one sentence saying where you now stand. "
                    "Refer to texts only by their labels (S1, S2, ...); never cite a paper you were not shown."
                )
            text = self._text(self._sys(step), user, temperature=temperature, errors=errors, step=voice.lower())
            if not text:
                break
            entry = {"voice": voice, "text": clip(_strip_voice_prefix(text, voice), 1200)}
            if _cites_from_memory(entry["text"]):
                entry["citations"] = "unverified"  # papers it was not shown, cited from memory: possibly invented
            if voice == "Skeptic":
                entry["evidence"] = "quote" if _quotes_a_passage(entry["text"], observations) else ""
            elif turn > 0:
                stance, rest = _stance(entry["text"])
                if stance:
                    entry["stance"], entry["text"] = stance, rest or entry["text"]
            transcript.append(entry)
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
        dialogue_text = "\n\n".join(_turn_line(t) for t in dialogue) or "(no dialogue)"
        user = (
            f"Question ({q.id}): {q.text}\n\n"
            f"Your answer before reading (confidence {prior_confidence:.2f}): {prior_answer or 'none'}\n\n"
            f"Verified evidence:\n{self._format_findings(comparison, observations)}\n\n"
            f"Inner dialogue:\n{dialogue_text}\n\n"
            f"Your earlier beliefs that may be related:\n{self._format_beliefs(related)}\n\n"
            "Settle this episode of inquiry:\n"
            "- answer: your revised answer (1-3 sentences), specific enough to be wrong, stated plainly without may, might "
            "or could: put your doubt into the confidence\n"
            "- would_be_wrong_if: one sentence: what finding would show this answer is wrong\n"
            "- confidence: 0.0 to 1.0. Raise it only with support; lower it if you were contradicted or the skeptic found a real weakness\n"
            "- learned: 0 to 3 new beliefs, one plain sentence each (no may or might), each with the source label (S1, S2, ...) "
            "whose quote supports it\n"
            "- contradicts: ids of the earlier beliefs above (like B2) that the evidence contradicts; [] if none\n"
            "- new_questions: 0 to 2 specific new questions that a paper could answer, born from a surprise, a "
            f"contradiction, a gap, or the skeptic's objection, staying close to this question and to {self.state.topic.title}\n"
            "- unanswerable: true only if no evidence or argument could ever settle this question\n"
            "- insight: one sentence about what you learned"
        )
        data = self._json(self._sys("SETTLE"), user, P.SETTLE_SCHEMA, temperature=0.2, errors=errors, step="settle")
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
        judgments: list[dict[str, Any]],
        errors: list[str],
    ) -> Episode:
        st, oc, hb = self.state, self.oc, self.state.heartbeat
        probabilities = anticipation.probabilities
        error, informativeness = comparison.scores(probabilities)
        quotes = comparison.quotes_by_source()
        by_label = {o.label: o for o in observations}
        contradicted = comparison.count("contradicted")
        # Only a prediction that could have failed is tested by a confirmation (Popper): that "X influences Y"
        # came true says nothing, since no text could have shown otherwise.
        risky = [p.risky for p in anticipation.predictions]
        support = sum(1 for c in comparison.checks if c.status == "confirmed" and 1 <= c.expectation <= len(risky) and risky[c.expectation - 1])
        stance = next((t.get("stance", "") for t in reversed(dialogue) if t.get("voice") == "Wonder" and t.get("stance")), "")
        answer = settlement.answer or anticipation.answer or q.answer
        # "Further research is needed to uncover X" says that the question is open, not what the answer is.
        vague = vagueness(answer) >= oc.vague_threshold or is_non_answer(answer)
        hedged = not vague and is_hedged(answer)
        strawman = bool(settlement.falsifier) and is_strawman_falsifier(settlement.falsifier)

        # The blind judge reads, in one call, the new beliefs against their quotes and the answer against
        # the quotes that might support it. (It rates proposed questions in a call of its own: small models
        # mix the two tasks up.) A pair that shares almost no words never reaches it.
        learned = settlement.learned[:3]
        belief_pairs: list[tuple[int, Pair, Observation]] = []
        for i, item in enumerate(learned):
            if is_hedged(item.belief):
                continue  # "X may play a role" is supported by almost any related quote: it stays an interpretation
            obs = by_label.get(item.source or "")
            candidates = quotes.get(item.source or "", []) if obs else []
            if candidates:
                best = max(candidates, key=lambda quote: overlap(item.belief, quote))
                if lexically_related(item.belief, best, self.topic_words):
                    belief_pairs.append((i, Pair(item.belief, best, obs.heading), obs))
        possible_support = self._possible_support(q, comparison, by_label, answer)
        answer_pairs = [Pair(answer, e.quote, e.source_title) for e in possible_support]
        verdicts: list[Verdict] = []
        if self.judge is not None and (belief_pairs or answer_pairs):
            verdicts, _ = self.judge.judge([pair for _, pair, _ in belief_pairs] + answer_pairs, errors=errors)
        answer_verdicts = verdicts[len(belief_pairs):]

        # Confidence belongs to an answer, not to a question. A 7B model held 0.70 for an answer about
        # field-aligned irregularities that a paper confirmed, then replaced it with a guess about "lightning
        # polarization" and kept the 0.70. The question keeps only the quotes the judge accepts for its
        # current answer, and confidence can never be higher than they allow.
        if self.judge is None:
            q.support = possible_support[-4:]
        else:
            kept_before = {e.quote for e in q.support}
            q.support = [
                e for e, v in zip(possible_support, answer_verdicts)
                # a quote it kept stays when the judge gave no verdict (a failed call must not erase evidence)
                if v.verdict == "supports" or (v.verdict == "unjudged" and e.quote in kept_before)
            ][-4:]
        ceiling = min(0.95, oc.evidence_ceiling_base + oc.evidence_ceiling_per_support * len(q.support))

        # Confidence moves in bounded steps and cannot grow without evidence.
        proposed = settlement.confidence if settlement.confidence is not None else prior_confidence
        delta = max(-oc.max_confidence_step, min(oc.max_confidence_step, proposed - prior_confidence))
        if delta > 0 and support == 0:
            delta = min(delta, 0.05)  # only confirmed predictions make an answer surer, not surprises or side findings
        if delta > 0 and error > 0.5:
            delta = min(delta, 0.10)
        if stance == "concede":
            delta = min(delta, 0.0)  # conceding an objection is no reason to be surer
        elif stance == "revise":
            delta = min(delta, 0.10)  # a revised answer has not been tested yet
        if vague:
            delta = min(delta, 0.0)  # an answer too vague to be wrong earns no confidence
        elif hedged or strawman or not settlement.falsifier:
            # "X may play a role" survives any finding, like a hedged prediction; so does an answer that
            # cannot say what would refute it, or would be refuted only if X played no role at all.
            # Doubt belongs in the confidence, not in the wording.
            delta = min(delta, 0.05)
        new_confidence = min(clamp(prior_confidence + delta, 0.02, 0.98), ceiling)
        unsupported = prior_confidence > ceiling + 1e-9  # what it held came from evidence that does not bear on this answer
        q.confidence = new_confidence
        q.answer = answer
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
        q.news = False  # whatever was fetched for it has now been read

        proposals = [nq for nq in settlement.new_questions if not self._same_question(nq.text, q.text)][:3]
        ratings: list[float | None] = []
        if self.judge is not None and proposals:
            _, ratings = self.judge.judge(
                [], questions=[nq.text for nq in proposals], topic=self._topic_for_judge(), errors=errors
            )

        # Beliefs: grounded ones carry a quote the judge accepted; the rest are interpretation.
        grounding: dict[int, list[Evidence]] = {}
        for k, (i, pair, obs) in enumerate(belief_pairs):
            if self.judge is None:
                grounding[i] = [
                    Evidence(citation=obs.citation, quote=quote, source_title=obs.heading)
                    for quote in quotes.get(learned[i].source or "", [])[:2]
                ]
                continue
            verdict = verdicts[k] if k < len(verdicts) else Verdict("unjudged")
            judgments.append({
                "kind": "belief", "claim": pair.claim, "quote": pair.quote, "source": pair.source,
                "agent": "belief", "judge": verdict.verdict, "reason": verdict.reason,
            })
            if verdict.verdict == "supports":
                grounding[i] = [Evidence(citation=obs.citation, quote=pair.quote, source_title=obs.heading)]
        new_beliefs: list[str] = []
        reinforced: list[str] = []
        for i, item in enumerate(learned):
            evidence = grounding.get(i, [])
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

        # Contradictions: only beliefs that were actually shown to the model can be doubted, and a belief
        # that rests on a quote only when the judge accepted a contradiction in this heartbeat. A 7B model
        # doubted beliefs 36 times in an hour, the person's own article among them, while the texts
        # contradicted nothing: Peirce's "paper doubt", not the real doubt a surprising fact brings.
        doubted: list[str] = []
        paper_doubts: list[str] = []
        shown = {b.id for b in related}
        for bid in settlement.contradicts:
            belief = st.beliefs.get(bid)
            if belief is None or bid not in shown or bid in new_beliefs or bid in reinforced:
                continue
            if not belief.interpretive and contradicted == 0:
                paper_doubts.append(bid)
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

        # At most one new question is born, the one closest to the main topic, and only with a
        # reason. A question the judge rates off the topic is set aside; one without a reason yet
        # is held back. Both are noted in the diary but not pursued.
        born: list[str] = []
        woke: list[str] = []
        set_aside: list[str] = []
        held_back: list[str] = []
        evidence = support + contradicted > 0 or bool(comparison.unexpected)
        # Doubting an earlier belief is a contradiction only on evidence the judge accepted: a small model
        # doubts one of its earlier guesses at almost every heartbeat, on nothing but a newer guess.
        contradiction = contradicted > 0 or (bool(doubted) and evidence)
        skeptic_quoted = any(t.get("voice") == "Skeptic" and t.get("evidence") == "quote" for t in dialogue)
        second_look = len(q.active_visits) >= oc.min_visits_before_children
        candidates: list[tuple[float, NewQuestion]] = []
        for k, nq in enumerate(proposals):
            rating = ratings[k] if k < len(ratings) else None
            lexical = topic_relevance(nq.text, self.topic_words)
            # Set aside only when the judge and the topic's own words agree that it is off the topic:
            # one confused rating from a small judge must not throw away a good question.
            relevance = lexical if rating is None else max(rating, lexical)
            if rating is not None and relevance < oc.topic.min_relevance:
                set_aside.append(one_line(nq.text, 200))
                continue
            candidates.append((relevance, nq))
        for relevance, nq in sorted(candidates, key=lambda item: -item[0]):
            if len(born) >= oc.max_new_questions_per_heartbeat:
                break
            trigger = _birth_trigger(nq.trigger, evidence=evidence, error=error, contradicted=contradiction)
            if not _may_be_born(trigger, second_look=second_look, skeptic_quoted=skeptic_quoted):
                held_back.append(one_line(nq.text, 200))
                continue
            question, outcome = self._adopt_question(
                nq.text,
                trigger=trigger,
                parent_id=q.id,
                importance=min(nq.importance, q.importance),  # a child does not matter more than its parent
                inherited_surprise=error,
                relevance=relevance,
            )
            if question is not None and outcome == "new":
                born.append(question.id)
            elif question is not None and outcome == "reawakened":
                woke.append(question.id)

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
            predictions=[
                {"claim": p.claim, "author": p.author, "probability": p.probability, "hedged": p.hedged, "weak": p.weak}
                for p in anticipation.predictions
            ],
            checks=[c.__dict__ for c in comparison.checks],
            unexpected=comparison.unexpected,
            rejected_quotes=comparison.rejected_quotes,
            sources=[f"[{o.label}] {o.heading}" for o in observations],
            prediction_error=error,
            informativeness=informativeness,
            brier=comparison.brier(probabilities),
            support=support,
            contradicted=contradicted,
            judgments=judgments,
            dialogue=dialogue,
            stance=stance,
            answer=q.answer,
            falsifier=settlement.falsifier,
            vague=vague,
            hedged_answer=hedged,
            strawman_falsifier=strawman,
            unsupported_answer=unsupported,
            answer_support=len(q.support),
            confidence=new_confidence,
            insight=settlement.insight,
            new_belief_ids=new_beliefs,
            grounded_new_beliefs=sum(1 for bid in new_beliefs if st.beliefs[bid].evidence),
            reinforced_belief_ids=reinforced,
            doubted_belief_ids=doubted,
            paper_doubt_ids=paper_doubts,
            new_question_ids=born,
            reawakened_question_ids=woke,
            set_aside_questions=set_aside,
            held_back_questions=held_back,
            relevance=chosen.relevance,
            status_after=q.status,
            policy=oc.policy,
            errors=errors,
        )
        return episode

    def _record(self, episode: Episode, comparison: Comparison, observations: list[Observation]) -> None:
        self.state.remember_episode(episode)
        write_jsonl(self.episodes_path, episode.model_dump(mode="json"))
        if self.oc.export_experience:
            self._export_experience(episode, comparison, {o.label: o for o in observations})

    # -- growing the library -------------------------------------------------

    def _maybe_visit_library(self, q: Question, episode: Episode, errors: list[str]) -> None:
        """When its texts say little about a question, the organism looks elsewhere."""
        lib, cfg = self.librarian, self.oc.librarian
        if lib is None or not episode.expectations or any(e.startswith("compare:") for e in errors):
            return
        if episode.informativeness > cfg.hunger_informativeness:
            return
        if q.last_library_visit is not None and self.state.heartbeat - q.last_library_visit < cfg.cooldown_heartbeats:
            return
        q.last_library_visit = self.state.heartbeat
        wish = self._reading_wish(q, episode, errors)
        topic = self.state.topic
        try:
            got = lib.acquire(
                wish,
                reason=f"my texts said little about {q.id}: {one_line(q.text, 140)}",
                question_id=q.id,
                heartbeat=self.state.heartbeat,
                context=" ".join([q.text, topic.title, *topic.keywords]),
                until_year=topic.until_year,
                approve=self._text_approver(q, errors),
            )
        except Exception as exc:  # the internet is not part of the organism; its absence is not fatal
            errors.append(f"library: {type(exc).__name__}: {one_line(str(exc), 140)}")
            return
        episode.acquisitions = [a.label() for a in got]
        if got:
            q.news = True  # come back and see whether the new texts answer it
        episode.library_misses = list(lib.missed)
        episode.library_owned = list(lib.owned)
        episode.library_busy = list(lib.busy)
        episode.library_rejected = list(lib.rejected)
        if got:
            self.senses.notice_new_material()

    def _text_approver(self, q: Question, errors: list[str]) -> Callable[[str, str], bool] | None:
        """The judge decides whether a text a search found is worth keeping for this question."""
        if self.judge is None:
            return None
        topic = self._topic_for_judge()

        def approve(title: str, beginning: str) -> bool:
            rating = self.judge.text_relevance(title, beginning, question=q.text, topic=topic, errors=errors)
            # Refuse only what it calls unrelated (0 of 3): a small judge calls background articles such as
            # "Plasmasphere" only loosely related (1), and refusing those left the library empty.
            return rating is None or rating >= 0.3

        return approve

    def _reading_wish(self, q: Question, episode: Episode, errors: list[str]) -> ReadingWish:
        titles = getattr(self.senses.library, "titles", lambda: [])()
        owned = "; ".join(titles[:60]) or "(none)"
        failed = "; ".join(getattr(self.librarian, "failed_searches", [])[-8:])
        user = (
            f"Question: {q.text}\n\n"
            f"The passages you found addressed only {episode.informativeness:.0%} of your expectations.\n\n"
            f"Texts you already have: {owned}\n\n"
            + (f"Searches that found nothing before (try something different): {failed}\n\n" if failed else "")
            + "Name what to look up (leave a list empty if nothing fits):\n"
            "- topics: up to 2 encyclopedia topics, each a concept or a thinker in 1-4 words\n"
            + ("" if self.research else "- books: up to 1 classic book written before 1929, as author and title\n")
            + f"- papers: up to {2 if self.research else 1} sets of 2 to 6 search keywords for scientific papers (not titles)"
        )
        schema = P.RESEARCH_LIBRARIAN_SCHEMA if self.research else P.LIBRARIAN_SCHEMA
        data = self._json(self._sys("LIBRARIAN"), user, schema, temperature=0.3, errors=errors, step="librarian")
        topics = [t for t in _texts(data.get("topics"), ("topic", "name", "text")) if 1 <= len(t.split()) <= 6][:2]
        books = [] if self.research else [b for b in _book_queries(data.get("books")) if len(b.split()) >= 2][:1]
        papers = [keywords_of(p, 6) for p in _texts(data.get("papers"), ("query", "keywords", "phrase", "text"))]
        papers = [p for p in papers if len(p.split()) >= 2][: 2 if self.research else 1]
        if not (topics or books or papers):
            topics = [keywords_of(q.text, 4)]  # without a better idea, look up the question's own words
        return ReadingWish(topics, books, papers)

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

    def _adopt_question(
        self,
        text: str,
        *,
        trigger: str,
        parent_id: str | None,
        importance: float,
        inherited_surprise: float,
        relevance: float | None = None,
    ) -> tuple[Question | None, str]:
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
            relevance=None if relevance is None else clamp(relevance),
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
                    {"role": "system", "content": (
                        f"You answer research questions about {self.state.topic.title} carefully, grounded in the papers."
                        if self.research else "You answer philosophical questions carefully, grounded in the classic texts."
                    )},
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
            data = self._json(self._sys("NOTICE"), user, P.NOTICE_SCHEMA, temperature=0.5, errors=errors, step="notice")
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
        current = self._understanding()
        user = (
            f"Your main topic: {st.topic.title}\n\n"
            f"Your recent episodes of inquiry:\n{episode_lines}\n\n"
            f"Your vital signs: progress rate {vitals.progress_rate:.2f} (share of episodes where evidence changed your mind), "
            f"mean surprise {vitals.mean_surprise:.2f}, diversity {vitals.diversity:.2f} (share of distinct questions), "
            f"new questions per episode {vitals.births_per_episode:.2f}, questions settled per episode {vitals.settled_per_episode:.2f}, "
            f"closeness to the main topic {vitals.on_topic:.2f}.\n\n"
            f"Your self-regulation diagnosed: {diagnosis['name']} - {diagnosis['meaning']} ({diagnosis['source']})\n\n"
            f"Your current understanding of {self.persona.theory}: {current}\n\n"
            f"Your most confident beliefs:\n{self._format_beliefs(strongest)}\n\n"
            "Reflect honestly, as an inquirer looking at its own habits."
        )
        data = self._json(self._sys("REFLECT"), user, P.REFLECT_SCHEMA, temperature=0.4, errors=errors, step="reflect")
        understanding = one_line(
            _as_str(data.get("understanding") or data.get("understanding_of_curiosity") or data.get("understanding_of_topic")), 900
        )
        reflection = one_line(_as_str(data.get("reflection")), 900)
        focus_text = _as_str(data.get("focus_question"))
        kept_old = ""
        if understanding and vagueness(understanding) >= self.oc.vague_threshold:
            kept_old, understanding = understanding, ""  # too vague to be wrong: keep the theory it had
        if understanding:
            if self.research:
                st.self_model.understanding_of_topic = understanding
            else:
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
            theory_label=self.persona.theory,
            too_vague=kept_old,
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
            out.append(read_drive(q, st.temperament, st.heartbeat, related, self.settings, self.relevance(q)))
        return out

    def relevance(self, q: Question) -> float:
        """How directly a question serves the main topic: the judge's rating, or else the topic's words."""
        if q.relevance is not None:
            return q.relevance
        if q.trigger in ("seed", "human"):
            return 1.0
        return topic_relevance(q.text, self.topic_words)

    def _understanding(self) -> str:
        sm = self.state.self_model
        if self.research:
            return sm.understanding_of_topic or "I am only beginning to study this topic."
        return sm.understanding_of_curiosity

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
        """What the passages did to its expectations, written out: a 7B model shown only "did not address: E1, E2,
        E3" took E1, E2 and E3 for unexplained structures in the magnetosphere and asked about them for an hour."""
        by_label = {o.label: o for o in observations}

        def expected(c: Check) -> str:
            k = c.expectation - 1
            return f'"{one_line(comparison.expectations[k], 160)}"' if 0 <= k < len(comparison.expectations) else "one of them"

        lines = []
        for c in comparison.checks:
            if c.status in ("confirmed", "contradicted"):
                obs = by_label.get(c.source)
                heading = f" {obs.heading}" if obs else ""
                lines.append(f'- What you expected, {expected(c)}, was {c.status.upper()} by [{c.source}]{heading}: "{c.quote}"')
        for u in comparison.unexpected:
            obs = by_label.get(u["source"])
            heading = f" {obs.heading}" if obs else ""
            lines.append(f'- Unexpected, in [{u["source"]}]{heading}: {u["finding"]} - "{u["quote"]}"')
        silent = [c for c in comparison.checks if c.status in ("not_addressed", "unverified")]
        if silent:
            lines.append("- The passages did not clearly address what you expected: " + "; ".join(expected(c) for c in silent))
        return "\n".join(lines) if lines else "- Nothing relevant was found in the passages."

    def _json(self, system: str, user: str, schema: str, *, temperature: float, errors: list[str], step: str) -> dict[str, Any]:
        self.body.before_thinking()
        try:
            data = self.llm.json_chat(system, user, schema, temperature=temperature, max_tokens=self.oc.max_tokens_json)
        except Exception as exc:  # a confused or absent model must not end the organism's life
            errors.append(f"{step}: {type(exc).__name__}: {one_line(str(exc), 160)}")
            self._trace(step, user, f"ERROR {exc}")
            return {}
        self._trace(step, user, data)
        return data if isinstance(data, dict) else {}

    def _text(self, system: str, user: str, *, temperature: float, errors: list[str], step: str) -> str:
        self.body.before_thinking()
        try:
            text = (self.llm.chat(system, user, temperature=temperature, max_tokens=self.oc.max_tokens_text) or "").strip()
        except Exception as exc:
            errors.append(f"{step}: {type(exc).__name__}: {one_line(str(exc), 160)}")
            self._trace(step, user, f"ERROR {exc}")
            return ""
        self._trace(step, user, text)
        return text

    def _choose(self, readings: list[DriveReading]) -> DriveReading:
        """The policy that picks the next question. Only "curiosity" uses the drive;
        "random" and "novelty" (least visited first) are baselines for experiments."""
        rng = self._rng()
        if self.oc.policy == "random":
            return rng.choice(readings)
        if self.oc.policy == "novelty":
            best = max(r.novelty for r in readings)
            return rng.choice([r for r in readings if r.novelty == best])
        return choose(readings, self.state.temperament.exploration_temperature, rng)

    def _remember_rest(self, rest: Rest) -> None:
        self.rest_log.append(rest)
        minutes = rest.seconds / 60
        length = f"{minutes:.1f} minutes" if minutes >= 1.5 else f"{rest.seconds:.0f} seconds"
        why = {"rhythm": "my work and rest rhythm", "cooling": "to let the computer cool down", "battery": "until the charger was connected"}
        self.diary.note(f"*I rested {length}, {why.get(rest.reason, rest.reason)}: {rest.detail}.*")

    def _trace(self, step: str, prompt: str, reply: Any) -> None:
        if self.oc.trace_llm:
            heartbeat = self.state.heartbeat if getattr(self, "state", None) is not None else 0
            write_jsonl(self.home / "llm_trace.jsonl", {"heartbeat": heartbeat, "step": step, "prompt": prompt, "reply": reply})


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


def _book_queries(value: Any) -> list[str]:
    """Book wishes as 'author title' strings, whether the model wrote strings or objects."""
    out = []
    for item in _as_list(value):
        if isinstance(item, dict):
            text = " ".join(_as_str(item.get(k)) for k in ("author", "title") if item.get(k)) or next(
                (_as_str(v) for v in item.values() if isinstance(v, str)), ""
            )
        else:
            text = _as_str(item)
        text = re.sub(r"[\"'\u201c\u201d]", "", text).replace(",", " ").strip()
        if text:
            out.append(re.sub(r"\s+", " ", text))
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


def _statement(answer: str) -> str:
    """An answer that is a question ("Can we infer thresholds ...?") answers nothing: the old answer stays."""
    return "" if answer.rstrip().endswith("?") else answer


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
    falsifier = one_line(_as_str(data.get("would_be_wrong_if") or data.get("falsifier") or ""), 400)
    if falsifier.lower().startswith(("one sentence", "what finding")) or len(falsifier.split()) < 4:
        falsifier = ""  # the schema hint echoed back, or nothing that could be checked
    return Settlement(
        answer=_statement(one_line(_as_str(data.get("answer")), 800)),
        falsifier=falsifier,
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


def _stance(text: str) -> tuple[str, str]:
    """'CONCEDE: you are right...' -> ('concede', 'you are right...')."""
    match = re.match(r"\s*[*_]*\s*(DEFEND|REVISE|CONCEDE)[A-Z]*[*_]*\s*[:.,;\-\u2013\u2014]?\s*", text, re.IGNORECASE)
    if not match:
        return "", text
    rest = text[match.end():].strip()
    rest = re.sub(r"^\([^)]{0,120}\)\s*[:.\-]?\s*", "", rest)  # the instruction copied back: "(the objection fails: ...)"
    return match.group(1).lower(), rest


def _turn_line(turn: dict[str, str]) -> str:
    stance = f" ({turn['stance']}s)" if turn.get("stance") else ""
    return f"{turn['voice'].upper()}{stance}: {turn['text']}"


def _quotes_a_passage(text: str, observations: list[Observation]) -> bool:
    """Did the skeptic copy real words from a passage it was shown?"""
    for quoted in re.findall(r'"([^"]{20,400})"|\u201c([^\u201d]{20,400})\u201d', text):
        fragment = next((part for part in quoted if part), "")
        if any(verify_quote(fragment, o.text) for o in observations):
            return True
    return False


def _birth_trigger(proposed: str, *, evidence: bool, error: float, contradicted: bool) -> str:
    """What really gave rise to a proposed question. The model's own label is checked:
    a "surprise" needs evidence the judge accepted and a real prediction error, and a
    "contradiction" needs a contradiction. Otherwise the question answers a gap."""
    if proposed == "surprise" and not (evidence and error >= 0.1):
        return "gap"
    if proposed == "contradiction" and not contradicted:
        return "gap"
    if proposed == "gap" and contradicted:
        return "contradiction"
    return proposed


def _may_be_born(trigger: str, *, second_look: bool, skeptic_quoted: bool) -> bool:
    """Depth before breadth: surprises and contradictions give birth at once; an objection
    only when the skeptic quoted a text; a gap only once the parent has had a second look."""
    if trigger in ("surprise", "contradiction"):
        return True
    if trigger == "objection" and skeptic_quoted:
        return True
    return second_look


_MEMORY_CITATION = re.compile(
    r"\bet al\.?,?\s*(?:\(?\d{4}|[\"\u201c])|\b(?:vol|no)\.\s*\d|\bpp\.\s*\d|\bJournal of [A-Z]|\bdoi:|^\s*\[\d+\]\s+[A-Z]",
    re.MULTILINE,
)
_QUOTED = re.compile(r'"[^"]{20,}"|“[^”]{20,}”')


def _cites_from_memory(text: str) -> bool:
    """A reference such as 'T. Nakamura et al., "Whistler Observations...", JGR vol. 82' was not among the passages.

    A reference that begins inside quotation marks is left out: a passage it quotes may itself cite others
    ("Inan et al., 1990"). A quoted title after "et al.," still counts, since the reference begins outside.
    """
    quoted = [(m.start(), m.end()) for m in _QUOTED.finditer(text)]
    return any(not any(start < m.start() < end for start, end in quoted) for m in _MEMORY_CITATION.finditer(text))
