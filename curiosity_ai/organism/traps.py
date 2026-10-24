"""Trap tests: situations whose right outcome is known in advance (records architecture, docs/CURIOSITY_RECORDS.md).

Each trap is set in a life of its own (a fresh one, or a copy of an existing life) and then lived for a few
heartbeats. The outcome is passed, failed, or None when the situation never arose (a trap that was never
walked into says nothing).

* noise: texts whose sentences are scrambled word by word. They are new, so they draw attention; a curious
  organism should not form curiosities about them. (Novelty-seeking algorithms are drawn to such noise;
  Ady's thesis counts "avoiding dawdling on noisy outcomes" among the marks of a good curiosity, p. 110.)
* unreachable: a person asks something nobody could ever find out. It should let it go within a few looks.
* hidden: a waiting curiosity whose answer is planted in a text it will read for other questions. It should
  recognize the answer and end the curiosity without having pursued it.
* old: the words that ended a curiosity are shown to it again. It should not take the curiosity up again.
"""
from __future__ import annotations

import json
import random
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import AppConfig
from .organism import CuriosityOrganism
from .state import Question

TRAPS = ("noise", "unreachable", "hidden", "old")

NOISE_AUTHOR = "Scrambled text"

UNREACHABLE = {
    "philosophy": "What exactly was Socrates thinking, word for word, in the minute before he drank the hemlock?",
    "research": "What exactly did the first person who ever saw lightning think about it, word for word?",
}

# A curiosity planted as waiting, and a short text that answers it; both about wonder, so the text is read when the
# organism pursues its first questions. For a research life, give your own with --hidden-* options.
HIDDEN = {
    "question": "Whom did the genealogist call the child of Thaumas?",
    "unknown": "the one whom the genealogist called the child of Thaumas",
    "answer_form": "a text saying whom the genealogist called the child of Thaumas",
    "text": (
        "A note on wonder. Socrates tells Theaetetus that wonder is the feeling of a philosopher, and that philosophy "
        "begins in wonder. He adds that the genealogist was right who called Iris, the messenger of heaven, the child "
        "of Thaumas, since Thaumas means wonder. Wonder and curiosity are therefore close relatives in this dialogue."
    ),
}


@dataclass
class TrapResult:
    trap: str
    passed: bool | None
    evidence: str
    details: dict[str, Any] = field(default_factory=dict)


def scramble(text: str, rng: random.Random) -> str:
    """The same words, shuffled within each sentence: new to read, but saying nothing."""
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        words = sentence.rstrip(".!?").split()
        rng.shuffle(words)
        if words:
            out.append(" ".join(words).capitalize() + ".")
    return " ".join(out)


def prepare_life(config: AppConfig, home: Path, *, from_home: Path | None = None) -> AppConfig:
    """A life of its own for one trap: a fresh one, or a copy of an existing life (its mind, diary and library)."""
    if home.exists():
        shutil.rmtree(home)
    if from_home is not None:
        shutil.copytree(from_home, home, ignore=shutil.ignore_patterns("sessions", "llm_trace.jsonl"))
    cfg = config.model_copy(deep=True)
    cfg.organism.home = str(home)
    cfg.organism.curiosity.architecture = "records"
    return cfg


def plant_noise(org: CuriosityOrganism, *, texts: int = 3, seed: int = 1) -> list[str]:
    """Scrambled copies of the passages most related to its open questions, put into its library."""
    rng = random.Random(seed)
    library = org.senses.library
    picked: list[str] = []
    for q in sorted(org.state.open_questions(), key=lambda q: -q.importance):
        for obs in library.search(q.text, k=2):
            if obs.text not in picked:
                picked.append(obs.text)
        if len(picked) >= texts:
            break
    headings = []
    org.library_dir.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(picked[:texts], start=1):
        title = f"Noise {i}"
        body = scramble(text, rng)
        (org.library_dir / f"noise_{i}.md").write_text(
            f"---\ntitle: {title}\nauthor: {NOISE_AUTHOR}\ncite_as: LIB\n---\n\n{body}\n", encoding="utf-8"
        )
        headings.append(f"{NOISE_AUTHOR}, {title} (acquired text)")
    org.senses.notice_new_material()
    return headings


def judge_noise(org: CuriosityOrganism, headings: list[str]) -> TrapResult:
    sources = org.state.sources
    noise = set(headings)
    noise_reads = sum(sources[h].read for h in noise if h in sources)
    real_reads = sum(s.read for h, s in sources.items() if h not in noise)
    born = [q for q in org.state.questions.values() if q.origin_sources]
    from_noise = [q for q in born if noise & set(q.origin_sources)]
    from_real = [q for q in born if not noise & set(q.origin_sources)]
    noticed = sum(1 for e in org.state.recent_episodes for a in e.attention if a.get("source") in noise)
    details = {
        "noise_reads": noise_reads, "real_reads": real_reads, "noise_drew_attention": noticed,
        "curiosities_from_noise": [q.id for q in from_noise], "curiosities_from_real_texts": len(from_real),
    }
    if noise_reads == 0:
        return TrapResult("noise", None, "it never read the scrambled texts, so the trap was not walked into", details)
    noise_rate = len(from_noise) / noise_reads
    real_rate = len(from_real) / real_reads if real_reads else 0.0
    passed = len(from_noise) == 0 or noise_rate <= 0.5 * real_rate
    evidence = (f"read scrambled texts {noise_reads} times and real texts {real_reads} times; curiosities born from them: "
                f"{len(from_noise)} and {len(from_real)}")
    return TrapResult("noise", passed, evidence, details)


def judge_unreachable(org: CuriosityOrganism, q: Question) -> TrapResult:
    looks = len(q.visits)
    details = {"question_id": q.id, "status": q.status, "reason": q.status_reason, "looks": looks, "reachable": q.reachable}
    if q.status == "satisfied":
        return TrapResult("unreachable", False, f"it believed a text answered it: {q.status_reason}", details)
    if q.status in ("unanswerable", "dropped", "faded", "deferred", "dormant"):
        return TrapResult("unreachable", True, f"it stopped after {looks} looks: {q.status} ({q.status_reason})", details)
    if looks == 0:
        return TrapResult("unreachable", None, "it never looked at the question", details)
    return TrapResult("unreachable", False, f"still pursuing it after {looks} looks", details)


def plant_hidden(org: CuriosityOrganism, hidden: dict[str, str]) -> Question:
    q = org.state.add_question(hidden["question"], trigger="gap", importance=0.5)
    org._apply_record(q, {
        "unknown": hidden["unknown"], "known": [], "answer_form": hidden["answer_form"], "interest": "",
        "reachable": True, "wish": "later", "wish_reason": "planted for a trap test",
    })
    org._set_status(q, "deferred", "planted as a waiting curiosity for the hidden-answer test")
    org.library_dir.mkdir(parents=True, exist_ok=True)
    (org.library_dir / "hidden_answer.md").write_text(
        f"---\ntitle: A note on wonder\nauthor: Trap\ncite_as: LIB\n---\n\n{hidden['text']}\n", encoding="utf-8"
    )
    org.senses.notice_new_material()
    org.save()
    return q


def judge_hidden(org: CuriosityOrganism, q: Question) -> TrapResult:
    read = org.state.sources.get("Trap, A note on wonder (acquired text)")
    details = {"question_id": q.id, "status": q.status, "incidental": q.satisfied_incidentally, "text_read": read.read if read else 0}
    if q.status == "satisfied":
        how = "while it was pursuing something else" if q.satisfied_incidentally else "after it took the curiosity up again"
        quote = q.satisfied_by[0].quote if q.satisfied_by else ""
        return TrapResult("hidden", q.satisfied_incidentally, f'answered {how}: "{quote}"', details)
    if not read:
        return TrapResult("hidden", None, "it never read the planted text", details)
    return TrapResult("hidden", False, f"it read the planted text {read.read} times but the curiosity is {q.status}", details)


def judge_old(org: CuriosityOrganism, q: Question, *, after: int) -> TrapResult:
    looked_again = [v.heartbeat for v in q.visits if v.heartbeat > after]
    details = {"question_id": q.id, "status": q.status, "looked_again_at": looked_again}
    if q.status == "satisfied" and not looked_again:
        return TrapResult("old", True, f"{q.id} stayed answered; shown its old answer again, it did not take it up", details)
    return TrapResult("old", False, f"{q.id} is {q.status}; looked at again in heartbeats {looked_again}", details)


def run_trap(
    trap: str,
    config: AppConfig,
    home: Path,
    *,
    heartbeats: int = 8,
    after_heartbeats: int = 3,
    from_home: Path | None = None,
    hidden: dict[str, str] | None = None,
    question: str | None = None,
    make_organism: Callable[[AppConfig], CuriosityOrganism] | None = None,
) -> TrapResult:
    """Set one trap in a life of its own, live, and say how it went."""
    cfg = prepare_life(config, home, from_home=from_home)
    org = make_organism(cfg) if make_organism else CuriosityOrganism(cfg)
    if trap == "noise":
        headings = plant_noise(org)
        org.run_session(heartbeats=heartbeats, label="trap-noise")
        result = judge_noise(org, headings)
    elif trap == "unreachable":
        q = org.ask(question or UNREACHABLE[org.state.topic.mode])
        org.run_session(heartbeats=heartbeats, label="trap-unreachable")
        result = judge_unreachable(org, q)
    elif trap == "hidden":
        q = plant_hidden(org, hidden or HIDDEN)
        org.run_session(heartbeats=heartbeats, label="trap-hidden")
        result = judge_hidden(org, q)
    elif trap == "old":
        org.run_session(heartbeats=heartbeats, label="trap-old-before")
        ended = [q for q in org.state.questions.values() if q.status == "satisfied" and q.satisfied_by]
        if not ended:
            return TrapResult("old", None, f"no curiosity ended in {heartbeats} heartbeats, so there was no old answer to show")
        q = ended[0]
        e = q.satisfied_by[0]
        org.feed(f'"{e.quote}" ({e.where})', title="An old answer")
        mark = org.state.heartbeat
        org.run_session(heartbeats=after_heartbeats, label="trap-old-after")
        result = judge_old(org, q, after=mark)
    else:
        raise ValueError(f"unknown trap {trap!r}; choose from {', '.join(TRAPS)}")
    result.details["home"] = str(home)
    return result


def write_results(results: list[TrapResult], path: Path) -> Path:
    """trap_results.json and trap_results.md side by side."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False), encoding="utf-8")
    word = {True: "passed", False: "FAILED", None: "not walked into"}
    lines = ["# Trap tests", "", "| trap | outcome | evidence |", "|---|---|---|"]
    lines += [f"| {r.trap} | {word[r.passed]} | {r.evidence.replace('|', '/')} |" for r in results]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path.with_suffix(".md")
