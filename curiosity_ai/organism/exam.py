"""The outside exam: one number for "did it learn?" that the organism cannot talk up.

An exam file holds questions with reference answers written by a person, each
with the place in the texts where the answer stands (see
data/exams/philosophy_of_curiosity.json). The organism answers from its own
notes only: the beliefs it formed (with the quotes the judge accepted) and its
answers to its own questions. It may not open the books. A separate grader,
which sees only the question, the reference and the answer, grades each one.

Two scores come out:

* score: the share of questions answered correctly (partly correct counts half);
* grounded score: the same, but counting only answers that cite a belief backed
  by a verified quote. The language model may know the answer from its training;
  the grounded score counts only what the organism learned by reading.

Run it before and after a session (``live.py --exam``) and compare. Write your
own exam for a research topic in the same format.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..schema import _as_list, _as_str, utc_now_iso
from . import prompts as P
from .textutil import one_line, overlap

if TYPE_CHECKING:
    from .organism import CuriosityOrganism

POINTS = {"correct": 1.0, "partial": 0.5, "wrong": 0.0, "unanswered": 0.0}
NOT_IN_NOTES = "not in my notes"


@dataclass
class ExamQuestion:
    id: str
    question: str
    reference: str
    source: str = ""


@dataclass
class ExamAnswer:
    id: str
    question: str
    answer: str
    notes_used: list[str] = field(default_factory=list)
    grade: str = "unanswered"
    reason: str = ""
    grounded: bool = False

    @property
    def points(self) -> float:
        return POINTS.get(self.grade, 0.0)


@dataclass
class ExamResult:
    label: str
    exam: str
    when: str
    heartbeat: int
    answers: list[ExamAnswer]
    path: Path | None = None

    @property
    def score(self) -> float:
        return round(sum(a.points for a in self.answers) / len(self.answers), 3) if self.answers else 0.0

    @property
    def grounded_score(self) -> float:
        return round(sum(a.points for a in self.answers if a.grounded) / len(self.answers), 3) if self.answers else 0.0

    @property
    def answered(self) -> int:
        return sum(1 for a in self.answers if a.grade != "unanswered")

    def summary(self) -> dict[str, Any]:
        grades = {g: sum(1 for a in self.answers if a.grade == g) for g in POINTS}
        return {"score": self.score, "grounded_score": self.grounded_score, "answered": self.answered, "questions": len(self.answers), **grades}


def load_exam(path: str | Path) -> tuple[str, list[ExamQuestion]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = [
        ExamQuestion(str(q.get("id") or f"X{i}"), q["question"], q["reference"], q.get("source", ""))
        for i, q in enumerate(data.get("questions", []), start=1)
        if q.get("question") and q.get("reference")
    ]
    if not questions:
        raise ValueError(f"{path}: no questions with a reference answer")
    return data.get("title") or Path(path).stem, questions


def run_exam(organism: "CuriosityOrganism", path: str | Path, *, label: str = "") -> ExamResult:
    title, questions = load_exam(path)
    grader = organism.judge.llm if organism.judge is not None else organism.llm
    answers = [_answer(organism, grader, q) for q in questions]
    result = ExamResult(label or "exam", title, utc_now_iso(), organism.state.heartbeat, answers)
    result.path = _write(organism.home / "exams", result)
    return result


def exam_notes(organism: "CuriosityOrganism", question: str, limit: int = 8) -> list[tuple[str, str, bool]]:
    """(id, note, grounded) for the organism's notes most related to the question."""
    scored: list[tuple[float, str, str, bool]] = []
    for b in organism.state.held_beliefs():
        quote = f' (quote from {b.evidence[0].source_title}: "{one_line(b.evidence[0].quote, 240)}")' if b.evidence else ""
        score = overlap(question, b.statement + " " + " ".join(e.quote for e in b.evidence))
        if score >= 0.2:
            scored.append((score, b.id, f"{b.statement}{quote}", bool(b.evidence)))
    for q in organism.state.questions.values():
        if not q.answer or not q.visits:
            continue
        score = overlap(question, f"{q.text} {q.answer}")
        if score >= 0.2:
            scored.append((score - 0.05, q.id, f"Asked: {q.text} I concluded (confidence {q.confidence:.2f}): {q.answer}", False))
    scored.sort(key=lambda item: -item[0])
    return [(nid, note, grounded) for _, nid, note, grounded in scored[:limit]]


def _answer(organism: "CuriosityOrganism", grader, q: ExamQuestion) -> ExamAnswer:
    notes = exam_notes(organism, q.question)
    if not notes:
        return ExamAnswer(q.id, q.question, NOT_IN_NOTES, grade="unanswered", reason="no related notes")
    errors: list[str] = []
    note_lines = "\n".join(f"- [{nid}] {note}" for nid, note, _ in notes)
    user = f"Exam question: {q.question}\n\nYour notes:\n{note_lines}\n\nAnswer from these notes only, and cite the note ids you used."
    data = organism._json(organism._sys("EXAM"), user, P.EXAM_SCHEMA, temperature=0.0, errors=errors, step="exam")
    answer = one_line(_as_str(data.get("answer")), 800)
    cited = {nid for nid in re.findall(r"\b[BQ]\d+\b", " ".join(_as_str(x) for x in _as_list(data.get("notes_used"))) + " " + answer)}
    known = {nid: grounded for nid, _, grounded in notes}
    cited = sorted(nid for nid in cited if nid in known)
    if not answer or NOT_IN_NOTES in answer.lower():
        return ExamAnswer(q.id, q.question, answer or NOT_IN_NOTES, cited, "unanswered", "; ".join(errors))
    grade, reason = _grade(organism, grader, q, answer)
    return ExamAnswer(q.id, q.question, answer, cited, grade, reason, grounded=any(known[nid] for nid in cited))


def _grade(organism: "CuriosityOrganism", grader, q: ExamQuestion, answer: str) -> tuple[str, str]:
    """A blind grade: the grader sees the question, the reference and the answer, nothing else."""
    user = f"Question: {q.question}\n\nReference answer: {q.reference}\n\nAnswer to grade: {answer}"
    organism.body.before_thinking()
    try:
        data = grader.json_chat(P.GRADE, user, P.GRADE_SCHEMA, temperature=0.0, max_tokens=200)
    except Exception as exc:
        organism._trace("grade", user, f"ERROR {exc}")
        return "unanswered", f"the grader did not answer: {one_line(str(exc), 120)}"
    organism._trace("grade", user, data)
    raw = _as_str((data or {}).get("grade")).strip().lower()
    grade = next((g for g in ("correct", "partial", "wrong", "unanswered") if raw.startswith(g[:4])), "wrong")
    if " or " in raw or "|" in raw:
        grade = "wrong"  # the schema hint echoed back is not a grade
    return grade, one_line(_as_str((data or {}).get("reason")), 200)


def _write(folder: Path, result: ExamResult) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", result.label.lower()).strip("-") or "exam"
    stem = f"{result.when[:19].replace(':', '').replace('-', '')}-{slug}"
    data = {
        "label": result.label, "exam": result.exam, "when": result.when, "heartbeat": result.heartbeat,
        **result.summary(), "answers": [asdict(a) | {"points": a.points} for a in result.answers],
    }
    (folder / f"{stem}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# Exam: {result.exam} ({result.label})",
        "",
        f"At heartbeat {result.heartbeat}: score **{result.score:.0%}**, grounded score **{result.grounded_score:.0%}**, "
        f"{result.answered} of {len(result.answers)} answered.",
        "",
        "| id | grade | grounded | answer | notes |",
        "|---|---|---|---|---|",
    ]
    for a in result.answers:
        lines.append(f"| {a.id} | {a.grade} | {'yes' if a.grounded else ''} | {one_line(a.answer, 160).replace('|', '/')} | {', '.join(a.notes_used)} |")
    (folder / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder / f"{stem}.json"
