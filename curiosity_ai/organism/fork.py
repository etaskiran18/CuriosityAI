"""What if it had pursued it? Fork a life at a deferred curiosity and live it both ways (records architecture).

A person cannot be asked to live the same afternoon twice; a program can. The life is copied twice: in one copy
it goes on as it decided (the curiosity waits), in the other it is made to pursue that curiosity, everything
else set aside. Comparing the two says whether deferring was a good decision.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from ..config import AppConfig
from .organism import CuriosityOrganism
from .state import CLOSED_STATUSES


@dataclass
class Branch:
    name: str  # "as decided" or "pursued"
    status: str
    satisfied: bool
    incidental: bool
    looks: int  # times it looked at the curiosity in this branch
    heartbeats: int
    home: str


def latest_deferred(org: CuriosityOrganism) -> str | None:
    """The curiosity it most recently decided to let wait (not one that made room for another)."""
    waiting = [
        q for q in org.state.questions.values()
        if q.status == "deferred" and q.has_record and "took its place" not in q.status_reason
    ]
    return max(waiting, key=lambda q: q.status_changed_at).id if waiting else None


def fork(
    config: AppConfig,
    home: Path,
    out: Path,
    *,
    question_id: str | None = None,
    heartbeats: int = 5,
    make_organism: Callable[[AppConfig], CuriosityOrganism] | None = None,
) -> tuple[str, list[Branch], str]:
    """Live ``heartbeats`` heartbeats both ways from the same moment. Returns (curiosity, branches, verdict)."""
    make = make_organism or (lambda cfg: CuriosityOrganism(cfg))
    if question_id is None:
        probe = make(_at(config, home))
        question_id = latest_deferred(probe)
        if question_id is None:
            raise ValueError("this life has no deferred curiosity to fork at")
    if out.exists():
        shutil.rmtree(out)
    branches: list[Branch] = []
    for name in ("as decided", "pursued"):
        target = out / name.replace(" ", "-")
        shutil.copytree(home, target, ignore=shutil.ignore_patterns("sessions", "llm_trace.jsonl"))
        org = make(_at(config, target))
        if question_id not in org.state.questions:
            raise ValueError(f"{question_id} is not a question of this life")
        start = org.state.heartbeat
        for _ in range(heartbeats):
            if name == "pursued":
                _pursue_only(org, question_id)
            org.heartbeat()
        org.save()
        q = org.state.questions[question_id]
        branches.append(Branch(
            name=name, status=q.status, satisfied=q.status == "satisfied", incidental=q.satisfied_incidentally,
            looks=sum(1 for v in q.visits if v.heartbeat > start), heartbeats=org.state.heartbeat - start, home=str(target),
        ))
    verdict = judge(branches)
    _write(out, question_id, branches, verdict)
    return question_id, branches, verdict


def judge(branches: list[Branch]) -> str:
    waited, pursued = branches
    if waited.satisfied and not pursued.satisfied:
        return "deferring was right: it was answered while it waited, and pursuing it did not answer it"
    if waited.satisfied:
        return "both answered it: deferring cost nothing"
    if pursued.satisfied:
        return f"deferring missed an answer: pursued, it was answered in {pursued.looks} looks"
    return f"neither answered it: deferring saved {pursued.looks} looks"


def _at(config: AppConfig, home: Path) -> AppConfig:
    cfg = config.model_copy(deep=True)
    cfg.organism.home = str(home)
    cfg.organism.curiosity.architecture = "records"
    return cfg


def _pursue_only(org: CuriosityOrganism, question_id: str) -> None:
    """In the pursued branch the curiosity is the only one pursued, until it ends."""
    q = org.state.questions[question_id]
    if q.status in CLOSED_STATUSES:
        return
    for other in org.state.open_questions():
        if other.id != question_id:
            org._set_status(other, "deferred", f"set aside: this copy of the life pursues {question_id}")
    if q.status != "open":
        org._set_status(q, "open", "pursued in this copy of the life, to see what deferring it cost")


def _write(out: Path, question_id: str, branches: list[Branch], verdict: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "fork_result.json").write_text(
        json.dumps({"question": question_id, "verdict": verdict, "branches": [asdict(b) for b in branches]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [f"# What if it had pursued {question_id}?", "", f"**{verdict}.**", "",
             "| branch | ended as | answered | looks at it | heartbeats |", "|---|---|---|---:|---:|"]
    lines += [f"| {b.name} | {b.status} | {'yes' + (' (while waiting)' if b.incidental else '') if b.satisfied else 'no'} | "
              f"{b.looks} | {b.heartbeats} |" for b in branches]
    (out / "fork_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
