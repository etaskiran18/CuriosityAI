"""Sessions: a bounded stretch of the organism's life, with a report that can be compared.

Every run of live.py that lets the organism think is a session. When the
session ends (normally, at its time limit, or with Ctrl+C) the folder
``<home>/sessions/<id>/`` receives:

* ``report.md``       what happened, in words and tables;
* ``metrics.json``    the numbers, for comparing experiments (scripts/compare_sessions.py);
* ``config.json``     the exact settings it ran with (model, policy, seed, temperament, ...);
* ``judge_check.csv`` up to 20 random claim/quote pairs the judge decided, with an
  empty column for your own verdict (scripts/judge_agreement.py scores it).

The life's ``research_map.md`` is rewritten too.
"""
from __future__ import annotations

import csv
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..schema import utc_now_iso
from ..utils import read_jsonl
from .drive import DIAGNOSES
from .state import Episode, MindState
from .textutil import one_line

if TYPE_CHECKING:
    from .body import Rest
    from .organism import CuriosityOrganism


@dataclass
class SessionReport:
    session_id: str
    directory: Path
    metrics: dict[str, Any]
    markdown: str


def snapshot(state: MindState) -> dict[str, Any]:
    return {
        "heartbeat": state.heartbeat,
        "questions": dict(Counter(q.status for q in state.questions.values())),
        "beliefs": dict(Counter(b.status for b in state.beliefs.values())),
        "temperament": state.temperament.model_dump(),
        "understanding": (
            state.self_model.understanding_of_topic if state.topic.mode == "research" else state.self_model.understanding_of_curiosity
        ),
    }


class Session:
    def __init__(self, organism: "CuriosityOrganism", *, label: str | None = None, minutes: float | None = None):
        self.organism = organism
        self.minutes = minutes
        self.started_at = utc_now_iso()
        self._t0 = time.monotonic()
        slug = re.sub(r"[^a-z0-9]+", "-", (label or "").lower()).strip("-")[:30]
        self.session_id = time.strftime("%Y%m%d-%H%M%S") + (f"-{slug}" if slug else "")
        self.start_heartbeat = organism.state.heartbeat
        self.before = snapshot(organism.state)
        self._rests_before = len(organism.rest_log)
        self._no_thought_before = organism.no_thought
        self.exam_before: Any = None  # ExamResult, when an exam was taken before living
        self.exam_after: Any = None
        self._t1: float | None = None
        self._rests_end: int | None = None

    def stop_clock(self) -> None:
        """Living is over; what follows (an exam, writing the report) is not part of the session."""
        self._t1 = time.monotonic()
        self._rests_end = len(self.organism.rest_log)

    def finish(self) -> SessionReport:
        org = self.organism
        episodes = [
            Episode.model_validate(row)
            for row in read_jsonl(org.episodes_path)
            if int(row.get("heartbeat", 0)) > self.start_heartbeat
        ]
        after = snapshot(org.state)
        vitals = [v for v in org.state.vitals_history if int(v.get("heartbeat", 0)) > self.start_heartbeat]
        meta = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": utc_now_iso(),
            "model": org.config.llm.model,
            "policy": org.oc.policy,
            "random_seed": org.oc.random_seed,
            "web": bool(org.librarian),
            "judge_model": (org.oc.judge.model or org.config.llm.model) if org.judge is not None else None,
            "mode": org.state.topic.mode,
            "topic": org.state.topic.title,
            "time_limit_minutes": self.minutes,
            "home": str(org.home),
        }
        metrics = compute_metrics(
            episodes,
            org.state,
            seconds=(self._t1 or time.monotonic()) - self._t0,
            rests=org.rest_log[self._rests_before:self._rests_end],
            no_thought=org.no_thought - self._no_thought_before,
            before=self.before,
            after=after,
            vitals=vitals,
        )
        metrics.update(exam_metrics(self.exam_before, self.exam_after))
        markdown = render_report(meta, metrics, episodes, org.state, self.before, after)
        directory = org.home / "sessions" / self.session_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.md").write_text(markdown, encoding="utf-8")
        write_judge_check(directory / "judge_check.csv", episodes, seed=self.session_id)
        try:
            from .research_map import write_research_map

            write_research_map(org)
        except Exception as exc:  # the map is a convenience; the session report must still be written
            (directory / "research_map_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        (directory / "metrics.json").write_text(json.dumps({**meta, **metrics}, indent=2, ensure_ascii=False), encoding="utf-8")
        (directory / "config.json").write_text(json.dumps(org.config.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
        return SessionReport(self.session_id, directory, {**meta, **metrics}, markdown)


def compute_metrics(
    episodes: list[Episode],
    state: MindState,
    *,
    seconds: float,
    rests: list["Rest"],
    no_thought: int,
    before: dict[str, Any],
    after: dict[str, Any],
    vitals: list[dict[str, Any]],
) -> dict[str, Any]:
    n = len(episodes)
    verified = sum(sum(1 for c in e.checks if c.get("status") in ("confirmed", "contradicted")) + len(e.unexpected) for e in episodes)
    rejected = sum(e.rejected_quotes for e in episodes)
    surprise = [e.prediction_error for e in episodes]
    half = n // 2
    born = [qid for e in episodes for qid in e.new_question_ids]
    by_question: dict[str, list[Episode]] = {}
    for e in episodes:
        by_question.setdefault(e.question_id, []).append(e)
    revisits = [eps for eps in by_question.values() if len(eps) >= 2]
    acquired = [label for e in episodes for label in e.acquisitions]
    rest_seconds = sum(r.seconds for r in rests)
    predictions = [pr for e in episodes for pr in e.predictions]
    hedged = sum(1 for pr in predictions if pr.get("hedged"))
    statuses = Counter(c.get("status") for e in episodes for c in e.checks)
    briers = [e.brier for e in episodes if e.brier is not None]
    judgments = [j for e in episodes for j in e.judgments]
    stances = Counter(e.stance for e in episodes if e.stance)
    debates = [e for e in episodes if any(t.get("voice") == "Skeptic" for t in e.dialogue)]

    def status_delta(kind: str, status: str) -> int:
        return after[kind].get(status, 0) - before[kind].get(status, 0)

    return {
        "minutes": round(seconds / 60, 1),
        "heartbeats": n,
        "heartbeats_per_hour": round(n / (seconds / 3600), 1) if seconds > 0 else 0.0,
        "failed_heartbeats": no_thought,
        "rest_minutes": round(rest_seconds / 60, 1),
        "cooling_rests": sum(1 for r in rests if r.reason == "cooling"),
        "distinct_questions": len(by_question),
        "questions_born": len(born),
        "questions_born_by_trigger": dict(Counter(state.questions[q].trigger for q in born if q in state.questions)),
        "questions_settled": max(0, status_delta("questions", "settled")),
        "questions_dormant": max(0, status_delta("questions", "dormant")),
        "questions_unanswerable": max(0, status_delta("questions", "unanswerable")),
        "beliefs_new": sum(len(e.new_belief_ids) for e in episodes),
        "beliefs_grounded": sum(e.grounded_new_beliefs for e in episodes),
        "beliefs_interpretive": sum(len(e.new_belief_ids) - e.grounded_new_beliefs for e in episodes),
        "beliefs_doubted": sum(len(e.doubted_belief_ids) for e in episodes),
        "beliefs_reinforced": sum(len(e.reinforced_belief_ids) for e in episodes),
        "quotes_verified": verified,
        "quotes_rejected": rejected,
        "fabrication_rate": round(rejected / (verified + rejected), 3) if verified + rejected else 0.0,
        "mean_surprise": _mean(surprise),
        "surprise_first_half": _mean(surprise[:half]),
        "surprise_second_half": _mean(surprise[half:]),
        "mean_informativeness": _mean([e.informativeness for e in episodes]),
        "mean_confidence_change": _mean([abs(e.confidence - e.prior_confidence) for e in episodes]),
        "revisited_questions": len(revisits),
        "surprise_change_on_revisits": _mean([eps[-1].prediction_error - eps[0].prediction_error for eps in revisits]),
        "predictions": len(predictions),
        "predictions_hedged": hedged,
        "hedged_rate": round(hedged / len(predictions), 3) if predictions else 0.0,
        "predictions_confirmed": statuses.get("confirmed", 0),
        "predictions_contradicted": statuses.get("contradicted", 0),
        # None when no prediction was addressed: 0.0 would read as perfect predictions.
        "mean_brier": _mean(briers) if briers else None,
        "brier_first_half": _mean(briers[: len(briers) // 2]) if len(briers) >= 2 else None,
        "brier_second_half": _mean(briers[len(briers) // 2:]) if len(briers) >= 2 else None,
        "judge_pairs": len(judgments),
        "judge_agreement": round(sum(1 for j in judgments if agrees(j)) / len(judgments), 3) if judgments else None,
        "judge_rejected": sum(1 for j in judgments if j.get("judge") == "neither"),
        "stances": dict(stances),
        "concession_rate": round(stances.get("concede", 0) / sum(stances.values()), 3) if stances else 0.0,
        "skeptic_quoted_rate": round(
            sum(1 for e in debates if any(t.get("evidence") == "quote" for t in e.dialogue)) / len(debates), 3
        ) if debates else 0.0,
        "vague_answers": sum(1 for e in episodes if e.vague),
        "mean_on_topic": _mean([e.relevance for e in episodes]),
        "questions_set_aside": sum(len(e.set_aside_questions) for e in episodes),
        "questions_held_back": sum(len(e.held_back_questions) for e in episodes),
        "dialogue_unverified_citations": sum(1 for e in episodes for t in e.dialogue if t.get("citations") == "unverified"),
        "library_visits": sum(1 for e in episodes if e.acquisitions or e.library_misses or e.library_owned or e.library_busy or e.library_rejected),
        "acquisitions": len(acquired),
        "acquisitions_by_kind": dict(Counter(label.split(":")[0] for label in acquired)),
        "library_busy": sum(len(e.library_busy) for e in episodes),
        "library_rejected": sum(len(e.library_rejected) for e in episodes),
        "diagnoses": dict(Counter(v.get("diagnosis", "") for v in vitals)),
    }


def agrees(judgment: dict[str, Any]) -> bool:
    """Did the organism's own claim match the judge? (A belief or a finding claims support.)"""
    claimed = {"confirmed": "supports", "contradicted": "contradicts"}.get(judgment.get("agent", ""), "supports")
    return judgment.get("judge") == claimed


def exam_metrics(before: Any, after: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if before is not None:
        out.update({"exam_before": before.score, "exam_grounded_before": before.grounded_score})
    if after is not None:
        out.update({"exam_after": after.score, "exam_grounded_after": after.grounded_score, "exam_questions": len(after.answers)})
    if before is not None and after is not None:
        out["exam_gain"] = round(after.score - before.score, 3)
        out["exam_grounded_gain"] = round(after.grounded_score - before.grounded_score, 3)
    return out


def write_judge_check(path: Path, episodes: list[Episode], *, seed: str, n: int = 20) -> None:
    """A random sample of the judge's decisions, for a person to check by hand."""
    rows = [(e.heartbeat, j) for e in episodes for j in e.judgments]
    if not rows:
        return
    sample = sorted(random.Random(seed).sample(rows, min(n, len(rows))), key=lambda row: row[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["heartbeat", "kind", "claim", "source", "quote", "organism_said", "judge_said", "judge_reason", "your_verdict"])
        for hb, j in sample:
            writer.writerow([hb, j.get("kind"), j.get("claim"), j.get("source"), j.get("quote"), j.get("agent"), j.get("judge"), j.get("reason"), ""])


def render_report(meta: dict[str, Any], m: dict[str, Any], episodes: list[Episode], state: MindState, before: dict[str, Any], after: dict[str, Any]) -> str:
    limit = f", time limit {meta['time_limit_minutes']:g} min" if meta.get("time_limit_minutes") else ""
    lines = [
        f"# Session {meta['session_id']}",
        "",
        f"Model `{meta['model']}`, policy **{meta['policy']}**, web library {'on' if meta['web'] else 'off'}, "
        f"seed {meta['random_seed']}{limit}. Life: `{meta['home']}`.",
        "",
        "## Summary",
        "",
        f"- **{m['heartbeats']} heartbeats in {m['minutes']:g} minutes** ({m['heartbeats_per_hour']:g} per hour); "
        f"rested {m['rest_minutes']:g} min ({m['cooling_rests']} cooling rests); {m['failed_heartbeats']} heartbeats where the model did not answer.",
        f"- **Questions:** visited {m['distinct_questions']}; new questions born: {m['questions_born']} "
        f"({_counts(m['questions_born_by_trigger'])}); settled {m['questions_settled']}, dormant {m['questions_dormant']}, "
        f"judged unanswerable {m['questions_unanswerable']}.",
        f"- **Beliefs:** new {m['beliefs_new']} ({m['beliefs_grounded']} grounded in verified quotes, "
        f"{m['beliefs_interpretive']} interpretations); earlier beliefs doubted {m['beliefs_doubted']}; reinforced {m['beliefs_reinforced']}.",
        f"- **Honesty:** {m['quotes_verified']} quotes verified in the sources, {m['quotes_rejected']} invented quotes caught "
        f"(fabrication rate {m['fabrication_rate']:.0%}).",
        f"- **Surprise:** mean {m['mean_surprise']:.2f} (first half {m['surprise_first_half']:.2f}, second half "
        f"{m['surprise_second_half']:.2f}); the texts addressed {m['mean_informativeness']:.0%} of its expectations on average.",
        f"- **Learning:** questions revisited: {m['revisited_questions']}; on them, surprise changed by "
        f"{m['surprise_change_on_revisits']:+.2f} from first to last visit (negative means it predicts the texts better).",
        f"- **Predictions:** {m['predictions']} made, {m['predictions_hedged']} of them hedged ({m['hedged_rate']:.0%}); "
        f"the texts confirmed {m['predictions_confirmed']} and contradicted {m['predictions_contradicted']}; {_brier_text(m)}.",
        _judge_line(m),
        f"- **Debate:** Wonder {_counts({_STANCE_WORD.get(k, k): v for k, v in m['stances'].items()})}; the skeptic quoted a passage in "
        f"{m['skeptic_quoted_rate']:.0%} of debates; {m['vague_answers']} answers were too vague to be wrong; "
        f"{m['dialogue_unverified_citations']} turns cited papers it was not shown (unverified).",
        f"- **Staying on topic and in depth:** mean closeness of the questions it worked on {m['mean_on_topic']:.2f}; "
        f"{m['questions_set_aside']} proposed questions set aside as off topic, {m['questions_held_back']} kept for a second look.",
        f"- **Library:** {m['library_visits']} visits, {m['acquisitions']} texts acquired ({_counts(m['acquisitions_by_kind'])}); "
        f"{m['library_rejected']} found but off topic; {m['library_busy']} searches impossible because a source was busy.",
        f"- **Self-regulation:** {_counts({DIAGNOSES.get(k, {}).get('name', k): v for k, v in m['diagnoses'].items()}) or 'no reflection in this session'}.",
    ]
    if "exam_after" in m or "exam_before" in m:
        took = f"{m['exam_before']:.0%} (grounded {m['exam_grounded_before']:.0%})" if "exam_before" in m else "not taken"
        then = f"{m['exam_after']:.0%} (grounded {m['exam_grounded_after']:.0%})" if "exam_after" in m else "not taken"
        lines.append(f"- **Exam:** before {took}, after {then}. Grounded answers cite a belief backed by a verified quote.")
    if before["understanding"] != after["understanding"]:
        heading = "What it thinks about its topic" if state.topic.mode == "research" else "Its theory of curiosity"
        lines += ["", f"## {heading}", "", f"- Before: {before['understanding']}", f"- After: {after['understanding']}"]
    grounded = [
        (bid, state.beliefs[bid]) for e in episodes for bid in e.new_belief_ids
        if bid in state.beliefs and state.beliefs[bid].evidence
    ]
    if grounded:
        lines += ["", "## New beliefs grounded in the texts", ""]
        lines += [f"- **{bid}** {b.statement} - {b.evidence[0].citation}" for bid, b in grounded[:10]]
    doubted = [(e.heartbeat, bid) for e in episodes for bid in e.doubted_belief_ids if bid in state.beliefs]
    if doubted:
        lines += ["", "## Beliefs it came to doubt", ""]
        lines += [f"- heartbeat {hb}: **{bid}** {state.beliefs[bid].statement}" for hb, bid in doubted[:10]]
    born = [(e.heartbeat, qid) for e in episodes for qid in e.new_question_ids if qid in state.questions]
    if born:
        lines += ["", "## Questions it gave birth to", ""]
        lines += [f"- heartbeat {hb} ({state.questions[qid].trigger}): **{qid}** {state.questions[qid].text}" for hb, qid in born[:12]]
    reads = [e for e in episodes if e.acquisitions or e.library_misses or e.library_owned or e.library_busy or e.library_rejected]
    if reads:
        lines += ["", "## Library visits", ""]
        for e in reads:
            line = f"- heartbeat {e.heartbeat}, for {e.question_id}: brought back {'; '.join(e.acquisitions) or 'nothing new'}"
            if e.library_owned:
                line += f"; already had: {'; '.join(e.library_owned)}"
            if e.library_misses:
                line += f"; not found: {'; '.join(e.library_misses)}"
            if e.library_rejected:
                line += f"; found but off topic: {'; '.join(e.library_rejected)}"
            if e.library_busy:
                line += f"; source busy: {'; '.join(e.library_busy)}"
            lines.append(line)
    lines += [
        "",
        "## Heartbeat by heartbeat",
        "",
        "| hb | question | surprise | addressed | confidence | beliefs | questions | library |",
        "|---:|---|---:|---:|---|---:|---:|---|",
    ]
    for e in episodes:
        lines.append(
            f"| {e.heartbeat} | {e.question_id}: {one_line(e.question, 70).replace('|', '/')} | {e.prediction_error:.2f} | "
            f"{e.informativeness:.0%} | {e.prior_confidence:.2f} -> {e.confidence:.2f} | {len(e.new_belief_ids)} | "
            f"{len(e.new_question_ids)} | {len(e.acquisitions)} |"
        )
    lines += [
        "",
        "Compare sessions: `python scripts/compare_sessions.py <session folder> <session folder> ...`",
        "",
    ]
    return "\n".join(lines)


_STANCE_WORD = {"defend": "defended", "revise": "revised", "concede": "conceded"}


def _brier_text(m: dict[str, Any]) -> str:
    if m.get("mean_brier") is None:
        return "no prediction was confirmed or contradicted, so there is no Brier score yet"
    halves = ""
    if m.get("brier_first_half") is not None:
        halves = f" (first half {m['brier_first_half']:.2f}, second half {m['brier_second_half']:.2f})"
    return f"mean Brier score {m['mean_brier']:.2f}{halves}; 0 is perfect, 0.25 is a coin flip"


def _judge_line(m: dict[str, Any]) -> str:
    if not m["judge_pairs"]:
        return "- **The judge:** checked nothing in this session (off, or no quotes to check)."
    return (
        f"- **The judge:** checked {m['judge_pairs']} claim/quote pairs and agreed with the organism on "
        f"{m['judge_agreement']:.0%}; {m['judge_rejected']} quotes were beside the point and did not count. "
        "Check a sample by hand: judge_check.csv."
    )


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])) or "none"
