"""Sessions: a bounded stretch of the organism's life, with a report that can be compared.

Every run of live.py that lets the organism think is a session. When the
session ends (normally, at its time limit, or with Ctrl+C) the folder
``<home>/sessions/<id>/`` receives:

* ``report.md``    what happened, in words and tables;
* ``metrics.json`` the numbers, for comparing experiments (scripts/compare_sessions.py);
* ``config.json``  the exact settings it ran with (model, policy, seed, temperament, ...).
"""
from __future__ import annotations

import json
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
        "understanding": state.self_model.understanding_of_curiosity,
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
            "time_limit_minutes": self.minutes,
            "home": str(org.home),
        }
        metrics = compute_metrics(
            episodes,
            org.state,
            seconds=time.monotonic() - self._t0,
            rests=org.rest_log[self._rests_before:],
            no_thought=org.no_thought - self._no_thought_before,
            before=self.before,
            after=after,
            vitals=vitals,
        )
        markdown = render_report(meta, metrics, episodes, org.state, self.before, after)
        directory = org.home / "sessions" / self.session_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.md").write_text(markdown, encoding="utf-8")
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
        "library_visits": sum(1 for e in episodes if e.acquisitions or e.library_misses or e.library_owned),
        "acquisitions": len(acquired),
        "acquisitions_by_kind": dict(Counter(label.split(":")[0] for label in acquired)),
        "diagnoses": dict(Counter(v.get("diagnosis", "") for v in vitals)),
    }


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
        f"- **Library:** {m['library_visits']} visits, {m['acquisitions']} texts acquired ({_counts(m['acquisitions_by_kind'])}).",
        f"- **Self-regulation:** {_counts({DIAGNOSES.get(k, {}).get('name', k): v for k, v in m['diagnoses'].items()}) or 'no reflection in this session'}.",
    ]
    if before["understanding"] != after["understanding"]:
        lines += ["", "## Its theory of curiosity", "", f"- Before: {before['understanding']}", f"- After: {after['understanding']}"]
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
    reads = [e for e in episodes if e.acquisitions or e.library_misses or e.library_owned]
    if reads:
        lines += ["", "## Library visits", ""]
        for e in reads:
            line = f"- heartbeat {e.heartbeat}, for {e.question_id}: brought back {'; '.join(e.acquisitions) or 'nothing new'}"
            if e.library_owned:
                line += f"; already had: {'; '.join(e.library_owned)}"
            if e.library_misses:
                line += f"; not found: {'; '.join(e.library_misses)}"
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


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])) or "none"
