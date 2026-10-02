"""The organism's diary: a Markdown record a human can read to watch it think."""
from __future__ import annotations

from pathlib import Path

from ..schema import utc_now_iso
from .state import Episode, MindState
from .textutil import one_line

_STATUS_MARK = {
    "confirmed": "confirmed",
    "contradicted": "CONTRADICTED",
    "not_addressed": "not addressed",
    "unverified": "claimed, but the quote was not found in the source (discarded)",
}


class Diary:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n\n")

    def birth(self, state: MindState) -> None:
        lines = [
            f"# Diary of {state.name}",
            "",
            f"*Born {state.born_at[:19].replace('T', ' ')} UTC.* I begin with these questions:",
            "",
        ]
        lines += [f"- **{q.id}** {q.text}" for q in state.questions.values()]
        self._append("\n".join(lines))

    def heartbeat(self, ep: Episode, state: MindState, ranking: list[tuple[str, float]]) -> None:
        q = state.questions.get(ep.question_id)
        origin = f"born from {ep.trigger}" + (f" in {q.parent_id}" if q and q.parent_id else "")
        d = ep.drive
        lines = [
            f"## Heartbeat {ep.heartbeat} - {ep.created_at[:16].replace('T', ' ')}",
            "",
            f"**I chose {ep.question_id}** ({origin}): {ep.question}",
            "",
            (
                f"*Why this one:* curiosity {d.get('total', 0):.2f} = gap {d.get('gap', 0):.2f} "
                f"(anchoring {d.get('anchoring', 0):.2f}), learning progress {d.get('learning_progress', 0):.2f}, "
                f"surprise {d.get('surprise', 0):.2f}, novelty {d.get('novelty', 0):.2f}, "
                f"importance {d.get('importance', 0):.2f}, boredom {d.get('boredom', 0):.2f}"
            ),
        ]
        if len(ranking) > 1:
            others = ", ".join(f"{qid} {score:.2f}" for qid, score in ranking[1:4])
            lines.append(f"*Other pulls:* {others}")
        lines += ["", f"**Before looking** (confidence {ep.prior_confidence:.2f}): {ep.prior_answer or 'I had no answer yet.'}"]
        if ep.expectations:
            lines.append("")
            lines += [f"- E{i} I expected: {e}" for i, e in enumerate(ep.expectations, start=1)]
        lines += ["", "**What I read:** " + ("; ".join(ep.sources) if ep.sources else "nothing relevant was found.")]
        if ep.checks:
            lines.append("")
            for c in ep.checks:
                mark = _STATUS_MARK.get(c.get("status", ""), c.get("status", ""))
                quote = f' - [{c.get("source")}] "{one_line(c.get("quote", ""), 220)}"' if c.get("quote") and c.get("status") != "unverified" else ""
                lines.append(f"- E{c.get('expectation')}: {mark}{quote}")
        for u in ep.unexpected:
            lines.append(f'- Unexpected: {u.get("finding", "")} - [{u.get("source")}] "{one_line(u.get("quote", ""), 220)}"')
        lines += [
            "",
            f"**Surprise** {ep.prediction_error:.2f} - **informativeness** {ep.informativeness:.2f}"
            + (f" - {ep.rejected_quotes} unverifiable quote(s) discarded" if ep.rejected_quotes else ""),
        ]
        if ep.dialogue:
            lines += ["", "**Inner dialogue**", ""]
            for turn in ep.dialogue:
                lines.append(f"> **{turn['voice']}:** {one_line(turn['text'], 1200)}")
                lines.append(">")
            lines.pop()
        lines += [
            "",
            f"**Now I think** (confidence {ep.prior_confidence:.2f} -> {ep.confidence:.2f}): {ep.answer}",
        ]
        if ep.insight:
            lines.append(f"*Insight:* {ep.insight}")
        for bid in ep.new_belief_ids:
            b = state.beliefs.get(bid)
            if b:
                kind = "interpretation" if b.interpretive else "; ".join(e.citation for e in b.evidence)
                lines.append(f"- New belief **{bid}** ({b.confidence:.2f}, {kind}): {b.statement}")
        for bid in ep.reinforced_belief_ids:
            b = state.beliefs.get(bid)
            if b:
                lines.append(f"- Reinforced **{bid}** -> {b.confidence:.2f}: {b.statement}")
        for bid in ep.doubted_belief_ids:
            b = state.beliefs.get(bid)
            if b:
                lines.append(f"- Now doubting **{bid}** -> {b.confidence:.2f}: {b.statement}")
        for qid in ep.new_question_ids:
            nq = state.questions.get(qid)
            if nq:
                lines.append(f"- New question **{qid}** ({nq.trigger}): {nq.text}")
        for qid in ep.reawakened_question_ids:
            nq = state.questions.get(qid)
            if nq:
                lines.append(f"- Reawakened **{qid}**: {nq.text}")
        if ep.acquisitions:
            lines.append(f"- My books said little here, so I went to the library and brought back: {'; '.join(ep.acquisitions)}")
        if ep.library_misses:
            lines.append(f"- I looked for, but did not find: {'; '.join(ep.library_misses)}")
        if ep.library_owned:
            lines.append(f"- I wished for, but already had: {'; '.join(ep.library_owned)}")
        if ep.status_after != "open":
            lines.append(f"- {ep.question_id} is now **{ep.status_after}**" + (f": {q.status_reason}" if q and q.status_reason else ""))
        if ep.errors:
            lines.append(f"- *Trouble this heartbeat:* {'; '.join(ep.errors)}")
        self._append("\n".join(lines))

    def noticed(self, title: str, question_ids: list[str], state: MindState) -> None:
        lines = [f"## A human shared something: *{title}*", ""]
        for qid in question_ids:
            q = state.questions.get(qid)
            if q:
                lines.append(f"- It made me wonder (**{qid}**): {q.text}")
        if not question_ids:
            lines.append("- I read it, but no new question came to me yet.")
        self._append("\n".join(lines))

    def asked(self, question_id: str, text: str, reused: bool) -> None:
        verb = "asked me something I was already wondering about" if reused else "asked me"
        self._append(f"## A human {verb}\n\n- **{question_id}** {text}")

    def reflection(self, heartbeat: int, vitals: dict, diagnosis: dict, changes: list[str], text: str, understanding: str, focus: str | None, woke: list[str], rested: list[str]) -> None:
        lines = [
            f"## Reflection after heartbeat {heartbeat} - {utc_now_iso()[:16].replace('T', ' ')}",
            "",
            f"**Vital signs:** {', '.join(f'{k} {v}' for k, v in vitals.items())}",
            "",
            f"**Diagnosis: {diagnosis['name']}** ({diagnosis['source']}): {diagnosis['meaning']} {diagnosis['response']}",
        ]
        if changes:
            lines.append(f"*Temperament adjusted:* {'; '.join(changes)}")
        if text:
            lines += ["", text]
        if understanding:
            lines += ["", f"**What I now think curiosity is:** {understanding}"]
        if focus:
            lines.append(f"**I want to pursue next:** {focus}")
        if woke:
            lines.append(f"*After incubation, these questions woke up again:* {', '.join(woke)}")
        if rested:
            lines.append(f"*I let these rest for now:* {', '.join(rested)}")
        self._append("\n".join(lines))

    def note(self, text: str) -> None:
        self._append(text)
