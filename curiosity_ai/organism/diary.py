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

_JUDGE_SAID = {"supports": "supports", "contradicts": "contradicts", "neither": "neither"}


def _check_line(c: dict) -> str:
    """One expectation's fate, with what the organism claimed and what the blind judge decided."""
    status, claimed, judge = c.get("status", ""), c.get("claimed", ""), c.get("judge", "")
    quote = f' - [{c.get("source")}] "{one_line(c.get("quote", ""), 220)}"' if c.get("quote") and status != "unverified" else ""
    if judge == "neither" and claimed in ("confirmed", "contradicted"):
        return f"- E{c.get('expectation')}: I said {claimed}, but the judge found the quote beside the point (not counted){quote}"
    if judge == "unjudged" and claimed in ("confirmed", "contradicted"):
        return f"- E{c.get('expectation')}: I said {claimed}, but the judge did not check it, so it is not counted{quote}"
    if judge in ("supports", "contradicts") and claimed in ("confirmed", "contradicted") and claimed != status:
        return f"- E{c.get('expectation')}: I said {claimed}; the judge read it as {_STATUS_MARK.get(status, status)}{quote}"
    if status == "own":
        return f"- E{c.get('expectation')}: your draft says so (your claim, not evidence){quote}"
    mark = _STATUS_MARK.get(status, status)
    if judge in ("supports", "contradicts"):
        mark += " (the judge agrees)"
    elif judge == "unjudged":
        mark += " (the judge could not check it)"
    return f"- E{c.get('expectation')}: {mark}{quote}"


class Diary:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n\n")

    def birth(self, state: MindState) -> None:
        born = f"*Born {state.born_at[:19].replace('T', ' ')} UTC.*"
        if state.topic.meaning:
            born += f" I take my topic to mean: {state.topic.meaning}\n\nI begin with these questions:"
        else:
            born += " I begin with these questions:"
        lines = [f"# Diary of {state.name}", "", born, ""]
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
                f"importance {d.get('importance', 0):.2f}, boredom {d.get('boredom', 0):.2f}, "
                f"on topic {d.get('relevance', 1):.2f}"
                + (", and **new texts arrived for it**" if d.get("news") else "")
            ),
        ]
        if len(ranking) > 1:
            others = ", ".join(f"{qid} {score:.2f}" for qid, score in ranking[1:4])
            lines.append(f"*Other pulls:* {others}")
        lines += ["", f"**Before looking** (confidence {ep.prior_confidence:.2f}): {ep.prior_answer or 'I had no answer yet.'}"]
        if ep.predictions:
            lines.append("")
            for i, pr in enumerate(ep.predictions, start=1):
                who = f"{pr.get('author')}: " if pr.get("author") and pr.get("author", "").lower() not in pr.get("claim", "").lower() else ""
                hedge = " *(hedged: it could never be wrong)*" if pr.get("hedged") else (
                    " *(says only that something has an influence: no text could contradict it)*" if pr.get("weak") else "")
                lines.append(f"- E{i} I predicted (p={pr.get('probability', 0):.2f}): {who}{pr.get('claim', '')}{hedge}")
        elif ep.expectations:
            lines.append("")
            lines += [f"- E{i} I expected: {e}" for i, e in enumerate(ep.expectations, start=1)]
        lines += ["", "**What I read:** " + ("; ".join(ep.sources) if ep.sources else "nothing relevant was found.")]
        if ep.checks:
            lines.append("")
            lines += [_check_line(c) for c in ep.checks]
        for u in ep.unexpected:
            lines.append(f'- Unexpected: {u.get("finding", "")} - [{u.get("source")}] "{one_line(u.get("quote", ""), 220)}"')
        dropped = [j for j in ep.judgments if j.get("kind") == "finding" and j.get("judge") in ("neither", "contradicts")]
        if dropped:
            lines.append(f"- The judge did not accept {len(dropped)} 'unexpected' finding(s): the quote did not say it.")
        brier = f" - **Brier score** {ep.brier:.2f}" if ep.brier is not None else ""
        lines += [
            "",
            f"**Surprise** {ep.prediction_error:.2f} - **informativeness** {ep.informativeness:.2f}{brier}"
            + (f" - {ep.rejected_quotes} unverifiable quote(s) discarded" if ep.rejected_quotes else ""),
        ]
        if ep.dialogue:
            lines += ["", "**Inner dialogue**", ""]
            for turn in ep.dialogue:
                stance = f" ({turn['stance']}s)" if turn.get("stance") else ""
                evidence = " *(quoting the text)*" if turn.get("evidence") == "quote" else ""
                if turn.get("citations") == "unverified":
                    evidence += " *(cites papers it was not shown: unverified, possibly invented)*"
                lines.append(f"> **{turn['voice']}{stance}:**{evidence} {one_line(turn['text'], 1200)}")
                lines.append(">")
            lines.pop()
        lines += [
            "",
            f"**Now I think** (confidence {ep.prior_confidence:.2f} -> {ep.confidence:.2f}): {ep.answer}",
        ]
        if ep.falsifier:
            lines.append(f"*I would be wrong if:* {ep.falsifier}")
        if ep.vague:
            lines.append("*My answer is too vague to be wrong, so I may not grow surer of it.*")
        elif ep.hedged_answer:
            lines.append("*My answer is hedged (may, could), so it could never be wrong: I may grow only a little surer of it.*")
        elif ep.strawman_falsifier:
            lines.append("*It would be wrong only if nothing at all, or one thing alone, were at play; no finding can show that, "
                         "so I may grow only a little surer of it.*")
        if ep.unsupported_answer:
            lines.append("*The quotes I had do not support this answer, so it cannot keep the confidence the old answer had.*")
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
        if ep.paper_doubt_ids:
            ids = ", ".join(f"**{bid}**" for bid in ep.paper_doubt_ids)
            lines.append(f"- I wanted to doubt {ids}, but no text contradicted it: a belief that rests on a text gives way only to a text.")
        for qid in ep.new_question_ids:
            nq = state.questions.get(qid)
            if nq:
                lines.append(f"- New question **{qid}** ({nq.trigger}): {nq.text}")
        for qid in ep.reawakened_question_ids:
            nq = state.questions.get(qid)
            if nq:
                lines.append(f"- Reawakened **{qid}**: {nq.text}")
        for text in ep.set_aside_questions:
            lines.append(f"- Set aside as off my topic: {text}")
        for text in ep.held_back_questions:
            lines.append(f"- Kept for a second look (I will ask it if the gap is still there): {text}")
        for text in ep.unknowable_questions:
            lines.append(f"- Not asked: it is about things nobody has identified yet, which no text can answer: {text}")
        if ep.acquisitions:
            lines.append(f"- My books said little here, so I went to the library and brought back: {'; '.join(ep.acquisitions)}")
        if ep.library_misses:
            lines.append(f"- I looked for, but did not find: {'; '.join(ep.library_misses)}")
        if ep.library_owned:
            lines.append(f"- I wished for, but already had: {'; '.join(ep.library_owned)}")
        if ep.library_rejected:
            lines.append(f"- I found, but did not keep (off my topic): {'; '.join(ep.library_rejected)}")
        if ep.library_busy:
            lines.append(f"- I could not look, the source was busy: {'; '.join(ep.library_busy)}")
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

    def reflection(
        self,
        heartbeat: int,
        vitals: dict,
        diagnosis: dict,
        changes: list[str],
        text: str,
        understanding: str,
        focus: str | None,
        woke: list[str],
        rested: list[str],
        *,
        theory_label: str = "curiosity itself",
        too_vague: str = "",
    ) -> None:
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
            label = "What I now think curiosity is" if theory_label == "curiosity itself" else "What I now think about my topic"
            lines += ["", f"**{label}:** {understanding}"]
        if too_vague:
            lines += ["", f"*I tried to restate my theory, but it was too vague to be wrong, so I kept the old one:* {one_line(too_vague, 400)}"]
        if focus:
            lines.append(f"**I want to pursue next:** {focus}")
        if woke:
            lines.append(f"*After incubation, these questions woke up again:* {', '.join(woke)}")
        if rested:
            lines.append(f"*I let these rest for now:* {', '.join(rested)}")
        self._append("\n".join(lines))

    def note(self, text: str) -> None:
        self._append(text)
