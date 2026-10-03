"""The research map: what a life of inquiry has found so far, in one page.

Written to ``<home>/research_map.md`` at the end of every session. It is built
from the mind and the episode log only (no language model), so it shows exactly
what the organism has, with every claim traceable to a quote:

* its current understanding of the topic;
* open questions, ranked by the pull they exert, with the current answer;
* hypotheses: answers it holds with some confidence, what would refute them,
  the grounded beliefs they rest on, and predictions the texts contradicted;
* surprises: predictions the texts contradicted (the judge agreeing);
* the person's own draft: its claims the organism met, and which other texts
  say something similar or the opposite (the draft is never evidence for itself);
* possible gaps: questions its texts kept silent about, even after looking
  elsewhere. Silence in its library is not proof that nobody has studied
  something, but it is where to look;
* conflicting evidence: beliefs it later came to doubt;
* the reading list: the texts its grounded beliefs rest on, and what it fetched.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from ..schema import utc_now_iso
from ..utils import read_jsonl
from .state import Episode
from .textutil import lexically_related, one_line

if TYPE_CHECKING:
    from .organism import CuriosityOrganism


def write_research_map(organism: "CuriosityOrganism") -> Path:
    path = organism.home / "research_map.md"
    path.write_text(render_research_map(organism), encoding="utf-8")
    return path


def render_research_map(organism: "CuriosityOrganism") -> str:
    st = organism.state
    episodes = [Episode.model_validate(row) for row in read_jsonl(organism.episodes_path)]
    topic = st.topic
    understanding = organism._understanding()
    lines = [
        f"# Research map: {topic.title}",
        "",
        f"*{st.name}, {st.heartbeat} heartbeats old. Written {utc_now_iso()[:16].replace('T', ' ')} UTC from its mind and episodes; "
        "every quote below was found in the source and accepted by the judge.*",
        "",
    ]
    if topic.description:
        lines += [f"**Topic:** {topic.description}", ""]
    if topic.meaning:
        lines += [f"**What it takes the topic to mean:** {topic.meaning}", ""]
    if topic.keywords:
        lines += [f"**Key terms:** {', '.join(topic.keywords)}", ""]
    if topic.until_year:
        lines += [f"**Reading only papers published up to {topic.until_year}.**", ""]
    lines += ["## What it now thinks", "", understanding or "(no theory yet)", ""]

    # Open questions by pull.
    drives = organism.drives()
    lines += ["## Open questions, by the pull they exert", ""]
    if drives:
        lines += ["| question | pull | on topic | confidence | visits | current answer |", "|---|---:|---:|---:|---:|---|"]
        for q, r in drives[:15]:
            lines.append(
                f"| **{q.id}** {_cell(q.text, 110)} | {r.total:.2f} | {r.relevance:.2f} | {q.confidence:.2f} | {len(q.visits)} | "
                f"{_cell(q.answer, 160) or '-'} |"
            )
    else:
        lines.append("- none open")
    lines.append("")

    # Hypotheses: answers that at least one quote the judge accepted supports. (An answer no quote supports
    # is held at 0.5 at most; it stays in the table of open questions above.)
    held = sorted(
        (q for q in st.questions.values() if q.answer and q.visits and q.support and q.confidence >= 0.5),
        key=lambda q: (-len(q.support), -q.confidence),
    )
    lines += ["## Hypotheses: answers the texts support", ""]
    if not held:
        lines += ["- none yet: none of its answers is supported by a quote the judge accepted", ""]
    falsifiers = {e.question_id: e.falsifier for e in episodes if e.falsifier}
    contra_by_q = _contradictions(episodes)
    for q in held[:12]:
        lines.append(f"### {q.id} ({q.status}, confidence {q.confidence:.2f}): {q.text}")
        lines.append("")
        lines.append(f"- **Answer:** {q.answer}")
        if q.id in falsifiers:
            lines.append(f"- **Would be wrong if:** {falsifiers[q.id]}")
        for e in q.support:
            lines.append(f'- **Supported by:** "{one_line(e.quote, 200)}" (*{e.where}*)')
        support = [st.beliefs[bid] for bid in q.related_beliefs if bid in st.beliefs and st.beliefs[bid].evidence]
        for b in support[:3]:
            e = b.evidence[0]
            lines.append(f'- Related belief ({b.id}, {b.status}): {b.statement} - "{one_line(e.quote, 200)}" (*{e.where}*)')
        for line in contra_by_q.get(q.id, [])[:3]:
            lines.append(f"- Against (a prediction the texts contradicted): {line}")
        lines.append("")

    # Surprises: contradicted predictions.
    surprises = [
        (e, e.question_id, e.expectations[c["expectation"] - 1], c)
        for e in episodes
        for c in e.checks
        if c.get("status") == "contradicted" and 1 <= c.get("expectation", 0) <= len(e.expectations)
    ]
    lines += _draft_section(episodes, organism)
    lines += ["## Surprises: what the texts contradicted", ""]
    if surprises:
        for e, qid, expected, c in surprises[-12:]:
            lines.append(f'- heartbeat {e.heartbeat} ({qid}): I expected "{one_line(expected, 160)}"; {_heading(e, c.get("source", ""))} said "{one_line(c.get("quote", ""), 200)}"')
    else:
        lines.append("- none yet: its predictions were confirmed or not addressed")
    lines.append("")

    # Possible gaps: silent texts, even after a library visit.
    lines += ["## Possible gaps: where its texts stayed silent", ""]
    gaps = []
    for q in st.questions.values():
        visits = q.visits
        if len(visits) >= 2 and sum(v.informativeness for v in visits) / len(visits) < 0.34:
            looked = any(e.question_id == q.id and (e.acquisitions or e.library_misses) for e in episodes)
            gaps.append((q, looked))
    if gaps:
        for q, looked in gaps[:12]:
            note = " It also looked elsewhere and found little." if looked else ""
            lines.append(f"- **{q.id}** {q.text} ({len(q.visits)} visits, status {q.status}).{note}")
        lines.append("")
        lines.append("*Silence in its library is not proof that nobody has studied this; it is where a person should look next.*")
    else:
        lines.append("- none yet (a gap needs at least two visits where the texts said little)")
    lines.append("")

    # Conflicting evidence: doubted beliefs.
    doubted = [b for b in st.beliefs.values() if b.status in ("doubted", "retracted")]
    lines += ["## Beliefs it came to doubt", ""]
    lines += [f"- **{b.id}** ({b.status}, {b.confidence:.2f}): {b.statement} - contradicted while investigating {', '.join(b.contradicted_by)}" for b in doubted[:12]] or ["- none"]
    lines.append("")

    # Reading list.
    cited = Counter(e.source_title for b in st.held_beliefs() for e in b.evidence if e.source_title)
    lines += ["## Reading list", "", "Texts its grounded beliefs rest on, most used first:", ""]
    lines += [f"- {title} ({n} belief{'s' if n > 1 else ''})" for title, n in cited.most_common(15)] or ["- none yet"]
    acquired = [row for row in read_jsonl(organism.library_dir / "acquisitions.jsonl")]
    if acquired:
        lines += ["", "What it fetched by itself, and why:", ""]
        for row in acquired[-15:]:
            lines.append(f"- {row.get('title')} ({row.get('kind')}, {row.get('source_url')}): {row.get('reason')}")
    settled = [q for q in st.questions.values() if q.status == "settled"]
    if settled:
        lines += ["", "## Settled questions", ""]
        lines += [f"- **{q.id}** {q.text} -> {q.answer} (confidence {q.confidence:.2f})" for q in settled]
    return "\n".join(lines) + "\n"


def _contradictions(episodes: list[Episode]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for e in episodes:
        for c in e.checks:
            if c.get("status") == "contradicted" and 1 <= c.get("expectation", 0) <= len(e.expectations):
                out.setdefault(e.question_id, []).append(
                    f'expected "{one_line(e.expectations[c["expectation"] - 1], 120)}", but {_heading(e, c.get("source", ""))} '
                    f'said "{one_line(c.get("quote", ""), 160)}" (heartbeat {e.heartbeat})'
                )
    return out


def _heading(e: Episode, label: str) -> str:
    """'S2' -> 'John Dewey, How We Think', from the sources the episode read."""
    for source in e.sources:
        if source.startswith(f"[{label}] "):
            return source[len(label) + 3:]
    return f"[{label}]"


def _cell(text: str, limit: int) -> str:
    return one_line(text or "", limit).replace("|", "/")


def _draft_section(episodes: list[Episode], organism: "CuriosityOrganism") -> list[str]:
    """The person's own claims it met, each with the other texts that say something similar or the opposite.

    Matching is by shared words (with the topic's own words left out), so it points to where to look; the
    agreeing and disagreeing quotes are ones the judge accepted for a prediction.
    """
    claims: dict[str, tuple[int, str, str]] = {}
    evidence: list[tuple[str, str, str, int]] = []  # (quote, status, source, heartbeat)
    for e in episodes:
        sources = {s.split("]", 1)[0].lstrip("["): s.split("]", 1)[-1].strip() for s in e.sources}
        for c in e.checks:
            quote = c.get("quote") or ""
            if not quote:
                continue
            if c.get("status") == "own":
                claims.setdefault(" ".join(quote.lower().split()), (e.heartbeat, e.question_id, quote))
            elif c.get("status") in ("confirmed", "contradicted"):
                evidence.append((quote, c["status"], sources.get(c.get("source", ""), ""), e.heartbeat))
    if not claims:
        return []
    lines = ["## Your draft's claims it met, and what the other texts say", "",
             "*Your draft is never evidence for itself: these are your claims, set against what the other texts said.*", ""]
    for heartbeat, qid, quote in sorted(claims.values())[:15]:
        lines.append(f'- "{one_line(quote, 220)}" (heartbeat {heartbeat}, {qid})')
        related = [ev for ev in evidence if lexically_related(quote, ev[0], organism.topic_words)]
        for text, status, source, hb in related[:3]:
            word = "says something similar" if status == "confirmed" else "says the opposite"
            lines.append(f'  - {source} {word}: "{one_line(text, 200)}" (heartbeat {hb})')
        if not related:
            lines.append("  - no other text it read says this yet")
    return lines + [""]

