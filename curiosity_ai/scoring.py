from __future__ import annotations

import re
from .config import AppConfig
from .schema import QuestionCandidate, CuriosityScore, Passage, WebResult
from .utils import jaccard, normalize_score


TENSION_TERMS = {
    "contradiction", "paradox", "tension", "conflict", "ambiguity", "uncertainty",
    "assumption", "limits", "failure", "problem", "doubt", "skepticism", "ignorance",
}
DEPTH_TERMS = {
    "why", "how", "under what conditions", "what follows", "if", "because", "ground",
    "foundation", "mechanism", "relation", "difference", "criterion",
}


class CuriosityScorer:
    def __init__(self, config: AppConfig):
        self.config = config

    def score(
        self,
        question: QuestionCandidate,
        recent_memory_texts: list[str],
        passages: list[Passage],
        web_results: list[WebResult],
    ) -> CuriosityScore:
        qtext = question.question + " " + question.philosophical_tension + " " + " ".join(question.assumptions)
        redundancy = max([jaccard(qtext, m) for m in recent_memory_texts] or [0.0])
        novelty = 1.0 - redundancy
        depth = self._depth_score(question)
        tension = self._tension_score(question)
        researchability = self._researchability_score(question, passages, web_results)
        generativity = self._generativity_score(question)
        primary_count = len([p for p in passages if p.source.source_type == "local"]) + len(web_results)
        memory_count = len([p for p in passages if p.source.source_type == "memory"])
        if primary_count >= self.config.depth.min_primary_sources:
            citation_grounding = 1.0
        elif primary_count > 0:
            citation_grounding = 0.55
        elif memory_count > 0:
            citation_grounding = 0.25
        else:
            citation_grounding = 0.05

        w = self.config.scoring
        raw = (
            w.novelty_weight * novelty
            + w.depth_weight * depth
            + w.epistemic_tension_weight * tension
            + w.researchability_weight * researchability
            + w.generativity_weight * generativity
            + w.citation_grounding_weight * citation_grounding
            - w.redundancy_penalty_weight * redundancy
        )
        total = normalize_score(raw)
        return CuriosityScore(
            novelty=normalize_score(novelty),
            depth=normalize_score(depth),
            epistemic_tension=normalize_score(tension),
            researchability=normalize_score(researchability),
            generativity=normalize_score(generativity),
            citation_grounding=normalize_score(citation_grounding),
            redundancy=normalize_score(redundancy),
            total=total,
            explanation=(
                f"Novelty={novelty:.2f}, depth={depth:.2f}, tension={tension:.2f}, "
                f"researchability={researchability:.2f}, generativity={generativity:.2f}, "
                f"redundancy={redundancy:.2f}."
            ),
        )

    def _depth_score(self, q: QuestionCandidate) -> float:
        text = (q.question + " " + q.why_interesting).lower()
        term_hits = sum(1 for t in DEPTH_TERMS if t in text)
        assumption_bonus = min(0.4, 0.1 * len(q.assumptions))
        length_bonus = min(0.25, len(q.question.split()) / 80)
        return normalize_score(0.2 + 0.1 * term_hits + assumption_bonus + length_bonus)

    def _tension_score(self, q: QuestionCandidate) -> float:
        text = (q.question + " " + q.philosophical_tension + " " + q.why_interesting).lower()
        hits = sum(1 for t in TENSION_TERMS if t in text)
        explicit = 0.35 if q.philosophical_tension.strip() else 0.0
        return normalize_score(0.15 + explicit + 0.12 * hits)

    def _researchability_score(self, q: QuestionCandidate, passages: list[Passage], web_results: list[WebResult]) -> float:
        primary_sources = len([p for p in passages if p.source.source_type == "local"]) + len(web_results)
        memory_sources = len([p for p in passages if p.source.source_type == "memory"])
        source_score = min(0.75, primary_sources * 0.16 + memory_sources * 0.04)
        likely = min(0.25, len(q.likely_sources) * 0.08)
        return normalize_score(0.15 + source_score + likely)

    def _generativity_score(self, q: QuestionCandidate) -> float:
        text = q.question.lower()
        bridge_bonus = 0.2 if re.search(r"\bbetween\b|\brelation\b|\bconnect", text) else 0.0
        what_if_bonus = 0.2 if "what if" in text or "under what" in text else 0.0
        tags_bonus = min(0.2, len(q.tags) * 0.04)
        return normalize_score(0.35 + bridge_bonus + what_if_bonus + tags_bonus)
