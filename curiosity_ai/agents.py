from __future__ import annotations

from typing import Any

from .config import AppConfig
from .llm import OllamaClient
from .schema import (
    _as_list,
    QuestionCandidate,
    Critique,
    Synthesis,
    Evaluation,
    PlanningDecision,
    CuriosityScore,
    SourceGateReport,
    TextualAnalysis,
    ArgumentReconstruction,
    ArgumentCritique,
    PhilosophicalDepthJudgment,
    RadicalCandidate,
    StrongPosition,
    BalanceStressTest,
    PhilosophicalMove,
    CuriosityTheoryState,
)
from .prompts import (
    READER_SYSTEM,
    EXPLORER_SYSTEM,
    RADICALIZER_SYSTEM,
    CRITIC_SYSTEM,
    RESEARCH_SYSTEM,
    TEXTUAL_ANALYST_SYSTEM,
    ARGUMENT_RECONSTRUCTOR_SYSTEM,
    POSITION_ENFORCER_SYSTEM,
    BALANCE_STRESS_TESTER_SYSTEM,
    SYNTHESIZER_SYSTEM,
    ARGUMENT_CRITIC_SYSTEM,
    PHILOSOPHICAL_DEPTH_JUDGE_SYSTEM,
    PHILOSOPHICAL_MOVE_DETECTOR_SYSTEM,
    CURIOSITY_THEORY_TRACKER_SYSTEM,
    EVALUATOR_SYSTEM,
    PLANNER_SYSTEM,
    PUBLISHER_SYSTEM,
)
from .grounding import format_passages, format_web_results


class ReaderAgent:
    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def distill(self, topic: str, passages_text: str, web_text: str, source_gate: SourceGateReport) -> str:
        user = f"""
Topic:
{topic}

Source gate:
{source_gate.model_dump()}

Retrieved local/memory passages:
{passages_text}

Retrieved web sources:
{web_text}

Distill the grounding into 5-8 source-aware conceptual notes.
Rules:
- Preserve citation labels exactly.
- Do not invent sources or philosopher claims not present above.
- If grounding is limited, state that explicitly.
- Separate primary-source evidence from system memory.
- Identify at least one possible conflict, paradox, or gap in the sources.
""".strip()
        return self.llm.chat(READER_SYSTEM, user, temperature=0.16)


class ExplorerAgent:
    def __init__(self, config: AppConfig, llm: OllamaClient):
        self.config = config
        self.llm = llm

    def generate_questions(self, topic: str, grounding_notes: str, recent_memory: str, source_gate: SourceGateReport, theory_state: CuriosityTheoryState | None = None) -> list[QuestionCandidate]:
        n = self.config.loop.candidate_questions_per_iteration
        schema = """
{
  "questions": [
    {
      "question": "string",
      "assumptions": ["string"],
      "why_interesting": "string",
      "philosophical_tension": "string",
      "likely_sources": ["citation label or source name"],
      "tags": ["string"]
    }
  ]
}
""".strip()
        user = f"""
Current topic:
{topic}

Current curiosity theory state:
{theory_state.model_dump() if theory_state else 'No curiosity theory state yet.'}

Source gate:
{source_gate.model_dump()}

Grounding notes:
{grounding_notes}

Recent memory summary:
{recent_memory}

Generate {n} candidate philosophical research questions.
Rules:
- Do not ask generic textbook questions.
- Each question must create a possible next inquiry path.
- Prefer questions that expose hidden assumptions, textual tensions, anachronism risks, or conceptual conflicts.
- Use source labels in likely_sources when relevant.
- If grounding is limited, propose questions that ask for better textual grounding rather than pretending evidence exists.
- At least half of the questions must explicitly explain how the question deepens curiosity, inquiry, doubt, wonder, learning, or knowledge-seeking.
""".strip()
        data = self.llm.json_chat(EXPLORER_SYSTEM, user, schema, temperature=0.42)
        return [QuestionCandidate.model_validate(q) for q in _as_list(data.get("questions", []))]


class RadicalQuestionAgent:
    def __init__(self, config: AppConfig, llm: OllamaClient):
        self.config = config
        self.llm = llm

    def generate_radicals(self, topic: str, grounding_notes: str, questions: list[QuestionCandidate], source_gate: SourceGateReport, theory_state: CuriosityTheoryState | None = None) -> list[RadicalCandidate]:
        n = self.config.aggressive_mode.radical_questions
        schema = """
{
  "candidates": [
    {
      "radical_question": "string",
      "radical_thesis": "string",
      "why_dangerous": "string",
      "required_sources": ["citation labels or source names"],
      "possible_objection": "string",
      "tags": ["radical", "string"]
    }
  ]
}
""".strip()
        user = f"""
Current topic:
{topic}

Curiosity theory state:
{theory_state.model_dump() if theory_state else 'No theory state yet.'}

Source gate:
{source_gate.model_dump()}

Grounding notes:
{grounding_notes}

Ordinary candidate questions:
{[q.model_dump() for q in questions]}

Generate {n} radical but defensible candidates.
Rules:
- Each candidate must be more provocative than the ordinary questions.
- Do not invent source support; cite only available labels or name the needed source.
- Each candidate must force a real conflict: paradox, contradiction, dilemma, reversal, or hidden assumption.
- At least one candidate must ask whether curiosity is epistemically dangerous.
- At least one candidate must ask whether certainty is necessary for curiosity rather than its enemy.
""".strip()
        data = self.llm.json_chat(RADICALIZER_SYSTEM, user, schema, temperature=0.58)
        return [RadicalCandidate.model_validate(x) for x in _as_list(data.get("candidates", []))]

    @staticmethod
    def as_questions(radicals: list[RadicalCandidate]) -> list[QuestionCandidate]:
        out: list[QuestionCandidate] = []
        for r in radicals:
            question = r.radical_question or r.radical_thesis
            if not question:
                continue
            out.append(QuestionCandidate(
                question=question,
                assumptions=[r.radical_thesis] if r.radical_thesis else [],
                why_interesting=r.why_dangerous,
                philosophical_tension=r.possible_objection,
                likely_sources=r.required_sources,
                tags=["radical", *r.tags],
            ))
        return out


class CriticAgent:
    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def critique(self, topic: str, questions: list[QuestionCandidate], grounding_notes: str, source_gate: SourceGateReport) -> Critique:
        q_dump = [q.model_dump() for q in questions]
        schema = """
{
  "accepted_questions": [
    {
      "question": "string",
      "assumptions": ["string"],
      "why_interesting": "string",
      "philosophical_tension": "string",
      "likely_sources": ["string"],
      "tags": ["string"]
    }
  ],
  "rejected_questions": [{"question": "string", "reason": "string"}],
  "main_objections": ["string"],
  "strongest_question": "string",
  "refinement": "string"
}
""".strip()
        user = f"""
Current topic:
{topic}

Source gate:
{source_gate.model_dump()}

Grounding notes:
{grounding_notes}

Candidate questions as JSON:
{q_dump}

Critique these questions. Accept only the strongest non-trivial ones.
Reject vague, repetitive, weakly grounded, fake-deep, merely technical, historically careless, or anachronistic items.
Return at least 3 main_objections unless the questions are genuinely excellent.
Make the strongest_question a philosophically meaningful research question that can be supported by the retrieved sources.
Important: do not reject a radical question merely because it is risky; reject it only if it is indefensible or ungrounded.
""".strip()
        data = self.llm.json_chat(CRITIC_SYSTEM, user, schema, temperature=0.16)
        return Critique.model_validate(data)


class ResearchAgent:
    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def connect(self, question: str, local_passages: str, web_results: str, recent_memory: str, source_gate: SourceGateReport) -> str:
        user = f"""
Question:
{question}

Source gate:
{source_gate.model_dump()}

Local/memory passages:
{local_passages}

Web results:
{web_results}

Recent memory:
{recent_memory}

Create a research brief:
1. What the sources support.
2. What remains speculative.
3. Which source passages need close reading.
4. Which tensions should drive the synthesis.
5. Which bold thesis could be defended without pretending it is directly stated by the sources.
Use citation labels exactly.
If the evidence is memory-only or limited, say this directly.
""".strip()
        return self.llm.chat(RESEARCH_SYSTEM, user, temperature=0.20)


class TextualAnalystAgent:
    def __init__(self, config: AppConfig, llm: OllamaClient):
        self.config = config
        self.llm = llm

    def analyze(self, question: str, passages_text: str, source_gate: SourceGateReport) -> list[TextualAnalysis]:
        schema = """
{
  "analyses": [
    {
      "source_label": "exact citation label",
      "source_role": "primary|memory|web|limited",
      "literal_claim": "what the passage literally says",
      "key_concepts": ["string"],
      "interpretive_analysis": "careful interpretation",
      "relevance_to_question": "how this passage bears on the question",
      "anachronism_risk": "risk of projecting modern concepts",
      "limitations": "what this passage does not establish"
    }
  ]
}
""".strip()
        user = f"""
Question:
{question}

Source gate:
{source_gate.model_dump()}

Passages to analyze:
{passages_text}

Analyze at least {self.config.depth.min_textual_analyses} passages if available.
Close-reading rules:
- Do not summarize all philosophy generally.
- Use only provided passages.
- Mention the exact source_label for each analysis.
- Explicitly identify anachronism risk.
- Identify whether the passage gives direct evidence or only indirect interpretive support.
""".strip()
        data = self.llm.json_chat(TEXTUAL_ANALYST_SYSTEM, user, schema, temperature=0.16)
        return [TextualAnalysis.model_validate(x) for x in _as_list(data.get("analyses", []))]


class ArgumentReconstructorAgent:
    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def reconstruct(self, question: str, research_brief: str, analyses: list[TextualAnalysis]) -> ArgumentReconstruction:
        schema = """
{
  "central_claim": "string",
  "premises": ["P1 with citation if source-dependent", "P2 with citation if source-dependent"],
  "conclusion": "string",
  "hidden_assumptions": ["string"],
  "inferential_gaps": ["string"],
  "strongest_objection": "string",
  "possible_reply": "string"
}
""".strip()
        user = f"""
Question:
{question}

Research brief:
{research_brief}

Textual analyses:
{[a.model_dump() for a in analyses]}

Reconstruct the strongest possible philosophical argument.
Rules:
- Use numbered premises.
- Mark weak links as inferential gaps.
- Do not hide speculative bridges.
- Every source-dependent premise should include a citation label.
- The central claim should be brave enough to be disputed, not a neutral summary.
""".strip()
        data = self.llm.json_chat(ARGUMENT_RECONSTRUCTOR_SYSTEM, user, schema, temperature=0.20)
        return ArgumentReconstruction.model_validate(data)


class PositionEnforcerAgent:
    def __init__(self, config: AppConfig, llm: OllamaClient):
        self.config = config
        self.llm = llm

    def enforce(self, topic: str, question: str, research_brief: str, analyses: list[TextualAnalysis], reconstruction: ArgumentReconstruction, radicals: list[RadicalCandidate]) -> StrongPosition:
        schema = """
{
  "position_type": "virtue|method|danger|conditional|revisionary",
  "thesis": "clear thesis sentence",
  "decisive_condition": "if conditional, state the decisive condition",
  "defended_claim": "what the report must defend",
  "concession": "strongest concession allowed without weakening the thesis",
  "source_labels": ["exact source labels"],
  "forbidden_hedges_to_remove": ["might", "could", "may"],
  "reasoning_strategy": "how to argue the position"
}
""".strip()
        user = f"""
Topic:
{topic}

Question:
{question}

Research brief:
{research_brief}

Textual analyses:
{[a.model_dump() for a in analyses]}

Argument reconstruction:
{reconstruction.model_dump()}

Radical candidates:
{[r.model_dump() for r in radicals]}

Force a defensible position. Do not return a neutral summary.
Choose whether curiosity is mainly a virtue, method, danger, conditional transformation, or revisionary concept.
The thesis must be strong enough that a critic can attack it.
""".strip()
        data = self.llm.json_chat(POSITION_ENFORCER_SYSTEM, user, schema, temperature=0.30)
        return StrongPosition.model_validate(data)


class BalanceStressTesterAgent:
    def __init__(self, config: AppConfig, llm: OllamaClient):
        self.config = config
        self.llm = llm

    def stress_test(self, question: str, position: StrongPosition | None, reconstruction: ArgumentReconstruction, analyses: list[TextualAnalysis]) -> BalanceStressTest:
        schema = """
{
  "balance_claim_detected": true,
  "what_is_balanced": "string",
  "who_or_what_decides_balance": "string",
  "criterion_for_virtue": "string",
  "criterion_for_danger": "string",
  "failure_modes": ["string"],
  "hard_question_for_synthesis": "string",
  "verdict": "pass|revise|block"
}
""".strip()
        user = f"""
Question:
{question}

Strong position:
{position.model_dump() if position else 'No strong position generated.'}

Argument reconstruction:
{reconstruction.model_dump()}

Textual analyses:
{[a.model_dump() for a in analyses]}

Stress-test any claim that uses balance, tempered curiosity, guided curiosity, moderation, or disciplined inquiry.
If no such claim is present, set balance_claim_detected=false but still identify whether a hidden balance claim is implied.
""".strip()
        data = self.llm.json_chat(BALANCE_STRESS_TESTER_SYSTEM, user, schema, temperature=0.12)
        return BalanceStressTest.model_validate(data)


class SynthesizerAgent:
    def __init__(self, config: AppConfig, llm: OllamaClient):
        self.config = config
        self.llm = llm

    def synthesize(
        self,
        topic: str,
        strongest_question: str,
        grounding_notes: str,
        research_brief: str,
        textual_analyses: list[TextualAnalysis],
        reconstruction: ArgumentReconstruction,
        score: CuriosityScore,
        source_gate: SourceGateReport,
        position: StrongPosition | None = None,
        theory_state: CuriosityTheoryState | None = None,
        balance_stress: BalanceStressTest | None = None,
    ) -> Synthesis:
        schema = """
{
  "title": "string",
  "central_question": "string",
  "thesis": "string",
  "argument": ["string with citation labels when source-grounded"],
  "counterarguments": ["string with citation labels when source-grounded"],
  "unresolved_tensions": ["string"],
  "next_directions": ["string"],
  "cited_passages": ["LOCAL or MEMORY citation labels actually used"],
  "cited_web": ["WEB citation labels actually used"]
}
""".strip()
        user = f"""
Input topic:
{topic}

Strongest question:
{strongest_question}

Required strong position:
{position.model_dump() if position else 'No strong position generated.'}

Current curiosity theory state:
{theory_state.model_dump() if theory_state else 'No theory state yet.'}

Balance stress test:
{balance_stress.model_dump() if balance_stress else 'No balance stress test generated.'}

Source gate:
{source_gate.model_dump()}

Grounding notes:
{grounding_notes}

Research brief:
{research_brief}

Textual analyses:
{[a.model_dump() for a in textual_analyses]}

Argument reconstruction:
{reconstruction.model_dump()}

Curiosity score:
{score.model_dump()}

Create the compact synthesis metadata.
Strict rule: if a claim depends on a source, cite it with the exact given label. Do not invent citations.
The thesis must take the strong position seriously; do not flatten it into 'both sides'.
""".strip()
        data = self.llm.json_chat(SYNTHESIZER_SYSTEM, user, schema, temperature=0.28)
        synthesis = Synthesis.model_validate(data)
        synthesis.report_markdown = self._write_deep_report(
            topic, strongest_question, synthesis, grounding_notes, research_brief,
            textual_analyses, reconstruction, score, source_gate, position, theory_state, balance_stress
        )
        return synthesis

    def _write_deep_report(
        self,
        topic: str,
        strongest_question: str,
        synthesis: Synthesis,
        grounding_notes: str,
        research_brief: str,
        textual_analyses: list[TextualAnalysis],
        reconstruction: ArgumentReconstruction,
        score: CuriosityScore,
        source_gate: SourceGateReport,
        position: StrongPosition | None,
        theory_state: CuriosityTheoryState | None,
        balance_stress: BalanceStressTest | None = None,
    ) -> str:
        user = f"""
Write a deep philosophical report in Markdown.

Target length: {self.config.depth.target_synthesis_words} words.
Minimum acceptable length: {self.config.depth.min_synthesis_words} words.
Maximum length: {self.config.depth.max_synthesis_words} words.

Input topic: {topic}
Central question: {strongest_question}
Required strong position: {position.model_dump() if position else 'No strong position available.'}
Current curiosity theory state: {theory_state.model_dump() if theory_state else 'No theory state yet.'}
Balance stress test: {balance_stress.model_dump() if balance_stress else 'No balance stress test generated.'}
Compact synthesis: {synthesis.model_dump()}
Source gate: {source_gate.model_dump()}
Grounding notes: {grounding_notes}
Research brief: {research_brief}
Textual analyses: {[a.model_dump() for a in textual_analyses]}
Argument reconstruction: {reconstruction.model_dump()}
Curiosity score: {score.model_dump()}

Required sections:
# {synthesis.title or 'Deep Philosophical Synthesis'}
## 1. Problem Framing
## 2. Textual Evidence and Close Reading
## 3. Conceptual Distinction
## 4. Real Philosophical Move
## 5. Reconstructed Argument
## 5a. Balance Stress Test (if balance/tempering is used)
## 6. Strong Objection
## 7. Reply
## 8. What This Changes About Curiosity
## 9. Unresolved Tensions
## 10. Next Research Directions

Rules:
- Use exact citation labels already provided.
- Do not invent new source labels.
- Do not rely on memory as if it were primary evidence.
- Mark limited grounding explicitly if source_gate decision is limited or block.
- Take a defensible position; do not hide behind 'could/might/may' except to mark a textual limitation.
- Include at least one real philosophical move: distinction, paradox, contradiction, dilemma, reversal, genealogy, hidden assumption, objection/reply, or criterion.
- Explicitly answer: what new property of curiosity/inquiry/doubt/wonder/learning was discovered?
- If the thesis uses balance/tempered/guided curiosity, explicitly answer the Balance Stress Test: what is balanced, who/what decides, criterion for virtue, criterion for danger, failure modes.
""".strip()
        return self.llm.chat(SYNTHESIZER_SYSTEM, user, temperature=0.38)

    def revise(self, synthesis: Synthesis, critique: ArgumentCritique, source_gate: SourceGateReport, position: StrongPosition | None = None, depth: PhilosophicalDepthJudgment | None = None, move: PhilosophicalMove | None = None, balance_stress: BalanceStressTest | None = None) -> Synthesis:
        if critique.verdict == "accept" and (not depth or depth.verdict == "pass") and (not move or move.verdict == "pass"):
            return synthesis
        user = f"""
Revise the following Markdown report according to the critiques.
Keep the same exact source citation labels. Do not invent new sources.
If the critique says grounding is limited, make that limitation explicit.
The revision must be more argumentative, less hedged, and must take the required position.

Required position:
{position.model_dump() if position else 'No strong position available.'}

Source gate:
{source_gate.model_dump()}

Argument critique:
{critique.model_dump()}

Philosophical depth judgment:
{depth.model_dump() if depth else 'No depth judgment yet.'}

Philosophical move judgment:
{move.model_dump() if move else 'No philosophical move judgment yet.'}

Balance stress test:
{balance_stress.model_dump() if balance_stress else 'No balance stress test generated.'}

Original report:
{synthesis.report_markdown}

Return only the revised Markdown report.
""".strip()
        revised = self.llm.chat(SYNTHESIZER_SYSTEM, user, temperature=0.30)
        synthesis.report_markdown = revised
        return synthesis


class ArgumentCriticAgent:
    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def critique(self, synthesis: Synthesis, source_gate: SourceGateReport, analyses: list[TextualAnalysis], reconstruction: ArgumentReconstruction, position: StrongPosition | None = None) -> ArgumentCritique:
        schema = """
{
  "verdict": "accept|revise|block",
  "major_objections": ["string"],
  "citation_problems": ["string"],
  "logical_gaps": ["string"],
  "anachronism_warnings": ["string"],
  "revision_instructions": ["string"]
}
""".strip()
        user = f"""
Source gate:
{source_gate.model_dump()}

Required strong position:
{position.model_dump() if position else 'No strong position generated.'}

Textual analyses:
{[a.model_dump() for a in analyses]}

Argument reconstruction:
{reconstruction.model_dump()}

Synthesis to critique:
{synthesis.model_dump()}

Attack the synthesis. Require revision if it is too short, too generic, under-cited, memory-only, anachronistic, excessively hedged, or if it refuses to take a position.
""".strip()
        data = self.llm.json_chat(ARGUMENT_CRITIC_SYSTEM, user, schema, temperature=0.10)
        return ArgumentCritique.model_validate(data)


class PhilosophicalDepthJudgeAgent:
    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def judge(
        self,
        topic: str,
        strongest_question: str,
        synthesis: Synthesis,
        source_gate: SourceGateReport,
        analyses: list[TextualAnalysis],
        reconstruction: ArgumentReconstruction,
        argument_critique: ArgumentCritique | None,
        position: StrongPosition | None = None,
    ) -> PhilosophicalDepthJudgment:
        schema = """
{
  "concept_definition": 0.0,
  "anachronism_control": 0.0,
  "genuine_tension": 0.0,
  "counterargument_strength": 0.0,
  "source_interpretation": 0.0,
  "curiosity_theme_progress": 0.0,
  "final_score": 0.0,
  "verdict": "pass|revise|block",
  "main_findings": ["string"],
  "depth_failures": ["string"],
  "required_revisions": ["string"],
  "curiosity_progress_note": "string"
}
""".strip()
        user = f"""
Original topic:
{topic}

Strongest question:
{strongest_question}

Required strong position:
{position.model_dump() if position else 'No strong position generated.'}

Source gate:
{source_gate.model_dump()}

Textual analyses:
{[a.model_dump() for a in analyses]}

Argument reconstruction:
{reconstruction.model_dump()}

Argument critic:
{argument_critique.model_dump() if argument_critique else 'No argument critique recorded.'}

Synthesis to judge:
{synthesis.model_dump()}

Judge philosophical depth strictly.
The key issue: did this report deepen the system's central curiosity project, or did it drift into ordinary philosopher comparison?
If the curiosity theme is not explicitly advanced, curiosity_theme_progress must be <= 0.45.
If the report lacks a strong counterargument, counterargument_strength must be <= 0.50.
If the report uses philosopher names without interpreting specific passages, source_interpretation must be <= 0.55.
If the report refuses to take a position despite the position requirement, final_score must be <= 0.60.
Verdict guidance:
- pass only if final_score >= 0.76 and curiosity_theme_progress >= 0.70
- revise if final_score is 0.45-0.75
- block if final_score < 0.45 or source grounding is misleading
""".strip()
        data = self.llm.json_chat(PHILOSOPHICAL_DEPTH_JUDGE_SYSTEM, user, schema, temperature=0.10)
        return PhilosophicalDepthJudgment.model_validate(data)


class PhilosophicalMoveDetectorAgent:
    def __init__(self, config: AppConfig, llm: OllamaClient):
        self.config = config
        self.llm = llm

    def detect(self, topic: str, question: str, synthesis: Synthesis, position: StrongPosition | None, depth: PhilosophicalDepthJudgment | None) -> PhilosophicalMove:
        schema = """
{
  "move_type": "distinction|paradox|contradiction|dilemma|reversal|genealogy|hidden_assumption|objection_reply|criterion|none",
  "move_statement": "string",
  "concept_definition": "string",
  "paradox_or_tension": "string",
  "source_support": ["citation labels"],
  "why_it_advances_curiosity": "string",
  "hedge_count": 0,
  "hedge_penalty": 0.0,
  "unsupported_inference_penalty": 0.0,
  "final_quality_score": 0.0,
  "verdict": "pass|revise|block",
  "required_revision": "string"
}
""".strip()
        user = f"""
Topic:
{topic}

Question:
{question}

Required position:
{position.model_dump() if position else 'No position.'}

Depth judgment:
{depth.model_dump() if depth else 'No depth judgment yet.'}

Hedge words to penalize:
{self.config.aggressive_mode.hedge_words}

Synthesis:
{synthesis.model_dump()}

Detect the real philosophical move and hedging weakness.
Rules:
- If there is no clear distinction/paradox/contradiction/dilemma/reversal/genealogy/hidden assumption/criterion, move_type must be none and verdict must be revise or block.
- If the report does not explain how the move advances curiosity/inquiry/doubt/wonder/learning, curiosity progress is inadequate and verdict must be revise.
- Penalize excessive hedging, but do not penalize explicit textual caution.
""".strip()
        data = self.llm.json_chat(PHILOSOPHICAL_MOVE_DETECTOR_SYSTEM, user, schema, temperature=0.08)
        move = PhilosophicalMove.model_validate(data)
        # Deterministic hedge check supplements the LLM judgment.
        report = (synthesis.report_markdown or "").lower()
        count = sum(report.count(h.lower()) for h in self.config.aggressive_mode.hedge_words)
        words = max(1, len(report.split()))
        ratio = count / words
        move.hedge_count = max(move.hedge_count, count)
        move.hedge_penalty = max(move.hedge_penalty, min(1.0, ratio / max(0.001, self.config.aggressive_mode.max_hedge_ratio)))
        if move.hedge_penalty > 0.80 and move.verdict == "pass":
            move.verdict = "revise"
            move.required_revision = (move.required_revision + " Excessive hedging detected; strengthen the thesis while preserving textual limits.").strip()
        return move


class CuriosityTheoryTrackerAgent:
    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def update(self, topic: str, synthesis: Synthesis, previous_state: CuriosityTheoryState | None, move: PhilosophicalMove | None, depth: PhilosophicalDepthJudgment | None) -> CuriosityTheoryState:
        schema = """
{
  "current_definition": "string",
  "properties_discovered": ["string"],
  "unresolved_tensions": ["string"],
  "latest_update": "string",
  "new_property_this_iteration": "string",
  "next_required_move": "string",
  "confidence": 0.0,
  "history": ["string"]
}
""".strip()
        user = f"""
Topic:
{topic}

Previous curiosity theory state:
{previous_state.model_dump() if previous_state else 'No previous state.'}

Philosophical move:
{move.model_dump() if move else 'No move judgment.'}

Depth judgment:
{depth.model_dump() if depth else 'No depth judgment.'}

Synthesis:
{synthesis.model_dump()}

Update the cumulative theory of curiosity.
Rules:
- Extract one new property of curiosity only if the iteration genuinely added one.
- Preserve unresolved tensions.
- Do not let the theory drift into generic virtue ethics unless it clarifies curiosity/inquiry/doubt/wonder/learning.
""".strip()
        data = self.llm.json_chat(CURIOSITY_THEORY_TRACKER_SYSTEM, user, schema, temperature=0.16)
        return CuriosityTheoryState.model_validate(data)


class EvaluatorAgent:
    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def evaluate(
        self,
        synthesis: Synthesis,
        score: CuriosityScore,
        strict_citations: bool,
        source_gate: SourceGateReport,
        argument_critique: ArgumentCritique | None,
        philosophical_depth: PhilosophicalDepthJudgment | None = None,
        philosophical_move: PhilosophicalMove | None = None,
        position: StrongPosition | None = None,
    ) -> Evaluation:
        schema = """
{
  "score": 0.0,
  "strengths": ["string"],
  "weaknesses": ["string"],
  "citation_issues": ["string"],
  "should_publish": false,
  "should_continue": true,
  "recommended_revision": "string"
}
""".strip()
        user = f"""
Synthesis:
{synthesis.model_dump()}

Required position:
{position.model_dump() if position else 'No required position.'}

Heuristic curiosity score:
{score.model_dump()}

Source gate:
{source_gate.model_dump()}

Argument critique:
{argument_critique.model_dump() if argument_critique else 'No argument critique recorded.'}

Philosophical depth judgment:
{philosophical_depth.model_dump() if philosophical_depth else 'No philosophical depth judgment recorded.'}

Philosophical move judgment:
{philosophical_move.model_dump() if philosophical_move else 'No philosophical move judgment recorded.'}

Strict citation mode: {strict_citations}

Evaluate whether this iteration produced genuine progress.
Publishing means: queue for human approval, NOT automatic public posting.
Penalize short, generic, under-grounded, memory-only work, weak counterarguments, anachronism, superficial philosopher-name comparison, drift away from the curiosity theme, excessive hedging, or failure to take a position.
If philosophical_depth.final_score or philosophical_move.final_quality_score is low, your evaluator score must also be low.
""".strip()
        data = self.llm.json_chat(EVALUATOR_SYSTEM, user, schema, temperature=0.10)
        return Evaluation.model_validate(data)


class PlanningAgent:
    def __init__(self, config: AppConfig, llm: OllamaClient):
        self.config = config
        self.llm = llm

    def plan_next(
        self,
        current_topic: str,
        synthesis: Synthesis,
        critique: Critique,
        score: CuriosityScore,
        recent_memory: str,
        source_gate: SourceGateReport,
        argument_critique: ArgumentCritique | None,
        philosophical_depth: PhilosophicalDepthJudgment | None = None,
        philosophical_move: PhilosophicalMove | None = None,
        theory_state: CuriosityTheoryState | None = None,
    ) -> PlanningDecision:
        n = self.config.loop.next_topics_per_iteration
        schema = """
{
  "chosen_next_topic": "string",
  "alternative_topics": ["string"],
  "reason": "string",
  "exploit_or_explore": "exploit|explore|bridge|stop",
  "expected_value": 0.0
}
""".strip()
        user = f"""
Current topic:
{current_topic}

Strongest question:
{critique.strongest_question}

Synthesis:
{synthesis.model_dump()}

Curiosity score:
{score.model_dump()}

Source gate:
{source_gate.model_dump()}

Argument critique:
{argument_critique.model_dump() if argument_critique else 'No argument critique recorded.'}

Philosophical depth judgment:
{philosophical_depth.model_dump() if philosophical_depth else 'No philosophical depth judgment recorded.'}

Philosophical move judgment:
{philosophical_move.model_dump() if philosophical_move else 'No philosophical move judgment recorded.'}

Curiosity theory state:
{theory_state.model_dump() if theory_state else 'No theory state yet.'}

Recent memory:
{recent_memory}

Choose the next topic. Provide {n} alternatives. Avoid repetition.
The chosen topic should be a specific research direction, not a broad field label.
Prefer a next topic that can retrieve concrete primary passages. If the current iteration was under-grounded or philosophically shallow, choose a topic that narrows to a source, passage, concept, or objection. Preserve the core curiosity project; avoid drifting into generic virtue ethics unless it directly explains curiosity, inquiry, wonder, doubt, or learning.
If the current iteration failed depth/move standards, select a revision-focused next topic rather than drifting.
""".strip()
        temp = self.config.agents.planner.exploration_temperature or 0.25
        data = self.llm.json_chat(PLANNER_SYSTEM, user, schema, temperature=temp)
        return PlanningDecision.model_validate(data)


class PublisherAgent:
    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def prepare_public_post(self, synthesis: Synthesis, evaluation: Evaluation) -> dict[str, str]:
        schema = """
{
  "title": "string",
  "body_markdown": "string"
}
""".strip()
        user = f"""
Internal synthesis:
{synthesis.model_dump()}

Evaluation:
{evaluation.model_dump()}

Prepare a concise public-facing post for human approval.
Constraints:
- 300-700 words.
- Keep citations already present.
- Do not claim consciousness or human-like understanding.
- Frame it as an experimental philosophical research note.
""".strip()
        return self.llm.json_chat(PUBLISHER_SYSTEM, user, schema, temperature=0.3)


def compact_recent_memory(records: list[dict[str, Any]]) -> str:
    if not records:
        return "No prior memory in this run/system."
    parts = []
    for item in records[-8:]:
        topic = item.get("input_topic", "")
        next_topic = item.get("next_topic", "")
        score = ((item.get("curiosity_score") or {}).get("total"))
        depth = ((item.get("philosophical_depth") or {}).get("final_score"))
        move = ((item.get("philosophical_move") or {}).get("move_statement"))
        synth = item.get("synthesis") or {}
        thesis = synth.get("thesis", "")
        parts.append(f"- topic={topic}; curiosity_score={score}; depth={depth}; thesis={thesis}; move={move}; next={next_topic}")
    return "\n".join(parts)


def prepare_grounding_text(passages, web_results) -> tuple[str, str]:
    return format_passages(passages), format_web_results(web_results)
